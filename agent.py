import asyncio
import json
import logging
import time
from datetime import date, timedelta

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    query,
)

import config
from approval import get_action_tier, is_auto_approved_path
from memory.processor import process_conversation
from memory.retriever import retrieve_context
from skills import get_skill_context, match_skills
from tools.calendar import calendar_server
from tools.contacts import contacts_server
from tools.gmail import gmail_server
from tools.imessage import imessage_server

log = logging.getLogger(__name__)


def load_identity_stack() -> str:
    """Read and concatenate all identity files into system prompt."""
    parts = []
    for path in config.IDENTITY_FILES:
        if path.exists():
            parts.append(f"<!-- {path.name} -->\n{path.read_text()}")

    # Add today's and yesterday's daily logs
    today = date.today()
    for d in [today, today - timedelta(days=1)]:
        log_path = config.WORKSPACE / "memory" / f"{d.isoformat()}.md"
        if log_path.exists():
            parts.append(f"<!-- Daily Log: {d.isoformat()} -->\n{log_path.read_text()}")

    return "\n\n---\n\n".join(parts)


def make_tool_checker(approval_manager, molly, chat_jid: str):
    """Create a can_use_tool callback for code-enforced action tiers.

    AUTO    → PermissionResultAllow (immediate)
    CONFIRM → WhatsApp approval flow → Allow/Deny
    BLOCKED → PermissionResultDeny (immediate)
    """

    async def can_use_tool(tool_name: str, tool_input: dict, _context) -> PermissionResultAllow | PermissionResultDeny:
        tier = get_action_tier(tool_name)
        t0 = time.time()

        if tier == "AUTO":
            _log_tool(tool_name, tool_input, True, t0)
            return PermissionResultAllow()

        if tier == "BLOCKED":
            log.warning("BLOCKED tool call: %s", tool_name)
            _log_tool(tool_name, tool_input, False, t0, "BLOCKED")
            return PermissionResultDeny(
                message=(
                    f"This action ({tool_name}) is blocked. "
                    f"It's classified as destructive or irreversible. "
                    f"Brian needs to do this himself."
                )
            )

        # CONFIRM tier
        # Check if this is a file write to an auto-approved path
        if is_auto_approved_path(tool_name, tool_input):
            log.info("Auto-approved %s to memory path: %s", tool_name, tool_input.get("file_path", ""))
            _log_tool(tool_name, tool_input, True, t0)
            return PermissionResultAllow()

        # Request approval via WhatsApp and wait
        log.info("CONFIRM tier — requesting approval for %s", tool_name)
        approved = await approval_manager.request_tool_approval(
            tool_name, tool_input, chat_jid, molly,
        )

        if approved:
            log.info("Tool approved: %s", tool_name)
            _log_tool(tool_name, tool_input, True, t0)
            return PermissionResultAllow()
        else:
            log.info("Tool denied: %s", tool_name)
            _log_tool(tool_name, tool_input, False, t0, "denied/timed out")
            return PermissionResultDeny(
                message="Brian denied this action or it timed out."
            )

    return can_use_tool


def _log_tool(tool_name: str, tool_input: dict, success: bool, t0: float, error: str = ""):
    """Log a tool call to operational memory (best-effort)."""
    try:
        from memory.retriever import get_vectorstore
        vs = get_vectorstore()
        params = json.dumps(tool_input, default=str)[:500]
        latency_ms = int((time.time() - t0) * 1000)
        vs.log_tool_call(tool_name, params, success, latency_ms, error)
    except Exception:
        log.debug("Failed to log tool call: %s", tool_name, exc_info=True)


async def _prompt_stream(text: str):
    """Wrap a string prompt as an async iterable for MCP server support.

    The SDK's string-prompt path calls end_input() immediately after writing,
    which closes stdin before MCP tool registration handshake messages can be
    processed. The async-iterable path uses stream_input() instead, which
    detects sdk_mcp_servers and waits for the first result before closing stdin.
    """
    yield {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": text},
        "parent_tool_use_id": None,
    }


async def handle_message(
    user_message: str,
    chat_id: str,
    session_id: str | None = None,
    approval_manager=None,
    molly_instance=None,
) -> tuple[str, str | None]:
    """Send a message through Claude and return (response_text, new_session_id)."""
    system_prompt = load_identity_stack()

    # Layer 2: semantic memory retrieval
    memory_context = retrieve_context(user_message)
    if memory_context:
        system_prompt += "\n\n---\n\n" + memory_context

    # Phase 3D: skill loading — match message against skill triggers
    matched_skills = match_skills(user_message)
    skill_context = get_skill_context(matched_skills)
    if skill_context:
        system_prompt += "\n\n---\n\n" + skill_context

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        model=config.CLAUDE_MODEL,
        permission_mode="bypassPermissions",
        allowed_tools=config.ALLOWED_TOOLS,
        mcp_servers={
            "google-calendar": calendar_server,
            "gmail": gmail_server,
            "apple-contacts": contacts_server,
            "imessage": imessage_server,
        },
        cwd=str(config.WORKSPACE),
    )
    if session_id:
        options.resume = session_id

    # Wire in code-enforced tool interception
    if approval_manager and molly_instance:
        options.can_use_tool = make_tool_checker(
            approval_manager, molly_instance, chat_id,
        )

    response_text = ""
    new_session_id = None

    try:
        # Use async iterable prompt instead of string to go through stream_input(),
        # which keeps stdin open for MCP server bidirectional communication.
        async for message in query(prompt=_prompt_stream(user_message), options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text
            elif isinstance(message, ResultMessage):
                new_session_id = message.session_id
                if message.is_error:
                    log.error("Claude returned error: %s", message.result)
    except Exception:
        log.error("Claude query failed for chat %s", chat_id, exc_info=True)
        if not response_text:
            response_text = "Something went wrong on my end. Try again in a moment."

    # Async post-processing: embed + store conversation turn
    if response_text and not response_text.startswith("Something went wrong"):
        asyncio.create_task(
            process_conversation(user_message, response_text, chat_id)
        )

    log.info(
        "Response for %s: %d chars, session=%s",
        chat_id,
        len(response_text),
        new_session_id,
    )
    return response_text, new_session_id
