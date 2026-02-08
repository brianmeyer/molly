import asyncio
import json
import logging
import time
from datetime import date, timedelta

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    HookMatcher,
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
from tools.grok import grok_server
from tools.kimi import kimi_server
from tools.whatsapp import whatsapp_server

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


def _normalize_tool_name(tool_name: str) -> str:
    """Normalize MCP-prefixed tool names to short names for tier lookup.

    The CLI sends tool names like 'mcp__gmail__gmail_send' for MCP tools
    but our ACTION_TIERS use short names like 'gmail_send'.
    Built-in tools ('Bash', 'Write') are passed through unchanged.
    """
    if tool_name.startswith("mcp__"):
        # mcp__servername__toolname → toolname
        parts = tool_name.split("__")
        if len(parts) >= 3:
            return "__".join(parts[2:])
    return tool_name


def make_tool_checker(approval_manager=None, molly=None, chat_jid: str = ""):
    """Create a can_use_tool callback for code-enforced action tiers.

    AUTO    → PermissionResultAllow (immediate)
    CONFIRM → WhatsApp approval flow → Allow/Deny (or deny if no approval_manager)
    BLOCKED → PermissionResultDeny (immediate)

    This callback only fires for tools NOT in allowed_tools (i.e., CONFIRM and
    BLOCKED tier tools). AUTO-tier tools are pre-approved via allowed_tools and
    never reach this callback.

    This callback MUST always be set so the SDK uses the stdio control protocol
    instead of interactive terminal prompts. When no approval_manager is available
    (heartbeat, terminal), CONFIRM-tier tools are denied with a message.
    """

    async def can_use_tool(tool_name: str, tool_input: dict, _context) -> PermissionResultAllow | PermissionResultDeny:
        short_name = _normalize_tool_name(tool_name)
        tier = get_action_tier(short_name)
        t0 = time.time()
        log.info("can_use_tool fired: %s (normalized: %s) → tier=%s", tool_name, short_name, tier)

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
        if is_auto_approved_path(short_name, tool_input):
            log.info("Auto-approved %s to memory path: %s", tool_name, tool_input.get("file_path", ""))
            _log_tool(tool_name, tool_input, True, t0)
            return PermissionResultAllow()

        # No approval manager (headless/heartbeat) — deny CONFIRM-tier
        if not approval_manager or not molly:
            log.info("No approval manager — denying CONFIRM tool: %s", tool_name)
            _log_tool(tool_name, tool_input, False, t0, "no_approval_manager")
            return PermissionResultDeny(
                message=(
                    f"This action ({tool_name}) requires Brian's approval, "
                    f"but no approval channel is available right now."
                )
            )

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


def _build_agents() -> dict[str, AgentDefinition]:
    """Define sub-agents for intelligent task routing.

    Opus orchestrates and delegates to these agents via the Task tool.
    Sub-agents inherit all available tools (tools=None).
    """
    return {
        "quick": AgentDefinition(
            description="Fast lookups, formatting, trivial subtasks",
            prompt=(
                "You are a fast helper. Handle simple lookups, formatting, "
                "unit conversions, quick calculations, and other mechanical "
                "subtasks. Be concise and direct."
            ),
            model="haiku",
        ),
        "worker": AgentDefinition(
            description="Email drafts, research synthesis, multi-step tool use",
            prompt=(
                "You are a capable worker. Handle substantial tasks like "
                "drafting emails, synthesizing research across multiple sources, "
                "multi-step tool workflows, and data gathering. Be thorough "
                "but efficient."
            ),
            model="sonnet",
        ),
        "analyst": AgentDefinition(
            description="Deep analysis, strategic thinking, complex reasoning",
            prompt=(
                "You are a deep analyst. Handle tasks requiring careful reasoning, "
                "strategic analysis, nuanced judgment, and complex multi-factor "
                "decisions. Take your time and be thorough."
            ),
            model="opus",
        ),
    }


async def _on_subagent_start(input, tool_use_id, context):
    """Log routing decision when a sub-agent is started."""
    agent_type = input.get("agent_type", "unknown") if isinstance(input, dict) else "unknown"
    try:
        from memory.retriever import get_vectorstore
        vs = get_vectorstore()
        vs.log_tool_call(
            f"routing:subagent_start:{agent_type}",
            json.dumps({"agent": agent_type}),
            True, 0, "",
        )
    except Exception:
        log.debug("Failed to log subagent start: %s", agent_type, exc_info=True)
    return {"continue_": True}


async def _on_subagent_stop(input, tool_use_id, context):
    """Log when a sub-agent completes."""
    agent_type = input.get("agent_type", "unknown") if isinstance(input, dict) else "unknown"
    try:
        from memory.retriever import get_vectorstore
        vs = get_vectorstore()
        vs.log_tool_call(
            f"routing:subagent_stop:{agent_type}",
            json.dumps({"agent": agent_type}),
            True, 0, "",
        )
    except Exception:
        log.debug("Failed to log subagent stop: %s", agent_type, exc_info=True)
    return {"continue_": True}


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

    # Only pre-approve AUTO-tier tools. CONFIRM/BLOCKED-tier tools remain
    # available (built-in + MCP) but require permission — which fires the
    # can_use_tool callback for our approval system and tool logging.
    auto_tools = sorted(config.ACTION_TIERS["AUTO"])

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        model=config.CLAUDE_MODEL,
        allowed_tools=auto_tools,
        agents=_build_agents(),
        hooks={
            "SubagentStart": [HookMatcher(hooks=[_on_subagent_start])],
            "SubagentStop": [HookMatcher(hooks=[_on_subagent_stop])],
        },
        mcp_servers={
            "google-calendar": calendar_server,
            "gmail": gmail_server,
            "apple-contacts": contacts_server,
            "imessage": imessage_server,
            "whatsapp-history": whatsapp_server,
            "kimi": kimi_server,
            "grok": grok_server,
        },
        cwd=str(config.WORKSPACE),
        # Always set can_use_tool so the SDK uses the stdio control protocol
        # (no interactive terminal prompts). Approval manager is optional —
        # without it, CONFIRM-tier tools are denied gracefully.
        can_use_tool=make_tool_checker(
            approval_manager, molly_instance, chat_id,
        ),
    )
    if session_id:
        options.resume = session_id

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
        task = asyncio.create_task(
            process_conversation(user_message, response_text, chat_id),
            name=f"post-process:{chat_id[:20]}",
        )
        task.add_done_callback(
            lambda t: log.error("Post-processing failed: %s", t.exception(), exc_info=t.exception())
            if not t.cancelled() and t.exception() else None
        )

    log.info(
        "Response for %s: %d chars, session=%s",
        chat_id,
        len(response_text),
        new_session_id,
    )
    return response_text, new_session_id
