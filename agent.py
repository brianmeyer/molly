import asyncio
import logging
from datetime import date, timedelta

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

import config
from memory.processor import process_conversation
from memory.retriever import retrieve_context

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


async def handle_message(
    user_message: str,
    chat_id: str,
    session_id: str | None = None,
) -> tuple[str, str | None]:
    """Send a message through Claude and return (response_text, new_session_id)."""
    system_prompt = load_identity_stack()

    # Layer 2: semantic memory retrieval
    memory_context = retrieve_context(user_message)
    if memory_context:
        system_prompt += "\n\n---\n\n" + memory_context

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        model=config.CLAUDE_MODEL,
        permission_mode="bypassPermissions",
        allowed_tools=config.ALLOWED_TOOLS,
        cwd=str(config.WORKSPACE),
    )
    if session_id:
        options.resume = session_id

    response_text = ""
    new_session_id = None

    try:
        async for message in query(prompt=user_message, options=options):
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
