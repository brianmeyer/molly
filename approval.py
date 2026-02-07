import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import config

log = logging.getLogger(__name__)

# Tag pattern for prompt-level approval (belt-and-suspenders with can_use_tool)
APPROVAL_TAG_RE = re.compile(
    r"\[APPROVAL_REQUIRED:\s*([^|]+?)\s*\|\s*(.+?)\s*\]",
    re.DOTALL,
)

YES_WORDS = frozenset({
    "yes", "y", "approve", "approved", "go", "proceed", "ok", "do it",
    "send", "yep", "yea", "yeah",
})
NO_WORDS = frozenset({
    "no", "n", "deny", "denied", "cancel", "stop", "don't", "dont",
    "nope", "nah",
})


# ---------------------------------------------------------------------------
# Action tier classification
# ---------------------------------------------------------------------------

def get_action_tier(tool_name: str) -> str:
    """Classify a tool into AUTO, CONFIRM, or BLOCKED.

    Unknown tools default to BLOCKED for safety.
    """
    for tier in ("AUTO", "CONFIRM", "BLOCKED"):
        if tool_name in config.ACTION_TIERS.get(tier, set()):
            return tier
    # Unknown tools are blocked by default
    return "BLOCKED"


def is_auto_approved_path(tool_name: str, tool_input: dict) -> bool:
    """Check if a CONFIRM-tier file operation targets an auto-approved path.

    Writes/edits to workspace/memory/ (daily logs, deep knowledge files)
    are auto-approved so Molly can manage her own memory without interrupting Brian.
    """
    if tool_name not in ("Write", "Edit"):
        return False

    file_path = tool_input.get("file_path", "")
    workspace = str(config.WORKSPACE)

    for safe_suffix in config.AUTO_APPROVE_PATHS:
        safe_path = f"{workspace}/{safe_suffix}"
        if file_path.startswith(safe_path):
            return True

    return False


def _log_approval_decision(
    tool_name: str,
    decision: str,
    response_time_s: float = 0.0,
):
    """Log an approval decision to operational memory (best-effort)."""
    try:
        from memory.retriever import get_vectorstore
        vs = get_vectorstore()
        vs.log_tool_call(
            tool_name=f"approval:{tool_name}",
            parameters=json.dumps({"decision": decision}),
            success=(decision == "approved"),
            latency_ms=int(response_time_s * 1000),
            error_message="" if decision == "approved" else decision,
        )
    except Exception:
        log.debug("Failed to log approval decision: %s", tool_name, exc_info=True)


def format_approval_message(tool_name: str, tool_input: dict) -> str:
    """Format a human-readable approval message for WhatsApp."""
    lines = ["Approval needed\n"]
    lines.append(f"Action: {tool_name}")

    if tool_name == "Write":
        lines.append(f"File: {tool_input.get('file_path', 'unknown')}")
        content = tool_input.get("content", "")
        preview = content[:200] + "..." if len(content) > 200 else content
        lines.append(f"Preview: {preview}")

    elif tool_name == "Edit":
        lines.append(f"File: {tool_input.get('file_path', 'unknown')}")
        old = tool_input.get("old_string", "")[:100]
        new = tool_input.get("new_string", "")[:100]
        lines.append(f"Replacing: {old}")
        lines.append(f"With: {new}")

    elif tool_name in ("gmail_send", "gmail_reply"):
        lines.append(f"To: {tool_input.get('to', 'unknown')}")
        lines.append(f"Subject: \"{tool_input.get('subject', '')}\"")
        body = tool_input.get("body", "")
        preview = body[:200] + "..." if len(body) > 200 else body
        lines.append(f"Preview: \"{preview}\"")

    elif tool_name == "gmail_draft":
        lines.append(f"To: {tool_input.get('to', 'unknown')}")
        lines.append(f"Subject: \"{tool_input.get('subject', '')}\"")

    elif tool_name in ("calendar_create", "calendar_update"):
        lines.append(f"Event: {tool_input.get('title', tool_input.get('summary', 'unknown'))}")
        lines.append(f"When: {tool_input.get('start', 'unknown')}")
        attendees = tool_input.get("attendees")
        if attendees:
            lines.append(f"Attendees: {', '.join(attendees)}")

    elif tool_name == "calendar_delete":
        lines.append(f"Event: {tool_input.get('event_id', 'unknown')}")

    else:
        # Generic format for other CONFIRM tools
        for key, value in list(tool_input.items())[:5]:
            val_str = str(value)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            lines.append(f"{key}: {val_str}")

    lines.append("\nReply YES to proceed, NO to cancel, or EDIT: [changes]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pending approval data
# ---------------------------------------------------------------------------

@dataclass
class PendingApproval:
    id: str
    category: str
    description: str
    chat_jid: str
    session_id: str | None
    future: asyncio.Future
    created_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Approval manager
# ---------------------------------------------------------------------------

class ApprovalManager:
    """Manages approval requests sent via WhatsApp and resolved by yes/no replies."""

    def __init__(self):
        self._pending: dict[str, PendingApproval] = {}  # chat_jid -> approval

    # --- Tag detection (prompt-level fallback) ---

    @staticmethod
    def find_approval_tag(text: str) -> tuple[str, str] | None:
        """If text contains [APPROVAL_REQUIRED: cat | desc], return (cat, desc)."""
        match = APPROVAL_TAG_RE.search(text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None

    @staticmethod
    def strip_approval_tag(text: str) -> str:
        """Remove the [APPROVAL_REQUIRED: ...] tag from text."""
        return APPROVAL_TAG_RE.sub("", text).strip()

    # --- Tool-level approval (code-enforced via can_use_tool) ---

    async def request_tool_approval(
        self,
        tool_name: str,
        tool_input: dict,
        chat_jid: str,
        molly,
    ) -> bool:
        """Send an approval request on WhatsApp for a tool call and wait.

        Returns True if approved, False if denied or timed out.
        """
        # Cancel any stale pending approval for this chat
        self._cancel_pending(chat_jid)

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        description = format_approval_message(tool_name, tool_input)
        approval = PendingApproval(
            id=str(uuid.uuid4()),
            category=tool_name,
            description=description,
            chat_jid=chat_jid,
            session_id=None,
            future=future,
        )
        self._pending[chat_jid] = approval

        # Send the structured approval message via WhatsApp
        molly._track_send(molly.wa.send_message(chat_jid, description))
        log.info("Tool approval requested [%s] in %s", tool_name, chat_jid)

        # Wait for yes/no/edit or timeout
        t0 = asyncio.get_event_loop().time()
        try:
            result = await asyncio.wait_for(
                future, timeout=config.APPROVAL_TIMEOUT
            )
            elapsed = asyncio.get_event_loop().time() - t0
            if result is True:
                _log_approval_decision(tool_name, "approved", elapsed)
                return True
            elif result is False:
                _log_approval_decision(tool_name, "denied", elapsed)
                return False
            elif isinstance(result, tuple) and result[0] == "edit":
                # Edit request — deny this call, agent will see the edit instruction
                # and retry with modified parameters
                _log_approval_decision(tool_name, "edited", elapsed)
                return False
            _log_approval_decision(tool_name, "denied", elapsed)
            return False
        except asyncio.TimeoutError:
            self._pending.pop(chat_jid, None)
            elapsed = asyncio.get_event_loop().time() - t0
            _log_approval_decision(tool_name, "timed_out", elapsed)
            molly._track_send(
                molly.wa.send_message(
                    chat_jid,
                    f"Approval timed out for: {tool_name}",
                )
            )
            log.info("Tool approval timed out: %s", tool_name)
            return False

    # --- Tag-based approval (prompt-level fallback) ---

    async def request(
        self,
        category: str,
        description: str,
        chat_jid: str,
        session_id: str | None,
        molly,
    ) -> bool:
        """Send a tag-based approval request and wait for yes/no.

        Returns True if approved, False if denied or timed out.
        """
        self._cancel_pending(chat_jid)

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        approval = PendingApproval(
            id=str(uuid.uuid4()),
            category=category,
            description=description,
            chat_jid=chat_jid,
            session_id=session_id,
            future=future,
        )
        self._pending[chat_jid] = approval

        msg = (
            f"Approval needed\n\n"
            f"{description}\n\n"
            f"Category: {category}\n"
            f"Reply YES to proceed, NO to cancel, or EDIT: [changes]"
        )
        molly._track_send(molly.wa.send_message(chat_jid, msg))
        log.info("Tag approval requested [%s]: %s", category, description)

        try:
            result = await asyncio.wait_for(
                future, timeout=config.APPROVAL_TIMEOUT
            )
            if result is True:
                return True
            return False
        except asyncio.TimeoutError:
            self._pending.pop(chat_jid, None)
            molly._track_send(
                molly.wa.send_message(
                    chat_jid, f"Approval timed out for: {description}"
                )
            )
            log.info("Tag approval timed out: %s", description)
            return False

    # --- Resolution ---

    def try_resolve(self, text: str, chat_jid: str) -> bool:
        """Check if an incoming message is a yes/no/edit for a pending approval.

        Returns True if the message was consumed as an approval response.
        """
        if chat_jid not in self._pending:
            return False

        normalized = text.strip().lower()

        if normalized in YES_WORDS:
            approval = self._pending.pop(chat_jid)
            if not approval.future.done():
                approval.future.set_result(True)
            log.info("Approval GRANTED: %s", approval.category)
            return True

        if normalized in NO_WORDS:
            approval = self._pending.pop(chat_jid)
            if not approval.future.done():
                approval.future.set_result(False)
            log.info("Approval DENIED: %s", approval.category)
            return True

        if normalized.startswith("edit:"):
            edit_instruction = text.strip()[5:].strip()
            approval = self._pending.pop(chat_jid)
            if not approval.future.done():
                approval.future.set_result(("edit", edit_instruction))
            log.info("Approval EDIT: %s → %s", approval.category, edit_instruction)
            return True

        return False

    # --- Query ---

    def has_pending(self, chat_jid: str) -> bool:
        return chat_jid in self._pending

    def get_pending(self, chat_jid: str) -> PendingApproval | None:
        return self._pending.get(chat_jid)

    def get_all_pending(self) -> list[PendingApproval]:
        return list(self._pending.values())

    def _cancel_pending(self, chat_jid: str):
        old = self._pending.pop(chat_jid, None)
        if old and not old.future.done():
            old.future.set_result(False)
