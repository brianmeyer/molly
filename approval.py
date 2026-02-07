import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import config

log = logging.getLogger(__name__)

APPROVAL_TAG_RE = re.compile(
    r"\[APPROVAL_REQUIRED:\s*([^|]+?)\s*\|\s*(.+?)\s*\]",
    re.DOTALL,
)

YES_WORDS = frozenset({
    "yes", "y", "approve", "approved", "go", "proceed", "ok", "do it", "send", "yep", "yea", "yeah",
})
NO_WORDS = frozenset({
    "no", "n", "deny", "denied", "cancel", "stop", "don't", "dont", "nope", "nah",
})


@dataclass
class PendingApproval:
    id: str
    category: str
    description: str
    chat_jid: str
    session_id: str | None
    future: asyncio.Future
    created_at: datetime = field(default_factory=datetime.now)


class ApprovalManager:
    """Manages approval requests sent via WhatsApp and resolved by yes/no replies."""

    def __init__(self):
        self._pending: dict[str, PendingApproval] = {}  # chat_jid → approval

    # --- Detection ---

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

    @staticmethod
    def is_pre_approved(category: str) -> bool:
        return category in config.APPROVED_ACTIONS

    # --- Request / resolve ---

    async def request(
        self,
        category: str,
        description: str,
        chat_jid: str,
        session_id: str | None,
        molly,
    ) -> bool:
        """Send an approval request on WhatsApp and wait for yes/no.

        Returns True if approved, False if denied or timed out.
        """
        if self.is_pre_approved(category):
            log.info("Action pre-approved (category=%s): %s", category, description)
            return True

        # Cancel any stale pending approval for this chat
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

        # Send the approval prompt via WhatsApp
        msg = (
            f"Approval needed\n\n"
            f"{description}\n\n"
            f"Category: {category}\n"
            f"Reply yes or no."
        )
        molly._track_send(molly.wa.send_message(chat_jid, msg))
        log.info("Approval requested [%s]: %s", category, description)

        # Wait for resolution or timeout
        try:
            return await asyncio.wait_for(future, timeout=config.APPROVAL_TIMEOUT)
        except asyncio.TimeoutError:
            self._pending.pop(chat_jid, None)
            molly._track_send(
                molly.wa.send_message(chat_jid, f"Approval timed out for: {description}")
            )
            log.info("Approval timed out: %s", description)
            return False

    def try_resolve(self, text: str, chat_jid: str) -> bool:
        """Check if an incoming message is a yes/no for a pending approval.

        Returns True if the message was consumed as an approval response.
        """
        if chat_jid not in self._pending:
            return False

        normalized = text.strip().lower()

        if normalized in YES_WORDS:
            approval = self._pending.pop(chat_jid)
            if not approval.future.done():
                approval.future.set_result(True)
            log.info("Approval GRANTED: %s", approval.description)
            return True

        if normalized in NO_WORDS:
            approval = self._pending.pop(chat_jid)
            if not approval.future.done():
                approval.future.set_result(False)
            log.info("Approval DENIED: %s", approval.description)
            return True

        return False

    def has_pending(self, chat_jid: str) -> bool:
        return chat_jid in self._pending

    def get_pending(self, chat_jid: str) -> PendingApproval | None:
        return self._pending.get(chat_jid)

    def _cancel_pending(self, chat_jid: str):
        old = self._pending.pop(chat_jid, None)
        if old and not old.future.done():
            old.future.set_result(False)
