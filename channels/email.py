"""Email channel adapter for Molly (Phase 5B).

Wraps the email processing from heartbeat.py (_check_email) into the
Channel abstraction.  Emails are processed via triage + batch embedding;
responses (if any) route through the owner's WhatsApp DM.
"""

from __future__ import annotations

import logging
from typing import Any

import config
from channels.base import Channel, InboundMessage, OutboundMessage, registry

log = logging.getLogger(__name__)


class EmailChannel(Channel):
    """Normalize email messages from Gmail API."""

    name = "email"

    # ------------------------------------------------------------------
    # Inbound normalization
    # ------------------------------------------------------------------

    def normalize_inbound(self, raw_data: dict, **kwargs) -> InboundMessage:
        """Convert email data to InboundMessage.

        Expected raw_data keys:
            subject, sender, sender_name, body, timestamp, message_id,
            thread_id, labels, snippet
        """
        subject = raw_data.get("subject", "")
        body = raw_data.get("body", "")
        sender = raw_data.get("sender", "")
        sender_name = raw_data.get("sender_name", sender)

        # Combine subject + body for processing
        text = f"Subject: {subject}\n\n{body}" if subject else body

        return InboundMessage(
            text=text,
            chat_id=f"email:{raw_data.get('thread_id', raw_data.get('message_id', sender))}",
            sender_id=sender,
            sender_name=sender_name,
            timestamp=raw_data.get("timestamp", ""),
            source="email",
            msg_id=raw_data.get("message_id", ""),
            is_owner=False,
            chat_mode="listen",  # Emails are triaged, not directly responded to
            raw=raw_data,
        )

    # ------------------------------------------------------------------
    # Outbound formatting
    # ------------------------------------------------------------------

    def format_outbound(self, message: OutboundMessage) -> str:
        """Email responses are routed to owner as WhatsApp notifications."""
        return message.text

    async def send(self, message: OutboundMessage, transport: Any = None) -> str | list[str] | None:
        """Route email notification to owner via WhatsApp.

        ``transport`` should be the molly instance.
        """
        if transport is None:
            log.warning("Email send called without transport")
            return None

        send_surface = getattr(transport, "send_surface_message", None)
        if not send_surface:
            return None

        owner_jid_fn = getattr(transport, "_get_owner_dm_jid", None)
        owner_jid = owner_jid_fn() if owner_jid_fn else ""
        if not owner_jid:
            return None

        send_surface(
            chat_jid=owner_jid,
            text=message.text,
            source="email",
            surfaced_summary=message.metadata.get("surfaced_summary", ""),
            sender_pattern=message.metadata.get("sender_pattern", ""),
        )
        return None

    # ------------------------------------------------------------------
    # Session & source helpers
    # ------------------------------------------------------------------

    def get_session_key(self, inbound: InboundMessage) -> str:
        return inbound.chat_id

    def get_source_label(self) -> str:
        return "email"

    def get_response_guidance(self) -> str | None:
        return (
            "Format for WhatsApp: use plain text, no markdown tables, "
            "no code blocks. Keep responses concise."
        )


# Auto-register
registry.register(EmailChannel())
