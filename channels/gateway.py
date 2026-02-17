"""Gateway channel adapter for Molly (Phase 5B).

Wraps the cron-scheduled gateway jobs (morning packet, noon digest,
evening wrap) and webhook-triggered events into the Channel abstraction.
The actual scheduling stays in ``gateway.py`` — this adapter provides
normalization for internal messages.
"""

from __future__ import annotations

import logging
from typing import Any

import config
from channels.base import Channel, InboundMessage, OutboundMessage, registry

log = logging.getLogger(__name__)


class GatewayChannel(Channel):
    """Normalize gateway-generated messages (cron + webhooks)."""

    name = "gateway"

    # ------------------------------------------------------------------
    # Inbound normalization
    # ------------------------------------------------------------------

    def normalize_inbound(self, raw_data: dict, **kwargs) -> InboundMessage:
        """Convert a gateway event into an InboundMessage.

        Expected raw_data keys:
            prompt     — the synthesized briefing/digest prompt
            job_id     — automation job identifier
            trigger    — "schedule" or "webhook"
            chat_id    — owner DM JID (always owner)
        """
        return InboundMessage(
            text=raw_data.get("prompt", ""),
            chat_id=raw_data.get("chat_id", "gateway:internal"),
            sender_id="gateway",
            sender_name="Gateway",
            source="gateway",
            is_owner=True,
            trigger_type=raw_data.get("trigger", "schedule"),
            chat_mode="owner_dm",
            raw=raw_data,
        )

    # ------------------------------------------------------------------
    # Outbound formatting
    # ------------------------------------------------------------------

    def format_outbound(self, message: OutboundMessage) -> str:
        """Gateway responses are plain text for WhatsApp delivery."""
        return message.text

    async def send(self, message: OutboundMessage, transport: Any = None) -> str | list[str] | None:
        """Send gateway output to owner via WhatsApp.

        ``transport`` should be the molly instance.
        """
        if transport is None:
            log.warning("Gateway send called without transport")
            return None

        wa = getattr(transport, "wa", None)
        if wa is None:
            return None

        owner_jid_fn = getattr(transport, "_get_owner_dm_jid", None)
        owner_jid = owner_jid_fn() if owner_jid_fn else ""
        if not owner_jid:
            return None

        # Use _track_send for echo avoidance
        track_send = getattr(transport, "_track_send", None)
        mid = wa.send_message(owner_jid, message.text)
        if track_send:
            track_send(mid)
        return mid

    # ------------------------------------------------------------------
    # Session & source helpers
    # ------------------------------------------------------------------

    def get_session_key(self, inbound: InboundMessage) -> str:
        return "gateway:summary"

    def get_source_label(self) -> str:
        return "gateway"

    def get_response_guidance(self) -> str | None:
        return (
            "Format for WhatsApp: use plain text, no markdown tables, "
            "no code blocks. Keep responses concise."
        )


# Auto-register
registry.register(GatewayChannel())
