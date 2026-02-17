"""Web/WebSocket channel adapter for Molly (Phase 5B).

Wraps the FastAPI WebSocket chat endpoint from ``web.py`` into the
Channel abstraction.  The web channel accepts full markdown without
the WhatsApp formatting restrictions.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

import config
from channels.base import Channel, InboundMessage, OutboundMessage, registry

log = logging.getLogger(__name__)


class WebChannel(Channel):
    """Normalize web/WebSocket messages."""

    name = "web"

    # ------------------------------------------------------------------
    # Inbound normalization
    # ------------------------------------------------------------------

    def normalize_inbound(self, raw_data: dict, **kwargs) -> InboundMessage:
        """Convert a WebSocket message to InboundMessage.

        Expected raw_data keys:
            text   — user message text
            token  — auth token (for session stability)
        """
        text = raw_data.get("text", "").strip()
        token = raw_data.get("token", "")

        # Stable session key from token hash
        if token and config.WEB_AUTH_TOKEN:
            stable_id = hashlib.sha256(token.encode()).hexdigest()[:8]
        else:
            stable_id = raw_data.get("session_id", uuid.uuid4().hex[:8])
        session_key = f"web:{stable_id}"

        return InboundMessage(
            text=text,
            chat_id=session_key,
            sender_id=session_key,
            sender_name="Web User",
            source="web",
            session_id=kwargs.get("session_id"),
            is_owner=True,  # Web UI is owner-only
            chat_mode="owner_dm",
            raw=raw_data,
        )

    # ------------------------------------------------------------------
    # Outbound formatting
    # ------------------------------------------------------------------

    def format_outbound(self, message: OutboundMessage) -> str:
        """Web accepts full markdown — return as-is."""
        return message.text

    async def send(self, message: OutboundMessage, transport: Any = None) -> str | list[str] | None:
        """Send through WebSocket connection.

        ``transport`` should be the WebSocket instance.
        """
        if transport is None:
            log.warning("Web send called without WebSocket transport")
            return None

        try:
            import json
            await transport.send_json({
                "type": "message",
                "text": message.text or "(no response)",
            })
        except Exception:
            log.error("Failed to send web message", exc_info=True)
        return None

    # ------------------------------------------------------------------
    # Session & source helpers
    # ------------------------------------------------------------------

    def get_session_key(self, inbound: InboundMessage) -> str:
        return inbound.chat_id

    def get_source_label(self) -> str:
        return "web"

    def get_response_guidance(self) -> str | None:
        """Web channel accepts full markdown — no restrictions."""
        return None


# Auto-register
registry.register(WebChannel())
