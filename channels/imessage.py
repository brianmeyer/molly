"""iMessage channel adapter for Molly (Phase 5B).

Wraps the iMessage mention detection from heartbeat.py into the
Channel abstraction.  iMessage is read-only (macOS chat.db) — all
responses are routed to the owner via WhatsApp.
"""

from __future__ import annotations

import logging
from typing import Any

import config
from channels.base import Channel, InboundMessage, OutboundMessage, registry

log = logging.getLogger(__name__)


class IMessageChannel(Channel):
    """Normalize iMessage mentions and heartbeat messages."""

    name = "imessage"

    # ------------------------------------------------------------------
    # Inbound normalization
    # ------------------------------------------------------------------

    def normalize_inbound(self, raw_data: dict, **kwargs) -> InboundMessage:
        """Convert iMessage data to InboundMessage.

        For @molly mentions (source="imessage-mention"):
            raw_data keys: text, sender, handle, is_from_me, timestamp,
                           chat_id, thread_context (assembled prompt)

        For heartbeat triage (source="heartbeat"):
            raw_data keys: text, sender, handle, is_from_me, timestamp
        """
        source = kwargs.get("source", "imessage")
        chat_id_raw = raw_data.get("chat_id", "")
        handle = raw_data.get("handle", "")

        # iMessage mentions use a synthetic chat ID for session isolation
        if source == "imessage-mention":
            chat_id = f"imessage:chat:{chat_id_raw}" if chat_id_raw else f"imessage:{handle}"
        else:
            chat_id = f"imessage:{handle}" if handle else f"imessage:unknown"

        return InboundMessage(
            text=raw_data.get("text", ""),
            chat_id=chat_id,
            sender_id=handle or raw_data.get("sender", ""),
            sender_name=raw_data.get("sender", ""),
            timestamp=raw_data.get("timestamp", ""),
            source=source,
            is_from_me=bool(raw_data.get("is_from_me", False)),
            is_owner=True,  # iMessage messages from Brian's Mac are always owner
            handle=handle,
            thread_context=raw_data.get("thread_context"),
            has_trigger=source == "imessage-mention",
            chat_mode="owner_dm",
            raw=raw_data,
        )

    # ------------------------------------------------------------------
    # Outbound formatting — responses go via WhatsApp
    # ------------------------------------------------------------------

    def format_outbound(self, message: OutboundMessage) -> str:
        """iMessage responses are plain text routed through WhatsApp."""
        return message.text

    async def send(self, message: OutboundMessage, transport: Any = None) -> str | list[str] | None:
        """Route response to owner via WhatsApp.

        ``transport`` should be the molly instance (which has .wa + .send_surface_message).
        """
        if transport is None:
            log.warning("iMessage send called without transport (molly instance)")
            return None

        wa = getattr(transport, "wa", None)
        if wa is None:
            log.warning("iMessage send: molly.wa not available")
            return None

        # Find owner DM JID
        owner_jid_fn = getattr(transport, "_get_owner_dm_jid", None)
        if owner_jid_fn:
            owner_jid = owner_jid_fn()
        else:
            owner_jid = ""

        if not owner_jid:
            log.warning("iMessage send: cannot determine owner JID")
            return None

        # Route through the surface message pipeline for preference learning
        send_surface = getattr(transport, "send_surface_message", None)
        if send_surface:
            send_surface(
                chat_jid=owner_jid,
                text=message.text,
                source="imessage",
                surfaced_summary=message.metadata.get("surfaced_summary", ""),
                sender_pattern=message.metadata.get("sender_pattern", "imessage:mention"),
            )
            return None  # send_surface_message tracks its own msg IDs

        # Fallback: direct WhatsApp send
        mid = wa.send_message(owner_jid, message.text)
        return mid

    # ------------------------------------------------------------------
    # Session & source helpers
    # ------------------------------------------------------------------

    def get_session_key(self, inbound: InboundMessage) -> str:
        return inbound.chat_id

    def get_source_label(self) -> str:
        return "imessage-mention"

    def get_response_guidance(self) -> str | None:
        return (
            "Format for WhatsApp: use plain text, no markdown tables, "
            "no code blocks. Keep responses concise."
        )


# Auto-register
registry.register(IMessageChannel())
