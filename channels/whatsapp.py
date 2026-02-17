"""WhatsApp channel adapter for Molly (Phase 5B).

Wraps the existing neonize WhatsAppClient and main.py message handling
into the Channel abstraction.  The actual WhatsApp connection management,
typing indicators, and send-tracking remain in ``whatsapp.py`` and
``main.py`` — this adapter provides normalization + formatting only.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import config
from channels.base import Channel, InboundMessage, OutboundMessage, registry

log = logging.getLogger(__name__)

# WhatsApp plain-text rendering patterns
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)_(.+?)_(?!\*)")
_HEADER_RE = re.compile(r"^#{1,3}\s+", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```\w*\n?")
_TABLE_ROW_RE = re.compile(r"^\|.*\|$", re.MULTILINE)
_TABLE_SEP_RE = re.compile(r"^\|[\s\-:|]+\|$", re.MULTILINE)

# Chunking defaults
_DEFAULT_CHUNK_CHARS = 3500
_SIGNATURE = "\n\n-MollyAI"


class WhatsAppChannel(Channel):
    """Normalize WhatsApp messages and format outputs for WhatsApp."""

    name = "whatsapp"

    # ------------------------------------------------------------------
    # Inbound normalization
    # ------------------------------------------------------------------

    def normalize_inbound(self, raw_data: dict, **kwargs) -> InboundMessage:
        """Convert a WhatsApp msg_data dict into an InboundMessage.

        Expected keys in raw_data (from whatsapp.py callback):
            msg_id, chat_jid, sender_jid, sender_name, content,
            timestamp, is_from_me, is_group
        """
        chat_jid = raw_data.get("chat_jid", "")
        sender_jid = raw_data.get("sender_jid", "")
        content = raw_data.get("content", "")

        # Detect @Molly trigger
        has_trigger = bool(config.TRIGGER_PATTERN.search(content))

        return InboundMessage(
            text=content,
            chat_id=chat_jid,
            sender_id=sender_jid,
            sender_name=raw_data.get("sender_name", ""),
            timestamp=raw_data.get("timestamp", ""),
            source="whatsapp",
            msg_id=raw_data.get("msg_id", ""),
            is_group=bool(raw_data.get("is_group", False)),
            is_from_me=bool(raw_data.get("is_from_me", False)),
            is_owner=kwargs.get("is_owner", False),
            has_trigger=has_trigger,
            chat_mode=kwargs.get("chat_mode", "store_only"),
            raw=raw_data,
        )

    # ------------------------------------------------------------------
    # Outbound formatting
    # ------------------------------------------------------------------

    def format_outbound(self, message: OutboundMessage) -> str:
        """Convert markdown response to WhatsApp-friendly plain text."""
        text = message.text
        if getattr(config, "WHATSAPP_PLAIN_RENDER", True):
            text = self._render_plain(text)
        return text

    def chunk_message(self, text: str) -> list[str]:
        """Split a long message into WhatsApp-sized chunks.

        Last chunk gets the MollyAI signature.
        """
        max_chars = getattr(config, "WHATSAPP_CHUNK_CHARS", _DEFAULT_CHUNK_CHARS)
        if len(text) <= max_chars:
            return [text + _SIGNATURE]

        chunks: list[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= max_chars:
                chunks.append(remaining)
                break

            # Find a good split point (paragraph break, sentence, or space)
            split_at = remaining.rfind("\n\n", 0, max_chars)
            if split_at < max_chars // 2:
                split_at = remaining.rfind(". ", 0, max_chars)
            if split_at < max_chars // 3:
                split_at = remaining.rfind(" ", 0, max_chars)
            if split_at < 1:
                split_at = max_chars

            chunk = remaining[:split_at].rstrip()
            remaining = remaining[split_at:].lstrip()
            if chunk:
                chunks.append(chunk)

        # Append signature to last chunk only
        if chunks:
            chunks[-1] += _SIGNATURE

        return chunks

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    async def send(self, message: OutboundMessage, transport: Any = None) -> str | list[str] | None:
        """Send through the WhatsApp client.

        ``transport`` should be the WhatsAppClient (wa) instance.
        Returns the message ID(s).
        """
        if transport is None:
            log.warning("WhatsApp send called without transport")
            return None

        formatted = self.format_outbound(message)
        chunking = getattr(config, "WHATSAPP_CHUNKING_ENABLED", True)

        if chunking:
            chunks = self.chunk_message(formatted)
        else:
            chunks = [formatted + _SIGNATURE]

        msg_ids: list[str] = []
        failed_chunks = 0
        for i, chunk in enumerate(chunks):
            try:
                mid = transport.send_message(message.chat_id, chunk)
                if mid:
                    if isinstance(mid, list):
                        msg_ids.extend(mid)
                    else:
                        msg_ids.append(mid)
            except Exception:
                failed_chunks += 1
                log.error(
                    "WhatsApp send failed for chunk %d/%d to %s",
                    i + 1, len(chunks), message.chat_id,
                    exc_info=True,
                )

        if failed_chunks:
            log.warning(
                "Partial WhatsApp delivery: %d/%d chunks sent to %s",
                len(chunks) - failed_chunks, len(chunks), message.chat_id,
            )

        return msg_ids if len(msg_ids) != 1 else msg_ids[0] if msg_ids else None

    # ------------------------------------------------------------------
    # Session & source helpers
    # ------------------------------------------------------------------

    def get_session_key(self, inbound: InboundMessage) -> str:
        return inbound.chat_id

    def get_source_label(self) -> str:
        return "whatsapp"

    def get_response_guidance(self) -> str | None:
        """WhatsApp-specific formatting constraints."""
        if not getattr(config, "WHATSAPP_PROMPT_GUARDRAILS", True):
            return None
        return (
            "Format for WhatsApp: use plain text, no markdown tables, "
            "no code blocks. Keep responses concise. Use bullet points "
            "with - instead of * for lists."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _render_plain(text: str) -> str:
        """Convert markdown to WhatsApp-friendly plain text."""
        text = _CODE_BLOCK_RE.sub("", text)
        text = _TABLE_SEP_RE.sub("", text)
        text = _TABLE_ROW_RE.sub(lambda m: m.group(0).replace("|", " ").strip(), text)
        text = _BOLD_RE.sub(r"*\1*", text)      # **bold** → *bold*
        text = _ITALIC_RE.sub(r"_\1_", text)     # _italic_ stays
        text = _HEADER_RE.sub("", text)           # Remove # headers
        # Clean up excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# Auto-register
registry.register(WhatsAppChannel())
