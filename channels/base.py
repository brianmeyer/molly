"""Base classes and registry for the channel abstraction layer.

InboundMessage normalizes data from any source (WhatsApp, iMessage, web,
gateway, email) into a single structure.  OutboundMessage carries the
response back through the originating channel with format-specific
rendering.

Channel implementations register themselves into the global
ChannelRegistry so that main.py can dispatch generically.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inbound message dataclass — the universal representation
# ---------------------------------------------------------------------------

@dataclass
class InboundMessage:
    """Normalized inbound message from any channel."""

    text: str
    chat_id: str
    sender_id: str
    sender_name: str = ""
    timestamp: str = ""           # ISO-8601
    source: str = "unknown"       # whatsapp | imessage | web | gateway | email | heartbeat
    session_id: str | None = None
    received_at: float = field(default_factory=time.time)

    # Channel-specific optional fields
    msg_id: str = ""              # WhatsApp message ID (for echo tracking)
    is_group: bool = False        # WhatsApp group flag
    is_from_me: bool = False      # Message sent by Molly/Brian
    is_owner: bool = False        # Sender is Brian
    thread_context: str | None = None   # iMessage preceding messages
    handle: str | None = None     # iMessage contact handle
    trigger_type: str = ""        # gateway: schedule | webhook
    has_trigger: bool = False     # @Molly mention detected
    attachments: list[dict] = field(default_factory=list)

    # Processing metadata
    chat_mode: str = "store_only"  # owner_dm | respond | listen | store_only
    raw: dict = field(default_factory=dict)  # Original channel-specific payload

    @property
    def clean_text(self) -> str:
        """Text with @Molly trigger removed."""
        import config
        if self.has_trigger:
            return config.TRIGGER_PATTERN.sub("", self.text).strip()
        return self.text.strip()


# ---------------------------------------------------------------------------
# Outbound message dataclass
# ---------------------------------------------------------------------------

@dataclass
class OutboundMessage:
    """Response to be sent through a channel."""

    text: str
    chat_id: str
    source: str = "unknown"
    session_id: str | None = None

    # Formatting hints
    format_type: str = "markdown"  # markdown | plain
    chunk_enabled: bool = False
    chunk_max_chars: int = 4000
    chunk_suffix: str = "\n\n-MollyAI"

    # Metadata
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract Channel base class
# ---------------------------------------------------------------------------

class Channel(ABC):
    """Base class for all channel implementations.

    Each channel knows how to:
    1. Convert raw channel data → InboundMessage
    2. Format OutboundMessage → channel-specific output
    3. Send formatted output through the channel transport
    """

    name: str = "base"

    @abstractmethod
    def normalize_inbound(self, raw_data: dict, **kwargs) -> InboundMessage:
        """Convert raw channel-specific data to an InboundMessage."""
        ...

    @abstractmethod
    def format_outbound(self, message: OutboundMessage) -> str:
        """Format an OutboundMessage for this channel's constraints."""
        ...

    @abstractmethod
    async def send(self, message: OutboundMessage, transport: Any = None) -> str | list[str] | None:
        """Send the formatted message through the channel transport.

        Returns the message ID(s) if the transport supports tracking.
        """
        ...

    def get_session_key(self, inbound: InboundMessage) -> str:
        """Build the session storage key for this channel + message."""
        return inbound.chat_id

    def get_source_label(self) -> str:
        """Source label passed to agent.handle_message()."""
        return self.name

    def get_response_guidance(self) -> str | None:
        """Optional response format guidance injected into agent prompt.

        Returns None for channels that accept raw markdown (like web).
        """
        return None


# ---------------------------------------------------------------------------
# Channel registry — singleton lookup by name
# ---------------------------------------------------------------------------

class ChannelRegistry:
    """Thread-safe registry mapping channel names to Channel instances."""

    def __init__(self):
        self._channels: dict[str, Channel] = {}
        self._lock = threading.Lock()

    def register(self, channel: Channel) -> None:
        """Register a channel by its .name attribute."""
        with self._lock:
            if channel.name in self._channels:
                log.warning("Overwriting channel registration: %s", channel.name)
            self._channels[channel.name] = channel
        log.debug("Registered channel: %s", channel.name)

    def get(self, name: str) -> Channel | None:
        """Look up a channel by name."""
        with self._lock:
            return self._channels.get(name)

    def list_channels(self) -> list[str]:
        """Return sorted list of registered channel names."""
        with self._lock:
            return sorted(self._channels.keys())

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._channels

    def __len__(self) -> int:
        with self._lock:
            return len(self._channels)


# Global singleton
registry = ChannelRegistry()
