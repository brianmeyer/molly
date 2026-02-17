"""Channel abstraction layer for Molly (Phase 5B).

Normalizes all inbound/outbound messages across WhatsApp, iMessage,
email, web, and gateway behind a unified interface.
"""

from channels.base import (
    Channel,
    ChannelRegistry,
    InboundMessage,
    OutboundMessage,
)

__all__ = [
    "Channel",
    "ChannelRegistry",
    "InboundMessage",
    "OutboundMessage",
]
