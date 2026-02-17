"""Tests for Phase 5B channels abstraction.

Tests cover:
  - InboundMessage / OutboundMessage dataclasses
  - Channel registry (register, lookup, list)
  - WhatsApp channel (normalize, format, chunk)
  - iMessage channel (normalize, session key)
  - Email channel (normalize)
  - Gateway channel (normalize, session key)
  - Web channel (normalize)
"""
from __future__ import annotations

import unittest

import config
from channels.base import (
    Channel,
    ChannelRegistry,
    InboundMessage,
    OutboundMessage,
    registry,
)

# Import channel modules to trigger auto-registration
import channels.whatsapp   # noqa: F401
import channels.imessage   # noqa: F401
import channels.email      # noqa: F401
import channels.gateway    # noqa: F401
import channels.web        # noqa: F401


# ---------------------------------------------------------------------------
# InboundMessage / OutboundMessage
# ---------------------------------------------------------------------------

class TestInboundMessage(unittest.TestCase):
    """Test InboundMessage creation and properties."""

    def test_basic_creation(self):
        msg = InboundMessage(
            text="Hello @Molly",
            chat_id="1234@s.whatsapp.net",
            sender_id="1234@s.whatsapp.net",
        )
        self.assertEqual(msg.text, "Hello @Molly")
        self.assertEqual(msg.source, "unknown")
        self.assertFalse(msg.is_group)
        self.assertFalse(msg.is_from_me)
        self.assertEqual(msg.chat_mode, "store_only")

    def test_clean_text_no_trigger(self):
        msg = InboundMessage(text="Hello", chat_id="x", sender_id="y")
        self.assertEqual(msg.clean_text, "Hello")

    def test_clean_text_with_trigger(self):
        msg = InboundMessage(
            text="@Molly what's on my calendar?",
            chat_id="x",
            sender_id="y",
            has_trigger=True,
        )
        cleaned = msg.clean_text
        self.assertNotIn("@Molly", cleaned)
        self.assertIn("calendar", cleaned)

    def test_default_values(self):
        msg = InboundMessage(text="t", chat_id="c", sender_id="s")
        self.assertEqual(msg.msg_id, "")
        self.assertIsNone(msg.thread_context)
        self.assertIsNone(msg.handle)
        self.assertEqual(msg.trigger_type, "")
        self.assertEqual(msg.attachments, [])
        self.assertEqual(msg.raw, {})
        self.assertIsNone(msg.session_id)
        self.assertGreater(msg.received_at, 0)


class TestOutboundMessage(unittest.TestCase):
    """Test OutboundMessage creation."""

    def test_basic_creation(self):
        msg = OutboundMessage(text="Response", chat_id="x")
        self.assertEqual(msg.text, "Response")
        self.assertEqual(msg.format_type, "markdown")
        self.assertFalse(msg.chunk_enabled)
        self.assertEqual(msg.metadata, {})


# ---------------------------------------------------------------------------
# Channel Registry
# ---------------------------------------------------------------------------

class TestChannelRegistry(unittest.TestCase):
    """Test channel registry operations."""

    def test_global_registry_has_channels(self):
        """Channels are auto-registered on import."""
        self.assertIn("whatsapp", registry)
        self.assertIn("imessage", registry)
        self.assertIn("email", registry)
        self.assertIn("gateway", registry)
        self.assertIn("web", registry)
        self.assertEqual(len(registry), 5)

    def test_get_channel(self):
        ch = registry.get("whatsapp")
        self.assertIsNotNone(ch)
        self.assertEqual(ch.name, "whatsapp")

    def test_get_unknown_returns_none(self):
        self.assertIsNone(registry.get("slack"))

    def test_list_channels(self):
        names = registry.list_channels()
        self.assertEqual(names, ["email", "gateway", "imessage", "web", "whatsapp"])

    def test_custom_registry(self):
        """Test a fresh registry (not the global one)."""
        r = ChannelRegistry()
        self.assertEqual(len(r), 0)

        class DummyChannel(Channel):
            name = "dummy"
            def normalize_inbound(self, raw_data, **kwargs):
                return InboundMessage(text="", chat_id="", sender_id="")
            def format_outbound(self, message):
                return message.text
            async def send(self, message, transport=None):
                return None

        r.register(DummyChannel())
        self.assertIn("dummy", r)
        self.assertEqual(r.get("dummy").name, "dummy")


# ---------------------------------------------------------------------------
# WhatsApp Channel
# ---------------------------------------------------------------------------

class TestWhatsAppChannel(unittest.TestCase):
    """Test WhatsApp channel normalize + format."""

    def setUp(self):
        self.ch = registry.get("whatsapp")
        self.assertIsNotNone(self.ch)

    def test_normalize_dm(self):
        msg = self.ch.normalize_inbound({
            "msg_id": "ABC123",
            "chat_jid": "1234@s.whatsapp.net",
            "sender_jid": "1234@s.whatsapp.net",
            "sender_name": "Brian",
            "content": "Hello",
            "timestamp": "2026-02-16T10:00:00Z",
            "is_from_me": False,
            "is_group": False,
        }, is_owner=True, chat_mode="owner_dm")

        self.assertEqual(msg.source, "whatsapp")
        self.assertEqual(msg.msg_id, "ABC123")
        self.assertEqual(msg.sender_name, "Brian")
        self.assertTrue(msg.is_owner)
        self.assertFalse(msg.is_group)
        self.assertEqual(msg.chat_mode, "owner_dm")
        self.assertFalse(msg.has_trigger)

    def test_normalize_group_with_trigger(self):
        msg = self.ch.normalize_inbound({
            "msg_id": "DEF456",
            "chat_jid": "group@g.us",
            "sender_jid": "1234@s.whatsapp.net",
            "sender_name": "Brian",
            "content": "@Molly check calendar",
            "timestamp": "2026-02-16T10:00:00Z",
            "is_from_me": False,
            "is_group": True,
        }, is_owner=True, chat_mode="respond")

        self.assertTrue(msg.is_group)
        self.assertTrue(msg.has_trigger)
        self.assertEqual(msg.chat_mode, "respond")
        self.assertIn("calendar", msg.clean_text)
        self.assertNotIn("@Molly", msg.clean_text)

    def test_format_outbound_strips_markdown(self):
        out = OutboundMessage(text="**Bold** and _italic_", chat_id="x")
        formatted = self.ch.format_outbound(out)
        # ** should become * (WhatsApp bold)
        self.assertIn("*Bold*", formatted)
        self.assertNotIn("**", formatted)

    def test_format_outbound_removes_headers(self):
        out = OutboundMessage(text="## Header\nContent", chat_id="x")
        formatted = self.ch.format_outbound(out)
        self.assertNotIn("##", formatted)
        self.assertIn("Content", formatted)

    def test_chunk_short_message(self):
        chunks = self.ch.chunk_message("Short message")
        self.assertEqual(len(chunks), 1)
        self.assertIn("-MollyAI", chunks[0])

    def test_chunk_long_message(self):
        # Create a message longer than default chunk size
        long_text = "Word " * 1000  # ~5000 chars
        chunks = self.ch.chunk_message(long_text)
        self.assertGreater(len(chunks), 1)
        # Only last chunk has signature
        self.assertIn("-MollyAI", chunks[-1])
        for chunk in chunks[:-1]:
            self.assertNotIn("-MollyAI", chunk)

    def test_session_key(self):
        msg = InboundMessage(text="t", chat_id="abc@s.whatsapp.net", sender_id="s")
        self.assertEqual(self.ch.get_session_key(msg), "abc@s.whatsapp.net")

    def test_response_guidance(self):
        guidance = self.ch.get_response_guidance()
        self.assertIsNotNone(guidance)
        self.assertIn("WhatsApp", guidance)


# ---------------------------------------------------------------------------
# iMessage Channel
# ---------------------------------------------------------------------------

class TestIMessageChannel(unittest.TestCase):
    """Test iMessage channel normalize."""

    def setUp(self):
        self.ch = registry.get("imessage")
        self.assertIsNotNone(self.ch)

    def test_normalize_mention(self):
        msg = self.ch.normalize_inbound({
            "text": "@molly check my calendar",
            "sender": "Brian",
            "handle": "+15551234567",
            "is_from_me": True,
            "timestamp": "2026-02-16T10:00:00Z",
            "chat_id": "42",
            "thread_context": "Person A: hello\nBrian: @molly check my calendar",
        }, source="imessage-mention")

        self.assertEqual(msg.source, "imessage-mention")
        self.assertEqual(msg.chat_id, "imessage:chat:42")
        self.assertTrue(msg.is_owner)
        self.assertTrue(msg.has_trigger)
        self.assertIsNotNone(msg.thread_context)
        self.assertEqual(msg.handle, "+15551234567")

    def test_normalize_heartbeat(self):
        msg = self.ch.normalize_inbound({
            "text": "Some iMessage text",
            "sender": "John",
            "handle": "john@example.com",
            "is_from_me": False,
            "timestamp": "2026-02-16T10:00:00Z",
        }, source="heartbeat")

        self.assertEqual(msg.source, "heartbeat")
        self.assertEqual(msg.chat_id, "imessage:john@example.com")
        self.assertFalse(msg.has_trigger)

    def test_session_key_synthetic(self):
        msg = InboundMessage(text="t", chat_id="imessage:chat:42", sender_id="s")
        self.assertEqual(self.ch.get_session_key(msg), "imessage:chat:42")


# ---------------------------------------------------------------------------
# Email Channel
# ---------------------------------------------------------------------------

class TestEmailChannel(unittest.TestCase):
    """Test Email channel normalize."""

    def setUp(self):
        self.ch = registry.get("email")
        self.assertIsNotNone(self.ch)

    def test_normalize(self):
        msg = self.ch.normalize_inbound({
            "subject": "Q4 Report",
            "sender": "john@company.com",
            "sender_name": "John Doe",
            "body": "Please review the attached report.",
            "timestamp": "2026-02-16T10:00:00Z",
            "message_id": "msg123",
            "thread_id": "thread456",
        })

        self.assertEqual(msg.source, "email")
        self.assertIn("Q4 Report", msg.text)
        self.assertIn("review", msg.text)
        self.assertEqual(msg.chat_id, "email:thread456")
        self.assertEqual(msg.sender_name, "John Doe")
        self.assertEqual(msg.chat_mode, "listen")


# ---------------------------------------------------------------------------
# Gateway Channel
# ---------------------------------------------------------------------------

class TestGatewayChannel(unittest.TestCase):
    """Test Gateway channel normalize."""

    def setUp(self):
        self.ch = registry.get("gateway")
        self.assertIsNotNone(self.ch)

    def test_normalize_schedule(self):
        msg = self.ch.normalize_inbound({
            "prompt": "Create morning briefing",
            "job_id": "morning-packet",
            "trigger": "schedule",
            "chat_id": "owner@s.whatsapp.net",
        })

        self.assertEqual(msg.source, "gateway")
        self.assertEqual(msg.text, "Create morning briefing")
        self.assertEqual(msg.trigger_type, "schedule")
        self.assertTrue(msg.is_owner)

    def test_session_key_always_gateway(self):
        msg = InboundMessage(text="t", chat_id="anything", sender_id="s")
        self.assertEqual(self.ch.get_session_key(msg), "gateway:summary")


# ---------------------------------------------------------------------------
# Web Channel
# ---------------------------------------------------------------------------

class TestWebChannel(unittest.TestCase):
    """Test Web channel normalize."""

    def setUp(self):
        self.ch = registry.get("web")
        self.assertIsNotNone(self.ch)

    def test_normalize(self):
        msg = self.ch.normalize_inbound({
            "text": "Hello from web",
            "token": "",
        })

        self.assertEqual(msg.source, "web")
        self.assertEqual(msg.text, "Hello from web")
        self.assertTrue(msg.is_owner)
        self.assertTrue(msg.chat_id.startswith("web:"))

    def test_no_response_guidance(self):
        """Web channel allows full markdown."""
        self.assertIsNone(self.ch.get_response_guidance())


if __name__ == "__main__":
    unittest.main()
