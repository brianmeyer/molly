"""Tests for Phase 5C.1 Voice Loop.

Tests cover:
  - VoiceLoop state machine
  - Budget tracking
  - System context loading
  - Tool declaration schemas
  - Tool call bridge (mocked)
  - Config values
  - Porcupine initialization (with real .ppn if available)
"""
from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import config


class TestVoiceLoopConfig(unittest.TestCase):
    """Test voice loop config values."""

    def test_voice_enabled_default(self):
        self.assertTrue(config.VOICE_ENABLED)

    def test_picovoice_access_key_set(self):
        # Key should be set from .env
        self.assertIsInstance(config.PICOVOICE_ACCESS_KEY, str)

    def test_porcupine_model_path(self):
        path = config.PORCUPINE_MODEL_PATH
        self.assertIn("molly_mac.ppn", path)

    def test_gemini_live_model(self):
        self.assertIn("gemini", config.GEMINI_LIVE_MODEL)

    def test_voice_budget_defaults(self):
        self.assertGreaterEqual(config.VOICE_MAX_SESSION_MINUTES, 1)
        self.assertGreaterEqual(config.VOICE_DAILY_BUDGET_MINUTES, 1)

    def test_voice_sensitivity(self):
        self.assertGreaterEqual(config.VOICE_SENSITIVITY, 0.0)
        self.assertLessEqual(config.VOICE_SENSITIVITY, 1.0)


class TestVoiceLoopStateMachine(unittest.TestCase):
    """Test VoiceLoop state machine."""

    def setUp(self):
        from voice_loop import VoiceLoop, VoiceState
        self.VoiceState = VoiceState
        self.vl = VoiceLoop()

    def test_initial_state(self):
        self.assertEqual(self.vl.state, self.VoiceState.LISTENING)

    def test_states_enum(self):
        self.assertEqual(self.VoiceState.LISTENING, "LISTENING")
        self.assertEqual(self.VoiceState.CONNECTING, "CONNECTING")
        self.assertEqual(self.VoiceState.CONVERSING, "CONVERSING")
        self.assertEqual(self.VoiceState.PAUSED, "PAUSED")

    def test_stop(self):
        self.assertTrue(self.vl._running)
        self.vl.stop()
        self.assertFalse(self.vl._running)


class TestVoiceLoopBudget(unittest.TestCase):
    """Test daily budget tracking."""

    def setUp(self):
        from voice_loop import VoiceLoop
        self.vl = VoiceLoop()

    def test_initial_budget(self):
        self.assertEqual(self.vl._daily_minutes_used, 0.0)
        self.assertTrue(self.vl._check_budget())

    def test_budget_exhausted(self):
        self.vl._daily_minutes_used = config.VOICE_DAILY_BUDGET_MINUTES + 1
        self.vl._daily_reset_date = __import__("time").strftime("%Y-%m-%d")
        self.assertFalse(self.vl._check_budget())

    def test_budget_resets_daily(self):
        self.vl._daily_minutes_used = 999
        self.vl._daily_reset_date = "2000-01-01"  # old date
        self.assertTrue(self.vl._check_budget())
        self.assertEqual(self.vl._daily_minutes_used, 0.0)


class TestVoiceLoopStats(unittest.TestCase):
    """Test stats reporting."""

    def test_get_stats(self):
        from voice_loop import VoiceLoop
        vl = VoiceLoop()
        stats = vl.get_stats()
        self.assertIn("state", stats)
        self.assertIn("daily_minutes_used", stats)
        self.assertIn("daily_budget_minutes", stats)
        self.assertEqual(stats["state"], "LISTENING")
        self.assertEqual(stats["daily_minutes_used"], 0.0)
        self.assertTrue(stats["running"])


class TestVoiceLoopContext(unittest.TestCase):
    """Test system context loading."""

    def test_load_system_context(self):
        from voice_loop import VoiceLoop
        vl = VoiceLoop()
        context = vl._load_system_context()
        # SOUL.md and USER.md should exist in workspace
        self.assertIsInstance(context, str)
        # If identity files exist, context should be non-empty
        if config.IDENTITY_FILES[0].exists():
            self.assertGreater(len(context), 0)


class TestVoiceToolDeclarations(unittest.TestCase):
    """Test Gemini Live tool declarations."""

    def test_get_declarations(self):
        from voice_loop import get_voice_tool_declarations
        decls = get_voice_tool_declarations()
        self.assertIsInstance(decls, list)
        self.assertGreater(len(decls), 0)
        names = {d["name"] for d in decls}
        self.assertIn("check_calendar", names)
        self.assertIn("send_message", names)
        self.assertIn("create_task", names)
        self.assertIn("search_memory", names)

    def test_declaration_schema(self):
        from voice_loop import get_voice_tool_declarations
        for decl in get_voice_tool_declarations():
            self.assertIn("name", decl)
            self.assertIn("description", decl)
            self.assertIn("parameters", decl)
            self.assertEqual(decl["parameters"]["type"], "object")


class TestVoiceToolBridge(unittest.TestCase):
    """Test tool call bridge (mocked)."""

    def test_search_memory_tool(self):
        from voice_loop import VoiceLoop
        vl = VoiceLoop()

        async def _test():
            with patch("memory.retriever.retrieve_context", new_callable=AsyncMock) as mock_retrieve:
                mock_retrieve.return_value = "Found context about calendar"
                result = await vl._tool_search_memory({"query": "calendar"})
                self.assertIn("context", result)

        try:
            import memory.retriever  # noqa: F401
        except ImportError:
            self.skipTest("memory.retriever not importable in test env")
        asyncio.run(_test())

    def test_unknown_tool(self):
        from voice_loop import VoiceLoop
        vl = VoiceLoop()

        async def _test():
            result = await vl._execute_tool("nonexistent_tool", {})
            self.assertIn("error", result)
            self.assertIn("not available via voice", result["error"])

        asyncio.run(_test())

    def test_confirm_tool_denied_via_voice(self):
        """Verify VOICE_CONFIRM_TOOLS are denied without approval channel."""
        from voice_loop import VoiceLoop
        vl = VoiceLoop()

        async def _test():
            result = await vl._execute_tool("send_message", {"text": "hi"})
            self.assertIn("error", result)
            self.assertIn("requires confirmation", result["error"])

        asyncio.run(_test())


class TestVoiceLoopCleanup(unittest.TestCase):
    """Test resource cleanup."""

    def test_cleanup_without_init(self):
        from voice_loop import VoiceLoop
        vl = VoiceLoop()
        # Should not raise even if nothing was initialized
        vl._cleanup()
        self.assertIsNone(vl.porcupine)
        self.assertIsNone(vl.recorder)


if __name__ == "__main__":
    unittest.main()
