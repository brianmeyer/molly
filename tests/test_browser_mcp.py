"""Tests for Phase 5C.2 Browser MCP.

Tests cover:
  - Browser MCP config generation
  - Tool specifications
  - Availability check
  - Safety constraints
  - Config values
"""
from __future__ import annotations

import unittest
from pathlib import Path

import config


class TestBrowserMCPConfig(unittest.TestCase):
    """Test browser MCP config values."""

    def test_browser_mcp_disabled_by_default(self):
        self.assertFalse(config.BROWSER_MCP_ENABLED)

    def test_browser_profile_dir(self):
        self.assertIsInstance(config.BROWSER_PROFILE_DIR, Path)
        self.assertIn("browser_profile", str(config.BROWSER_PROFILE_DIR))


class TestBrowserMCPServerConfig(unittest.TestCase):
    """Test browser MCP server configuration."""

    def test_get_config(self):
        from tools.browser_mcp import get_browser_mcp_config
        cfg = get_browser_mcp_config()
        self.assertEqual(cfg["name"], "browser")
        self.assertEqual(cfg["command"], "npx")
        self.assertIn("@anthropic/browser-mcp", cfg["args"])
        self.assertIn("--headless", cfg["args"])
        self.assertIn("BROWSER_PROFILE_DIR", cfg["env"])

    def test_get_tool_specs(self):
        from tools.browser_mcp import get_browser_tool_specs
        specs = get_browser_tool_specs()
        self.assertIsInstance(specs, list)
        self.assertGreater(len(specs), 0)
        names = {s["name"] for s in specs}
        self.assertIn("browser_navigate", names)
        self.assertIn("browser_click", names)
        self.assertIn("browser_type", names)
        self.assertIn("browser_screenshot", names)
        self.assertIn("browser_extract_text", names)

    def test_tool_spec_schema(self):
        from tools.browser_mcp import get_browser_tool_specs
        for spec in get_browser_tool_specs():
            self.assertIn("name", spec)
            self.assertIn("description", spec)
            self.assertIn("input_schema", spec)
            self.assertEqual(spec["input_schema"]["type"], "object")

    def test_availability_check(self):
        from tools.browser_mcp import is_browser_available
        # Should be False since BROWSER_MCP_ENABLED defaults to False
        self.assertFalse(is_browser_available())


class TestBrowserSafety(unittest.TestCase):
    """Test browser safety constraints."""

    def test_blocked_actions(self):
        from tools.browser_mcp import BROWSER_BLOCKED_ACTIONS
        self.assertIn("credential_entry", BROWSER_BLOCKED_ACTIONS)
        self.assertIn("password_fill", BROWSER_BLOCKED_ACTIONS)
        self.assertIn("payment_form", BROWSER_BLOCKED_ACTIONS)

    def test_timeouts(self):
        from tools.browser_mcp import BROWSER_ACTION_TIMEOUT, BROWSER_TASK_TIMEOUT
        self.assertEqual(BROWSER_ACTION_TIMEOUT, 30)
        self.assertEqual(BROWSER_TASK_TIMEOUT, 120)


class TestBrowserWorkerProfile(unittest.TestCase):
    """Test that browser worker profile is wired up."""

    def test_browser_profile_has_mcp_server(self):
        from workers import WORKER_PROFILES
        browser_profile = WORKER_PROFILES.get("browser")
        self.assertIsNotNone(browser_profile)
        self.assertIn("browser-mcp", browser_profile["mcp_servers"])


if __name__ == "__main__":
    unittest.main()
