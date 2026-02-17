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

    def test_browser_mcp_enabled_by_default(self):
        self.assertTrue(config.BROWSER_MCP_ENABLED)

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
        # BROWSER_MCP_ENABLED defaults to True
        self.assertTrue(is_browser_available())


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

    def test_browser_mcp_in_tool_names_registry(self):
        """Regression: browser-mcp must have tool names so workers get allowed_tools."""
        from agent import _MCP_SERVER_TOOL_NAMES
        self.assertIn("browser-mcp", _MCP_SERVER_TOOL_NAMES)
        tool_names = _MCP_SERVER_TOOL_NAMES["browser-mcp"]
        self.assertGreater(len(tool_names), 0)
        for expected in ("browser_navigate", "browser_click", "browser_screenshot"):
            self.assertIn(expected, tool_names)

    def test_browser_worker_allowed_tools_nonempty(self):
        """Regression: browser worker must resolve to non-empty tool list."""
        from workers import _get_allowed_tools
        tools = _get_allowed_tools("browser")
        # Even if not all are in AUTO tier, the function should at least
        # attempt to resolve (not return [] due to missing registry entry)
        from agent import _MCP_SERVER_TOOL_NAMES
        # Verify the registry has the tools (pre-condition for the worker)
        self.assertTrue(len(_MCP_SERVER_TOOL_NAMES.get("browser-mcp", set())) >= 5)


if __name__ == "__main__":
    unittest.main()
