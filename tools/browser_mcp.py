"""Browser MCP server configuration for Molly (Phase 5C.2).

Provides headless Playwright + Chromium MCP server for the browser
worker profile.  Uses @anthropic/browser-mcp in headless mode with
a sandboxed Chromium profile (no main browser cookies).

Safety:
  - Sandboxed profile dir (no access to main browser cookies)
  - No credential entry (approval tier DENY)
  - Full action logging
  - 30s/action + 2min/task timeouts
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP Server Configuration
# ---------------------------------------------------------------------------

def get_browser_mcp_config() -> dict[str, Any]:
    """Return the MCP server spec for the browser tool.

    This follows the same shape as entries in agent.py's _MCP_SERVER_SPECS
    so it can be registered alongside existing MCP servers.
    """
    profile_dir = config.BROWSER_PROFILE_DIR
    profile_dir.mkdir(parents=True, exist_ok=True)

    return {
        "name": "browser",
        "command": "npx",
        "args": ["@anthropic/browser-mcp", "--headless"],
        "env": {
            "BROWSER_PROFILE_DIR": str(profile_dir),
        },
        "timeout": 120,  # 2-minute overall task timeout
    }


def get_browser_tool_specs() -> list[dict[str, Any]]:
    """Return tool specifications for the browser MCP server.

    These define what tools the browser worker can use via the
    @anthropic/browser-mcp server.
    """
    return [
        {
            "name": "browser_navigate",
            "description": "Navigate to a URL in the headless browser.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to navigate to",
                    },
                },
                "required": ["url"],
            },
        },
        {
            "name": "browser_click",
            "description": "Click on an element in the page.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the element to click",
                    },
                },
                "required": ["selector"],
            },
        },
        {
            "name": "browser_type",
            "description": "Type text into an input field.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the input element",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type",
                    },
                },
                "required": ["selector", "text"],
            },
        },
        {
            "name": "browser_screenshot",
            "description": "Take a screenshot of the current page.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "full_page": {
                        "type": "boolean",
                        "description": "Whether to capture the full page",
                        "default": False,
                    },
                },
            },
        },
        {
            "name": "browser_extract_text",
            "description": "Extract visible text content from the current page.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector to extract text from (optional, defaults to body)",
                    },
                },
            },
        },
    ]


def is_browser_available() -> bool:
    """Check if the browser MCP server can be started.

    Requires: npx available + BROWSER_MCP_ENABLED=True.
    """
    if not config.BROWSER_MCP_ENABLED:
        return False

    import shutil
    if not shutil.which("npx"):
        log.debug("Browser MCP unavailable: npx not found")
        return False

    return True


# ---------------------------------------------------------------------------
# Safety constraints
# ---------------------------------------------------------------------------

# Actions that are always blocked for browser worker
BROWSER_BLOCKED_ACTIONS = {
    "credential_entry",
    "password_fill",
    "payment_form",
    "account_creation",
}

# Maximum time for a single browser action (seconds)
BROWSER_ACTION_TIMEOUT = 30

# Maximum time for a complete browser task (seconds)
BROWSER_TASK_TIMEOUT = 120
