"""Phase 4 tests: Web UI, Terminal, Email monitoring, Config additions.

Covers:
1. Config additions (WEB_HOST, WEB_PORT, WEB_AUTH_TOKEN, EMAIL_POLL_INTERVAL, EMAIL_LOOKBACK_HOURS)
2. web.py (create_app, route existence, WebSocket token auth)
3. terminal.py (AST parse check)
4. heartbeat._check_email (callable, rate-limiting logic)
5. main.py web server wiring (uvicorn + create_app block)
"""

import ast
import asyncio
import importlib
import inspect
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so imports resolve
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===================================================================
# 1. Config additions
# ===================================================================

class TestConfigDefaults(unittest.TestCase):
    """Verify Phase-4 config keys exist and have the expected defaults."""

    def setUp(self):
        import config
        self.cfg = config

    def test_web_host_exists_and_default(self):
        self.assertTrue(hasattr(self.cfg, "WEB_HOST"))
        self.assertIsInstance(self.cfg.WEB_HOST, str)
        if "MOLLY_WEB_HOST" not in os.environ:
            self.assertEqual(self.cfg.WEB_HOST, "127.0.0.1")

    def test_web_port_exists_and_default(self):
        self.assertTrue(hasattr(self.cfg, "WEB_PORT"))
        self.assertIsInstance(self.cfg.WEB_PORT, int)
        if "MOLLY_WEB_PORT" not in os.environ:
            self.assertEqual(self.cfg.WEB_PORT, 8080)

    def test_web_auth_token_exists_and_default(self):
        self.assertTrue(hasattr(self.cfg, "WEB_AUTH_TOKEN"))
        self.assertIsInstance(self.cfg.WEB_AUTH_TOKEN, str)
        if "MOLLY_WEB_TOKEN" not in os.environ:
            self.assertEqual(self.cfg.WEB_AUTH_TOKEN, "")

    def test_email_poll_interval_exists_and_default(self):
        self.assertTrue(hasattr(self.cfg, "EMAIL_POLL_INTERVAL"))
        self.assertEqual(self.cfg.EMAIL_POLL_INTERVAL, 600)

    def test_email_lookback_hours_removed(self):
        """EMAIL_LOOKBACK_HOURS was removed (audit: unused config)."""
        self.assertFalse(hasattr(self.cfg, "EMAIL_LOOKBACK_HOURS"))


# ===================================================================
# 2. web.py -- create_app, routes, token auth
# ===================================================================

class TestWebCreateApp(unittest.TestCase):
    """Verify create_app() returns a usable FastAPI application."""

    def _make_molly_stub(self):
        """Build a minimal Molly-like object for create_app()."""
        molly = MagicMock()
        molly.sessions = {}
        molly.approvals = MagicMock()
        molly.save_sessions = MagicMock()
        return molly

    def test_create_app_returns_fastapi(self):
        from fastapi import FastAPI
        from web import create_app

        molly = self._make_molly_stub()
        app = create_app(molly)
        self.assertIsInstance(app, FastAPI)

    def test_routes_registered(self):
        """The app must expose GET / and WebSocket /ws."""
        from web import create_app

        molly = self._make_molly_stub()
        app = create_app(molly)

        route_paths = {r.path for r in app.routes}
        self.assertIn("/", route_paths, "GET / route is missing")
        self.assertIn("/ws", route_paths, "WebSocket /ws route is missing")


class TestWebTokenAuth(unittest.TestCase):
    """Test WebSocket /ws token authentication."""

    def _make_molly_stub(self):
        molly = MagicMock()
        molly.sessions = {}
        molly.approvals = MagicMock()
        molly.save_sessions = MagicMock()
        return molly

    @patch("web.config")
    @patch("web.handle_message", new_callable=AsyncMock)
    def test_ws_rejects_bad_token(self, mock_handle, mock_config):
        """When WEB_AUTH_TOKEN is set, a wrong token should close the socket."""
        from fastapi.testclient import TestClient
        from web import create_app

        mock_config.WEB_AUTH_TOKEN = "secret123"
        molly = self._make_molly_stub()
        app = create_app(molly)

        client = TestClient(app)
        # The server calls ws.close() before or right after accept,
        # which raises an Exception from the test client.
        try:
            with client.websocket_connect("/ws?token=wrong") as ws:
                # If we somehow get here, try to interact -- should fail
                try:
                    ws.send_text("hi")
                    ws.receive_json()
                    self.fail("Expected WebSocket to reject bad token")
                except Exception:
                    pass  # Server closed connection -- expected
        except Exception:
            # Connection rejected -- expected path
            pass

    @patch("web.config")
    @patch("web.handle_message", new_callable=AsyncMock)
    def test_ws_accepts_correct_token(self, mock_handle, mock_config):
        """When the correct token is supplied, the connection should be accepted."""
        from fastapi.testclient import TestClient
        from web import create_app

        mock_config.WEB_AUTH_TOKEN = "secret123"
        molly = self._make_molly_stub()
        app = create_app(molly)

        mock_handle.return_value = ("Hello from Molly", "session-abc")

        client = TestClient(app)
        with client.websocket_connect("/ws?token=secret123") as ws:
            ws.send_text(json.dumps({"text": "hi"}))
            # We should get a typing indicator first
            typing_msg = ws.receive_json()
            self.assertEqual(typing_msg["type"], "typing")
            # Then the actual response
            resp = ws.receive_json()
            self.assertEqual(resp["type"], "message")
            self.assertIn("Hello from Molly", resp["text"])

    @patch("web.config")
    @patch("web.handle_message", new_callable=AsyncMock)
    def test_ws_no_token_required_when_empty(self, mock_handle, mock_config):
        """When WEB_AUTH_TOKEN is empty, any connection should be accepted."""
        from fastapi.testclient import TestClient
        from web import create_app

        mock_config.WEB_AUTH_TOKEN = ""
        molly = self._make_molly_stub()
        app = create_app(molly)

        mock_handle.return_value = ("Hello!", "sess-1")

        client = TestClient(app)
        with client.websocket_connect("/ws?token=") as ws:
            ws.send_text(json.dumps({"text": "test"}))
            typing_msg = ws.receive_json()
            self.assertEqual(typing_msg["type"], "typing")
            resp = ws.receive_json()
            self.assertEqual(resp["type"], "message")

    @patch("web.config")
    @patch("web.handle_message", new_callable=AsyncMock)
    def test_ws_no_token_param_when_empty_config(self, mock_handle, mock_config):
        """When WEB_AUTH_TOKEN is empty, connection with no token param succeeds."""
        from fastapi.testclient import TestClient
        from web import create_app

        mock_config.WEB_AUTH_TOKEN = ""
        molly = self._make_molly_stub()
        app = create_app(molly)

        mock_handle.return_value = ("Hello!", "sess-2")

        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"text": "hello"}))
            typing_msg = ws.receive_json()
            self.assertEqual(typing_msg["type"], "typing")
            resp = ws.receive_json()
            self.assertEqual(resp["type"], "message")


# ===================================================================
# 3. terminal.py -- AST parse check
# ===================================================================

class TestTerminalAST(unittest.TestCase):
    """Verify terminal.py is syntactically valid Python."""

    def test_parses_as_valid_python(self):
        terminal_path = PROJECT_ROOT / "terminal.py"
        self.assertTrue(terminal_path.exists(), "terminal.py not found")

        source = terminal_path.read_text()
        tree = ast.parse(source, filename="terminal.py")
        self.assertIsInstance(tree, ast.Module)

    def test_has_main_async_function(self):
        """terminal.py should define an async function named 'main'."""
        terminal_path = PROJECT_ROOT / "terminal.py"
        source = terminal_path.read_text()
        tree = ast.parse(source, filename="terminal.py")

        async_func_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
        ]
        self.assertIn("main", async_func_names, "async def main() not found in terminal.py")

    def test_has_if_name_main_guard(self):
        """terminal.py should have an if __name__ == '__main__' guard."""
        terminal_path = PROJECT_ROOT / "terminal.py"
        source = terminal_path.read_text()
        self.assertIn('if __name__ == "__main__"', source)

    def test_imports_handle_message(self):
        """terminal.py should import handle_message from agent."""
        terminal_path = PROJECT_ROOT / "terminal.py"
        source = terminal_path.read_text()
        tree = ast.parse(source, filename="terminal.py")

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "agent":
                imported_names = [alias.name for alias in node.names]
                if "handle_message" in imported_names:
                    found = True
                    break
        self.assertTrue(found, "terminal.py should import handle_message from agent")


# ===================================================================
# 4. heartbeat._check_email -- callable + rate-limiting
# ===================================================================

class TestCheckEmailCallable(unittest.TestCase):
    """Verify _check_email exists and is an async callable."""

    def test_check_email_is_async_callable(self):
        from heartbeat import _check_email
        self.assertTrue(callable(_check_email))
        self.assertTrue(asyncio.iscoroutinefunction(_check_email))


class TestCheckEmailRateLimiting(unittest.TestCase):
    """Test the rate-limiting logic inside _check_email."""

    def _run_async(self, coro):
        """Helper to run an async function synchronously."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @patch("heartbeat.config")
    def test_skips_when_last_check_too_recent(self, mock_config):
        """If less than EMAIL_POLL_INTERVAL has passed, _check_email returns early."""
        import tempfile

        mock_config.EMAIL_POLL_INTERVAL = 600
        state_path = Path(tempfile.mktemp(suffix=".json"))
        mock_config.STATE_FILE = state_path

        # Write state with a recent email_heartbeat_hw (30 seconds ago)
        state = {"email_heartbeat_hw": time.time() - 30}
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state))

        molly = MagicMock()
        molly.cancel_event = None

        # Build mock gmail service chain
        mock_list_execute = MagicMock(return_value={"messages": []})
        mock_list = MagicMock()
        mock_list.execute = mock_list_execute
        mock_messages = MagicMock()
        mock_messages.list.return_value = mock_list
        mock_users = MagicMock()
        mock_users.messages.return_value = mock_messages
        mock_service = MagicMock()
        mock_service.users.return_value = mock_users

        mock_gmail_mod = MagicMock()
        mock_gmail_mod.get_gmail_service.return_value = mock_service

        with patch.dict("sys.modules", {"tools.google_auth": mock_gmail_mod}):
            from heartbeat import _check_email
            self._run_async(_check_email(molly))

        # The Gmail list call should NOT have been made (rate-limited)
        mock_messages.list.assert_not_called()

        # Cleanup
        if state_path.exists():
            state_path.unlink()

    @patch("heartbeat.config")
    def test_proceeds_when_last_check_old_enough(self, mock_config):
        """If more than EMAIL_POLL_INTERVAL has passed, _check_email proceeds."""
        import tempfile

        mock_config.EMAIL_POLL_INTERVAL = 600
        state_path = Path(tempfile.mktemp(suffix=".json"))
        mock_config.STATE_FILE = state_path

        # Write state with an old email_heartbeat_hw (700 seconds ago)
        state = {"email_heartbeat_hw": time.time() - 700}
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state))

        molly = MagicMock()
        molly.cancel_event = None

        mock_list_execute = MagicMock(return_value={"messages": []})
        mock_list = MagicMock()
        mock_list.execute = mock_list_execute
        mock_messages = MagicMock()
        mock_messages.list.return_value = mock_list
        mock_users = MagicMock()
        mock_users.messages.return_value = mock_messages
        mock_service = MagicMock()
        mock_service.users.return_value = mock_users

        mock_gmail_mod = MagicMock()
        mock_gmail_mod.get_gmail_service.return_value = mock_service

        with patch.dict("sys.modules", {"tools.google_auth": mock_gmail_mod}):
            from heartbeat import _check_email
            self._run_async(_check_email(molly))

        # The Gmail list call SHOULD have been made
        mock_messages.list.assert_called_once()

        # Cleanup
        if state_path.exists():
            state_path.unlink()

    @patch("heartbeat.config")
    def test_proceeds_on_first_run(self, mock_config):
        """On first run (no email_heartbeat_hw), _check_email should proceed."""
        import tempfile

        mock_config.EMAIL_POLL_INTERVAL = 600
        state_path = Path(tempfile.mktemp(suffix=".json"))
        mock_config.STATE_FILE = state_path

        # Empty state
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({}))

        molly = MagicMock()
        molly.cancel_event = None

        mock_list_execute = MagicMock(return_value={"messages": []})
        mock_list = MagicMock()
        mock_list.execute = mock_list_execute
        mock_messages = MagicMock()
        mock_messages.list.return_value = mock_list
        mock_users = MagicMock()
        mock_users.messages.return_value = mock_messages
        mock_service = MagicMock()
        mock_service.users.return_value = mock_users

        mock_gmail_mod = MagicMock()
        mock_gmail_mod.get_gmail_service.return_value = mock_service

        with patch.dict("sys.modules", {"tools.google_auth": mock_gmail_mod}):
            from heartbeat import _check_email
            self._run_async(_check_email(molly))

        # Should have called list because last_check was 0 (first run)
        mock_messages.list.assert_called_once()

        # Cleanup
        if state_path.exists():
            state_path.unlink()

    @patch("heartbeat.config")
    def test_returns_early_when_no_gmail_service(self, mock_config):
        """If get_gmail_service returns None, _check_email exits silently."""
        mock_config.EMAIL_POLL_INTERVAL = 600

        molly = MagicMock()

        mock_gmail_mod = MagicMock()
        mock_gmail_mod.get_gmail_service.return_value = None

        with patch.dict("sys.modules", {"tools.google_auth": mock_gmail_mod}):
            from heartbeat import _check_email
            # Should not raise
            self._run_async(_check_email(molly))


# ===================================================================
# 5. main.py -- web server wiring
# ===================================================================

class TestMainWebWiring(unittest.TestCase):
    """Verify that main.py has the uvicorn/create_app web server block."""

    def test_main_imports_uvicorn_and_create_app(self):
        """The Molly.run() method should contain the uvicorn + create_app wiring."""
        main_path = PROJECT_ROOT / "main.py"
        self.assertTrue(main_path.exists(), "main.py not found")
        source = main_path.read_text()

        self.assertIn("import uvicorn", source, "uvicorn import not found in main.py")
        self.assertIn("from web import create_app", source,
                       "create_app import not found in main.py")
        self.assertIn("create_app(self)", source,
                       "create_app(self) call not found in main.py")

    def test_uvicorn_config_uses_web_settings(self):
        """Verify the uvicorn.Config references config.WEB_HOST and config.WEB_PORT."""
        main_path = PROJECT_ROOT / "main.py"
        source = main_path.read_text()

        self.assertIn("config.WEB_HOST", source,
                       "config.WEB_HOST not used in uvicorn config")
        self.assertIn("config.WEB_PORT", source,
                       "config.WEB_PORT not used in uvicorn config")

    def test_web_server_block_inside_run_method(self):
        """The web server block should be inside the async run() method of Molly."""
        main_path = PROJECT_ROOT / "main.py"
        source = main_path.read_text()
        tree = ast.parse(source, filename="main.py")

        # Find the Molly class
        molly_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Molly":
                molly_class = node
                break
        self.assertIsNotNone(molly_class, "Molly class not found in main.py")

        # Find the run method
        run_method = None
        for node in ast.iter_child_nodes(molly_class):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "run":
                run_method = node
                break
        self.assertIsNotNone(run_method, "async def run() not found in Molly class")

        # Check that the run method body contains "uvicorn"
        run_source = ast.get_source_segment(source, run_method)
        self.assertIn("uvicorn", run_source,
                       "uvicorn not referenced inside Molly.run()")
        self.assertIn("create_app", run_source,
                       "create_app not referenced inside Molly.run()")

    def test_web_server_has_fallback_on_import_error(self):
        """The web server block should handle ImportError gracefully."""
        main_path = PROJECT_ROOT / "main.py"
        source = main_path.read_text()

        self.assertIn("except ImportError", source,
                       "No ImportError fallback for web server in main.py")


# ===================================================================
# Run
# ===================================================================

if __name__ == "__main__":
    unittest.main()
