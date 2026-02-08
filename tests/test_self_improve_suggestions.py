import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from self_improve import SelfImprovementEngine


class TestSelfImproveSuggestions(unittest.TestCase):
    def setUp(self):
        self.engine = SelfImprovementEngine()

    def test_pattern_steps_parses_string_sequence(self):
        steps = self.engine._pattern_steps({"steps": "alpha -> beta -> gamma"})
        self.assertEqual(steps, ["alpha", "beta", "gamma"])

    def test_build_failure_diagnostic_tool_generates_valid_python(self):
        tool_name, tool_code, test_code = self.engine._build_failure_diagnostic_tool(
            source_tool_name="mcp__grok__grok_reason",
            failures=4,
            sample_error="dependency missing",
        )

        self.assertEqual(tool_name, "Diagnose mcp-grok-grok-reason failures")
        compile(tool_code, "<generated_tool>", "exec")
        compile(test_code, "<generated_test>", "exec")

    def test_detect_tool_gap_candidates_filters_by_failure_count(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="self-improve-tool-gaps-"))
        db_path = tmpdir / "mollygraph.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE tool_calls (
                tool_name TEXT,
                success INTEGER,
                error_message TEXT,
                created_at TEXT
            )
            """
        )
        now = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            "INSERT INTO tool_calls (tool_name, success, error_message, created_at) VALUES (?, ?, ?, ?)",
            [
                ("mcp__grok__grok_reason", 0, "dependency missing", now),
                ("mcp__grok__grok_reason", 0, "dependency missing", now),
                ("mcp__grok__grok_reason", 0, "dependency missing", now),
                ("mcp__gmail__gmail_search", 0, "timeout", now),
                ("approval:Bash", 0, "denied", now),
            ],
        )
        conn.commit()
        conn.close()

        old_path = config.MOLLYGRAPH_PATH
        config.MOLLYGRAPH_PATH = db_path
        try:
            candidates = self.engine._detect_tool_gap_candidates(days=30, min_failures=3)
        finally:
            config.MOLLYGRAPH_PATH = old_path
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["tool_name"], "mcp__grok__grok_reason")
        self.assertEqual(candidates[0]["failures"], 3)

    def test_detect_tool_gap_candidates_default_threshold_and_window(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="self-improve-tool-gap-defaults-"))
        db_path = tmpdir / "mollygraph.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE tool_calls (
                tool_name TEXT,
                success INTEGER,
                error_message TEXT,
                created_at TEXT
            )
            """
        )
        now = datetime.now(timezone.utc)
        in_window = now.isoformat()
        out_of_window = (now - timedelta(days=8)).isoformat()
        conn.executemany(
            "INSERT INTO tool_calls (tool_name, success, error_message, created_at) VALUES (?, ?, ?, ?)",
            [
                ("mcp__grok__grok_reason", 0, "dependency missing", in_window),
                ("mcp__grok__grok_reason", 0, "dependency missing", in_window),
                ("mcp__grok__grok_reason", 0, "dependency missing", in_window),
                ("mcp__grok__grok_reason", 0, "dependency missing", in_window),
                ("mcp__grok__grok_reason", 0, "dependency missing", out_of_window),
            ],
        )
        conn.commit()
        conn.close()

        old_path = config.MOLLYGRAPH_PATH
        config.MOLLYGRAPH_PATH = db_path
        try:
            # Default threshold/window is 5 failures in 7 days; this should not trigger.
            candidates = self.engine._detect_tool_gap_candidates()
        finally:
            config.MOLLYGRAPH_PATH = old_path
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertEqual(candidates, [])


class TestSelfImproveSuggestionFlow(unittest.IsolatedAsyncioTestCase):
    async def test_tool_gap_rejection_stops_before_drafting_code(self):
        engine = SelfImprovementEngine()
        candidate = {
            "tool_name": "mcp__example__unstable",
            "failures": 6,
            "last_failed_at": datetime.now(timezone.utc).isoformat(),
            "sample_error": "transient failure",
        }

        with patch.object(engine, "_detect_tool_gap_candidates", return_value=[candidate]), \
                patch.object(engine, "_has_recent_event", return_value=False), \
                patch.object(engine, "_request_owner_decision", new=AsyncMock(return_value=False)) as ask_mock, \
                patch.object(engine, "propose_tool", new=AsyncMock()) as propose_mock, \
                patch.object(engine, "_log_negative_preference_signal") as neg_pref_mock:
            result = await engine._propose_tool_updates_from_failures()

        self.assertEqual(result["status"], "skipped")
        self.assertTrue(ask_mock.await_count >= 1)
        propose_mock.assert_not_awaited()
        neg_pref_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
