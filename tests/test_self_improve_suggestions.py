import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import db_pool
from foundry_adapter import FoundrySequenceSignal
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
        conn = db_pool.sqlite_connect(str(db_path))
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
        conn = db_pool.sqlite_connect(str(db_path))
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

    def test_detect_workflow_patterns_filters_primitives_and_keeps_tool_call_fallback(self):
        rows = [
            {"tool_name": "Write", "created_at": "2026-02-09T10:00:00+00:00"},
            {"tool_name": "WebSearch", "created_at": "2026-02-09T10:01:00+00:00"},
            {"tool_name": "kimi_research", "created_at": "2026-02-09T10:02:00+00:00"},
            {"tool_name": "worker_agent", "created_at": "2026-02-09T10:03:00+00:00"},
            {"tool_name": "approval:Bash", "created_at": "2026-02-09T10:04:00+00:00"},
            {"tool_name": "WebSearch", "created_at": "2026-02-09T10:05:00+00:00"},
            {"tool_name": "kimi_research", "created_at": "2026-02-09T10:06:00+00:00"},
            {"tool_name": "worker_agent", "created_at": "2026-02-09T10:07:00+00:00"},
        ]
        with patch.object(self.engine, "_rows", return_value=rows), \
                patch.object(self.engine, "_load_foundry_sequence_signals", return_value={}), \
                patch.object(self.engine, "_existing_automation_ids", return_value=set()):
            patterns = self.engine._detect_workflow_patterns(days=30, min_occurrences=2)

        target = next(
            (item for item in patterns if item.get("steps_text") == "WebSearch -> kimi_research -> worker_agent"),
            None,
        )
        self.assertIsNotNone(target)
        self.assertEqual(target["source"], "tool_calls")
        self.assertEqual(target["tool_call_count"], 2)
        self.assertEqual(target["foundry_count"], 0)
        for item in patterns:
            for step in item.get("steps", []):
                lowered = str(step).lower()
                self.assertNotIn(lowered, {"write", "edit", "bash"})
                self.assertFalse(lowered.startswith("approval:"))

    def test_detect_workflow_patterns_adds_foundry_signal_to_reach_threshold(self):
        rows = [
            {"tool_name": "WebSearch", "created_at": "2026-02-09T10:01:00+00:00"},
            {"tool_name": "kimi_research", "created_at": "2026-02-09T10:02:00+00:00"},
            {"tool_name": "worker_agent", "created_at": "2026-02-09T10:03:00+00:00"},
            {"tool_name": "WebSearch", "created_at": "2026-02-09T10:05:00+00:00"},
            {"tool_name": "kimi_research", "created_at": "2026-02-09T10:06:00+00:00"},
            {"tool_name": "worker_agent", "created_at": "2026-02-09T10:07:00+00:00"},
        ]
        key = "WebSearch -> kimi_research -> worker_agent"
        foundry = {
            key: FoundrySequenceSignal(
                steps=("WebSearch", "kimi_research", "worker_agent"),
                count=1,
                successes=1,
                latest_at="2026-02-09T13:49:27+00:00",
            )
        }
        with patch.object(self.engine, "_rows", return_value=rows), \
                patch.object(self.engine, "_load_foundry_sequence_signals", return_value=foundry), \
                patch.object(self.engine, "_existing_automation_ids", return_value=set()):
            patterns = self.engine._detect_workflow_patterns(days=30, min_occurrences=3)

        target = next((item for item in patterns if item.get("steps_text") == key), None)
        self.assertIsNotNone(target)
        self.assertEqual(target["source"], "tool_calls+foundry")
        self.assertEqual(target["count"], 3)
        self.assertEqual(target["tool_call_count"], 2)
        self.assertEqual(target["foundry_count"], 1)
        self.assertGreater(target["confidence"], 0.8)


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


class TestSelfImproveSkillGapWiring(unittest.IsolatedAsyncioTestCase):
    def _setup_skill_gap_schema(self, db_path: Path):
        conn = db_pool.sqlite_connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skill_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_key TEXT NOT NULL,
                status TEXT,
                addressed INTEGER DEFAULT 0,
                cooldown_until TEXT,
                proposal_id TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS self_improvement_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                payload TEXT,
                status TEXT DEFAULT 'proposed',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    async def test_skill_gap_cluster_creates_pending_proposal_and_marks_addressed(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="self-improve-skill-gap-"))
        db_path = tmpdir / "mollygraph.db"
        self._setup_skill_gap_schema(db_path)

        conn = db_pool.sqlite_connect(str(db_path))
        conn.executemany(
            "INSERT INTO skill_gaps (cluster_key, status, addressed) VALUES (?, ?, ?)",
            [
                ("calendar-followups", "open", 0),
                ("calendar-followups", "open", 0),
                ("calendar-followups", "open", 0),
            ],
        )
        conn.commit()
        conn.close()

        hook = AsyncMock(return_value={"status": "pending", "proposal_id": "track-a-1"})
        fake_molly = SimpleNamespace(
            track_a_hooks=SimpleNamespace(draft_pending_skill_proposal=hook),
        )
        engine = SelfImprovementEngine(molly=fake_molly)

        old_path = config.MOLLYGRAPH_PATH
        config.MOLLYGRAPH_PATH = db_path
        rows = []
        events = 0
        try:
            result = await engine._propose_skill_updates_from_gap_clusters(min_cluster_size=3)
            conn = db_pool.sqlite_connect(str(db_path))
            rows = conn.execute(
                "SELECT status, addressed, proposal_id, cooldown_until FROM skill_gaps WHERE cluster_key = ?",
                ("calendar-followups",),
            ).fetchall()
            events = conn.execute(
                "SELECT COUNT(*) FROM self_improvement_events WHERE category = ? AND title = ?",
                ("skill-gap", "Skill gap cluster: calendar-followups"),
            ).fetchone()[0]
            conn.close()
        finally:
            config.MOLLYGRAPH_PATH = old_path
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["drafted"], 1)
        hook.assert_awaited_once()

        self.assertEqual(len(rows), 3)
        self.assertTrue(all(str(row[0]).lower() == "proposed" for row in rows))
        self.assertTrue(all(int(row[1] or 0) == 1 for row in rows))
        self.assertTrue(all(str(row[2] or "") == "track-a-1" for row in rows))
        self.assertTrue(all(str(row[3] or "").strip() for row in rows))
        self.assertGreaterEqual(int(events or 0), 1)

    async def test_skill_gap_cluster_skips_when_cooldown_or_duplicate_exists(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="self-improve-skill-gap-cooldown-"))
        db_path = tmpdir / "mollygraph.db"
        self._setup_skill_gap_schema(db_path)
        now = datetime.now(timezone.utc)
        future = (now + timedelta(days=2)).isoformat()

        conn = db_pool.sqlite_connect(str(db_path))
        conn.executemany(
            "INSERT INTO skill_gaps (cluster_key, status, addressed, cooldown_until) VALUES (?, ?, ?, ?)",
            [
                ("ops-workflow", "open", 0, future),
                ("ops-workflow", "open", 0, future),
                ("ops-workflow", "open", 0, future),
            ],
        )
        conn.executemany(
            """
            INSERT INTO self_improvement_events
            (id, event_type, category, title, payload, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "evt-1",
                    "proposal",
                    "skill-gap",
                    "Skill gap cluster: duplicate-cluster",
                    "{}",
                    "pending",
                    now.isoformat(),
                    now.isoformat(),
                )
            ],
        )
        conn.executemany(
            "INSERT INTO skill_gaps (cluster_key, status, addressed) VALUES (?, ?, ?)",
            [
                ("duplicate-cluster", "open", 0),
                ("duplicate-cluster", "open", 0),
                ("duplicate-cluster", "open", 0),
            ],
        )
        conn.commit()
        conn.close()

        hook = AsyncMock(return_value={"status": "pending", "proposal_id": "track-a-2"})
        fake_molly = SimpleNamespace(
            track_a_hooks=SimpleNamespace(draft_pending_skill_proposal=hook),
        )
        engine = SelfImprovementEngine(molly=fake_molly)

        old_path = config.MOLLYGRAPH_PATH
        config.MOLLYGRAPH_PATH = db_path
        try:
            result = await engine._propose_skill_updates_from_gap_clusters(min_cluster_size=3)
        finally:
            config.MOLLYGRAPH_PATH = old_path
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertEqual(result["status"], "skipped")
        hook.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
