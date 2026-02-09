import asyncio
import json
import shutil
import sqlite3
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "numpy" not in sys.modules:
    sys.modules["numpy"] = types.SimpleNamespace(array=lambda x: x, dot=lambda _a, _b: 0.0)
if "sqlite_vec" not in sys.modules:
    sys.modules["sqlite_vec"] = types.SimpleNamespace(load=lambda _conn: None)

import agent
import config
from skill_analytics import (
    get_skill_gap_clusters,
    get_skill_stats,
    get_underperforming_skills,
)


class _GapStore:
    def __init__(self):
        self.rows: list[dict] = []

    def log_skill_gap(self, user_message: str, tools_used: list[str], session_id: str = "", addressed: bool = False):
        row = {
            "id": len(self.rows) + 1,
            "user_message": user_message,
            "tools_used": list(tools_used),
            "session_id": session_id,
            "addressed": int(bool(addressed)),
        }
        self.rows.append(row)
        return row["id"]


def _create_analytics_tables(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE skill_executions (
            id TEXT PRIMARY KEY,
            skill_name TEXT,
            trigger TEXT,
            outcome TEXT,
            user_approval TEXT,
            edits_made TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE skill_gaps (
            id INTEGER PRIMARY KEY,
            user_message TEXT,
            tools_used TEXT,
            session_id TEXT,
            created_at TEXT,
            addressed INTEGER DEFAULT 0
        )
        """
    )
    conn.commit()
    conn.close()


class TestSkillGapSchema(unittest.TestCase):
    def test_vectorstore_declares_skill_gaps_table(self):
        src = (PROJECT_ROOT / "memory" / "vectorstore.py").read_text()
        self.assertIn("CREATE TABLE IF NOT EXISTS skill_gaps", src)
        self.assertIn("tools_used TEXT", src)
        self.assertIn("addressed INTEGER DEFAULT 0", src)
        self.assertIn("def log_skill_gap", src)


class TestSkillGapDetection(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        agent._CHAT_RUNTIMES.clear()

    async def asyncTearDown(self):
        agent._CHAT_RUNTIMES.clear()
        await asyncio.sleep(0)

    async def test_handle_message_logs_gap_only_for_current_turn(self):
        store = _GapStore()
        turn_calls = [
            ["gmail_search", "calendar_search"],
            [
                "gmail_search",
                "routing:subagent_start:worker",
                "approval:Bash",
                "memory_search",
                "gmail_draft",
                "gmail_send",
            ],
        ]

        async def fake_query(runtime, _turn_prompt):
            for tool_name in turn_calls.pop(0):
                agent._record_turn_tool_call(runtime.request_state, tool_name)
            return "ok", "session-abc"

        with patch("agent.load_identity_stack", return_value="identity"), \
                patch("agent.retrieve_context", return_value=""), \
                patch("agent.match_skills", return_value=[]), \
                patch("agent.get_skill_context", return_value=""), \
                patch("agent._ensure_connected_runtime", new=AsyncMock()), \
                patch("agent._disconnect_runtime", new=AsyncMock()), \
                patch("agent._query_with_client", side_effect=fake_query), \
                patch("agent.process_conversation", new=AsyncMock()), \
                patch("memory.retriever.get_vectorstore", return_value=store):
            await agent.handle_message("first turn", "chat-gap")
            await agent.handle_message("second turn", "chat-gap")

        self.assertEqual(len(store.rows), 1)
        self.assertEqual(store.rows[0]["user_message"], "second turn")
        self.assertEqual(
            store.rows[0]["tools_used"],
            ["gmail_search", "gmail_draft", "gmail_send"],
        )

    async def test_handle_message_skips_gap_when_skill_matches(self):
        store = _GapStore()

        async def fake_query(runtime, _turn_prompt):
            for tool_name in ["gmail_search", "gmail_draft", "gmail_send", "calendar_search"]:
                agent._record_turn_tool_call(runtime.request_state, tool_name)
            return "ok", "session-has-skill"

        with patch("agent.load_identity_stack", return_value="identity"), \
                patch("agent.retrieve_context", return_value=""), \
                patch("agent.match_skills", return_value=[object()]), \
                patch("agent.get_skill_context", return_value="skill context"), \
                patch("agent._ensure_connected_runtime", new=AsyncMock()), \
                patch("agent._disconnect_runtime", new=AsyncMock()), \
                patch("agent._query_with_client", side_effect=fake_query), \
                patch("agent.process_conversation", new=AsyncMock()), \
                patch("memory.retriever.get_vectorstore", return_value=store):
            await agent.handle_message("with skill", "chat-skill")

        self.assertEqual(store.rows, [])

    def test_filtering_excludes_meta_and_baseline_calls(self):
        filtered = agent._filter_workflow_tool_calls(
            [
                "routing:subagent_start:worker",
                "approval:Bash",
                "memory_search",
                "mcp__memory__memory_search",
                "gmail_search",
                "calendar_search",
            ]
        )
        self.assertEqual(filtered, ["gmail_search", "calendar_search"])


class TestSkillAnalytics(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="skill-analytics-"))
        self.db_path = self.tmpdir / "mollygraph.db"
        _create_analytics_tables(self.db_path)
        self.old_path = config.MOLLYGRAPH_PATH
        config.MOLLYGRAPH_PATH = self.db_path

    def tearDown(self):
        config.MOLLYGRAPH_PATH = self.old_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_underperforming_and_skill_stats(self):
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(str(self.db_path))
        conn.executemany(
            """
            INSERT INTO skill_executions (id, skill_name, trigger, outcome, user_approval, edits_made, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("a1", "email-followup", "", "success", "", "", now),
                ("a2", "email-followup", "", "failed", "", "", now),
                ("a3", "email-followup", "", "failed", "", "", now),
                ("b1", "calendar-cleanup", "", "completed", "", "", now),
                ("b2", "calendar-cleanup", "", "failed", "", "", now),
                ("b3", "calendar-cleanup", "", "success", "", "", now),
                ("b4", "calendar-cleanup", "", "failed", "", "", now),
                ("b5", "calendar-cleanup", "", "failed", "", "", now),
                ("c1", "notes-refactor", "", "success", "", "", now),
                ("c2", "notes-refactor", "", "success", "", "", now),
                ("c3", "notes-refactor", "", "success", "", "", now),
                ("c4", "notes-refactor", "", "success", "", "", now),
                ("c5", "notes-refactor", "", "success", "", "", now),
                ("c6", "notes-refactor", "", "success", "", "", now),
            ],
        )
        conn.commit()
        conn.close()

        stats = get_skill_stats("calendar-cleanup")
        self.assertEqual(stats["invocations"], 5)
        self.assertEqual(stats["successes"], 2)
        self.assertEqual(stats["failures"], 3)
        self.assertAlmostEqual(stats["success_rate"], 0.4)

        underperforming = get_underperforming_skills(min_invocations=5, max_success_rate=0.6)
        self.assertEqual([item["skill_name"] for item in underperforming], ["calendar-cleanup"])

    def test_gap_clusters_are_deterministic(self):
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=40)).isoformat()
        conn = sqlite3.connect(str(self.db_path))
        conn.executemany(
            """
            INSERT INTO skill_gaps (user_message, tools_used, session_id, created_at, addressed)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    "Draft follow up email to candidate for interview status",
                    json.dumps(["gmail_search", "gmail_draft", "gmail_send"]),
                    "s1",
                    now.isoformat(),
                    0,
                ),
                (
                    "Send follow up email to recruiter about interview response",
                    json.dumps(["gmail_search", "gmail_draft", "gmail_send"]),
                    "s1",
                    now.isoformat(),
                    0,
                ),
                (
                    "Plan weekly grocery list and reminders for home",
                    json.dumps(["reminders", "notes", "calendar"]),
                    "s2",
                    now.isoformat(),
                    0,
                ),
                (
                    "Set reminders for grocery shopping and family meals",
                    json.dumps(["reminders", "calendar", "notes"]),
                    "s3",
                    now.isoformat(),
                    0,
                ),
                (
                    "Old email request should be out of range",
                    json.dumps(["gmail_search", "gmail_draft", "gmail_send"]),
                    "s4",
                    old,
                    0,
                ),
                (
                    "Addressed gap should be ignored",
                    json.dumps(["gmail_search", "gmail_draft", "gmail_send"]),
                    "s5",
                    now.isoformat(),
                    1,
                ),
            ],
        )
        conn.commit()
        conn.close()

        first = get_skill_gap_clusters(days=30)
        second = get_skill_gap_clusters(days=30)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(first[0]["gap_count"], 2)
        self.assertIn("email", first[0]["top_keywords"])
        self.assertEqual(first[1]["gap_count"], 2)
        self.assertIn("grocery", first[1]["top_keywords"])


if __name__ == "__main__":
    unittest.main()
