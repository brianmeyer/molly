import sqlite3
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from self_improve import SelfImprovementEngine


class _RecordingVectorStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        conn = sqlite3.connect(str(self.db_path))
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skill_executions (
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
        conn.commit()
        conn.close()

    def log_self_improvement_event(
        self,
        event_type: str,
        category: str,
        title: str,
        payload: str = "",
        status: str = "proposed",
    ) -> str:
        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT INTO self_improvement_events
               (id, event_type, category, title, payload, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_id, event_type, category, title, payload, status, now, now),
        )
        conn.commit()
        conn.close()
        return event_id

    def update_self_improvement_event_status(self, event_id: str, status: str):
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "UPDATE self_improvement_events SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, event_id),
        )
        conn.commit()
        conn.close()

    def log_skill_execution(self, *args, **kwargs):
        raise AssertionError("Skill lifecycle flow must not log into skill_executions")


class TestSelfImproveSkillLifecycle(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="self-improve-skill-lifecycle-"))
        self.skills_dir = self.temp_dir / "workspace-skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.temp_dir / "mollygraph.db"
        self.vectorstore = _RecordingVectorStore(self.db_path)

        self.old_skills_dir = config.SKILLS_DIR
        self.old_mollygraph_path = config.MOLLYGRAPH_PATH
        config.SKILLS_DIR = self.skills_dir
        config.MOLLYGRAPH_PATH = self.db_path

        self.vectorstore_patcher = patch(
            "memory.retriever.get_vectorstore",
            return_value=self.vectorstore,
        )
        self.vectorstore_patcher.start()

    def tearDown(self):
        self.vectorstore_patcher.stop()
        config.SKILLS_DIR = self.old_skills_dir
        config.MOLLYGRAPH_PATH = self.old_mollygraph_path

    def _proposal(self, name: str = "Workflow Review skill") -> dict:
        return {
            "name": name,
            "trigger": ["When owner asks for workflow review"],
            "tools": ["Bash", "Write"],
            "steps": ["Collect inputs", "Run checks", "Summarize findings"],
            "guardrails": ["Keep output concise", "Confirm before writes"],
            "source": "test",
            "metadata": {"origin": "unit-test"},
        }

    def _read_events(self) -> list[tuple[str, str, str, str]]:
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute(
            "SELECT event_type, category, title, status FROM self_improvement_events ORDER BY created_at"
        ).fetchall()
        conn.close()
        return [(str(a), str(b), str(c), str(d)) for a, b, c, d in rows]

    def _skill_executions_count(self) -> int:
        conn = sqlite3.connect(str(self.db_path))
        value = int(conn.execute("SELECT COUNT(*) FROM skill_executions").fetchone()[0])
        conn.close()
        return value

    async def test_new_skill_pending_yes_activates_and_logs_in_self_improvement_events(self):
        engine = SelfImprovementEngine()
        proposal = self._proposal()
        live_path = self.skills_dir / "workflow-review-skill.md"
        pending_path = self.skills_dir / "workflow-review-skill.md.pending"

        with patch.object(engine, "initialize", new=AsyncMock()), \
                patch.object(engine, "_request_owner_decision", new=AsyncMock(return_value=True)):
            result = await engine.propose_skill_lifecycle(proposal)

        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["mode"], "pending")
        self.assertTrue(live_path.exists())
        self.assertFalse(pending_path.exists())
        self.assertEqual(self._skill_executions_count(), 0)

        events = self._read_events()
        self.assertTrue(any(row[0] == "proposal" and row[1] == "skill" and row[3] == "completed" for row in events))
        self.assertTrue(any(row[0] == "lifecycle" and row[1] == "skill" and row[3] == "activated" for row in events))

    async def test_new_skill_pending_no_rejects_and_records_cooldown_fingerprint(self):
        engine = SelfImprovementEngine()
        proposal = self._proposal(name="Pipeline Mapper skill")
        live_path = self.skills_dir / "pipeline-mapper-skill.md"
        pending_path = self.skills_dir / "pipeline-mapper-skill.md.pending"

        with patch.object(engine, "initialize", new=AsyncMock()), \
                patch.object(
                    engine,
                    "_request_owner_decision",
                    new=AsyncMock(return_value=("deny", "too broad")),
                ):
            result = await engine.propose_skill_lifecycle(proposal)

        self.assertEqual(result["status"], "rejected")
        self.assertIn("too broad", result["reason"])
        self.assertFalse(live_path.exists())
        self.assertFalse(pending_path.exists())

        events = self._read_events()
        self.assertTrue(any(row[1] == "skill" and row[3] == "rejected" for row in events))
        self.assertTrue(any(row[1] == "skill-cooldown" and row[3] == "rejected" for row in events))

    async def test_new_skill_pending_edit_updates_pending_then_approves(self):
        engine = SelfImprovementEngine()
        proposal = self._proposal(name="Weekly Ops skill")
        live_path = self.skills_dir / "weekly-ops-skill.md"

        decision_mock = AsyncMock(side_effect=["steps: Gather data -> Validate results -> Share summary", True])
        with patch.object(engine, "initialize", new=AsyncMock()), \
                patch.object(engine, "_request_owner_decision", new=decision_mock):
            result = await engine.propose_skill_lifecycle(proposal)

        self.assertEqual(result["status"], "approved")
        self.assertEqual(decision_mock.await_count, 2)
        text = live_path.read_text()
        self.assertIn("1. Gather data", text)
        self.assertIn("2. Validate results", text)
        self.assertIn("3. Share summary", text)

    async def test_pending_edit_yes_replaces_original_skill(self):
        engine = SelfImprovementEngine()
        base = self._proposal(name="Release Review skill")
        live_path = self.skills_dir / "release-review-skill.md"
        live_path.write_text(engine._render_skill_markdown(base))

        updated = dict(base)
        updated["steps"] = ["Collect release notes", "Validate dependencies", "Publish rollout summary"]
        pending_edit_path = self.skills_dir / "release-review-skill.md.pending-edit"

        with patch.object(engine, "initialize", new=AsyncMock()), \
                patch.object(engine, "_request_owner_decision", new=AsyncMock(return_value=True)):
            result = await engine.propose_skill_lifecycle(updated)

        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["mode"], "pending-edit")
        self.assertFalse(pending_edit_path.exists())
        text = live_path.read_text()
        self.assertIn("1. Collect release notes", text)
        self.assertIn("2. Validate dependencies", text)
        self.assertIn("3. Publish rollout summary", text)

    async def test_pending_edit_no_discards_copy_and_preserves_original(self):
        engine = SelfImprovementEngine()
        base = self._proposal(name="Incident Triage skill")
        live_path = self.skills_dir / "incident-triage-skill.md"
        original_text = engine._render_skill_markdown(base)
        live_path.write_text(original_text)

        updated = dict(base)
        updated["steps"] = ["Collect incidents", "Rank by severity", "Send escalation note"]
        pending_edit_path = self.skills_dir / "incident-triage-skill.md.pending-edit"

        with patch.object(engine, "initialize", new=AsyncMock()), \
                patch.object(
                    engine,
                    "_request_owner_decision",
                    new=AsyncMock(return_value=("deny", "keep current wording")),
                ):
            result = await engine.propose_skill_lifecycle(updated)

        self.assertEqual(result["status"], "rejected")
        self.assertFalse(pending_edit_path.exists())
        self.assertEqual(live_path.read_text(), original_text)

    async def test_rejection_cooldown_persists_and_blocks_reproposal_for_30_days(self):
        proposal = self._proposal(name="Context Harvest skill")

        engine_a = SelfImprovementEngine()
        decision_mock_a = AsyncMock(return_value=("deny", "not needed"))
        with patch.object(engine_a, "initialize", new=AsyncMock()), \
                patch.object(engine_a, "_request_owner_decision", new=decision_mock_a):
            first = await engine_a.propose_skill_lifecycle(proposal)
        self.assertEqual(first["status"], "rejected")

        engine_b = SelfImprovementEngine()
        decision_mock_b = AsyncMock(return_value=True)
        with patch.object(engine_b, "initialize", new=AsyncMock()), \
                patch.object(engine_b, "_request_owner_decision", new=decision_mock_b):
            second = await engine_b.propose_skill_lifecycle(proposal)

        self.assertEqual(second["status"], "skipped")
        self.assertEqual(second["reason"], "rejection cooldown active")
        self.assertEqual(decision_mock_b.await_count, 0)


if __name__ == "__main__":
    unittest.main()
