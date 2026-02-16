import sqlite3
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import config
import db_pool
import health


class TestHealthSkillObservability(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="health-skill-observability-"))
        self.db_path = self.temp_dir / "mollygraph.sqlite"
        conn = db_pool.sqlite_connect(str(self.db_path))
        conn.executescript(
            """
            CREATE TABLE skill_executions (
                id TEXT PRIMARY KEY,
                skill_name TEXT,
                trigger TEXT,
                outcome TEXT,
                user_approval TEXT,
                edits_made TEXT,
                created_at TEXT
            );
            CREATE TABLE tool_calls (
                id TEXT PRIMARY KEY,
                tool_name TEXT,
                parameters TEXT,
                success INTEGER,
                latency_ms INTEGER,
                error_message TEXT,
                user_feedback TEXT,
                created_at TEXT
            );
            """
        )
        conn.commit()
        conn.close()

    def _insert_skill(self, outcome: str):
        conn = db_pool.sqlite_connect(str(self.db_path))
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO skill_executions
            (id, skill_name, trigger, outcome, user_approval, edits_made, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), "daily-digest", "whatsapp:prompt", outcome, "", "", now),
        )
        conn.commit()
        conn.close()

    def _insert_tool_call(self, tool_name: str):
        conn = db_pool.sqlite_connect(str(self.db_path))
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO tool_calls
            (id, tool_name, parameters, success, latency_ms, error_message, user_feedback, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), tool_name, "{}", 1, 0, "", "", now),
        )
        conn.commit()
        conn.close()

    def test_skill_execution_volume_red_when_no_events(self):
        with patch.object(config, "MOLLYGRAPH_PATH", self.db_path):
            status, detail = health.HealthDoctor()._skill_execution_volume_check()

        self.assertEqual(status, "red")
        self.assertIn("executions=0", detail)

    def test_skill_execution_volume_yellow_when_below_low_watermark(self):
        self._insert_skill("success")
        self._insert_skill("success")

        with patch.object(config, "MOLLYGRAPH_PATH", self.db_path), patch.object(
            health, "HEALTH_SKILL_LOW_WATERMARK", 3
        ):
            status, detail = health.HealthDoctor()._skill_execution_volume_check()

        self.assertEqual(status, "yellow")
        self.assertIn("executions=2", detail)

    def test_skill_vs_bash_ratio_red_when_bash_dominates(self):
        self._insert_skill("success")
        for _ in range(5):
            self._insert_tool_call("Bash")

        with patch.object(config, "MOLLYGRAPH_PATH", self.db_path), patch.object(
            health, "HEALTH_SKILL_BASH_RATIO_RED", 0.30
        ), patch.object(health, "HEALTH_SKILL_BASH_RATIO_YELLOW", 0.75):
            status, detail = health.HealthDoctor()._skill_vs_direct_bash_ratio_check()

        self.assertEqual(status, "red")
        self.assertIn("ratio=0.20", detail)

    def test_skill_vs_bash_ratio_green_when_skills_keep_up(self):
        for _ in range(4):
            self._insert_skill("success")
        for _ in range(2):
            self._insert_tool_call("Bash")

        with patch.object(config, "MOLLYGRAPH_PATH", self.db_path), patch.object(
            health, "HEALTH_SKILL_BASH_RATIO_RED", 0.30
        ), patch.object(health, "HEALTH_SKILL_BASH_RATIO_YELLOW", 0.75):
            status, detail = health.HealthDoctor()._skill_vs_direct_bash_ratio_check()

        self.assertEqual(status, "green")
        self.assertIn("ratio=2.00", detail)


if __name__ == "__main__":
    unittest.main()
