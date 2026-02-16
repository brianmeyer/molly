import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import config
import db_pool
from monitoring.agents.track_f_preprod import (
    _check_result,
    _foundry_ingestion_health_check,
    _parser_compatibility_check,
    _promotion_drift_status_check,
    _skill_telemetry_presence_check,
    run_track_f_audit,
)
from monitoring.health import HealthDoctor


def _seed_operational_tables(db_path: Path):
    conn = db_pool.sqlite_connect(str(db_path))
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


def _insert_skill_execution(db_path: Path, created_at: str):
    conn = db_pool.sqlite_connect(str(db_path))
    conn.execute(
        """
        INSERT INTO skill_executions
        (id, skill_name, trigger, outcome, user_approval, edits_made, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("skill-1", "Workflow skill", "pattern", "ok", "approved", "none", created_at),
    )
    conn.commit()
    conn.close()


def _insert_self_improvement_events(db_path: Path, rows: list[tuple[str, str]]):
    conn = db_pool.sqlite_connect(str(db_path))
    now = datetime.now(timezone.utc).isoformat()
    for idx, (category, status) in enumerate(rows, start=1):
        conn.execute(
            """
            INSERT INTO self_improvement_events
            (id, event_type, category, title, payload, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"evt-{idx}",
                "proposal",
                category,
                f"title-{idx}",
                "{}",
                status,
                now,
                now,
            ),
        )
    conn.commit()
    conn.close()


class TestTrackFConfigDefaults(unittest.TestCase):
    def test_defaults_are_report_only_and_non_enforcing(self):
        self.assertTrue(config.TRACK_F_REPORT_ONLY)
        self.assertFalse(config.TRACK_F_ENFORCE_PARSER_COMPAT)
        self.assertFalse(config.TRACK_F_ENFORCE_SKILL_TELEMETRY)
        self.assertFalse(config.TRACK_F_ENFORCE_FOUNDRY_INGESTION)
        self.assertFalse(config.TRACK_F_ENFORCE_PROMOTION_DRIFT)


class TestTrackFPreprodAuditChecks(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path(tempfile.mkdtemp(prefix="track-f-audit-tests-"))
        self.mollygraph = self.temp_root / "mollygraph.db"
        self.health_dir = self.temp_root / "health"
        self.audit_dir = self.temp_root / "audits"

    def test_parser_compatibility_is_green(self):
        check = _parser_compatibility_check()
        self.assertEqual(check.status, "green")

    def test_skill_telemetry_missing_table_is_report_only_by_default(self):
        with patch.object(
            config, "MOLLYGRAPH_PATH", self.mollygraph
        ), patch.object(config, "TRACK_F_REPORT_ONLY", True), patch.object(
            config, "TRACK_F_ENFORCE_SKILL_TELEMETRY", False
        ):
            check = _skill_telemetry_presence_check()

        self.assertEqual(check.status, "yellow")
        self.assertFalse(check.action_required)
        self.assertIn("report-only", check.detail)

    def test_skill_telemetry_missing_table_can_be_hard_enforced(self):
        with patch.object(
            config, "MOLLYGRAPH_PATH", self.mollygraph
        ), patch.object(config, "TRACK_F_REPORT_ONLY", False), patch.object(
            config, "TRACK_F_ENFORCE_SKILL_TELEMETRY", True
        ):
            check = _skill_telemetry_presence_check()

        self.assertEqual(check.status, "red")
        self.assertTrue(check.action_required)
        self.assertIn("enforced", check.detail)

    def test_foundry_and_promotion_checks_go_green_with_healthy_data(self):
        _seed_operational_tables(self.mollygraph)
        _insert_self_improvement_events(
            self.mollygraph,
            [
                ("skill", "proposed"),
                ("skill", "approved"),
                ("tool", "proposed"),
                ("tool", "approved"),
                ("core", "deployed"),
            ],
        )

        with patch.object(
            config, "MOLLYGRAPH_PATH", self.mollygraph
        ), patch.object(config, "TRACK_F_REPORT_ONLY", True), patch.object(
            config, "TRACK_F_ENFORCE_FOUNDRY_INGESTION", False
        ), patch.object(config, "TRACK_F_ENFORCE_PROMOTION_DRIFT", False):
            foundry = _foundry_ingestion_health_check()
            drift = _promotion_drift_status_check()

        self.assertEqual(foundry.status, "green")
        self.assertEqual(drift.status, "green")

    def test_run_track_f_preprod_audit_writes_markdown_report(self):
        _seed_operational_tables(self.mollygraph)
        now = datetime.now(timezone.utc).isoformat()
        _insert_skill_execution(self.mollygraph, created_at=now)
        _insert_self_improvement_events(
            self.mollygraph,
            [
                ("skill", "proposed"),
                ("skill", "approved"),
                ("tool", "proposed"),
                ("tool", "approved"),
                ("core", "deployed"),
            ],
        )

        with patch.object(config, "HEALTH_REPORT_DIR", self.health_dir), patch.object(
            config, "TRACK_F_AUDIT_DIR", self.audit_dir
        ), patch.object(config, "MOLLYGRAPH_PATH", self.mollygraph):
            report_path = run_track_f_audit()

        self.assertTrue(report_path.exists())
        text = report_path.read_text()
        self.assertIn("trackf.parser_compatibility", text)
        self.assertIn("trackf.skill_telemetry_presence", text)
        self.assertIn("trackf.foundry_ingestion_health", text)
        self.assertIn("trackf.promotion_drift_status", text)
        self.assertIn("## Verdict:", text)


if __name__ == "__main__":
    unittest.main()
