import sqlite3
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.issue_registry import (
    build_issue_fingerprint,
    ensure_issue_registry_tables,
    should_notify,
    upsert_issue,
)


class TestIssueRegistrySchema(unittest.TestCase):
    def test_tables_and_indexes_created(self):
        conn = sqlite3.connect(":memory:")
        ensure_issue_registry_tables(conn)

        issue_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(maintenance_issues)").fetchall()
        }
        self.assertTrue(
            {
                "fingerprint",
                "check_id",
                "severity",
                "status",
                "first_seen",
                "last_seen",
                "consecutive_failures",
                "last_detail",
                "source",
            }.issubset(issue_cols)
        )

        event_cols = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(maintenance_issue_events)"
            ).fetchall()
        }
        self.assertTrue(
            {
                "issue_fingerprint",
                "event_type",
                "created_at",
                "payload",
            }.issubset(event_cols)
        )

        issue_indexes = {
            row[1]
            for row in conn.execute("PRAGMA index_list(maintenance_issues)").fetchall()
        }
        event_indexes = {
            row[1]
            for row in conn.execute(
                "PRAGMA index_list(maintenance_issue_events)"
            ).fetchall()
        }
        self.assertIn("idx_maintenance_issues_status", issue_indexes)
        self.assertIn("idx_maintenance_issues_last_seen", issue_indexes)
        self.assertIn("idx_maintenance_issue_events_fingerprint", event_indexes)
        self.assertIn("idx_maintenance_issue_events_created_at", event_indexes)

        conn.close()

    def test_schema_backfill_adds_missing_columns(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE maintenance_issues (
                fingerprint TEXT PRIMARY KEY,
                check_id TEXT
            );
            CREATE TABLE maintenance_issue_events (
                issue_fingerprint TEXT,
                event_type TEXT
            );
            """
        )
        conn.commit()

        ensure_issue_registry_tables(conn)

        issue_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(maintenance_issues)").fetchall()
        }
        event_cols = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(maintenance_issue_events)"
            ).fetchall()
        }
        self.assertIn("status", issue_cols)
        self.assertIn("consecutive_failures", issue_cols)
        self.assertIn("last_detail", issue_cols)
        self.assertIn("payload", event_cols)
        self.assertIn("created_at", event_cols)
        conn.close()


class TestIssueFingerprint(unittest.TestCase):
    def test_fingerprint_normalizes_noisy_detail(self):
        detail_a = (
            "Timeout for host 10.0.0.8 at 2026-02-09T10:11:12Z "
            "request_id=123e4567-e89b-12d3-a456-426614174000 code=503"
        )
        detail_b = (
            "Timeout for host 10.9.9.2 at 2026-02-09T13:42:59Z "
            "request_id=923e4567-e89b-12d3-a456-426614174999 code=500"
        )
        fp_a = build_issue_fingerprint(
            check_id="quality.operational_tables",
            detail=detail_a,
            source="Health",
        )
        fp_b = build_issue_fingerprint(
            check_id="quality.operational_tables",
            detail=detail_b,
            source="health",
        )
        self.assertEqual(fp_a, fp_b)


class TestIssueLifecycle(unittest.TestCase):
    def test_upsert_tracks_failures_and_events(self):
        conn = sqlite3.connect(":memory:")
        ensure_issue_registry_tables(conn)

        t1 = datetime(2026, 2, 9, 10, 0, tzinfo=timezone.utc)
        t2 = t1 + timedelta(minutes=5)
        t3 = t1 + timedelta(minutes=10)

        created = upsert_issue(
            conn,
            check_id="quality.operational_tables",
            severity="red",
            detail="skill_executions=0 corrections=0",
            source="health",
            observed_at=t1,
        )
        self.assertEqual(created["event_type"], "created")
        self.assertEqual(created["status"], "open")
        self.assertEqual(created["consecutive_failures"], 1)

        observed = upsert_issue(
            conn,
            check_id="quality.operational_tables",
            severity="red",
            detail="skill_executions=0 corrections=0 at 2026-02-09T10:05:00Z",
            source="health",
            observed_at=t2,
        )
        self.assertEqual(observed["event_type"], "observed")
        self.assertEqual(observed["consecutive_failures"], 2)

        resolved = upsert_issue(
            conn,
            check_id="quality.operational_tables",
            severity="green",
            detail="skill_executions=12 corrections=2",
            source="health",
            observed_at=t3,
        )
        self.assertEqual(resolved["event_type"], "status_changed")
        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(resolved["consecutive_failures"], 0)
        self.assertTrue(resolved["severity_changed"])

        issue_row = conn.execute(
            """
            SELECT severity, status, first_seen, last_seen, consecutive_failures
            FROM maintenance_issues
            WHERE fingerprint = ?
            """,
            (created["fingerprint"],),
        ).fetchone()
        self.assertEqual(issue_row[0], "green")
        self.assertEqual(issue_row[1], "resolved")
        self.assertEqual(issue_row[2], t1.isoformat())
        self.assertEqual(issue_row[3], t3.isoformat())
        self.assertEqual(issue_row[4], 0)

        event_types = [
            row[0]
            for row in conn.execute(
                """
                SELECT event_type
                FROM maintenance_issue_events
                WHERE issue_fingerprint = ?
                ORDER BY created_at
                """,
                (created["fingerprint"],),
            ).fetchall()
        ]
        self.assertEqual(event_types, ["created", "observed", "status_changed"])
        conn.close()


class TestShouldNotify(unittest.TestCase):
    def test_requires_fingerprint(self):
        self.assertFalse(
            should_notify(
                fingerprint="",
                cooldown_hours=4,
                last_notified_at=None,
                severity_changed=False,
            )
        )

    def test_severity_change_bypasses_cooldown(self):
        now = datetime.now(timezone.utc)
        self.assertTrue(
            should_notify(
                fingerprint="abc123",
                cooldown_hours=24,
                last_notified_at=now.isoformat(),
                severity_changed=True,
            )
        )

    def test_cooldown_blocks_recent_repeat(self):
        recent = datetime.now(timezone.utc) - timedelta(minutes=20)
        self.assertFalse(
            should_notify(
                fingerprint="abc123",
                cooldown_hours=1,
                last_notified_at=recent.isoformat(),
                severity_changed=False,
            )
        )

    def test_cooldown_allows_after_window(self):
        old = datetime.now(timezone.utc) - timedelta(hours=5)
        self.assertTrue(
            should_notify(
                fingerprint="abc123",
                cooldown_hours=1,
                last_notified_at=old.isoformat(),
                severity_changed=False,
            )
        )

    def test_invalid_last_notified_allows(self):
        self.assertTrue(
            should_notify(
                fingerprint="abc123",
                cooldown_hours=2,
                last_notified_at="not-a-timestamp",
                severity_changed=False,
            )
        )


class TestVectorStoreHook(unittest.TestCase):
    def test_vectorstore_init_calls_issue_registry_schema(self):
        source = (PROJECT_ROOT / "memory" / "vectorstore.py").read_text()
        self.assertIn("ensure_issue_registry_tables(self.conn)", source)


if __name__ == "__main__":
    unittest.main()
