import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from automations import AutomationEngine, propose_automation


class TestAutomationProposalMining(unittest.IsolatedAsyncioTestCase):
    def _build_recurring_logs(self, occurrences: int) -> list[dict]:
        base = datetime(2026, 1, 5, 14, 10, tzinfo=timezone.utc)
        logs: list[dict] = []
        for i in range(occurrences):
            start = base + timedelta(days=i)
            logs.append(
                {
                    "tool_name": "gmail_search",
                    "created_at": start.isoformat(),
                    "success": 1,
                    "latency_ms": 220,
                    "parameters": '{"query":"is:unread newer_than:1d"}',
                }
            )
            logs.append(
                {
                    "tool_name": "gmail_read",
                    "created_at": (start + timedelta(minutes=1)).isoformat(),
                    "success": 1,
                    "latency_ms": 180,
                    "parameters": '{"id":"sample-message-id"}',
                }
            )
            logs.append(
                {
                    "tool_name": "calendar_list",
                    "created_at": (start + timedelta(minutes=2)).isoformat(),
                    "success": True,
                    "latency_ms": 160,
                }
            )
            logs.append(
                {
                    "tool_name": "whatsapp_search",
                    "created_at": (start + timedelta(minutes=20)).isoformat(),
                    "success": True,
                }
            )
        return logs

    def test_empty_logs_returns_none(self):
        self.assertIsNone(propose_automation([]))

    def test_low_signal_logs_returns_none(self):
        logs = [
            {"tool_name": "gmail_search", "created_at": "2026-01-01T10:00:00+00:00", "success": 1},
            {"tool_name": "calendar_list", "created_at": "2026-01-01T10:03:00+00:00", "success": 1},
            {"tool_name": "imessage_recent", "created_at": "2026-01-01T10:10:00+00:00", "success": 1},
            {"tool_name": "contacts", "created_at": "2026-01-01T10:20:00+00:00", "success": 1},
        ]
        self.assertIsNone(propose_automation(logs))

    def test_malformed_logs_are_ignored(self):
        logs = [
            None,
            {"foo": "bar"},
            {"tool_name": "gmail_search", "created_at": "not-a-date", "success": True},
            {"tool_name": "", "created_at": "2026-01-01T10:00:00+00:00"},
        ]
        self.assertIsNone(propose_automation(logs))

    async def test_strong_pattern_generates_data_driven_proposal_and_loads(self):
        min_occurrences = max(3, int(getattr(config, "AUTOMATION_MIN_PATTERN_COUNT", 3)))
        logs = self._build_recurring_logs(occurrences=min_occurrences + 3)
        proposal_yaml = propose_automation(logs)

        self.assertIsNotNone(proposal_yaml)
        parsed = yaml.safe_load(proposal_yaml)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed.get("trigger", {}).get("type"), "schedule")
        self.assertIn("cron", parsed.get("trigger", {}))
        self.assertIsInstance(parsed.get("conditions"), list)
        self.assertIsInstance(parsed.get("pipeline"), list)
        self.assertGreaterEqual(len(parsed.get("pipeline", [])), 2)

        meta = parsed.get("meta", {})
        self.assertIsInstance(meta, dict)
        self.assertIn("why_proposed", meta)
        self.assertIn("proposal_confidence", meta)
        self.assertIn("evidence", meta)
        evidence = meta.get("evidence", {})
        self.assertGreaterEqual(evidence.get("top_pattern", {}).get("occurrence_count", 0), min_occurrences)
        self.assertGreater(len(evidence.get("sample_events", [])), 0)

        confidence = float(meta.get("proposal_confidence", 0.0))
        threshold = float(meta.get("auto_enable_threshold", 1.0))
        if confidence < threshold:
            self.assertFalse(bool(parsed.get("enabled")))

        tmpdir = Path(tempfile.mkdtemp(prefix="automation-proposal-test-"))
        old_dir = config.AUTOMATIONS_DIR
        old_state = config.AUTOMATIONS_STATE_FILE
        try:
            config.AUTOMATIONS_DIR = tmpdir
            config.AUTOMATIONS_STATE_FILE = tmpdir / "state.json"
            (tmpdir / "proposal.yaml").write_text(proposal_yaml)

            engine = AutomationEngine(molly=None)
            await engine.load_automations()
            self.assertEqual(len(engine._automations), 1)
            loaded = next(iter(engine._automations.values()))
            self.assertTrue(loaded.trigger_cfg.get("cron"))
            self.assertIsInstance(loaded.conditions, list)
            self.assertIsInstance(loaded.pipeline, list)
        finally:
            config.AUTOMATIONS_DIR = old_dir
            config.AUTOMATIONS_STATE_FILE = old_state
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
