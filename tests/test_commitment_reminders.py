import unittest
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from approval import get_action_tier
from automations import _extract_commitment_title, _extract_due_datetime, _titles_similar


class TestCommitmentParsing(unittest.TestCase):
    def test_extract_commitment_title_strips_due_phrase(self):
        text = "remind me to call Dave about the proposal tomorrow at 3pm"
        title = _extract_commitment_title(text)
        self.assertEqual(title.lower(), "call dave about the proposal")

    def test_extract_due_datetime_tomorrow_with_time(self):
        now_utc = datetime(2026, 2, 8, 15, 0, tzinfo=timezone.utc)
        due = _extract_due_datetime("remind me to call Dave tomorrow at 3pm", now_utc)
        self.assertIsNotNone(due)
        # 3pm America/New_York on Feb 9, 2026 -> 20:00 UTC
        self.assertEqual(due.isoformat(), "2026-02-09T20:00:00+00:00")

    def test_title_similarity_handles_compact_variation(self):
        self.assertTrue(
            _titles_similar(
                "Call Dave about the proposal",
                "call dave proposal",
            )
        )


class TestReminderTiering(unittest.TestCase):
    def test_reminders_list_operation_auto(self):
        tier = get_action_tier("reminders", {"operation": "list"})
        self.assertEqual(tier, "AUTO")

    def test_reminders_create_operation_confirm(self):
        tier = get_action_tier("reminders", {"operation": "create"})
        self.assertEqual(tier, "CONFIRM")

    def test_list_reminders_alias_auto(self):
        tier = get_action_tier("list_reminders", {})
        self.assertEqual(tier, "AUTO")


if __name__ == "__main__":
    unittest.main()

