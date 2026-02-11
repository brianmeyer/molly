import asyncio
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from approval import get_action_tier
from automations import (
    _extract_commitment_title,
    _extract_due_datetime,
    _extract_due_datetime_with_llm,
    _titles_similar,
)


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

    def test_extract_title_strips_later(self):
        """Vague time words like 'later' should be stripped from titles."""
        text = "remind me to call Bob later"
        title = _extract_commitment_title(text)
        self.assertEqual(title.lower(), "call bob")

    def test_extract_title_strips_soon(self):
        text = "I'll review the PR soon"
        title = _extract_commitment_title(text)
        self.assertNotIn("soon", title.lower())

    def test_extract_title_strips_eventually(self):
        text = "remind me to clean my desk eventually"
        title = _extract_commitment_title(text)
        self.assertNotIn("eventually", title.lower())


class TestVagueTimeParsing(unittest.TestCase):
    """Tests for BUG-01 fix: vague time words like 'later', 'soon'."""

    def test_later_before_7pm_defaults_to_tonight(self):
        # 2 PM ET = 19:00 UTC
        now_utc = datetime(2026, 2, 10, 19, 0, tzinfo=timezone.utc)
        due = _extract_due_datetime("remind me later", now_utc)
        self.assertIsNotNone(due, "'later' should produce a due date, not None")
        # 7 PM ET on Feb 10 = midnight UTC Feb 11
        self.assertEqual(due.isoformat(), "2026-02-11T00:00:00+00:00")

    def test_later_after_7pm_defaults_to_tomorrow_9am(self):
        # 8 PM ET = 01:00 UTC next day
        now_utc = datetime(2026, 2, 11, 1, 0, tzinfo=timezone.utc)
        due = _extract_due_datetime("remind me later", now_utc)
        self.assertIsNotNone(due, "'later' should produce a due date, not None")
        # 9 AM ET on Feb 11 = 14:00 UTC Feb 11
        self.assertEqual(due.isoformat(), "2026-02-11T14:00:00+00:00")

    def test_soon_defaults_to_tonight(self):
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)  # 1 PM ET
        due = _extract_due_datetime("I'll do that soon", now_utc)
        self.assertIsNotNone(due, "'soon' should produce a due date")

    def test_in_a_bit_defaults_to_tonight(self):
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)  # 1 PM ET
        due = _extract_due_datetime("remind me in a bit", now_utc)
        self.assertIsNotNone(due, "'in a bit' should produce a due date")

    def test_unrecognized_phrase_still_returns_none(self):
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)
        due = _extract_due_datetime("remind me about the thing", now_utc)
        self.assertIsNone(due, "Non-vague text without a date should return None")

    def test_explicit_date_takes_precedence_over_later(self):
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)
        due = _extract_due_datetime("remind me tomorrow at 3pm, or later if needed", now_utc)
        self.assertIsNotNone(due)
        # "tomorrow" should take precedence
        self.assertIn("2026-02-11", due.isoformat())

    def test_sooner_is_not_matched_as_soon(self):
        """'sooner' should NOT trigger vague-time matching (word boundary check)."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)
        due = _extract_due_datetime("we need to do this sooner rather than, well, you know", now_utc)
        self.assertIsNone(due, "'sooner' should not match 'soon'")

    def test_laterally_is_not_matched_as_later(self):
        """'laterally' should NOT trigger vague-time matching."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)
        due = _extract_due_datetime("think laterally about this problem", now_utc)
        self.assertIsNone(due, "'laterally' should not match 'later'")


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

    def test_google_tasks_create_confirm(self):
        """Google Tasks create should be CONFIRM tier (called directly, not via MCP)."""
        tier = get_action_tier("tasks_create", {})
        self.assertEqual(tier, "CONFIRM")


class TestSyncCommitmentStatus(unittest.TestCase):
    """Tests for the _sync_commitment_status title-fallback fix (CRITICAL audit finding)."""

    def test_stale_apple_id_falls_back_to_title_match(self):
        """Commitments with stale Apple Reminder IDs should match via title."""
        from automations import AutomationEngine

        engine = AutomationEngine.__new__(AutomationEngine)
        engine._state_lock = asyncio.Lock()
        engine._state = {
            "automations": {
                "commitment-tracker": {
                    "commitments": [
                        {
                            "id": "cmt-test-001",
                            "title": "Call Dave about the proposal",
                            "status": "open",
                            "reminder_id": "apple-old-id-12345",  # stale Apple ID
                            "reminder_title": "",
                            "reminder_completed": False,
                            "due_at": "2026-02-10T20:00:00+00:00",
                            "completed_at": "",
                        },
                    ],
                    "completion_events": [],
                }
            }
        }
        engine._last_commitment_sync_at = None

        # Mock _list_molly_reminders to return Google Tasks (different IDs)
        google_tasks = [
            {
                "id": "google-task-xyz789",
                "title": "Call Dave about the proposal",
                "completed": True,
                "due_at": "2026-02-10T00:00:00.000Z",
                "completed_at": "2026-02-10T21:00:00.000Z",
                "notes": "",
            },
        ]
        engine._list_molly_reminders = AsyncMock(return_value=google_tasks)
        engine._save_state_locked = AsyncMock()

        now_utc = datetime(2026, 2, 11, 15, 0, tzinfo=timezone.utc)
        asyncio.run(engine._sync_commitment_status(now_utc))

        record = engine._state["automations"]["commitment-tracker"]["commitments"][0]
        self.assertEqual(record["status"], "completed", "Should sync via title-based fallback")
        self.assertEqual(record["reminder_id"], "google-task-xyz789", "Should update to Google Task ID")

    def test_sync_matches_by_id_when_available(self):
        """When the ID matches a Google Task, it should use the ID match directly."""
        from automations import AutomationEngine

        engine = AutomationEngine.__new__(AutomationEngine)
        engine._state_lock = asyncio.Lock()
        engine._state = {
            "automations": {
                "commitment-tracker": {
                    "commitments": [
                        {
                            "id": "cmt-test-002",
                            "title": "Review PR",
                            "status": "open",
                            "reminder_id": "google-task-abc",
                            "reminder_title": "",
                            "reminder_completed": False,
                            "due_at": "",
                            "completed_at": "",
                        },
                    ],
                    "completion_events": [],
                }
            }
        }
        engine._last_commitment_sync_at = None

        google_tasks = [
            {
                "id": "google-task-abc",
                "title": "Review PR",
                "completed": True,
                "due_at": "",
                "completed_at": "2026-02-10T15:00:00.000Z",
                "notes": "",
            },
        ]
        engine._list_molly_reminders = AsyncMock(return_value=google_tasks)
        engine._save_state_locked = AsyncMock()

        now_utc = datetime(2026, 2, 11, 15, 0, tzinfo=timezone.utc)
        asyncio.run(engine._sync_commitment_status(now_utc))

        record = engine._state["automations"]["commitment-tracker"]["commitments"][0]
        self.assertEqual(record["status"], "completed")


class TestQuietHoursNudge(unittest.TestCase):
    """Tests for quiet hours enforcement in _check_due_commitments."""

    def test_in_quiet_hours_wrapping_midnight(self):
        from heartbeat import _in_quiet_hours
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")

        # 11 PM ET (within 22:00-07:00 quiet hours)
        late_night = datetime(2026, 2, 10, 23, 0, tzinfo=tz)
        self.assertTrue(_in_quiet_hours(late_night), "11 PM should be in quiet hours")

        # 2 AM ET
        early_morning = datetime(2026, 2, 11, 2, 0, tzinfo=tz)
        self.assertTrue(_in_quiet_hours(early_morning), "2 AM should be in quiet hours")

        # 6:59 AM ET
        before_end = datetime(2026, 2, 11, 6, 59, tzinfo=tz)
        self.assertTrue(_in_quiet_hours(before_end), "6:59 AM should be in quiet hours")

    def test_not_in_quiet_hours_during_day(self):
        from heartbeat import _in_quiet_hours
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")

        # 2 PM ET (outside 22:00-07:00 quiet hours)
        afternoon = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        self.assertFalse(_in_quiet_hours(afternoon), "2 PM should not be in quiet hours")

        # 7:00 AM ET (boundary - should be outside)
        morning_boundary = datetime(2026, 2, 11, 7, 0, tzinfo=tz)
        self.assertFalse(_in_quiet_hours(morning_boundary), "7:00 AM should not be in quiet hours")

        # 9 PM ET (before 10 PM start)
        before_start = datetime(2026, 2, 10, 21, 0, tzinfo=tz)
        self.assertFalse(_in_quiet_hours(before_start), "9 PM should not be in quiet hours")

    def test_check_due_commitments_skips_quiet_hours(self):
        """_check_due_commitments should not send nudges during quiet hours."""
        from heartbeat import _check_due_commitments

        molly = MagicMock()
        molly.automations = MagicMock()
        molly.automations._state_lock = asyncio.Lock()
        molly.automations._state = {
            "automations": {
                "commitment-tracker": {
                    "commitments": [
                        {
                            "id": "cmt-test",
                            "title": "Overdue task",
                            "status": "open",
                            "due_at": "2026-02-10T20:00:00+00:00",
                            "last_nudged_at": "",
                        },
                    ]
                }
            }
        }

        # 11:30 PM ET = 04:30 UTC next day
        with patch("heartbeat.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 11, 4, 30, tzinfo=timezone.utc)
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            asyncio.run(
                _check_due_commitments(molly, "test-jid")
            )

        # send_surface_message should NOT have been called
        molly.send_surface_message.assert_not_called()


class TestTasklistCacheInvalidation(unittest.TestCase):
    """Tests for stale tasklist cache invalidation on 404."""

    def test_invalidate_tasklist_cache(self):
        from automations import AutomationEngine

        engine = AutomationEngine.__new__(AutomationEngine)
        engine._molly_tasklist_id = "some-cached-id"

        engine._invalidate_tasklist_cache()
        self.assertFalse(hasattr(engine, "_molly_tasklist_id"))

    def test_invalidate_tasklist_cache_no_attr(self):
        """Should not raise if cache is already empty."""
        from automations import AutomationEngine

        engine = AutomationEngine.__new__(AutomationEngine)
        engine._invalidate_tasklist_cache()  # Should not raise


class TestApplyReminderToRecord(unittest.TestCase):
    """Tests for _apply_reminder_to_record state transitions."""

    def test_midnight_due_at_does_not_overwrite_precise_value(self):
        """Google Tasks midnight stub should not overwrite a precise due_at."""
        from automations import AutomationEngine

        engine = AutomationEngine.__new__(AutomationEngine)
        record = {
            "id": "cmt-1",
            "status": "open",
            "due_at": "2026-02-10T19:00:00+00:00",  # 7PM precise
            "reminder_id": "",
            "reminder_title": "",
            "reminder_completed": False,
            "completed_at": "",
        }
        reminder = {
            "id": "task-abc",
            "title": "Call Bob",
            "due_at": "2026-02-10T00:00:00.000Z",  # midnight stub from Google
            "completed": False,
            "completed_at": "",
        }
        engine._apply_reminder_to_record(record, reminder, "2026-02-11T00:00:00+00:00")
        self.assertEqual(record["due_at"], "2026-02-10T19:00:00+00:00",
                         "Midnight stub should not overwrite precise due_at")

    def test_due_at_updated_when_record_empty(self):
        """If record has no due_at, even a midnight stub should be applied."""
        from automations import AutomationEngine

        engine = AutomationEngine.__new__(AutomationEngine)
        record = {
            "id": "cmt-2",
            "status": "open",
            "due_at": "",  # no existing value
            "reminder_id": "",
            "reminder_title": "",
            "reminder_completed": False,
            "completed_at": "",
        }
        reminder = {
            "id": "task-xyz",
            "title": "Review PR",
            "due_at": "2026-02-10T00:00:00.000Z",
            "completed": False,
            "completed_at": "",
        }
        engine._apply_reminder_to_record(record, reminder, "2026-02-11T00:00:00+00:00")
        self.assertEqual(record["due_at"], "2026-02-10T00:00:00.000Z",
                         "Empty due_at should be populated from Google Tasks")

    def test_completion_sets_status_and_completed_at(self):
        """Completing a reminder should set status=completed and completed_at."""
        from automations import AutomationEngine

        engine = AutomationEngine.__new__(AutomationEngine)
        record = {
            "id": "cmt-3",
            "status": "open",
            "due_at": "",
            "reminder_id": "task-1",
            "reminder_title": "Test",
            "reminder_completed": False,
            "completed_at": "",
        }
        reminder = {
            "id": "task-1",
            "title": "Test",
            "due_at": "",
            "completed": True,
            "completed_at": "2026-02-10T15:00:00+00:00",
        }
        changed = engine._apply_reminder_to_record(record, reminder, "2026-02-11T00:00:00+00:00")
        self.assertTrue(changed)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["completed_at"], "2026-02-10T15:00:00+00:00")

    def test_uncompletion_reopens_record(self):
        """If a completed reminder is uncompleted, record should reopen."""
        from automations import AutomationEngine

        engine = AutomationEngine.__new__(AutomationEngine)
        record = {
            "id": "cmt-4",
            "status": "completed",
            "due_at": "",
            "reminder_id": "task-2",
            "reminder_title": "Test",
            "reminder_completed": True,
            "completed_at": "2026-02-10T15:00:00+00:00",
        }
        reminder = {
            "id": "task-2",
            "title": "Test",
            "due_at": "",
            "completed": False,
            "completed_at": "",
        }
        changed = engine._apply_reminder_to_record(record, reminder, "2026-02-11T00:00:00+00:00")
        self.assertTrue(changed)
        self.assertEqual(record["status"], "open")
        self.assertEqual(record["completed_at"], "")


class TestQuietHoursBoundaries(unittest.TestCase):
    """Additional boundary tests for quiet hours."""

    def test_exact_start_boundary_is_in_quiet_hours(self):
        from heartbeat import _in_quiet_hours
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        at_start = datetime(2026, 2, 10, 22, 0, tzinfo=tz)
        self.assertTrue(_in_quiet_hours(at_start), "22:00 should be in quiet hours")

    def test_one_minute_before_start_is_not_quiet(self):
        from heartbeat import _in_quiet_hours
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        before_start = datetime(2026, 2, 10, 21, 59, tzinfo=tz)
        self.assertFalse(_in_quiet_hours(before_start), "21:59 should not be in quiet hours")


class TestParseTimeOffset(unittest.TestCase):
    """Deterministic tests for _parse_time_offset (no model required)."""

    def _now(self):
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        # Tuesday Feb 10, 2026 2:00 PM ET
        return datetime(2026, 2, 10, 14, 0, tzinfo=tz)

    def test_relative_hours(self):
        from memory.triage import _parse_time_offset

        result = _parse_time_offset("+2h", self._now())
        self.assertIsNotNone(result)
        # 2 PM ET + 2h = 4 PM ET = 21:00 UTC
        self.assertEqual(result.hour, 21)
        self.assertEqual(result.day, 10)

    def test_relative_days_and_time(self):
        from memory.triage import _parse_time_offset

        result = _parse_time_offset("+0d17:00", self._now())
        self.assertIsNotNone(result)
        # 5 PM ET on Feb 10 = 22:00 UTC
        self.assertEqual(result.isoformat(), "2026-02-10T22:00:00+00:00")

    def test_next_day_target(self):
        from memory.triage import _parse_time_offset

        result = _parse_time_offset("+1d09:00", self._now())
        self.assertIsNotNone(result)
        # 9 AM ET on Feb 11 = 14:00 UTC
        self.assertEqual(result.isoformat(), "2026-02-11T14:00:00+00:00")

    def test_day_name_saturday(self):
        from memory.triage import _parse_time_offset

        # Feb 10 is Tuesday, so Saturday is 4 days ahead (Feb 14)
        result = _parse_time_offset("+SAT10:00", self._now())
        self.assertIsNotNone(result)
        self.assertIn("2026-02-14", result.isoformat())
        # 10 AM ET = 15:00 UTC
        self.assertEqual(result.hour, 15)

    def test_day_name_monday_wraps_week(self):
        from memory.triage import _parse_time_offset

        # Feb 10 is Tuesday, so next Monday is 6 days ahead (Feb 16)
        result = _parse_time_offset("+MON09:00", self._now())
        self.assertIsNotNone(result)
        self.assertIn("2026-02-16", result.isoformat())

    def test_same_day_name_advances_full_week(self):
        from memory.triage import _parse_time_offset

        # Feb 10 is Tuesday; +TUE should go to *next* Tuesday (Feb 17)
        result = _parse_time_offset("+TUE10:00", self._now())
        self.assertIsNotNone(result)
        self.assertIn("2026-02-17", result.isoformat())

    def test_none_response(self):
        from memory.triage import _parse_time_offset

        self.assertIsNone(_parse_time_offset("NONE", self._now()))

    def test_empty_response(self):
        from memory.triage import _parse_time_offset

        self.assertIsNone(_parse_time_offset("", self._now()))

    def test_garbage_response(self):
        from memory.triage import _parse_time_offset

        self.assertIsNone(_parse_time_offset("I think maybe around 5pm?", self._now()))


class TestExtractDueDatetimeWithLLM(unittest.TestCase):
    """Tests for the async LLM-fallback wrapper."""

    def test_regex_match_skips_llm(self):
        """When regex succeeds, the LLM should not be called."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)
        with patch("memory.triage.parse_time_local_async") as mock_llm:
            # "tomorrow at 3pm" is handled by regex
            result = asyncio.run(
                _extract_due_datetime_with_llm("remind me tomorrow at 3pm", now_utc)
            )
            self.assertIsNotNone(result)
            mock_llm.assert_not_called()

    def test_vague_regex_match_skips_llm(self):
        """The 8 known vague words still use the fast regex path."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)
        with patch("memory.triage.parse_time_local_async") as mock_llm:
            result = asyncio.run(
                _extract_due_datetime_with_llm("remind me later", now_utc)
            )
            self.assertIsNotNone(result)
            mock_llm.assert_not_called()

    def test_llm_fallback_called_for_unknown_phrase(self):
        """When regex returns None, the LLM fallback should be invoked."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)
        expected = datetime(2026, 2, 10, 22, 0, tzinfo=timezone.utc)

        async def mock_parse(text, now_local):
            return expected

        with patch("memory.triage.parse_time_local_async", side_effect=mock_parse):
            result = asyncio.run(
                _extract_due_datetime_with_llm("remind me end of day", now_utc)
            )
            self.assertEqual(result, expected)

    def test_llm_failure_returns_none(self):
        """If the LLM raises, the wrapper should return None gracefully."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)

        async def mock_parse_fail(text, now_local):
            raise RuntimeError("model exploded")

        with patch("memory.triage.parse_time_local_async", side_effect=mock_parse_fail):
            result = asyncio.run(
                _extract_due_datetime_with_llm("remind me end of day", now_utc)
            )
            self.assertIsNone(result)


class TestTitleStripsNewPhrases(unittest.TestCase):
    """Verify that the expanded trailing-due regex strips LLM-handled phrases."""

    def test_strips_end_of_day(self):
        title = _extract_commitment_title("remind me to call Dave end of day")
        self.assertEqual(title.lower(), "call dave")

    def test_strips_after_lunch(self):
        title = _extract_commitment_title("remind me to review the doc after lunch")
        self.assertEqual(title.lower(), "review the doc")

    def test_strips_this_weekend(self):
        title = _extract_commitment_title("I'll clean the garage this weekend")
        self.assertEqual(title.lower(), "clean the garage")

    def test_strips_in_a_couple_hours(self):
        title = _extract_commitment_title("remind me to check on it in a couple hours")
        self.assertEqual(title.lower(), "check on it")

    def test_strips_before_dinner(self):
        title = _extract_commitment_title("I need to call mom before dinner")
        self.assertEqual(title.lower(), "call mom")

    def test_strips_next_week(self):
        title = _extract_commitment_title("I'll follow up on the proposal next week")
        self.assertEqual(title.lower(), "follow up on the proposal")


class TestParseTimeOffsetExtended(unittest.TestCase):
    """Extended edge-case tests for _parse_time_offset (Round 1 audit fixes)."""

    def _now(self):
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        # Tuesday Feb 10, 2026 2:00 PM ET
        return datetime(2026, 2, 10, 14, 0, tzinfo=tz)

    def test_think_tags_stripped(self):
        """Qwen3 <think>...</think> blocks should be stripped before parsing."""
        from memory.triage import _parse_time_offset

        raw = "<think>The user wants end of day so 5 PM.</think>+0d17:00"
        result = _parse_time_offset(raw, self._now())
        self.assertIsNotNone(result)
        self.assertEqual(result.isoformat(), "2026-02-10T22:00:00+00:00")

    def test_think_tags_multiline(self):
        """Multi-line <think> blocks should also be stripped."""
        from memory.triage import _parse_time_offset

        raw = "<think>\nLet me think about this.\nEnd of day = 5pm.\n</think>\n+0d17:00"
        result = _parse_time_offset(raw, self._now())
        self.assertIsNotNone(result)
        self.assertEqual(result.isoformat(), "2026-02-10T22:00:00+00:00")

    def test_think_tags_only_returns_none(self):
        """If only <think> tags with no offset, should return None."""
        from memory.triage import _parse_time_offset

        raw = "<think>I don't know when they mean.</think>"
        result = _parse_time_offset(raw, self._now())
        self.assertIsNone(result)

    def test_relative_minutes(self):
        """Test +Nm format for sub-hour resolution."""
        from memory.triage import _parse_time_offset

        result = _parse_time_offset("+30m", self._now())
        self.assertIsNotNone(result)
        # 2:00 PM ET + 30m = 2:30 PM ET = 19:30 UTC
        self.assertEqual(result.isoformat(), "2026-02-10T19:30:00+00:00")

    def test_relative_minutes_min_variant(self):
        """Test +NmIN variant."""
        from memory.triage import _parse_time_offset

        result = _parse_time_offset("+45min", self._now())
        self.assertIsNotNone(result)
        # 2:00 PM ET + 45m = 2:45 PM ET = 19:45 UTC
        self.assertEqual(result.isoformat(), "2026-02-10T19:45:00+00:00")

    def test_zero_hours(self):
        """Test +0h edge case: should return current time."""
        from memory.triage import _parse_time_offset

        result = _parse_time_offset("+0h", self._now())
        self.assertIsNotNone(result)
        # 2 PM ET + 0h = 2 PM ET = 19:00 UTC
        self.assertEqual(result.isoformat(), "2026-02-10T19:00:00+00:00")

    def test_zero_minutes(self):
        """Test +0m edge case."""
        from memory.triage import _parse_time_offset

        result = _parse_time_offset("+0m", self._now())
        self.assertIsNotNone(result)
        self.assertEqual(result.isoformat(), "2026-02-10T19:00:00+00:00")

    def test_hours_cap_at_8760(self):
        """Hours should be capped at 8760 (1 year) to prevent OverflowError."""
        from memory.triage import _parse_time_offset

        result = _parse_time_offset("+999999h", self._now())
        self.assertIsNotNone(result)
        # Should not raise; verify it caps at 8760 hours
        expected_max = self._now() + timedelta(hours=8760)
        self.assertEqual(result, expected_max.astimezone(timezone.utc))

    def test_days_cap_at_365(self):
        """Days should be capped at 365 to prevent OverflowError."""
        from memory.triage import _parse_time_offset

        result = _parse_time_offset("+99999d09:00", self._now())
        self.assertIsNotNone(result)
        # Should not raise; capped at 365 days

    def test_minutes_cap_at_1440(self):
        """Minutes should be capped at 1440 (24 hours) to prevent extreme values."""
        from memory.triage import _parse_time_offset

        result = _parse_time_offset("+999999m", self._now())
        self.assertIsNotNone(result)
        expected_max = self._now() + timedelta(minutes=1440)
        self.assertEqual(result, expected_max.astimezone(timezone.utc))

    def test_day_name_full_monday(self):
        """Test +MONDAY format (full day name)."""
        from memory.triage import _parse_time_offset

        # (?:DAY)? captures "DAY" suffix, so MONDAY -> MON + DAY
        result = _parse_time_offset("+MONDAY09:00", self._now())
        self.assertIsNotNone(result)
        # Feb 10 is Tuesday, next Monday is Feb 16
        self.assertIn("2026-02-16", result.isoformat())

    def test_day_name_with_space(self):
        """Test +SAT 10:00 (with space before time)."""
        from memory.triage import _parse_time_offset

        result = _parse_time_offset("+SAT 10:00", self._now())
        self.assertIsNotNone(result)
        self.assertIn("2026-02-14", result.isoformat())
        self.assertEqual(result.hour, 15)  # 10 AM ET = 15:00 UTC

    def test_hour_minute_clamped_to_valid_ranges(self):
        """Hours > 23 and minutes > 59 should be clamped."""
        from memory.triage import _parse_time_offset

        result = _parse_time_offset("+0d25:99", self._now())
        self.assertIsNotNone(result)
        # hour clamped to 23, minute clamped to 59

    def test_lowercase_input_works(self):
        """Lower-case input should be uppercased and parsed."""
        from memory.triage import _parse_time_offset

        result = _parse_time_offset("+2h", self._now())
        self.assertIsNotNone(result)
        self.assertEqual(result.hour, 21)  # 2 PM + 2h = 4 PM ET = 21:00 UTC


class TestParseTimeLocalOrchestrator(unittest.TestCase):
    """Tests for the parse_time_local sync orchestrator function."""

    def _now(self):
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        return datetime(2026, 2, 10, 14, 0, tzinfo=tz)

    def test_model_not_loaded_returns_none(self):
        """If _load_model returns None, parse_time_local should return None."""
        from memory.triage import parse_time_local

        with patch("memory.triage._load_model", return_value=None):
            result = parse_time_local("end of day", self._now())
            self.assertIsNone(result)

    def test_chat_completion_success(self):
        """When chat completion returns valid offset, result should parse."""
        from memory.triage import parse_time_local

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "+0d17:00"}}]
        }

        with patch("memory.triage._load_model", return_value=mock_model):
            result = parse_time_local("end of day", self._now())
            self.assertIsNotNone(result)
            # 5 PM ET = 22:00 UTC
            self.assertEqual(result.isoformat(), "2026-02-10T22:00:00+00:00")
            mock_model.create_chat_completion.assert_called_once()

    def test_chat_failure_falls_back_to_completion(self):
        """When chat_completion raises, should fall back to create_completion."""
        from memory.triage import parse_time_local

        mock_model = MagicMock()
        mock_model.create_chat_completion.side_effect = RuntimeError("chat broken")
        mock_model.create_completion.return_value = {
            "choices": [{"text": "+2h"}]
        }

        with patch("memory.triage._load_model", return_value=mock_model):
            result = parse_time_local("in a couple hours", self._now())
            self.assertIsNotNone(result)
            # 2 PM + 2h = 4 PM ET = 21:00 UTC
            self.assertEqual(result.hour, 21)
            mock_model.create_completion.assert_called_once()

    def test_both_completions_fail_returns_none(self):
        """If both chat and plain completion fail, should return None."""
        from memory.triage import parse_time_local

        mock_model = MagicMock()
        mock_model.create_chat_completion.side_effect = RuntimeError("chat broken")
        mock_model.create_completion.side_effect = RuntimeError("completion broken")

        with patch("memory.triage._load_model", return_value=mock_model):
            result = parse_time_local("end of day", self._now())
            self.assertIsNone(result)

    def test_model_returns_none_response(self):
        """If model returns NONE offset, should return None."""
        from memory.triage import parse_time_local

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "NONE"}}]
        }

        with patch("memory.triage._load_model", return_value=mock_model):
            result = parse_time_local("grab groceries on the way home", self._now())
            self.assertIsNone(result)

    def test_model_returns_empty_string(self):
        """If model returns empty string from both paths, should return None."""
        from memory.triage import parse_time_local

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": ""}}]
        }
        mock_model.create_completion.return_value = {
            "choices": [{"text": ""}]
        }

        with patch("memory.triage._load_model", return_value=mock_model):
            result = parse_time_local("something vague", self._now())
            self.assertIsNone(result)


class TestIntegrationRegexLLMChain(unittest.TestCase):
    """Integration tests for the full regex→LLM→parser chain."""

    def test_end_of_day_full_chain(self):
        """'end of day' should go through LLM path and produce 5 PM due date."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)  # 1 PM ET

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "+0d17:00"}}]
        }

        with patch("memory.triage._load_model", return_value=mock_model):
            result = asyncio.run(
                _extract_due_datetime_with_llm("remind me to call Dave end of day", now_utc)
            )
            self.assertIsNotNone(result)
            # 5 PM ET = 22:00 UTC
            self.assertEqual(result.isoformat(), "2026-02-10T22:00:00+00:00")
            mock_model.create_chat_completion.assert_called_once()

    def test_after_lunch_full_chain(self):
        """'after lunch' should go through LLM and produce 1 PM."""
        now_utc = datetime(2026, 2, 10, 16, 0, tzinfo=timezone.utc)  # 11 AM ET

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "+0d13:00"}}]
        }

        with patch("memory.triage._load_model", return_value=mock_model):
            result = asyncio.run(
                _extract_due_datetime_with_llm("review the doc after lunch", now_utc)
            )
            self.assertIsNotNone(result)
            # 1 PM ET = 18:00 UTC
            self.assertEqual(result.isoformat(), "2026-02-10T18:00:00+00:00")

    def test_this_weekend_full_chain(self):
        """'this weekend' should go through LLM and produce Saturday 10 AM."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)  # Tue 1 PM ET

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "+SAT10:00"}}]
        }

        with patch("memory.triage._load_model", return_value=mock_model):
            result = asyncio.run(
                _extract_due_datetime_with_llm("clean the garage this weekend", now_utc)
            )
            self.assertIsNotNone(result)
            # Saturday Feb 14, 10 AM ET = 15:00 UTC
            self.assertIn("2026-02-14", result.isoformat())
            self.assertEqual(result.hour, 15)

    def test_regex_still_takes_precedence_in_chain(self):
        """'remind me later' should use regex fast path, not LLM."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)

        mock_model = MagicMock()

        with patch("memory.triage._load_model", return_value=mock_model):
            result = asyncio.run(
                _extract_due_datetime_with_llm("remind me later", now_utc)
            )
            self.assertIsNotNone(result)
            # Regex handled it — model should not be called
            mock_model.create_chat_completion.assert_not_called()

    def test_tomorrow_explicit_skips_llm(self):
        """'tomorrow at 3pm' should use regex, not LLM."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)

        mock_model = MagicMock()

        with patch("memory.triage._load_model", return_value=mock_model):
            result = asyncio.run(
                _extract_due_datetime_with_llm("call Dave tomorrow at 3pm", now_utc)
            )
            self.assertIsNotNone(result)
            mock_model.create_chat_completion.assert_not_called()

    def test_llm_think_tags_in_chain(self):
        """LLM response with <think> tags should be parsed correctly."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "<think>end of day is 5pm</think>+0d17:00"}}]
        }

        with patch("memory.triage._load_model", return_value=mock_model):
            result = asyncio.run(
                _extract_due_datetime_with_llm("finish the report end of day", now_utc)
            )
            self.assertIsNotNone(result)
            self.assertEqual(result.isoformat(), "2026-02-10T22:00:00+00:00")

    def test_llm_returns_none_in_chain(self):
        """If LLM returns NONE, the whole chain should return None."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "NONE"}}]
        }

        with patch("memory.triage._load_model", return_value=mock_model):
            result = asyncio.run(
                _extract_due_datetime_with_llm("grab groceries on the way home", now_utc)
            )
            self.assertIsNone(result)

    def test_import_failure_returns_none(self):
        """If memory.triage import fails, wrapper returns None gracefully."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)

        with patch("memory.triage.parse_time_local_async", side_effect=ImportError("no module")):
            result = asyncio.run(
                _extract_due_datetime_with_llm("remind me end of day", now_utc)
            )
            self.assertIsNone(result)


class TestVagueTimeBoundary(unittest.TestCase):
    """Test the 7 PM boundary for vague time word defaults."""

    def test_at_exactly_7pm_defaults_to_tomorrow(self):
        """At exactly 7 PM ET (hour==19), should default to tomorrow 9 AM."""
        # 7 PM ET = 00:00 UTC next day
        now_utc = datetime(2026, 2, 11, 0, 0, tzinfo=timezone.utc)
        due = _extract_due_datetime("remind me later", now_utc)
        self.assertIsNotNone(due)
        # hour < 19 is False when hour==19, so falls to tomorrow 9 AM ET = 14:00 UTC
        self.assertEqual(due.isoformat(), "2026-02-11T14:00:00+00:00")

    def test_at_7pm_01_defaults_to_tomorrow(self):
        """At 7:01 PM ET, should default to tomorrow 9 AM."""
        # 7:01 PM ET = 00:01 UTC next day
        now_utc = datetime(2026, 2, 11, 0, 1, tzinfo=timezone.utc)
        due = _extract_due_datetime("remind me later", now_utc)
        self.assertIsNotNone(due)
        # Should be 9 AM ET tomorrow = 14:00 UTC
        self.assertEqual(due.isoformat(), "2026-02-11T14:00:00+00:00")


class TestDSTTransition(unittest.TestCase):
    """Test time parsing across DST boundaries."""

    def test_parse_time_offset_spring_forward(self):
        """Verify offset parsing works during spring-forward DST."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        # March 8, 2026 1:30 AM ET — just before spring-forward at 2 AM
        now = datetime(2026, 3, 8, 1, 30, tzinfo=tz)
        result = _parse_time_offset("+2h", now)
        self.assertIsNotNone(result)
        # 1:30 AM EST = 06:30 UTC, + 2 absolute hours = 08:30 UTC
        # (UTC-add avoids wall-clock DST surprises)
        self.assertEqual(result.hour, 8)  # UTC
        self.assertEqual(result.minute, 30)

    def test_parse_time_offset_fall_back(self):
        """Verify offset parsing works during fall-back DST."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        # Nov 1, 2026 1:30 AM EDT — just before fall-back at 2 AM
        now = datetime(2026, 11, 1, 0, 30, tzinfo=tz)
        result = _parse_time_offset("+3h", now)
        self.assertIsNotNone(result)
        # Should produce a valid UTC datetime 3 hours later


class TestRound2Fixes(unittest.TestCase):
    """Tests for Round 2 audit fixes."""

    # ── HIGH: +Nm regex no longer matches day name prefixes ───────────
    def test_minutes_regex_does_not_match_day_name(self):
        """'+1MON09:00' should parse as Monday 9 AM, not as 1 minute."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        # Tuesday Feb 10 2026, 2 PM ET
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+1MON09:00", now)
        self.assertIsNotNone(result)
        # Should be next Monday Feb 16, 9 AM ET = 14:00 UTC
        self.assertEqual(result.isoformat(), "2026-02-16T14:00:00+00:00")

    def test_minutes_regex_still_works_for_plain_minutes(self):
        """'+30m' should still parse correctly as 30 minutes."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+30m", now)
        self.assertIsNotNone(result)
        expected = now + timedelta(minutes=30)
        self.assertEqual(result, expected.astimezone(timezone.utc))

    def test_minutes_regex_min_variant_still_works(self):
        """'+15MIN' should still parse as 15 minutes."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+15MIN", now)
        self.assertIsNotNone(result)
        expected = now + timedelta(minutes=15)
        self.assertEqual(result, expected.astimezone(timezone.utc))

    # ── HIGH: Past-time guard in _extract_due_datetime_with_llm ───────
    def test_past_time_guard_bumps_to_next_day(self):
        """If LLM returns 'end of day' (5 PM) but it's 9 PM, bump to tomorrow 5 PM."""
        # 9 PM ET = 02:00 UTC next day (Feb 11)
        now_utc = datetime(2026, 2, 11, 2, 0, tzinfo=timezone.utc)
        # LLM would return +0d17:00 → 5 PM same day = already past
        # The mock returns 5 PM Feb 10 = 22:00 UTC Feb 10
        past_result = datetime(2026, 2, 10, 22, 0, tzinfo=timezone.utc)

        async def fake_parse(*_a, **_kw):
            return past_result

        with patch("memory.triage.parse_time_local_async", side_effect=fake_parse):
            result = asyncio.run(
                _extract_due_datetime_with_llm("remind me end of day", now_utc)
            )
        self.assertIsNotNone(result)
        # Should be bumped to tomorrow 5 PM ET = Feb 11 22:00 UTC
        self.assertGreater(result, now_utc)
        # Verify the time-of-day is preserved (5 PM ET = hour 22 in UTC during EST)
        from zoneinfo import ZoneInfo

        result_local = result.astimezone(ZoneInfo("America/New_York"))
        self.assertEqual(result_local.hour, 17)
        self.assertEqual(result_local.minute, 0)

    def test_future_time_not_bumped(self):
        """If LLM returns a future time, it should NOT be bumped."""
        # 2 PM ET = 19:00 UTC
        now_utc = datetime(2026, 2, 10, 19, 0, tzinfo=timezone.utc)
        # LLM returns 5 PM same day = 22:00 UTC (future)
        future_result = datetime(2026, 2, 10, 22, 0, tzinfo=timezone.utc)

        async def fake_parse(*_a, **_kw):
            return future_result

        with patch("memory.triage.parse_time_local_async", side_effect=fake_parse):
            result = asyncio.run(
                _extract_due_datetime_with_llm("remind me end of day", now_utc)
            )
        self.assertIsNotNone(result)
        self.assertEqual(result, future_result)

    def test_past_time_guard_exact_now_bumps(self):
        """If LLM returns exactly now, bump to next day (edge case)."""
        now_utc = datetime(2026, 2, 10, 22, 0, tzinfo=timezone.utc)
        exact_now = datetime(2026, 2, 10, 22, 0, tzinfo=timezone.utc)

        async def fake_parse(*_a, **_kw):
            return exact_now

        with patch("memory.triage.parse_time_local_async", side_effect=fake_parse):
            result = asyncio.run(
                _extract_due_datetime_with_llm("remind me end of day", now_utc)
            )
        self.assertIsNotNone(result)
        self.assertGreater(result, now_utc)

    # ── MEDIUM: Day name regex handles all full names ─────────────────
    def test_full_day_name_saturday(self):
        """'+SATURDAY10:00' should parse correctly."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        # Tuesday Feb 10 2026
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+SATURDAY10:00", now)
        self.assertIsNotNone(result)
        # Saturday Feb 14, 10 AM ET = 15:00 UTC
        self.assertEqual(result.isoformat(), "2026-02-14T15:00:00+00:00")

    def test_full_day_name_tuesday(self):
        """'+TUESDAY09:00' from a Tuesday should advance to next week."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        # Tuesday Feb 10 2026
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+TUESDAY09:00", now)
        self.assertIsNotNone(result)
        # Same day → advance full week → Feb 17 Tuesday, 9 AM ET = 14:00 UTC
        self.assertEqual(result.isoformat(), "2026-02-17T14:00:00+00:00")

    def test_full_day_name_wednesday(self):
        """'+WEDNESDAY14:00' should parse correctly."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        # Tuesday Feb 10 2026
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+WEDNESDAY14:00", now)
        self.assertIsNotNone(result)
        # Wednesday Feb 11, 2 PM ET = 19:00 UTC
        self.assertEqual(result.isoformat(), "2026-02-11T19:00:00+00:00")

    def test_full_day_name_thursday(self):
        """'+THURSDAY 08:30' should parse correctly (with space)."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        # Tuesday Feb 10 2026
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+THURSDAY 08:30", now)
        self.assertIsNotNone(result)
        # Thursday Feb 12, 8:30 AM ET = 13:30 UTC
        self.assertEqual(result.isoformat(), "2026-02-12T13:30:00+00:00")

    # ── MEDIUM: Title strip now handles "in N minutes" ────────────────
    def test_strips_in_30_minutes(self):
        """'call Dave in 30 minutes' → 'call Dave'."""
        title = _extract_commitment_title("remind me to call Dave in 30 minutes")
        self.assertEqual(title.lower(), "call dave")

    def test_strips_in_a_few_minutes(self):
        """'check email in a few minutes' → 'check email'."""
        title = _extract_commitment_title("remind me to check email in a few minutes")
        self.assertEqual(title.lower(), "check email")

    def test_strips_in_a_couple_minutes(self):
        """'review PR in a couple minutes' → 'review PR'."""
        title = _extract_commitment_title("remind me to review PR in a couple minutes")
        self.assertEqual(title.lower(), "review pr")


class TestRound3Fixes(unittest.TestCase):
    """Tests for Round 3 audit fixes."""

    # ── CRITICAL: +Nm regex handles MINS/MINUTES variants ─────────────
    def test_minutes_regex_handles_mins(self):
        """'+30MINS' should parse as 30 minutes."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+30MINS", now)
        self.assertIsNotNone(result)
        expected = now + timedelta(minutes=30)
        self.assertEqual(result, expected.astimezone(timezone.utc))

    def test_minutes_regex_handles_minutes(self):
        """'+15MINUTES' should parse as 15 minutes."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+15MINUTES", now)
        self.assertIsNotNone(result)
        expected = now + timedelta(minutes=15)
        self.assertEqual(result, expected.astimezone(timezone.utc))

    def test_minutes_regex_handles_minute(self):
        """'+1MINUTE' should parse as 1 minute."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+1MINUTE", now)
        self.assertIsNotNone(result)
        expected = now + timedelta(minutes=1)
        self.assertEqual(result, expected.astimezone(timezone.utc))

    # ── HIGH: +Nh regex handles HOUR/HOURS variants ───────────────────
    def test_hours_regex_handles_hours(self):
        """'+2HOURS' should parse as 2 hours."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+2HOURS", now)
        self.assertIsNotNone(result)
        expected = now + timedelta(hours=2)
        self.assertEqual(result, expected.astimezone(timezone.utc))

    def test_hours_regex_handles_hour(self):
        """'+1HOUR' should parse as 1 hour."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+1HOUR", now)
        self.assertIsNotNone(result)
        expected = now + timedelta(hours=1)
        self.assertEqual(result, expected.astimezone(timezone.utc))

    def test_hours_regex_rejects_hybrid_format(self):
        """'+17H:00' should NOT match as 17 hours (looks like malformed +NdHH:MM)."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+17H:00", now)
        # Should not match +Nh due to (?![A-Z\d:]) lookahead — falls through
        self.assertIsNone(result)

    # ── HIGH: Title strip handles "in N hours" and "in N days" ────────
    def test_strips_in_2_hours(self):
        """'call Dave in 2 hours' → 'call Dave'."""
        title = _extract_commitment_title("remind me to call Dave in 2 hours")
        self.assertEqual(title.lower(), "call dave")

    def test_strips_in_3_days(self):
        """'review PR in 3 days' → 'review PR'."""
        title = _extract_commitment_title("remind me to review PR in 3 days")
        self.assertEqual(title.lower(), "review pr")

    def test_strips_in_5_hours(self):
        """'check email in 5 hours' → 'check email'."""
        title = _extract_commitment_title("remind me to check email in 5 hours")
        self.assertEqual(title.lower(), "check email")

    def test_strips_in_1_day(self):
        """'do laundry in 1 day' → 'do laundry'."""
        title = _extract_commitment_title("remind me to do laundry in 1 day")
        self.assertEqual(title.lower(), "do laundry")

    # ── MEDIUM: Day-name regex rejects non-day words ──────────────────
    def test_day_name_regex_rejects_monitor(self):
        """'MONITOR 10:00' should NOT match as Monday."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("MONITOR 10:00", now)
        self.assertIsNone(result)

    def test_day_name_regex_rejects_sunlight(self):
        """'SUNLIGHT 09:00' should NOT match as Sunday."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("SUNLIGHT 09:00", now)
        self.assertIsNone(result)

    def test_day_name_regex_rejects_thunder(self):
        """'THUNDER 15:00' should NOT match as Thursday."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("THUNDER 15:00", now)
        self.assertIsNone(result)

    # ── MEDIUM: Past-time guard uses now_local date, exact assertions ─
    def test_past_time_guard_exact_bumped_datetime(self):
        """Verify the exact UTC datetime after a past-time bump."""
        # 9 PM ET Feb 10 = 02:00 UTC Feb 11
        now_utc = datetime(2026, 2, 11, 2, 0, tzinfo=timezone.utc)
        # LLM returns 5 PM ET Feb 10 (already past)
        past_result = datetime(2026, 2, 10, 22, 0, tzinfo=timezone.utc)

        async def fake_parse(*_a, **_kw):
            return past_result

        with patch("memory.triage.parse_time_local_async", side_effect=fake_parse):
            result = asyncio.run(
                _extract_due_datetime_with_llm("remind me end of day", now_utc)
            )
        self.assertIsNotNone(result)
        # Bumped to tomorrow (Feb 11) at 5 PM ET = 22:00 UTC Feb 11
        self.assertEqual(result.isoformat(), "2026-02-11T22:00:00+00:00")

    def test_past_time_guard_far_past_uses_now_date(self):
        """If LLM returns a date several days in the past, bump uses now's date."""
        # Feb 11 2026, 2 PM ET = 19:00 UTC
        now_utc = datetime(2026, 2, 11, 19, 0, tzinfo=timezone.utc)
        # LLM hallucinated a result from Feb 5 at 5 PM ET = 22:00 UTC
        far_past = datetime(2026, 2, 5, 22, 0, tzinfo=timezone.utc)

        async def fake_parse(*_a, **_kw):
            return far_past

        with patch("memory.triage.parse_time_local_async", side_effect=fake_parse):
            result = asyncio.run(
                _extract_due_datetime_with_llm("remind me end of day", now_utc)
            )
        self.assertIsNotNone(result)
        # Should bump to Feb 12 (now + 1 day) at 5 PM ET = 22:00 UTC
        self.assertEqual(result.isoformat(), "2026-02-12T22:00:00+00:00")
        self.assertGreater(result, now_utc)

    # ── QA: Strengthen fall-back DST test ─────────────────────────────
    def test_fall_back_dst_offset_produces_correct_utc(self):
        """Verify exact UTC time for offset during fall-back DST."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        # Nov 1, 2026 0:30 AM EDT (before fall-back at 2 AM)
        now = datetime(2026, 11, 1, 0, 30, tzinfo=tz)
        result = _parse_time_offset("+3h", now)
        self.assertIsNotNone(result)
        # 0:30 AM EDT = 04:30 UTC, + 3 absolute hours = 07:30 UTC
        # (UTC-add ensures exact elapsed time regardless of DST transitions)
        self.assertEqual(result.hour, 7)
        self.assertEqual(result.minute, 30)


class TestRound4DeferredFixes(unittest.TestCase):
    """Tests for Round 4 deferred fixes (M1-M8, L2, L5, L6)."""

    # ── M1: DST UTC-add for relative offsets ─────────────────────────
    def test_spring_forward_relative_offset_utc_add(self):
        """During spring-forward, +2h should be exactly 2 absolute hours."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        # 1:30 AM EST, March 8 2026 = 06:30 UTC
        now = datetime(2026, 3, 8, 1, 30, tzinfo=tz)
        result = _parse_time_offset("+2h", now)
        self.assertIsNotNone(result)
        # 06:30 UTC + 2h = 08:30 UTC (absolute, no DST surprise)
        self.assertEqual(result.hour, 8)
        self.assertEqual(result.minute, 30)

    def test_fall_back_relative_offset_utc_add(self):
        """During fall-back, +3h should be exactly 3 absolute hours."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        # 0:30 AM EDT, Nov 1 2026 = 04:30 UTC
        now = datetime(2026, 11, 1, 0, 30, tzinfo=tz)
        result = _parse_time_offset("+3h", now)
        self.assertIsNotNone(result)
        # 04:30 UTC + 3h = 07:30 UTC (absolute, not wall-clock)
        self.assertEqual(result.hour, 7)
        self.assertEqual(result.minute, 30)

    def test_spring_forward_minutes_utc_add(self):
        """During spring-forward, +30m should be exactly 30 absolute minutes."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 3, 8, 1, 45, tzinfo=tz)  # 06:45 UTC
        result = _parse_time_offset("+30m", now)
        self.assertIsNotNone(result)
        # 06:45 UTC + 30m = 07:15 UTC
        self.assertEqual(result.hour, 7)
        self.assertEqual(result.minute, 15)

    # ── M2: +Nh handles HR/HRS ──────────────────────────────────────
    def test_parse_offset_hr(self):
        """'+2HR' should parse as 2 hours."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+2HR", now)
        self.assertIsNotNone(result)
        now_utc = now.astimezone(timezone.utc)
        expected = now_utc + timedelta(hours=2)
        self.assertEqual(result, expected)

    def test_parse_offset_hrs(self):
        """'+3HRS' should parse as 3 hours."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+3HRS", now)
        self.assertIsNotNone(result)
        now_utc = now.astimezone(timezone.utc)
        expected = now_utc + timedelta(hours=3)
        self.assertEqual(result, expected)

    # ── M3: Day-name handles TUES/THURS/WEDS ────────────────────────
    def test_parse_offset_tues(self):
        """'+TUES09:00' should parse as next Tuesday 9 AM."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        # Monday Feb 9, 2026
        now = datetime(2026, 2, 9, 10, 0, tzinfo=tz)
        result = _parse_time_offset("+TUES09:00", now)
        self.assertIsNotNone(result)
        # Next Tuesday = Feb 10
        local_result = result.astimezone(tz)
        self.assertEqual(local_result.weekday(), 1)  # Tuesday
        self.assertEqual(local_result.hour, 9)

    def test_parse_offset_thurs(self):
        """'+THURS10:00' should parse as next Thursday 10 AM."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 9, 10, 0, tzinfo=tz)  # Monday
        result = _parse_time_offset("+THURS10:00", now)
        self.assertIsNotNone(result)
        local_result = result.astimezone(tz)
        self.assertEqual(local_result.weekday(), 3)  # Thursday
        self.assertEqual(local_result.hour, 10)

    def test_parse_offset_weds(self):
        """'+WEDS14:00' should parse as next Wednesday 2 PM."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 9, 10, 0, tzinfo=tz)  # Monday
        result = _parse_time_offset("+WEDS14:00", now)
        self.assertIsNotNone(result)
        local_result = result.astimezone(tz)
        self.assertEqual(local_result.weekday(), 2)  # Wednesday
        self.assertEqual(local_result.hour, 14)

    def test_parse_offset_thur(self):
        """'+THUR08:00' should parse as next Thursday."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 9, 10, 0, tzinfo=tz)  # Monday
        result = _parse_time_offset("+THUR08:00", now)
        self.assertIsNotNone(result)
        local_result = result.astimezone(tz)
        self.assertEqual(local_result.weekday(), 3)  # Thursday

    # ── M4: Title strip 'day after tomorrow' ─────────────────────────
    def test_title_strips_day_after_tomorrow(self):
        """'call Dave day after tomorrow' should strip to 'call Dave'."""
        from automations import _COMMITMENT_TRAILING_DUE_RE

        title = "call Dave day after tomorrow"
        stripped = _COMMITMENT_TRAILING_DUE_RE.sub("", title)
        self.assertEqual(stripped, "call Dave")

    # ── M5: Title strip 'in a minute' / 'in a day' ──────────────────
    def test_title_strips_in_a_minute(self):
        """'check email in a minute' should strip to 'check email'."""
        from automations import _COMMITMENT_TRAILING_DUE_RE

        title = "check email in a minute"
        stripped = _COMMITMENT_TRAILING_DUE_RE.sub("", title)
        self.assertEqual(stripped, "check email")

    def test_title_strips_in_a_day(self):
        """'review doc in a day' should strip to 'review doc'."""
        from automations import _COMMITMENT_TRAILING_DUE_RE

        title = "review doc in a day"
        stripped = _COMMITMENT_TRAILING_DUE_RE.sub("", title)
        self.assertEqual(stripped, "review doc")

    # ── M6: Compound +1H30M → None ──────────────────────────────────
    def test_compound_offset_returns_none(self):
        """'+1H30M' is an unsupported compound format and should return None."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        # +1H30M: the +1H pattern matches but its lookahead (?![A-Z\d:])
        # should see '3' (a digit) and fail, then +30M might match.
        # Actually +1H30M: after H there's '3' (digit), so H pattern fails.
        # Then +30M pattern sees +1H30M... the \+ matches at pos 0, then
        # (\d+) matches "1", then M... wait.  Actually re.search scans.
        # Let me think: "+1H30M" uppercased.
        # +Nm search: \+\s*(\d+)M... → finds "+1" then expects M but sees H → no.
        #   re.search keeps scanning... finds "30M" but no \+ before it → no match.
        #   Actually \+\s*(\d+) requires \+ at start. "30M" has no +. So no match.
        # +Nh search: \+\s*(\d+)H... → finds "+1H", then (?:(?:OU)?RS?)? → optional,
        #   then (?![A-Z\d:]) lookahead → next char is '3' (digit) → FAILS. Good.
        # +Nd search: no D. Day search: no day name. → returns None.
        result = _parse_time_offset("+1H30M", now)
        self.assertIsNone(result)

    # ── M7: Title strip 'in an hour' ────────────────────────────────
    def test_title_strips_in_an_hour(self):
        """'call Dave in an hour' should strip to 'call Dave'."""
        from automations import _COMMITMENT_TRAILING_DUE_RE

        title = "call Dave in an hour"
        stripped = _COMMITMENT_TRAILING_DUE_RE.sub("", title)
        self.assertEqual(stripped, "call Dave")

    # ── M8: Regex fast-path past-time guard ──────────────────────────
    def test_regex_past_time_guard_bumps_to_tomorrow(self):
        """'today at 3pm' when it's 5 PM should bump to tomorrow 3 PM."""
        # 5 PM ET = 22:00 UTC
        now_utc = datetime(2026, 2, 10, 22, 0, tzinfo=timezone.utc)
        due = _extract_due_datetime("remind me today at 3pm", now_utc)
        self.assertIsNotNone(due)
        # 3 PM today is past; should bump to 3 PM tomorrow (Feb 11)
        # 3 PM ET Feb 11 = 20:00 UTC Feb 11
        self.assertEqual(due.isoformat(), "2026-02-11T20:00:00+00:00")
        self.assertGreater(due, now_utc)

    def test_regex_past_time_guard_does_not_bump_future(self):
        """'today at 5pm' when it's 3 PM should NOT bump."""
        # 3 PM ET = 20:00 UTC
        now_utc = datetime(2026, 2, 10, 20, 0, tzinfo=timezone.utc)
        due = _extract_due_datetime("remind me today at 5pm", now_utc)
        self.assertIsNotNone(due)
        # 5 PM ET today = 22:00 UTC — in the future, no bump
        self.assertEqual(due.isoformat(), "2026-02-10T22:00:00+00:00")

    def test_regex_past_time_guard_tonight_at_6pm_when_past(self):
        """'tonight at 6pm' when it's 8 PM should bump to tomorrow 6 PM."""
        # 8 PM ET = 01:00 UTC next day (Feb 11)
        now_utc = datetime(2026, 2, 11, 1, 0, tzinfo=timezone.utc)
        due = _extract_due_datetime("remind me tonight at 6pm", now_utc)
        self.assertIsNotNone(due)
        # 6 PM tonight is past; should bump to 6 PM tomorrow
        self.assertGreater(due, now_utc)

    # ── L2: Title strip 'in a couple of hours' ──────────────────────
    def test_title_strips_in_a_couple_of_hours(self):
        """'call Dave in a couple of hours' should strip to 'call Dave'."""
        from automations import _COMMITMENT_TRAILING_DUE_RE

        title = "call Dave in a couple of hours"
        stripped = _COMMITMENT_TRAILING_DUE_RE.sub("", title)
        self.assertEqual(stripped, "call Dave")

    def test_title_strips_in_a_couple_hours(self):
        """'call Dave in a couple hours' (no 'of') should also strip."""
        from automations import _COMMITMENT_TRAILING_DUE_RE

        title = "call Dave in a couple hours"
        stripped = _COMMITMENT_TRAILING_DUE_RE.sub("", title)
        self.assertEqual(stripped, "call Dave")

    # ── L5: _parse_time_offset(None) ─────────────────────────────────
    def test_parse_time_offset_none_input(self):
        """_parse_time_offset(None, now) should return None without error."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset(None, now)
        self.assertIsNone(result)

    def test_parse_time_offset_empty_string(self):
        """_parse_time_offset('', now) should return None without error."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("", now)
        self.assertIsNone(result)

    # ── L6: Space after + in offsets ─────────────────────────────────
    def test_space_after_plus_hours(self):
        """'+ 2h' (space after +) should parse as 2 hours."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+ 2h", now)
        self.assertIsNotNone(result)
        now_utc = now.astimezone(timezone.utc)
        expected = now_utc + timedelta(hours=2)
        self.assertEqual(result, expected)

    def test_space_after_plus_minutes(self):
        """'+ 30m' (space after +) should parse as 30 minutes."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+ 30m", now)
        self.assertIsNotNone(result)
        now_utc = now.astimezone(timezone.utc)
        expected = now_utc + timedelta(minutes=30)
        self.assertEqual(result, expected)

    def test_space_after_plus_days(self):
        """'+ 1d09:00' (space after +) should parse as tomorrow 9 AM."""
        from memory.triage import _parse_time_offset
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 2, 10, 14, 0, tzinfo=tz)
        result = _parse_time_offset("+ 1d09:00", now)
        self.assertIsNotNone(result)
        local_result = result.astimezone(tz)
        self.assertEqual(local_result.day, 11)
        self.assertEqual(local_result.hour, 9)


class TestRound5Fixes(unittest.TestCase):
    """Tests for Round 5 audit findings."""

    # ── CRITICAL: "day after tomorrow" → +2 days ────────────────────
    def test_day_after_tomorrow_produces_plus_2_days(self):
        """'day after tomorrow' should be +2 days, not +1."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)  # 1 PM ET
        due = _extract_due_datetime("remind me to call Dave day after tomorrow", now_utc)
        self.assertIsNotNone(due)
        # Feb 10 + 2 = Feb 12 at 9 AM ET (default) = 14:00 UTC
        self.assertEqual(due.isoformat(), "2026-02-12T14:00:00+00:00")

    def test_day_after_tomorrow_with_time(self):
        """'day after tomorrow at 3pm' should be +2 days at 3 PM."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)
        due = _extract_due_datetime("call Dave day after tomorrow at 3pm", now_utc)
        self.assertIsNotNone(due)
        # Feb 12 at 3 PM ET = 20:00 UTC
        self.assertEqual(due.isoformat(), "2026-02-12T20:00:00+00:00")

    def test_plain_tomorrow_still_works(self):
        """'tomorrow' should still produce +1 day (no regression)."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)
        due = _extract_due_datetime("remind me to call Dave tomorrow", now_utc)
        self.assertIsNotNone(due)
        # Feb 11 at 9 AM ET = 14:00 UTC
        self.assertEqual(due.isoformat(), "2026-02-11T14:00:00+00:00")

    def test_tomorrow_at_3pm_still_works(self):
        """'tomorrow at 3pm' should still produce +1 day at 3 PM."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)
        due = _extract_due_datetime("call Dave tomorrow at 3pm", now_utc)
        self.assertIsNotNone(due)
        # Feb 11 at 3 PM ET = 20:00 UTC
        self.assertEqual(due.isoformat(), "2026-02-11T20:00:00+00:00")

    # ── MEDIUM: "in 2 hours" falls through to LLM (confirm regex returns None)
    def test_in_n_hours_returns_none_from_regex(self):
        """'in 2 hours' should return None from regex fast-path (defers to LLM)."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)
        due = _extract_due_datetime("remind me in 2 hours", now_utc)
        self.assertIsNone(due)

    def test_in_30_minutes_returns_none_from_regex(self):
        """'in 30 minutes' should return None from regex (defers to LLM)."""
        now_utc = datetime(2026, 2, 10, 18, 0, tzinfo=timezone.utc)
        due = _extract_due_datetime("remind me in 30 minutes", now_utc)
        self.assertIsNone(due)


if __name__ == "__main__":
    unittest.main()
