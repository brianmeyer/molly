import asyncio
import json
import sys
import time
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import config
from main import Molly
from memory import triage

# ---------------------------------------------------------------------------
# Pre-mock tools.calendar so deferred `from tools.calendar import ...` works
# even without claude_agent_sdk installed in the test environment.
# ---------------------------------------------------------------------------
_mock_calendar_mod = MagicMock()
_mock_calendar_mod.calendar_search = AsyncMock(return_value={"content": []})
_mock_calendar_mod.calendar_create = AsyncMock(return_value={"content": []})
sys.modules.setdefault("tools.calendar", _mock_calendar_mod)

CHAT_JID = "15850000000@s.whatsapp.net"


def _new_molly() -> Molly:
    """Create a bare Molly instance for testing (no __init__ side effects)."""
    molly = object.__new__(Molly)
    molly._auto_calendar_lock = None
    molly._auto_calendar_seen = {}
    molly._auto_create_undo_map = {}
    molly._sent_ids = {}
    molly.wa = MagicMock()
    molly.wa.send_message.return_value = "reply-1"
    molly.notify_auto_created_tool_result = MagicMock()
    molly._get_owner_dm_jid = MagicMock(return_value=CHAT_JID)
    molly.registered_chats = {}
    return molly


# ---------------------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------------------
class TestCalendarExtraction(unittest.IsolatedAsyncioTestCase):
    async def test_extract_calendar_event_async_extracts_title_time_and_location(self):
        event_dt = datetime(2026, 2, 14, 15, 0, tzinfo=timezone.utc)
        message = (
            "From: Local School <school@example.com>\n"
            "Subject: Valentine's Day Breakfast\n"
            "Location: School Campus\n"
            "Join us Friday Feb 14 at 10am for breakfast."
        )

        with patch("memory.triage.parse_time_local_async", new=AsyncMock(return_value=event_dt)):
            extracted = await triage.extract_calendar_event_async(
                message,
                sender_name="Local School",
                channel="email",
            )

        self.assertIsNotNone(extracted)
        self.assertEqual(extracted["title"], "Valentine's Day Breakfast")
        self.assertEqual(extracted["location"], "School Campus")
        self.assertEqual(extracted["datetime"], event_dt)

    async def test_extract_calendar_event_async_returns_none_without_time(self):
        message = "Subject: School Breakfast\nFriday at 10am"
        with patch("memory.triage.parse_time_local_async", new=AsyncMock(return_value=None)):
            extracted = await triage.extract_calendar_event_async(message, channel="email")
        self.assertIsNone(extracted)

    async def test_extract_returns_none_for_empty_body(self):
        """Empty or whitespace-only message returns None without calling LLM."""
        result = await triage.extract_calendar_event_async("", channel="email")
        self.assertIsNone(result)

        result2 = await triage.extract_calendar_event_async("   \n  ", channel="email")
        self.assertIsNone(result2)

    async def test_extract_returns_none_when_no_event_patterns(self):
        """Message without time/date patterns returns None."""
        result = await triage.extract_calendar_event_async(
            "Hello, please review the attached document.", channel="email"
        )
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Dedup tests (H1 fix: mock tools.calendar via sys.modules)
# ---------------------------------------------------------------------------
class TestCalendarDedup(unittest.IsolatedAsyncioTestCase):
    async def test_calendar_event_is_duplicate_uses_cycle_cache(self):
        event_dt = datetime(2026, 2, 20, 10, 0, tzinfo=timezone.utc)
        event = {"title": "Parent Teacher Conference", "datetime": event_dt, "location": None}
        dedup_key = triage.calendar_event_dedup_key(event["title"], event_dt)
        cycle_seen = {dedup_key}

        mock_search = AsyncMock()
        with patch.object(_mock_calendar_mod, "calendar_search", mock_search):
            is_dup = await triage.calendar_event_is_duplicate_async(event, cycle_seen=cycle_seen)

        self.assertTrue(is_dup)
        mock_search.assert_not_awaited()

    async def test_calendar_event_is_duplicate_matches_calendar_search_window(self):
        event_dt = datetime(2026, 2, 20, 10, 0, tzinfo=timezone.utc)
        event = {"title": "Parent Teacher Conference", "datetime": event_dt, "location": None}
        cycle_seen: set[tuple[str, str]] = set()

        search_payload = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        [
                            {
                                "summary": "Parent-Teacher Conference",
                                "start": "2026-02-20T10:20:00+00:00",
                            }
                        ]
                    ),
                }
            ]
        }

        mock_search = AsyncMock(return_value=search_payload)
        with patch.object(_mock_calendar_mod, "calendar_search", mock_search):
            is_dup = await triage.calendar_event_is_duplicate_async(event, cycle_seen=cycle_seen)

        self.assertTrue(is_dup)
        dedup_key = triage.calendar_event_dedup_key(event["title"], event_dt)
        self.assertIn(dedup_key, cycle_seen)

    async def test_calendar_event_is_duplicate_returns_false_when_no_match(self):
        event_dt = datetime(2026, 2, 20, 10, 0, tzinfo=timezone.utc)
        event = {"title": "Parent Teacher Conference", "datetime": event_dt, "location": None}

        search_payload = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        [
                            {
                                "summary": "Team Lunch",
                                "start": "2026-02-20T15:00:00+00:00",
                            }
                        ]
                    ),
                }
            ]
        }

        mock_search = AsyncMock(return_value=search_payload)
        with patch.object(_mock_calendar_mod, "calendar_search", mock_search):
            is_dup = await triage.calendar_event_is_duplicate_async(event, cycle_seen=set())

        self.assertFalse(is_dup)

    async def test_dedup_boundary_30min_exact_is_duplicate(self):
        """Event at exactly +30 minutes should be a duplicate (<=)."""
        event_dt = datetime(2026, 2, 20, 10, 0, tzinfo=timezone.utc)
        event = {"title": "Standup", "datetime": event_dt, "location": None}
        # Existing event at exactly +30 min
        boundary_time = event_dt + timedelta(minutes=30)

        search_payload = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps([{
                        "summary": "Standup",
                        "start": boundary_time.isoformat(),
                    }]),
                }
            ]
        }

        mock_search = AsyncMock(return_value=search_payload)
        with patch.object(_mock_calendar_mod, "calendar_search", mock_search):
            is_dup = await triage.calendar_event_is_duplicate_async(event, cycle_seen=set())
        self.assertTrue(is_dup)

    async def test_dedup_boundary_31min_is_not_duplicate(self):
        """Event at +31 minutes should NOT be a duplicate."""
        event_dt = datetime(2026, 2, 20, 10, 0, tzinfo=timezone.utc)
        event = {"title": "Standup", "datetime": event_dt, "location": None}
        outside_time = event_dt + timedelta(minutes=31)

        search_payload = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps([{
                        "summary": "Standup",
                        "start": outside_time.isoformat(),
                    }]),
                }
            ]
        }

        mock_search = AsyncMock(return_value=search_payload)
        with patch.object(_mock_calendar_mod, "calendar_search", mock_search):
            is_dup = await triage.calendar_event_is_duplicate_async(event, cycle_seen=set())
        self.assertFalse(is_dup)


# ---------------------------------------------------------------------------
# Dedup error path tests (H2)
# ---------------------------------------------------------------------------
class TestDedupErrorPaths(unittest.IsolatedAsyncioTestCase):
    async def test_dedup_search_exception_returns_true_safe(self):
        """When calendar_search raises, dedup should return True (fail-closed)."""
        event_dt = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)
        event = {"title": "Brunch", "datetime": event_dt, "location": None}

        mock_search = AsyncMock(side_effect=Exception("API timeout"))
        with patch.object(_mock_calendar_mod, "calendar_search", mock_search):
            is_dup = await triage.calendar_event_is_duplicate_async(event, cycle_seen=set())
        self.assertTrue(is_dup)

    async def test_dedup_search_is_error_returns_true_safe(self):
        """When calendar_search returns is_error, dedup should return True (fail-closed)."""
        event_dt = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)
        event = {"title": "Brunch", "datetime": event_dt, "location": None}

        mock_search = AsyncMock(return_value={"is_error": True, "content": []})
        with patch.object(_mock_calendar_mod, "calendar_search", mock_search):
            is_dup = await triage.calendar_event_is_duplicate_async(event, cycle_seen=set())
        self.assertTrue(is_dup)


# ---------------------------------------------------------------------------
# _auto_action_from_triage tests
# ---------------------------------------------------------------------------
class TestAutoActionFlagGate(unittest.IsolatedAsyncioTestCase):
    async def test_auto_action_noops_when_feature_flag_disabled(self):
        molly = _new_molly()
        triage_result = SimpleNamespace(classification="relevant", reason="calendar_event")

        with patch.object(config, "AUTO_CALENDAR_EXTRACTION_ENABLED", False), \
                patch("memory.triage.extract_calendar_event_async", new=AsyncMock()) as mock_extract:
            created = await Molly._auto_action_from_triage(
                molly,
                message_text="Dinner Friday at 7pm",
                triage_result=triage_result,
                channel="whatsapp",
                sender_name="John",
                source_chat_jid=CHAT_JID,
            )

        self.assertFalse(created)
        mock_extract.assert_not_awaited()

    async def test_auto_action_creates_when_enabled_and_not_duplicate(self):
        molly = _new_molly()
        triage_result = SimpleNamespace(classification="relevant", reason="calendar_event")
        event_dt = datetime(2026, 2, 20, 19, 0, tzinfo=timezone.utc)
        extracted_event = {
            "title": "Dinner with John",
            "datetime": event_dt,
            "location": "Doya",
        }
        create_result = {
            "content": [
                {
                    "type": "text",
                    "text": (
                        '{"status":"created","event":{'
                        '"id":"evt-123","summary":"Dinner with John",'
                        '"start":"2026-02-20T19:00:00+00:00","location":"Doya"}}'
                    ),
                }
            ]
        }

        mock_create = AsyncMock(return_value=create_result)
        with patch.object(config, "AUTO_CALENDAR_EXTRACTION_ENABLED", True), \
                patch("memory.triage.extract_calendar_event_async", new=AsyncMock(return_value=extracted_event)), \
                patch("memory.triage.calendar_event_is_duplicate_async", new=AsyncMock(return_value=False)), \
                patch.object(_mock_calendar_mod, "calendar_create", mock_create):
            created = await Molly._auto_action_from_triage(
                molly,
                message_text="Dinner Friday at 7pm at Doya",
                triage_result=triage_result,
                channel="email",
                sender_name="John",
                source_chat_jid=CHAT_JID,
            )

        self.assertTrue(created)
        mock_create.assert_awaited_once()
        tool_input = mock_create.await_args.args[0]
        self.assertEqual(tool_input["summary"], "Dinner with John")
        self.assertEqual(tool_input["_auto_source"], "email")
        molly.notify_auto_created_tool_result.assert_called_once()


# ---------------------------------------------------------------------------
# Classification and EVENT_PATTERNS gate tests (M3/M4)
# ---------------------------------------------------------------------------
class TestAutoActionGates(unittest.IsolatedAsyncioTestCase):
    async def test_noise_classification_short_circuits(self):
        """Messages classified as noise should NOT trigger extraction."""
        molly = _new_molly()
        triage_result = SimpleNamespace(classification="noise", reason="spam")

        with patch.object(config, "AUTO_CALENDAR_EXTRACTION_ENABLED", True), \
                patch("memory.triage.extract_calendar_event_async", new=AsyncMock()) as mock_extract:
            created = await Molly._auto_action_from_triage(
                molly,
                message_text="Dinner Friday at 7pm",
                triage_result=triage_result,
                channel="email",
                sender_name="Spam Co",
                source_chat_jid=CHAT_JID,
            )

        self.assertFalse(created)
        mock_extract.assert_not_awaited()

    async def test_background_classification_short_circuits(self):
        """Messages classified as background should NOT trigger extraction."""
        molly = _new_molly()
        triage_result = SimpleNamespace(classification="background", reason="newsletter")

        with patch.object(config, "AUTO_CALENDAR_EXTRACTION_ENABLED", True), \
                patch("memory.triage.extract_calendar_event_async", new=AsyncMock()) as mock_extract:
            created = await Molly._auto_action_from_triage(
                molly,
                message_text="Dinner Friday at 7pm",
                triage_result=triage_result,
                channel="email",
                sender_name="Newsletter",
                source_chat_jid=CHAT_JID,
            )

        self.assertFalse(created)
        mock_extract.assert_not_awaited()

    async def test_no_event_patterns_short_circuits(self):
        """Relevant message without event patterns should NOT trigger extraction."""
        molly = _new_molly()
        triage_result = SimpleNamespace(classification="relevant", reason="calendar_event")

        with patch.object(config, "AUTO_CALENDAR_EXTRACTION_ENABLED", True), \
                patch("memory.triage.extract_calendar_event_async", new=AsyncMock()) as mock_extract:
            created = await Molly._auto_action_from_triage(
                molly,
                # No time/date patterns in this message
                message_text="Please review the attached document and let me know your thoughts.",
                triage_result=triage_result,
                channel="email",
                sender_name="John",
                source_chat_jid=CHAT_JID,
            )

        self.assertFalse(created)
        mock_extract.assert_not_awaited()


# ---------------------------------------------------------------------------
# Error path tests (H2)
# ---------------------------------------------------------------------------
class TestAutoActionErrorPaths(unittest.IsolatedAsyncioTestCase):
    async def test_extraction_exception_returns_false(self):
        """When extract_calendar_event_async raises, should return False gracefully."""
        molly = _new_molly()
        triage_result = SimpleNamespace(classification="relevant", reason="calendar_event")

        with patch.object(config, "AUTO_CALENDAR_EXTRACTION_ENABLED", True), \
                patch("memory.triage.extract_calendar_event_async", new=AsyncMock(side_effect=Exception("LLM down"))):
            created = await Molly._auto_action_from_triage(
                molly,
                message_text="Meeting Thursday at 2pm in conference room",
                triage_result=triage_result,
                channel="email",
                sender_name="Boss",
                source_chat_jid=CHAT_JID,
            )

        self.assertFalse(created)
        molly.notify_auto_created_tool_result.assert_not_called()

    async def test_calendar_create_exception_returns_false(self):
        """When calendar_create raises, should return False gracefully."""
        molly = _new_molly()
        triage_result = SimpleNamespace(classification="relevant", reason="calendar_event")
        event_dt = datetime(2026, 3, 5, 14, 0, tzinfo=timezone.utc)
        extracted_event = {"title": "Team Meeting", "datetime": event_dt, "location": "Room A"}

        mock_create = AsyncMock(side_effect=Exception("Google API down"))
        with patch.object(config, "AUTO_CALENDAR_EXTRACTION_ENABLED", True), \
                patch("memory.triage.extract_calendar_event_async", new=AsyncMock(return_value=extracted_event)), \
                patch("memory.triage.calendar_event_is_duplicate_async", new=AsyncMock(return_value=False)), \
                patch.object(_mock_calendar_mod, "calendar_create", mock_create):
            created = await Molly._auto_action_from_triage(
                molly,
                message_text="Team Meeting Thursday at 2pm in Room A",
                triage_result=triage_result,
                channel="email",
                sender_name="Boss",
                source_chat_jid=CHAT_JID,
            )

        self.assertFalse(created)
        molly.notify_auto_created_tool_result.assert_not_called()

    async def test_calendar_create_is_error_returns_false(self):
        """When calendar_create returns is_error, should return False."""
        molly = _new_molly()
        triage_result = SimpleNamespace(classification="relevant", reason="calendar_event")
        event_dt = datetime(2026, 3, 5, 14, 0, tzinfo=timezone.utc)
        extracted_event = {"title": "Team Meeting", "datetime": event_dt, "location": "Room A"}

        error_result = {"is_error": True, "content": [{"type": "text", "text": "Quota exceeded"}]}
        mock_create = AsyncMock(return_value=error_result)
        with patch.object(config, "AUTO_CALENDAR_EXTRACTION_ENABLED", True), \
                patch("memory.triage.extract_calendar_event_async", new=AsyncMock(return_value=extracted_event)), \
                patch("memory.triage.calendar_event_is_duplicate_async", new=AsyncMock(return_value=False)), \
                patch.object(_mock_calendar_mod, "calendar_create", mock_create):
            created = await Molly._auto_action_from_triage(
                molly,
                message_text="Team Meeting Thursday at 2pm in Room A",
                triage_result=triage_result,
                channel="email",
                sender_name="Boss",
                source_chat_jid=CHAT_JID,
            )

        self.assertFalse(created)
        molly.notify_auto_created_tool_result.assert_not_called()

    async def test_extraction_returns_none_no_creation(self):
        """When extraction returns None, should not attempt creation."""
        molly = _new_molly()
        triage_result = SimpleNamespace(classification="relevant", reason="calendar_event")

        mock_create = AsyncMock()
        with patch.object(config, "AUTO_CALENDAR_EXTRACTION_ENABLED", True), \
                patch("memory.triage.extract_calendar_event_async", new=AsyncMock(return_value=None)), \
                patch.object(_mock_calendar_mod, "calendar_create", mock_create):
            created = await Molly._auto_action_from_triage(
                molly,
                message_text="Maybe dinner sometime at 7pm next week",
                triage_result=triage_result,
                channel="whatsapp",
                sender_name="Friend",
                source_chat_jid=CHAT_JID,
            )

        self.assertFalse(created)
        mock_create.assert_not_awaited()


# ---------------------------------------------------------------------------
# Owner DM routing test
# ---------------------------------------------------------------------------
class TestOwnerDmRouting(unittest.IsolatedAsyncioTestCase):
    async def test_owner_dm_mention_routes_to_handle_message_for_multistep(self):
        chat_jid = "15850000000@s.whatsapp.net"
        msg_data = {
            "msg_id": "msg-1",
            "chat_jid": chat_jid,
            "sender_jid": "15850000000@s.whatsapp.net",
            "sender_name": "Brian",
            "content": "@molly schedule dinner with John Friday",
            "timestamp": "2026-02-15T12:00:00Z",
            "is_from_me": False,
            "is_group": False,
        }

        molly = object.__new__(Molly)
        molly.db = MagicMock()
        molly._sent_ids = {}
        molly._last_responses = {}
        molly._auto_create_undo_map = {}
        molly.registered_chats = {}
        molly.sessions = {}
        molly.approvals = SimpleNamespace(
            try_resolve=MagicMock(return_value=False),
            find_approval_tag=MagicMock(return_value=None),
        )
        molly.automations = SimpleNamespace(on_message=AsyncMock(return_value=None))
        molly.self_improvement = SimpleNamespace(
            should_trigger_owner_skill_phrase=MagicMock(return_value=False),
            propose_skill_from_owner_phrase=AsyncMock(return_value=None),
        )
        molly.wa = MagicMock()
        molly.wa.send_typing = MagicMock()
        molly.wa.send_typing_stopped = MagicMock()
        molly.wa.send_message = MagicMock(return_value="out-1")
        molly._track_send = MagicMock()
        molly._maybe_handle_auto_create_undo = AsyncMock(return_value=False)
        molly._log_preference_signal_if_dismissive = AsyncMock(return_value=None)
        molly._detect_and_log_correction = AsyncMock(return_value=None)
        molly.build_agent_chat_context = MagicMock(return_value=None)
        molly.save_sessions = MagicMock()
        molly._get_chat_mode = MagicMock(return_value="owner_dm")
        molly._is_owner = MagicMock(return_value=True)

        def _spawn_bg(coro, *, name: str = ""):
            if hasattr(coro, "close"):
                coro.close()
            return asyncio.create_task(asyncio.sleep(0))

        molly._spawn_bg = _spawn_bg

        with patch("main.handle_message", new=AsyncMock(return_value=("Done", "sess-1"))) as mock_handle:
            await Molly.process_message(molly, msg_data)

        mock_handle.assert_awaited_once()
        called_message = mock_handle.await_args.args[0]
        self.assertEqual(called_message, "schedule dinner with John Friday")
        self.assertEqual(molly.sessions[chat_jid], "sess-1")
        molly.wa.send_message.assert_called_once_with(chat_jid, "Done")


if __name__ == "__main__":
    unittest.main()
