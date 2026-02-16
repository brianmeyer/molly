import unittest
import time
import logging
from unittest.mock import AsyncMock, MagicMock, patch

from main import Molly


def _new_molly() -> Molly:
    """Create a bare Molly instance for testing (no __init__ side effects)."""
    molly = object.__new__(Molly)
    molly.wa = MagicMock()
    molly.wa.send_message.return_value = "reply-1"
    molly._sent_ids = {}
    molly._auto_create_undo_map = {}
    molly.registered_chats = {}
    return molly


CHAT_JID = "15850000000@s.whatsapp.net"
OTHER_JID = "15559999999@s.whatsapp.net"


# ---------------------------------------------------------------------------
# Notification tests
# ---------------------------------------------------------------------------
class TestAutoCreateNotifications(unittest.TestCase):
    def test_calendar_create_sends_whatsapp_notification(self):
        molly = _new_molly()
        molly.wa.send_message.return_value = "msg-1"
        response = {
            "content": [
                {
                    "type": "text",
                    "text": (
                        '{"status":"created","event":{"id":"evt-123","summary":"Focus Session",'
                        '"start":"2026-02-16T10:30:00-05:00","location":"HQ"}}'
                    ),
                }
            ]
        }

        with patch.object(Molly, "_get_owner_dm_jid", return_value=CHAT_JID):
            molly.notify_auto_created_tool_result(
                source_chat_jid=CHAT_JID,
                tool_name="calendar_create",
                tool_input={"summary": "Fallback title"},
                tool_response=response,
            )

        sent_text = molly.wa.send_message.call_args[0][1]
        self.assertIn("Auto-created: Focus Session on 2026-02-16 at 10:30 AM at HQ.", sent_text)
        self.assertIn("Reply 'undo' to remove.", sent_text)
        self.assertIn("msg-1", molly._sent_ids)
        self.assertEqual(molly._auto_create_undo_map["msg-1"]["resource_id"], "evt-123")

    def test_tasks_create_uses_input_fallbacks(self):
        molly = _new_molly()
        molly.wa.send_message.return_value = "msg-2"

        with patch.object(Molly, "_get_owner_dm_jid", return_value=CHAT_JID):
            molly.notify_auto_created_tool_result(
                source_chat_jid=CHAT_JID,
                tool_name="tasks_create",
                tool_input={"title": "Pay Amex", "due": "2026-02-17"},
                tool_response={
                    "content": [
                        {
                            "type": "text",
                            "text": '{"created":{"id":"task-456","title":"Pay Amex","due":"2026-02-17"}}',
                        }
                    ]
                },
            )

        sent_text = molly.wa.send_message.call_args[0][1]
        self.assertIn("Auto-created: Pay Amex on 2026-02-17 at all day at no location.", sent_text)
        self.assertIn("Reply 'undo' to remove.", sent_text)
        self.assertIn("msg-2", molly._sent_ids)
        self.assertEqual(molly._auto_create_undo_map["msg-2"]["resource_id"], "task-456")

    def test_non_create_tools_do_not_notify(self):
        molly = _new_molly()
        molly.notify_auto_created_tool_result(
            source_chat_jid=CHAT_JID,
            tool_name="calendar_update",
            tool_input={},
            tool_response={},
        )
        molly.wa.send_message.assert_not_called()

    def test_notification_failure_logs_warning(self):
        """M1: When send_message returns None, a WARNING is logged about orphaned resource."""
        molly = _new_molly()
        molly.wa.send_message.return_value = None  # notification failure
        response = {
            "content": [
                {
                    "type": "text",
                    "text": '{"status":"created","event":{"id":"evt-orphan","summary":"Lost Event","start":"2026-03-01T09:00:00"}}',
                }
            ]
        }
        with patch.object(Molly, "_get_owner_dm_jid", return_value=CHAT_JID):
            with self.assertLogs("molly", level="WARNING") as cm:
                molly.notify_auto_created_tool_result(
                    source_chat_jid=CHAT_JID,
                    tool_name="calendar_create",
                    tool_input={"summary": "Lost Event"},
                    tool_response=response,
                )
        # Should have logged a warning about the orphaned resource
        warning_found = any("orphaned resource" in msg.lower() for msg in cm.output)
        self.assertTrue(warning_found, f"Expected orphaned resource warning in: {cm.output}")
        # No undo entry should exist since notification failed
        self.assertEqual(len(molly._auto_create_undo_map), 0)


# ---------------------------------------------------------------------------
# Undo tests — happy paths
# ---------------------------------------------------------------------------
class TestAutoCreateUndo(unittest.IsolatedAsyncioTestCase):
    @patch.object(Molly, "_save_undo_map")  # avoid disk I/O in tests
    async def test_undo_calendar_uses_exact_event_id(self, mock_save):
        molly = _new_molly()
        now = time.time()
        molly._auto_create_undo_map = {
            "notif-evt": {
                "chat_jid": CHAT_JID,
                "resource_type": "calendar",
                "resource_id": "evt-abc-123",
                "tasklist_id": "",
                "title": "Focus Session",
                "created_ts": now,
            }
        }

        mock_delete = AsyncMock(return_value={"content": []})
        with patch.object(Molly, "_undo_auto_created_entry", return_value=(True, "")) as mock_undo:
            handled = await molly._maybe_handle_auto_create_undo("undo", CHAT_JID)

        self.assertTrue(handled)
        mock_undo.assert_awaited_once()
        # Verify the entry passed to _undo was the correct one
        call_entry = mock_undo.call_args[0][0]
        self.assertEqual(call_entry["resource_id"], "evt-abc-123")
        self.assertNotIn("notif-evt", molly._auto_create_undo_map)
        self.assertIn("Undid auto-created", molly.wa.send_message.call_args[0][1])

    @patch.object(Molly, "_save_undo_map")
    async def test_undo_task_uses_exact_task_id_and_tasklist_id(self, mock_save):
        molly = _new_molly()
        now = time.time()
        molly._auto_create_undo_map = {
            "notif-older": {
                "chat_jid": CHAT_JID,
                "resource_type": "task",
                "resource_id": "task-old",
                "tasklist_id": "@default",
                "title": "Pay Rent",
                "created_ts": now - 1,
            },
            "notif-latest": {
                "chat_jid": CHAT_JID,
                "resource_type": "task",
                "resource_id": "task-new-789",
                "tasklist_id": "tasklist-xyz",
                "title": "Pay Amex",
                "created_ts": now,
            },
        }

        with patch.object(Molly, "_undo_auto_created_entry", return_value=(True, "")) as mock_undo:
            handled = await molly._maybe_handle_auto_create_undo("undo notif-latest", CHAT_JID)

        self.assertTrue(handled)
        mock_undo.assert_awaited_once()
        call_entry = mock_undo.call_args[0][0]
        self.assertEqual(call_entry["resource_id"], "task-new-789")
        self.assertEqual(call_entry["tasklist_id"], "tasklist-xyz")
        self.assertNotIn("notif-latest", molly._auto_create_undo_map)
        self.assertIn("notif-older", molly._auto_create_undo_map)


# ---------------------------------------------------------------------------
# Undo tests — failure paths (M2)
# ---------------------------------------------------------------------------
class TestAutoCreateUndoEdgeCases(unittest.IsolatedAsyncioTestCase):
    @patch.object(Molly, "_save_undo_map")
    async def test_double_undo_second_attempt_reports_nothing(self, mock_save):
        """Double-undo: after first undo succeeds, second should say 'no item found'."""
        molly = _new_molly()
        now = time.time()
        molly._auto_create_undo_map = {
            "notif-1": {
                "chat_jid": CHAT_JID,
                "resource_type": "calendar",
                "resource_id": "evt-once",
                "tasklist_id": "",
                "title": "One-time event",
                "created_ts": now,
            }
        }

        with patch.object(Molly, "_undo_auto_created_entry", return_value=(True, "")):
            first = await molly._maybe_handle_auto_create_undo("undo", CHAT_JID)
        self.assertTrue(first)
        self.assertNotIn("notif-1", molly._auto_create_undo_map)

        # Second undo — map is now empty, bare "undo" should fall through (H1 fix)
        second = await molly._maybe_handle_auto_create_undo("undo", CHAT_JID)
        self.assertFalse(second)  # falls through to normal processing

    @patch.object(Molly, "_save_undo_map")
    async def test_expired_ttl_entry_pruned(self, mock_save):
        """Entries older than TTL should be pruned and undo should report nothing."""
        molly = _new_molly()
        from main import AUTO_CREATE_UNDO_TTL_SECONDS
        expired_ts = time.time() - AUTO_CREATE_UNDO_TTL_SECONDS - 100
        molly._auto_create_undo_map = {
            "notif-expired": {
                "chat_jid": CHAT_JID,
                "resource_type": "calendar",
                "resource_id": "evt-old",
                "tasklist_id": "",
                "title": "Expired Event",
                "created_ts": expired_ts,
            }
        }

        # Bare "undo" — after pruning the map should be empty, falls through
        handled = await molly._maybe_handle_auto_create_undo("undo", CHAT_JID)
        self.assertFalse(handled)
        self.assertEqual(len(molly._auto_create_undo_map), 0)

    @patch.object(Molly, "_save_undo_map")
    async def test_undo_api_error_re_inserts_entry(self, mock_save):
        """When the delete API fails, the entry should be re-inserted for retry."""
        molly = _new_molly()
        now = time.time()
        molly._auto_create_undo_map = {
            "notif-fail": {
                "chat_jid": CHAT_JID,
                "resource_type": "calendar",
                "resource_id": "evt-fail",
                "tasklist_id": "",
                "title": "Stubborn Event",
                "created_ts": now,
            }
        }

        with patch.object(Molly, "_undo_auto_created_entry", return_value=(False, "API down")):
            handled = await molly._maybe_handle_auto_create_undo("undo", CHAT_JID)

        self.assertTrue(handled)
        # Entry should be re-inserted so user can retry
        self.assertIn("notif-fail", molly._auto_create_undo_map)
        sent_text = molly.wa.send_message.call_args[0][1]
        self.assertIn("Undo failed", sent_text)

    @patch.object(Molly, "_save_undo_map")
    async def test_empty_map_bare_undo_falls_through(self, mock_save):
        """H1 fix: Bare 'undo' with empty map should NOT be consumed."""
        molly = _new_molly()
        molly._auto_create_undo_map = {}

        handled = await molly._maybe_handle_auto_create_undo("undo", CHAT_JID)
        self.assertFalse(handled)  # falls through to normal processing
        molly.wa.send_message.assert_not_called()

    @patch.object(Molly, "_save_undo_map")
    async def test_cross_jid_isolation(self, mock_save):
        """Undo from a different chat_jid should NOT touch another chat's entries."""
        molly = _new_molly()
        now = time.time()
        molly._auto_create_undo_map = {
            "notif-other-chat": {
                "chat_jid": OTHER_JID,
                "resource_type": "calendar",
                "resource_id": "evt-other",
                "tasklist_id": "",
                "title": "Other Chat Event",
                "created_ts": now,
            }
        }

        # Bare "undo" from CHAT_JID — map has entries but not for this chat
        # With H1 fix, map is non-empty so bare "undo" IS consumed but reports "no item found"
        handled = await molly._maybe_handle_auto_create_undo("undo", CHAT_JID)
        self.assertTrue(handled)
        # The other chat's entry should remain untouched
        self.assertIn("notif-other-chat", molly._auto_create_undo_map)
        sent_text = molly.wa.send_message.call_args[0][1]
        self.assertIn("No recent auto-created item found", sent_text)

    @patch.object(Molly, "_save_undo_map")
    async def test_undo_specific_id_not_found_consumed(self, mock_save):
        """'undo <specific_id>' where ID doesn't exist should be consumed with error message."""
        molly = _new_molly()
        molly._auto_create_undo_map = {}

        handled = await molly._maybe_handle_auto_create_undo("undo notif-bogus-id", CHAT_JID)
        self.assertTrue(handled)
        sent_text = molly.wa.send_message.call_args[0][1]
        self.assertIn("notif-bogus-id", sent_text)
        self.assertIn("No recent auto-created item found", sent_text)

    @patch.object(Molly, "_save_undo_map")
    async def test_undo_message_with_other_content_falls_through(self, mock_save):
        """'undo <something>' is treated as a specific-ID lookup and consumed."""
        molly = _new_molly()
        molly._auto_create_undo_map = {}

        # "undo notif-xyz" — specific ID lookup, always consumed even if not found
        handled = await molly._maybe_handle_auto_create_undo("undo notif-xyz", CHAT_JID)
        self.assertTrue(handled)  # specific ID path always consumed

    async def test_non_undo_message_passes_through(self):
        """Messages not starting with 'undo' are never consumed."""
        molly = _new_molly()
        molly._auto_create_undo_map = {"some-notif": {"chat_jid": CHAT_JID, "created_ts": time.time()}}

        handled = await molly._maybe_handle_auto_create_undo("hello world", CHAT_JID)
        self.assertFalse(handled)
        molly.wa.send_message.assert_not_called()


# ---------------------------------------------------------------------------
# Persistence tests (H2)
# ---------------------------------------------------------------------------
class TestUndoMapPersistence(unittest.TestCase):
    def test_record_entry_triggers_save(self):
        """Recording an undo entry should persist to disk."""
        molly = _new_molly()
        with patch.object(Molly, "_save_undo_map") as mock_save:
            molly._record_auto_create_undo_entry(
                "notif-persist",
                CHAT_JID,
                resource_type="calendar",
                resource_id="evt-persist",
                title="Persisted Event",
            )
        mock_save.assert_called_once()
        self.assertIn("notif-persist", molly._auto_create_undo_map)


# ---------------------------------------------------------------------------
# Delete dispatch tests (M2 — exercise _undo_auto_created_entry)
# ---------------------------------------------------------------------------
class TestUndoDeleteDispatch(unittest.IsolatedAsyncioTestCase):
    """Test the actual _undo_auto_created_entry method (not mocked away)."""

    async def test_calendar_delete_dispatches_correctly(self):
        molly = _new_molly()
        entry = {
            "resource_type": "calendar",
            "resource_id": "evt-dispatch-test",
            "tasklist_id": "",
        }
        mock_delete = AsyncMock(return_value={"content": [{"type": "text", "text": "deleted"}]})
        with patch.dict("sys.modules", {}):
            with patch("main.Molly._undo_auto_created_entry", wraps=molly._undo_auto_created_entry):
                # Patch the import target inside the method
                import tools.calendar as cal_mod
                original = getattr(cal_mod, "calendar_delete", None)
                cal_mod.calendar_delete = mock_delete
                try:
                    success, detail = await molly._undo_auto_created_entry(entry)
                finally:
                    if original is not None:
                        cal_mod.calendar_delete = original
                    elif hasattr(cal_mod, "calendar_delete"):
                        del cal_mod.calendar_delete

        self.assertTrue(success)
        mock_delete.assert_awaited_once_with({"event_id": "evt-dispatch-test"})

    async def test_task_delete_dispatches_with_tasklist_id(self):
        molly = _new_molly()
        entry = {
            "resource_type": "task",
            "resource_id": "task-dispatch-test",
            "tasklist_id": "mylist-123",
        }
        mock_delete = AsyncMock(return_value={"content": [{"type": "text", "text": "deleted"}]})
        import tools.google_tasks as tasks_mod
        original = getattr(tasks_mod, "tasks_delete", None)
        tasks_mod.tasks_delete = mock_delete
        try:
            success, detail = await molly._undo_auto_created_entry(entry)
        finally:
            if original is not None:
                tasks_mod.tasks_delete = original

        self.assertTrue(success)
        mock_delete.assert_awaited_once_with({
            "task_id": "task-dispatch-test",
            "tasklist_id": "mylist-123",
        })

    async def test_missing_resource_id_returns_failure(self):
        molly = _new_molly()
        entry = {"resource_type": "calendar", "resource_id": "", "tasklist_id": ""}
        success, detail = await molly._undo_auto_created_entry(entry)
        self.assertFalse(success)
        self.assertIn("missing resource id", detail)

    async def test_unsupported_resource_type_returns_failure(self):
        molly = _new_molly()
        entry = {"resource_type": "note", "resource_id": "note-123", "tasklist_id": ""}
        success, detail = await molly._undo_auto_created_entry(entry)
        self.assertFalse(success)
        self.assertIn("unsupported resource type", detail)

    async def test_delete_api_exception_returns_failure(self):
        molly = _new_molly()
        entry = {
            "resource_type": "calendar",
            "resource_id": "evt-fail",
            "tasklist_id": "",
        }
        mock_delete = AsyncMock(side_effect=Exception("API unavailable"))
        import tools.calendar as cal_mod
        original = getattr(cal_mod, "calendar_delete", None)
        cal_mod.calendar_delete = mock_delete
        try:
            success, detail = await molly._undo_auto_created_entry(entry)
        finally:
            if original is not None:
                cal_mod.calendar_delete = original

        self.assertFalse(success)
        self.assertIn("API unavailable", detail)

    async def test_delete_returns_is_error(self):
        molly = _new_molly()
        entry = {
            "resource_type": "calendar",
            "resource_id": "evt-err",
            "tasklist_id": "",
        }
        mock_delete = AsyncMock(return_value={"is_error": True, "content": [{"type": "text", "text": "Not found"}]})
        import tools.calendar as cal_mod
        original = getattr(cal_mod, "calendar_delete", None)
        cal_mod.calendar_delete = mock_delete
        try:
            success, detail = await molly._undo_auto_created_entry(entry)
        finally:
            if original is not None:
                cal_mod.calendar_delete = original

        self.assertFalse(success)


# ---------------------------------------------------------------------------
# Notification error path tests (M3)
# ---------------------------------------------------------------------------
class TestNotificationErrorPaths(unittest.TestCase):
    def test_is_error_response_skipped(self):
        """Tool response with is_error=True should not trigger notification."""
        molly = _new_molly()
        molly.notify_auto_created_tool_result(
            source_chat_jid=CHAT_JID,
            tool_name="calendar_create",
            tool_input={"summary": "Test"},
            tool_response={"is_error": True, "content": [{"type": "text", "text": "Error"}]},
        )
        molly.wa.send_message.assert_not_called()

    def test_empty_resource_id_skipped_with_warning(self):
        """Tool response with no resource id should log warning and skip."""
        molly = _new_molly()
        molly.wa.send_message.return_value = "msg-skip"
        response = {
            "content": [
                {
                    "type": "text",
                    "text": '{"status":"created","event":{"summary":"No ID Event","start":"2026-03-01"}}',
                }
            ]
        }
        with patch.object(Molly, "_get_owner_dm_jid", return_value=CHAT_JID):
            with self.assertLogs("molly", level="WARNING") as cm:
                molly.notify_auto_created_tool_result(
                    source_chat_jid=CHAT_JID,
                    tool_name="calendar_create",
                    tool_input={"summary": "No ID Event"},
                    tool_response=response,
                )
        # Should have warned about missing resource id
        self.assertTrue(any("without resource id" in msg for msg in cm.output))
        # No WhatsApp message should be sent
        molly.wa.send_message.assert_not_called()

    def test_record_entry_with_empty_notification_id_discarded(self):
        """_record_auto_create_undo_entry with empty notification_id should silently discard."""
        molly = _new_molly()
        with patch.object(Molly, "_save_undo_map") as mock_save:
            molly._record_auto_create_undo_entry(
                "",  # empty notification_id
                CHAT_JID,
                resource_type="calendar",
                resource_id="evt-123",
                title="Test",
            )
        mock_save.assert_not_called()
        self.assertEqual(len(molly._auto_create_undo_map), 0)

    def test_record_entry_with_empty_resource_id_discarded(self):
        """_record_auto_create_undo_entry with empty resource_id should silently discard."""
        molly = _new_molly()
        with patch.object(Molly, "_save_undo_map") as mock_save:
            molly._record_auto_create_undo_entry(
                "notif-123",
                CHAT_JID,
                resource_type="calendar",
                resource_id="",  # empty
                title="Test",
            )
        mock_save.assert_not_called()
        self.assertEqual(len(molly._auto_create_undo_map), 0)


# ---------------------------------------------------------------------------
# Cross-JID specific-ID and max-entry tests (M4)
# ---------------------------------------------------------------------------
class TestUndoMapAdvancedEdgeCases(unittest.IsolatedAsyncioTestCase):
    @patch.object(Molly, "_save_undo_map")
    async def test_specific_id_cross_jid_rejected(self, mock_save):
        """'undo <specific_id>' from wrong chat should reject (not delete other chat's entry)."""
        molly = _new_molly()
        now = time.time()
        molly._auto_create_undo_map = {
            "notif-other": {
                "chat_jid": OTHER_JID,
                "resource_type": "calendar",
                "resource_id": "evt-other",
                "tasklist_id": "",
                "title": "Other Chat Event",
                "created_ts": now,
            }
        }

        handled = await molly._maybe_handle_auto_create_undo("undo notif-other", CHAT_JID)
        self.assertTrue(handled)
        # Entry should NOT have been removed (belongs to other chat)
        self.assertIn("notif-other", molly._auto_create_undo_map)
        sent_text = molly.wa.send_message.call_args[0][1]
        self.assertIn("No recent auto-created item found", sent_text)

    def test_max_entry_eviction(self):
        """When undo map exceeds MAX_ENTRIES, oldest entries should be evicted."""
        from main import AUTO_CREATE_UNDO_MAX_ENTRIES
        molly = _new_molly()
        now = time.time()

        # Fill map past capacity
        for i in range(AUTO_CREATE_UNDO_MAX_ENTRIES + 10):
            molly._auto_create_undo_map[f"notif-{i}"] = {
                "chat_jid": CHAT_JID,
                "resource_type": "calendar",
                "resource_id": f"evt-{i}",
                "tasklist_id": "",
                "title": f"Event {i}",
                "created_ts": now + i,  # newer as i increases
            }

        molly._prune_auto_create_undo_map()
        self.assertEqual(len(molly._auto_create_undo_map), AUTO_CREATE_UNDO_MAX_ENTRIES)
        # The oldest entries (lowest i/ts) should be evicted, newest kept
        self.assertNotIn("notif-0", molly._auto_create_undo_map)
        self.assertIn(f"notif-{AUTO_CREATE_UNDO_MAX_ENTRIES + 9}", molly._auto_create_undo_map)


# ---------------------------------------------------------------------------
# save_json / load_json direct tests (QA iteration-3 M1/M2)
# Adapted for Phase 1 refactor: save_json now delegates to utils.atomic_write_json
# ---------------------------------------------------------------------------
import json
import os
import tempfile
from pathlib import Path

from main import save_json, load_json
from utils import atomic_write_json, load_json as utils_load_json


class TestSaveJsonAtomicity(unittest.TestCase):
    """Verify save_json writes atomically via .tmp + os.replace."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.target = Path(self._tmpdir) / "data.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_basic_write_and_read_back(self):
        data = {"key": "value", "n": 42}
        save_json(self.target, data)
        self.assertTrue(self.target.exists())
        read_back = json.loads(self.target.read_text())
        self.assertEqual(read_back, data)

    def test_tmp_file_cleaned_up(self):
        """After successful save, no .tmp sibling file should remain."""
        save_json(self.target, {"a": 1})
        # Check for any .tmp files in the directory
        tmp_files = list(Path(self._tmpdir).glob("*.tmp"))
        self.assertEqual(len(tmp_files), 0, f"Leftover tmp files: {tmp_files}")

    def test_uses_os_replace(self):
        """Verify os.replace is called (the POSIX-atomic rename)."""
        with patch("utils.os.replace", wraps=os.replace) as mock_replace:
            save_json(self.target, {"b": 2})
        mock_replace.assert_called_once()
        # First arg should be the .tmp path, second the target
        args = mock_replace.call_args[0]
        self.assertTrue(str(args[0]).endswith(".tmp"))
        self.assertEqual(Path(args[1]), self.target)

    def test_parent_dir_created_if_missing(self):
        nested = Path(self._tmpdir) / "sub" / "dir" / "data.json"
        save_json(nested, {"c": 3})
        self.assertTrue(nested.exists())
        self.assertEqual(json.loads(nested.read_text()), {"c": 3})

    def test_overwrite_preserves_old_on_write_failure(self):
        """If write_text raises, the original file should still be intact."""
        save_json(self.target, {"original": True})
        with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                save_json(self.target, {"new": True})
        # Original should still be intact
        read_back = json.loads(self.target.read_text())
        self.assertEqual(read_back, {"original": True})


class TestLoadJsonErrorHandling(unittest.TestCase):
    """Verify load_json handles missing files, corrupt JSON, and OS errors."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_missing_file_returns_default(self):
        path = Path(self._tmpdir) / "missing.json"
        result = load_json(path, {"default": True})
        self.assertEqual(result, {"default": True})

    def test_missing_file_returns_empty_dict_when_no_default(self):
        path = Path(self._tmpdir) / "missing.json"
        result = load_json(path)
        self.assertEqual(result, {})

    def test_corrupt_json_returns_default(self):
        path = Path(self._tmpdir) / "bad.json"
        path.write_text("{not valid json!!!")
        result = load_json(path, {"fallback": True})
        self.assertEqual(result, {"fallback": True})

    def test_valid_json_loaded_correctly(self):
        path = Path(self._tmpdir) / "good.json"
        data = {"items": [1, 2, 3], "nested": {"a": "b"}}
        path.write_text(json.dumps(data))
        result = load_json(path, {})
        self.assertEqual(result, data)

    def test_os_error_returns_default(self):
        path = Path(self._tmpdir) / "locked.json"
        path.write_text('{"ok": true}')
        with patch("pathlib.Path.read_text", side_effect=OSError("permission denied")):
            result = load_json(path, {"safe": True})
        self.assertEqual(result, {"safe": True})


if __name__ == "__main__":
    unittest.main()
