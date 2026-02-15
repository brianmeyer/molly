import unittest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from main import Molly


class TestAutoCreateNotifications(unittest.TestCase):
    def _new_molly(self) -> Molly:
        molly = object.__new__(Molly)
        molly.wa = MagicMock()
        molly._sent_ids = {}
        molly._auto_create_undo_map = {}
        molly.registered_chats = {}
        return molly

    def test_calendar_create_sends_whatsapp_notification(self):
        molly = self._new_molly()
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

        with patch.object(Molly, "_get_owner_dm_jid", return_value="15850000000@s.whatsapp.net"):
            molly.notify_auto_created_tool_result(
                source_chat_jid="15850000000@s.whatsapp.net",
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
        molly = self._new_molly()
        molly.wa.send_message.return_value = "msg-2"

        with patch.object(Molly, "_get_owner_dm_jid", return_value="15850000000@s.whatsapp.net"):
            molly.notify_auto_created_tool_result(
                source_chat_jid="15850000000@s.whatsapp.net",
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
        molly = self._new_molly()
        molly.notify_auto_created_tool_result(
            source_chat_jid="15850000000@s.whatsapp.net",
            tool_name="calendar_update",
            tool_input={},
            tool_response={},
        )
        molly.wa.send_message.assert_not_called()


class TestAutoCreateUndo(unittest.IsolatedAsyncioTestCase):
    def _new_molly(self) -> Molly:
        molly = object.__new__(Molly)
        molly.wa = MagicMock()
        molly.wa.send_message.return_value = "reply-1"
        molly._sent_ids = {}
        molly._auto_create_undo_map = {}
        return molly

    async def test_undo_calendar_uses_exact_event_id(self):
        chat_jid = "15850000000@s.whatsapp.net"
        molly = self._new_molly()
        now = time.time()
        molly._auto_create_undo_map = {
            "notif-evt": {
                "chat_jid": chat_jid,
                "resource_type": "calendar",
                "resource_id": "evt-abc-123",
                "tasklist_id": "",
                "title": "Focus Session",
                "created_ts": now,
            }
        }

        with patch("tools.calendar.calendar_delete", new=AsyncMock(return_value={"content": []})) as delete_mock:
            handled = await molly._maybe_handle_auto_create_undo("undo", chat_jid)

        self.assertTrue(handled)
        delete_mock.assert_awaited_once_with({"event_id": "evt-abc-123"})
        self.assertNotIn("notif-evt", molly._auto_create_undo_map)
        self.assertIn("Undid auto-created event", molly.wa.send_message.call_args[0][1])

    async def test_undo_task_uses_exact_task_id_and_tasklist_id(self):
        chat_jid = "15850000000@s.whatsapp.net"
        molly = self._new_molly()
        now = time.time()
        molly._auto_create_undo_map = {
            "notif-older": {
                "chat_jid": chat_jid,
                "resource_type": "task",
                "resource_id": "task-old",
                "tasklist_id": "@default",
                "title": "Pay Rent",
                "created_ts": now - 1,
            },
            "notif-latest": {
                "chat_jid": chat_jid,
                "resource_type": "task",
                "resource_id": "task-new-789",
                "tasklist_id": "tasklist-xyz",
                "title": "Pay Amex",
                "created_ts": now,
            },
        }

        with patch("tools.google_tasks.tasks_delete", new=AsyncMock(return_value={"content": []})) as delete_mock:
            handled = await molly._maybe_handle_auto_create_undo("undo notif-latest", chat_jid)

        self.assertTrue(handled)
        delete_mock.assert_awaited_once_with({"task_id": "task-new-789", "tasklist_id": "tasklist-xyz"})
        self.assertNotIn("notif-latest", molly._auto_create_undo_map)
        self.assertIn("notif-older", molly._auto_create_undo_map)
