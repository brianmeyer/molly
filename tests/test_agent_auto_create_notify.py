import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent import _maybe_notify_auto_created


class TestAgentAutoCreateNotify(unittest.TestCase):
    def test_notifies_for_create_tools(self):
        notifier = MagicMock()
        runtime = SimpleNamespace(molly_instance=SimpleNamespace(notify_auto_created_tool_result=notifier))

        _maybe_notify_auto_created(
            runtime=runtime,
            chat_id="15850000000@s.whatsapp.net",
            tool_name="tasks_create",
            tool_input={"title": "Pay rent"},
            tool_response={"content": []},
        )

        notifier.assert_called_once()

    def test_ignores_non_create_tools(self):
        notifier = MagicMock()
        runtime = SimpleNamespace(molly_instance=SimpleNamespace(notify_auto_created_tool_result=notifier))

        _maybe_notify_auto_created(
            runtime=runtime,
            chat_id="15850000000@s.whatsapp.net",
            tool_name="calendar_update",
            tool_input={},
            tool_response={},
        )

        notifier.assert_not_called()
