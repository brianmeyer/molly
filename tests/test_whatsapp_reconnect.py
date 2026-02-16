import threading
import unittest
from unittest.mock import MagicMock, patch

from neonize.exc import SendMessageError

from whatsapp import WhatsAppClient


class TestWhatsAppReconnect(unittest.TestCase):
    def _new_client(self) -> WhatsAppClient:
        wa = object.__new__(WhatsAppClient)
        wa.client = MagicMock()
        wa.connected = True
        wa._non_whatsapp_sender = None
        wa._reconnect_lock = threading.Lock()
        wa._last_reconnect_kick = 0.0
        return wa

    def test_request_reconnect_throttles_disconnect(self):
        wa = self._new_client()
        wa._request_reconnect("first")
        wa._request_reconnect("second")
        self.assertEqual(wa.client.disconnect.call_count, 1)

    @patch.object(WhatsAppClient, "_parse_jid", return_value=object())
    @patch.object(WhatsAppClient, "_prepare_outbound_chunks", return_value=["hello"])
    @patch.object(WhatsAppClient, "_request_reconnect")
    def test_send_message_marks_disconnected_on_websocket_error(
        self,
        reconnect_mock,
        _prepare_chunks_mock,
        _parse_jid_mock,
    ):
        wa = self._new_client()
        wa.client.send_message.side_effect = SendMessageError(
            "failed to send message node: websocket not connected"
        )

        result = wa.send_message("15555550100@s.whatsapp.net", "hello")

        self.assertIsNone(result)
        self.assertFalse(wa.connected)
        reconnect_mock.assert_called_once_with("send failed: websocket not connected")
