import unittest
from unittest.mock import patch

import config
from whatsapp import WhatsAppClient


class _DummyResponse:
    def __init__(self, msg_id: str):
        self.ID = msg_id


class _DummyNeonizeClient:
    def __init__(self):
        self.sent: list[tuple[object, str]] = []

    def send_message(self, jid, text: str):
        self.sent.append((jid, text))
        return _DummyResponse(f"msg-{len(self.sent)}")


class TestWhatsAppSendFormatting(unittest.TestCase):
    def _new_client(self) -> WhatsAppClient:
        client = object.__new__(WhatsAppClient)
        client.client = _DummyNeonizeClient()
        client._non_whatsapp_sender = None
        return client

    @patch.object(WhatsAppClient, "_parse_jid", return_value=object())
    def test_send_message_renders_and_chunks_when_enabled(self, _mock_parse_jid):
        wa = self._new_client()
        text = (
            "## Snapshot\n\n"
            "| Item | Value |\n"
            "| --- | --- |\n"
            "| Long | " + ("x" * 360) + " |\n"
        )

        with patch.object(config, "WHATSAPP_PLAIN_RENDER", True), patch.object(
            config, "WHATSAPP_CHUNKING_ENABLED", True
        ), patch.object(config, "WHATSAPP_CHUNK_CHARS", 220):
            result = wa.send_message("15555550100@s.whatsapp.net", text)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 1)
        self.assertEqual(len(wa.client.sent), len(result))

        joined_payload = "\n".join(payload for _, payload in wa.client.sent)
        self.assertNotIn("| --- |", joined_payload)
        self.assertNotIn("## Snapshot", joined_payload)
        self.assertTrue(wa.client.sent[-1][1].endswith("-MollyAI"))

    @patch.object(WhatsAppClient, "_parse_jid", return_value=object())
    def test_send_message_keeps_raw_text_when_rendering_disabled(self, _mock_parse_jid):
        wa = self._new_client()
        text = "| A | B |\n| --- | --- |\n| 1 | 2 |"

        with patch.object(config, "WHATSAPP_PLAIN_RENDER", False), patch.object(
            config, "WHATSAPP_CHUNKING_ENABLED", True
        ), patch.object(config, "WHATSAPP_CHUNK_CHARS", 4000):
            result = wa.send_message("15555550100@s.whatsapp.net", text)

        self.assertIsInstance(result, str)
        self.assertEqual(len(wa.client.sent), 1)
        payload = wa.client.sent[0][1]
        self.assertIn("| A | B |", payload)
        self.assertTrue(payload.endswith("-MollyAI"))

    @patch.object(WhatsAppClient, "_parse_jid", return_value=object())
    def test_send_message_can_disable_chunking(self, _mock_parse_jid):
        wa = self._new_client()
        text = "x" * 800

        with patch.object(config, "WHATSAPP_PLAIN_RENDER", True), patch.object(
            config, "WHATSAPP_CHUNKING_ENABLED", False
        ), patch.object(config, "WHATSAPP_CHUNK_CHARS", 220):
            result = wa.send_message("15555550100@s.whatsapp.net", text)

        self.assertIsInstance(result, str)
        self.assertEqual(len(wa.client.sent), 1)


if __name__ == "__main__":
    unittest.main()
