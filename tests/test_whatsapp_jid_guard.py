import unittest

from whatsapp import WhatsAppClient


class TestWhatsAppJidGuard(unittest.TestCase):
    def test_parse_jid_ignores_non_whatsapp_ids(self):
        self.assertIsNone(WhatsAppClient._parse_jid("web:8d94e06e"))
        self.assertIsNone(WhatsAppClient._parse_jid("imessage"))

    def test_parse_jid_accepts_whatsapp_ids(self):
        jid = WhatsAppClient._parse_jid("15555550100@s.whatsapp.net")
        self.assertIsNotNone(jid)


if __name__ == "__main__":
    unittest.main()
