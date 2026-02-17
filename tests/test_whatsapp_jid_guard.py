import unittest

from whatsapp import WhatsAppClient


class TestWhatsAppJidGuard(unittest.TestCase):
    def test_parse_jid_ignores_non_whatsapp_ids(self):
        self.assertIsNone(WhatsAppClient._parse_jid("web:8d94e06e"))
        self.assertIsNone(WhatsAppClient._parse_jid("imessage"))
        self.assertIsNone(WhatsAppClient._parse_jid("foo@example.com"))

    def test_parse_jid_accepts_whatsapp_ids(self):
        jid = WhatsAppClient._parse_jid("15555550100@s.whatsapp.net")
        self.assertIsNotNone(jid)

    def test_parse_jid_accepts_lid_jids(self):
        jid = WhatsAppClient._parse_jid("99900000000000@lid")
        self.assertIsNotNone(jid)

    def test_parse_jid_handles_required_formats_without_crashing(self):
        samples = (
            "web:8d94e06e",
            "15551234567@s.whatsapp.net",
            "99900000000000@lid",
        )
        for sample in samples:
            try:
                WhatsAppClient._parse_jid(sample)
            except Exception as exc:  # pragma: no cover - regression guard
                self.fail(f"_parse_jid crashed for {sample}: {exc}")


if __name__ == "__main__":
    unittest.main()
