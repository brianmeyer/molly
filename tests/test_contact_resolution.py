"""Tests for contacts.py and scripts/import_contacts.py."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from contacts import ContactResolver, _normalize_phone

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestNormalizePhone(unittest.TestCase):
    def test_strips_formatting(self):
        self.assertEqual(_normalize_phone("+1 (212) 555-1234"), "2125551234")

    def test_drops_leading_country_code(self):
        self.assertEqual(_normalize_phone("12125551234"), "2125551234")

    def test_already_10_digits(self):
        self.assertEqual(_normalize_phone("2125551234"), "2125551234")

    def test_international_number_last_10(self):
        self.assertEqual(_normalize_phone("+442071234567"), "2071234567")

    def test_short_number_passthrough(self):
        self.assertEqual(_normalize_phone("555"), "555")

    def test_empty_string(self):
        self.assertEqual(_normalize_phone(""), "")


class TestContactResolver(unittest.TestCase):
    SAMPLE_CONTACTS = {
        "2125551234": {"name": "Alice Smith", "email": "alice@gmail.com", "phone_raw": "+12125551234"},
        "3105559876": {"name": "Bob Jones", "email": "bob@work.com", "phone_raw": "+13105559876"},
    }

    def _make_resolver(self, contacts=None):
        """Create a resolver with mock contacts data."""
        with patch.object(ContactResolver, "_load_contacts"):
            resolver = ContactResolver()
        resolver._contacts = contacts if contacts is not None else self.SAMPLE_CONTACTS.copy()
        return resolver

    def test_resolve_phone_hit(self):
        resolver = self._make_resolver()
        self.assertEqual(resolver.resolve_phone("2125551234"), "Alice Smith")

    def test_resolve_phone_with_country_code(self):
        resolver = self._make_resolver()
        self.assertEqual(resolver.resolve_phone("12125551234"), "Alice Smith")

    def test_resolve_phone_miss(self):
        resolver = self._make_resolver()
        self.assertIsNone(resolver.resolve_phone("9995550000"))

    def test_resolve_phone_entry(self):
        resolver = self._make_resolver()
        entry = resolver.resolve_phone_entry("+1-212-555-1234")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["name"], "Alice Smith")
        self.assertEqual(entry["email"], "alice@gmail.com")

    def test_cache_pushname_new(self):
        resolver = self._make_resolver(contacts={})
        resolver.cache_pushname("9995550000", "Charlie")
        self.assertEqual(resolver.resolve_phone("9995550000"), "Charlie")

    def test_cache_pushname_no_overwrite(self):
        resolver = self._make_resolver()
        resolver.cache_pushname("2125551234", "Not Alice")
        # Should keep original contact name
        self.assertEqual(resolver.resolve_phone("2125551234"), "Alice Smith")

    def test_graceful_missing_file(self):
        """Resolver works with empty dict when contacts file doesn't exist."""
        with patch("contacts.CONTACTS_FILE", Path("/nonexistent/contacts.json")):
            resolver = ContactResolver()
        self.assertIsNone(resolver.resolve_phone("2125551234"))


class TestSourceContracts(unittest.TestCase):
    """Verify integration points exist."""

    def test_contact_of_in_valid_rel_types(self):
        source = (PROJECT_ROOT / "memory" / "graph.py").read_text()
        self.assertIn('"CONTACT_OF"', source)

    def test_resolver_has_required_methods(self):
        self.assertTrue(hasattr(ContactResolver, "resolve_phone"))
        self.assertTrue(hasattr(ContactResolver, "resolve_phone_entry"))
        self.assertTrue(hasattr(ContactResolver, "cache_pushname"))
        self.assertTrue(hasattr(ContactResolver, "enrich_graph"))


class TestImportScript(unittest.TestCase):
    def test_parse_vcf(self):
        from scripts.import_contacts import parse_vcf

        vcf_content = (
            "BEGIN:VCARD\n"
            "VERSION:3.0\n"
            "FN:Alice Smith\n"
            "TEL;TYPE=CELL:+1 (212) 555-1234\n"
            "TEL;TYPE=WORK:+1 (212) 555-5678\n"
            "EMAIL;TYPE=HOME:alice@gmail.com\n"
            "END:VCARD\n"
            "BEGIN:VCARD\n"
            "VERSION:3.0\n"
            "FN:Bob Jones\n"
            "TEL;TYPE=CELL:+1-310-555-9876\n"
            "END:VCARD\n"
        )

        tmp = Path("/tmp/test_contacts.vcf")
        tmp.write_text(vcf_content)
        try:
            result = parse_vcf(tmp)
            self.assertIn("2125551234", result)
            self.assertIn("2125555678", result)
            self.assertIn("3105559876", result)
            self.assertEqual(result["2125551234"]["name"], "Alice Smith")
            self.assertEqual(result["2125551234"]["email"], "alice@gmail.com")
            self.assertEqual(result["3105559876"]["name"], "Bob Jones")
            self.assertEqual(result["3105559876"]["email"], "")
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
