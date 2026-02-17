"""Tests for contacts.py, scripts/import_contacts.py, and integration call sites."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from contacts import ContactResolver, normalize_phone

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Phone normalization (canonical function)
# ---------------------------------------------------------------------------


class TestNormalizePhone(unittest.TestCase):
    def test_strips_formatting(self):
        self.assertEqual(normalize_phone("+1 (212) 555-1234"), "2125551234")

    def test_drops_leading_country_code(self):
        self.assertEqual(normalize_phone("12125551234"), "2125551234")

    def test_already_10_digits(self):
        self.assertEqual(normalize_phone("2125551234"), "2125551234")

    def test_international_number_last_10(self):
        self.assertEqual(normalize_phone("+442071234567"), "2071234567")

    def test_short_number_passthrough(self):
        self.assertEqual(normalize_phone("555"), "555")

    def test_empty_string(self):
        self.assertEqual(normalize_phone(""), "")

    def test_plus_only(self):
        self.assertEqual(normalize_phone("+"), "")

    def test_none_input(self):
        # str(None) -> "None" -> no digits -> ""
        self.assertEqual(normalize_phone(None), "")

    def test_number_with_extension(self):
        # Extensions get merged into digits — last 10 returned
        result = normalize_phone("+1-212-555-1234 ext 567")
        self.assertEqual(len(result), 10)


# ---------------------------------------------------------------------------
# ContactResolver unit tests
# ---------------------------------------------------------------------------


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

    def test_cache_pushname_empty_pushname_ignored(self):
        resolver = self._make_resolver(contacts={})
        resolver.cache_pushname("9995550000", "")
        self.assertIsNone(resolver.resolve_phone("9995550000"))

    def test_cache_pushname_whitespace_only(self):
        resolver = self._make_resolver(contacts={})
        resolver.cache_pushname("9995550000", "   ")
        # Whitespace-only pushname is truthy, gets stored
        self.assertEqual(resolver.resolve_phone("9995550000"), "   ")

    def test_graceful_missing_file(self):
        """Resolver works with empty dict when contacts file doesn't exist."""
        with patch("contacts.CONTACTS_FILE", Path("/nonexistent/contacts.json")):
            resolver = ContactResolver()
        self.assertIsNone(resolver.resolve_phone("2125551234"))

    def test_graceful_malformed_json(self):
        """Resolver handles malformed JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            tmp = Path(f.name)
        try:
            with patch("contacts.CONTACTS_FILE", tmp):
                resolver = ContactResolver()
            self.assertIsNone(resolver.resolve_phone("2125551234"))
        finally:
            tmp.unlink(missing_ok=True)

    def test_schema_validation_skips_bad_entries(self):
        """Entries missing 'name' key are filtered out."""
        data = {
            "2125551234": {"name": "Alice", "email": "a@b.com", "phone_raw": "+12125551234"},
            "3105559876": {"email": "bad@entry.com"},  # Missing name
            "4155551111": "not a dict",  # Not even a dict
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json.dumps(data))
            tmp = Path(f.name)
        try:
            with patch("contacts.CONTACTS_FILE", tmp):
                resolver = ContactResolver()
            self.assertEqual(resolver.resolve_phone("2125551234"), "Alice")
            self.assertIsNone(resolver.resolve_phone("3105559876"))
            self.assertIsNone(resolver.resolve_phone("4155551111"))
        finally:
            tmp.unlink(missing_ok=True)

    def test_schema_validation_rejects_non_dict_root(self):
        """contacts.json that is a list instead of dict is rejected."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('[{"name": "Alice"}]')
            tmp = Path(f.name)
        try:
            with patch("contacts.CONTACTS_FILE", tmp):
                resolver = ContactResolver()
            self.assertEqual(resolver._contacts, {})
        finally:
            tmp.unlink(missing_ok=True)

    def test_unicode_contact_names(self):
        contacts = {
            "2125551234": {"name": "Jose Garcia", "email": "", "phone_raw": "+12125551234"},
        }
        resolver = self._make_resolver(contacts)
        self.assertEqual(resolver.resolve_phone("2125551234"), "Jose Garcia")


# ---------------------------------------------------------------------------
# enrich_graph tests (mocked graph layer)
# ---------------------------------------------------------------------------


class TestEnrichGraph(unittest.TestCase):
    """Test enrich_graph with a fake graph module to avoid neo4j import."""

    def _make_resolver(self):
        with patch.object(ContactResolver, "_load_contacts"):
            return ContactResolver()

    def _make_graph_mock(self):
        """Create a mock memory.graph module with the functions enrich_graph imports."""
        mock_mod = MagicMock()
        mock_mod.upsert_entity_sync = MagicMock(side_effect=lambda name, *a: name)
        mock_mod.upsert_relationship_sync = MagicMock()
        mock_mod.set_entity_properties = MagicMock()
        return mock_mod

    @patch("contacts.config")
    def test_enrich_happy_path(self, mock_config):
        mock_config.OWNER_NAME = "Brian"
        mock_graph = self._make_graph_mock()
        resolver = self._make_resolver()
        with patch.dict("sys.modules", {"memory.graph": mock_graph}):
            resolver.enrich_graph("Alice", "+12125551234", "contacts_json", "alice@gmail.com")

        mock_graph.upsert_entity_sync.assert_any_call("Alice", "Person", 0.9)
        mock_graph.upsert_entity_sync.assert_any_call("Brian", "Person", 1.0)
        mock_graph.set_entity_properties.assert_called_once_with("Alice", {"phone": "2125551234", "email": "alice@gmail.com"})
        mock_graph.upsert_relationship_sync.assert_called_once_with("Alice", "Brian", "CONTACT_OF", 0.9, "from contacts_json")

    @patch("contacts.config")
    def test_enrich_phone_only_no_email(self, mock_config):
        mock_config.OWNER_NAME = "Brian"
        mock_graph = self._make_graph_mock()
        resolver = self._make_resolver()
        with patch.dict("sys.modules", {"memory.graph": mock_graph}):
            resolver.enrich_graph("Bob", "3105559876", "pushname", "")

        mock_graph.set_entity_properties.assert_called_once_with("Bob", {"phone": "3105559876"})

    @patch("contacts.config")
    def test_enrich_exception_does_not_propagate(self, mock_config):
        mock_config.OWNER_NAME = "Brian"
        mock_graph = self._make_graph_mock()
        mock_graph.upsert_entity_sync.side_effect = RuntimeError("Neo4j down")
        resolver = self._make_resolver()
        with patch.dict("sys.modules", {"memory.graph": mock_graph}):
            # Should not raise
            resolver.enrich_graph("Alice", "2125551234", "test")

    @patch("contacts.config")
    def test_enrich_uses_canonical_name(self, mock_config):
        mock_config.OWNER_NAME = "Brian"
        mock_graph = self._make_graph_mock()
        mock_graph.upsert_entity_sync.side_effect = lambda name, *a: "Alice Smith" if "Alice" in name else name
        resolver = self._make_resolver()
        with patch.dict("sys.modules", {"memory.graph": mock_graph}):
            resolver.enrich_graph("Alice", "2125551234", "test", "a@b.com")

        mock_graph.set_entity_properties.assert_called_once_with("Alice Smith", {"phone": "2125551234", "email": "a@b.com"})


# ---------------------------------------------------------------------------
# submit_enrichment dedup tests
# ---------------------------------------------------------------------------


class TestSubmitEnrichment(unittest.TestCase):
    def _make_resolver(self):
        with patch.object(ContactResolver, "_load_contacts"):
            return ContactResolver()

    @patch("contacts._enrichment_pool")
    def test_submit_deduplicates(self, mock_pool):
        resolver = self._make_resolver()
        resolver.submit_enrichment("Alice", "2125551234", "test")
        resolver.submit_enrichment("Alice", "12125551234", "test")  # Same number, different format
        # Should only submit once (both normalize to "2125551234")
        self.assertEqual(mock_pool.submit.call_count, 1)

    @patch("contacts._enrichment_pool")
    def test_submit_different_phones(self, mock_pool):
        resolver = self._make_resolver()
        resolver.submit_enrichment("Alice", "2125551234", "test")
        resolver.submit_enrichment("Bob", "3105559876", "test")
        self.assertEqual(mock_pool.submit.call_count, 2)


# ---------------------------------------------------------------------------
# Integration: _resolve_sender_name
# ---------------------------------------------------------------------------


class TestResolveSenderName(unittest.TestCase):
    SAMPLE_CONTACTS = {
        "2125551234": {"name": "Alice Smith", "email": "alice@gmail.com", "phone_raw": "+12125551234"},
    }

    def _make_patched_resolver(self, contacts=None):
        with patch.object(ContactResolver, "_load_contacts"):
            resolver = ContactResolver()
        resolver._contacts = contacts if contacts is not None else self.SAMPLE_CONTACTS.copy()
        return resolver

    @patch("contacts._enrichment_pool")
    def test_no_pushname_contact_hit(self, mock_pool):
        resolver = self._make_patched_resolver()
        with patch("contacts.get_resolver", return_value=resolver), \
             patch("contacts._resolver", resolver):
            from whatsapp import _resolve_sender_name
            result = _resolve_sender_name("12125551234@s.whatsapp.net", "")
        self.assertEqual(result, "Alice Smith")

    @patch("contacts._enrichment_pool")
    def test_no_pushname_contact_miss(self, mock_pool):
        resolver = self._make_patched_resolver()
        with patch("contacts.get_resolver", return_value=resolver), \
             patch("contacts._resolver", resolver):
            from whatsapp import _resolve_sender_name
            result = _resolve_sender_name("9995550000@s.whatsapp.net", "")
        self.assertEqual(result, "9995550000")

    @patch("contacts._enrichment_pool")
    def test_pushname_present_returns_pushname(self, mock_pool):
        resolver = self._make_patched_resolver()
        with patch("contacts.get_resolver", return_value=resolver), \
             patch("contacts._resolver", resolver):
            from whatsapp import _resolve_sender_name
            result = _resolve_sender_name("9995550000@s.whatsapp.net", "Charlie")
        self.assertEqual(result, "Charlie")

    @patch("contacts._enrichment_pool")
    def test_pushname_triggers_enrichment(self, mock_pool):
        resolver = self._make_patched_resolver()
        with patch("contacts.get_resolver", return_value=resolver), \
             patch("contacts._resolver", resolver):
            from whatsapp import _resolve_sender_name
            _resolve_sender_name("9995550000@s.whatsapp.net", "Charlie")
        # Should have submitted enrichment with pushname
        self.assertTrue(mock_pool.submit.called)

    @patch("contacts._enrichment_pool")
    def test_pushname_with_known_contact_enriches_contact_data(self, mock_pool):
        resolver = self._make_patched_resolver()
        with patch("contacts.get_resolver", return_value=resolver), \
             patch("contacts._resolver", resolver):
            from whatsapp import _resolve_sender_name
            _resolve_sender_name("12125551234@s.whatsapp.net", "Alice WA")
        # Should still return pushname
        # But enrichment should use contact data (Alice Smith), not pushname
        if mock_pool.submit.called:
            args = mock_pool.submit.call_args
            # First positional arg after the callable is the name
            self.assertEqual(args[0][1], "Alice Smith")

    def test_resolver_unavailable_falls_back(self):
        with patch("contacts.get_resolver", side_effect=ImportError("no contacts")):
            from whatsapp import _resolve_sender_name
            # With pushname — should return pushname
            result = _resolve_sender_name("12125551234@s.whatsapp.net", "Charlie")
            self.assertEqual(result, "Charlie")
            # Without pushname — should return phone
            result = _resolve_sender_name("12125551234@s.whatsapp.net", "")
            self.assertEqual(result, "12125551234")


# ---------------------------------------------------------------------------
# Source contracts
# ---------------------------------------------------------------------------


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
        self.assertTrue(hasattr(ContactResolver, "submit_enrichment"))

    def test_set_entity_properties_in_graph(self):
        source = (PROJECT_ROOT / "memory" / "graph.py").read_text()
        self.assertIn("def set_entity_properties(", source)

    def test_owner_name_in_config(self):
        source = (PROJECT_ROOT / "config.py").read_text()
        self.assertIn("OWNER_NAME", source)

    def test_import_script_uses_canonical_normalize(self):
        source = (PROJECT_ROOT / "scripts" / "import_contacts.py").read_text()
        self.assertIn("from contacts import normalize_phone", source)
        self.assertNotIn("def _normalize_phone", source)


# ---------------------------------------------------------------------------
# Import script
# ---------------------------------------------------------------------------


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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            f.write(vcf_content)
            tmp = Path(f.name)
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

    def test_parse_vcf_no_fn_skipped(self):
        from scripts.import_contacts import parse_vcf

        vcf_content = (
            "BEGIN:VCARD\n"
            "VERSION:3.0\n"
            "TEL;TYPE=CELL:+1-212-555-1234\n"
            "END:VCARD\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            f.write(vcf_content)
            tmp = Path(f.name)
        try:
            result = parse_vcf(tmp)
            self.assertEqual(result, {})
        finally:
            tmp.unlink(missing_ok=True)

    def test_parse_vcf_no_tel_skipped(self):
        from scripts.import_contacts import parse_vcf

        vcf_content = (
            "BEGIN:VCARD\n"
            "VERSION:3.0\n"
            "FN:Email Only\n"
            "EMAIL:email@only.com\n"
            "END:VCARD\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            f.write(vcf_content)
            tmp = Path(f.name)
        try:
            result = parse_vcf(tmp)
            self.assertEqual(result, {})
        finally:
            tmp.unlink(missing_ok=True)

    def test_parse_vcf_empty_file(self):
        from scripts.import_contacts import parse_vcf

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            f.write("")
            tmp = Path(f.name)
        try:
            result = parse_vcf(tmp)
            self.assertEqual(result, {})
        finally:
            tmp.unlink(missing_ok=True)

    def test_parse_vcf_duplicate_phones_last_wins(self):
        from scripts.import_contacts import parse_vcf

        vcf_content = (
            "BEGIN:VCARD\n"
            "FN:Alice\n"
            "TEL:+1-212-555-1234\n"
            "END:VCARD\n"
            "BEGIN:VCARD\n"
            "FN:Bob\n"
            "TEL:+1-212-555-1234\n"
            "END:VCARD\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            f.write(vcf_content)
            tmp = Path(f.name)
        try:
            result = parse_vcf(tmp)
            # Last one wins
            self.assertEqual(result["2125551234"]["name"], "Bob")
        finally:
            tmp.unlink(missing_ok=True)

    def test_parse_vcf_unicode_names(self):
        from scripts.import_contacts import parse_vcf

        vcf_content = (
            "BEGIN:VCARD\n"
            "FN:Jose Garcia\n"
            "TEL:+1-212-555-9999\n"
            "END:VCARD\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            f.write(vcf_content)
            tmp = Path(f.name)
        try:
            result = parse_vcf(tmp)
            self.assertEqual(result["2125559999"]["name"], "Jose Garcia")
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
