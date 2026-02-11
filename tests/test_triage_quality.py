"""Tests for triage quality improvements: prefilter, sender tiers, channel prompts."""

import sqlite3
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Shim native-extension modules that may not be installed in test env
if "numpy" not in sys.modules:
    sys.modules["numpy"] = types.SimpleNamespace(
        array=lambda x: x,
        asarray=lambda x: x,
        dot=lambda _a, _b: 0.0,
        isscalar=lambda obj: isinstance(obj, (int, float, complex, bool, str, bytes)),
        bool_=bool,
        ndarray=tuple,
    )
if "sqlite_vec" not in sys.modules:
    sys.modules["sqlite_vec"] = types.SimpleNamespace(load=lambda _conn: None)


def _read(path: str) -> str:
    return (PROJECT_ROOT / path).read_text()


# ---------------------------------------------------------------------------
# Source-level contract tests (no imports needed)
# ---------------------------------------------------------------------------


class TestTriageSourceContracts(unittest.TestCase):
    """Verify expected symbols exist in triage.py source."""

    def setUp(self):
        self.src = _read("memory/triage.py")

    def test_automated_sender_patterns_defined(self):
        self.assertIn("AUTOMATED_SENDER_PATTERNS", self.src)

    def test_noise_subject_patterns_defined(self):
        self.assertIn("NOISE_SUBJECT_PATTERNS", self.src)

    def test_channel_prompts_defined(self):
        self.assertIn("SYSTEM_PROMPT_EMAIL", self.src)
        self.assertIn("SYSTEM_PROMPT_IMESSAGE", self.src)
        self.assertIn("SYSTEM_PROMPT_GROUP", self.src)

    def test_backward_compat_alias(self):
        self.assertIn("SYSTEM_PROMPT = SYSTEM_PROMPT_GROUP", self.src)

    def test_prefilter_function_defined(self):
        self.assertIn("def _check_prefilter(", self.src)

    def test_get_channel_prompt_defined(self):
        self.assertIn("def _get_channel_prompt(", self.src)

    def test_run_model_accepts_system_prompt(self):
        self.assertIn("system_prompt", self.src)

    def test_build_context_accepts_sender_group(self):
        self.assertIn(
            'def _build_context(sender_name: str = "", group_name: str = "")',
            self.src,
        )


class TestVectorstoreSourceContracts(unittest.TestCase):
    """Verify sender_tiers table and methods exist in vectorstore.py source."""

    def setUp(self):
        self.src = _read("memory/vectorstore.py")

    def test_sender_tiers_table(self):
        self.assertIn("CREATE TABLE IF NOT EXISTS sender_tiers", self.src)

    def test_upsert_method(self):
        self.assertIn("def upsert_sender_tier(", self.src)

    def test_get_method(self):
        self.assertIn("def get_sender_tier(", self.src)

    def test_list_method(self):
        self.assertIn("def get_sender_tiers(", self.src)

    def test_context_signals_method(self):
        self.assertIn("def get_triage_context_signals(", self.src)

    def test_migration_method(self):
        self.assertIn("def _ensure_sender_tiers_table(", self.src)


class TestConfigCommands(unittest.TestCase):
    """Verify new commands are registered in config."""

    def setUp(self):
        self.src = _read("config.py")

    def test_upgrade_command(self):
        self.assertIn("/upgrade", self.src)

    def test_downgrade_command(self):
        self.assertIn("/downgrade", self.src)

    def test_mute_command(self):
        self.assertIn("/mute", self.src)

    def test_tiers_command(self):
        self.assertIn("/tiers", self.src)


class TestCommandsSource(unittest.TestCase):
    """Verify command handlers exist in commands.py."""

    def setUp(self):
        self.src = _read("commands.py")

    def test_upgrade_handler(self):
        self.assertIn('cmd == "/upgrade"', self.src)

    def test_downgrade_handler(self):
        self.assertIn('cmd == "/downgrade"', self.src)

    def test_mute_handler(self):
        self.assertIn('cmd == "/mute"', self.src)

    def test_tiers_handler(self):
        self.assertIn('cmd == "/tiers"', self.src)

    def test_help_lists_new_commands(self):
        self.assertIn("/upgrade", self.src)
        self.assertIn("/downgrade", self.src)
        self.assertIn("/mute", self.src)
        self.assertIn("/tiers", self.src)


# ---------------------------------------------------------------------------
# Behavioral tests: prefilter logic
# ---------------------------------------------------------------------------


class TestPrefilter(unittest.TestCase):
    """Test _check_prefilter deterministic routing."""

    def setUp(self):
        # Patch config.VIP_CONTACTS for test isolation
        self.vip_patch = patch(
            "memory.triage.config.VIP_CONTACTS",
            [{"name": "Mom", "email": "mom@family.com"}],
        )
        self.vip_patch.start()

        # Patch vectorstore access — _check_prefilter does
        # "from memory.retriever import get_vectorstore" inside the function
        self.mock_vs = MagicMock()
        self.mock_vs.get_sender_tier.return_value = None
        self.vs_patch = patch(
            "memory.retriever.get_vectorstore",
            return_value=self.mock_vs,
        )
        self.vs_patch.start()

    def tearDown(self):
        self.vs_patch.stop()
        self.vip_patch.stop()

    def _pf(self):
        from memory.triage import _check_prefilter
        return _check_prefilter

    def test_vip_name_returns_urgent(self):
        pf = self._pf()
        result = pf("hey brian", "Mom", "iMessage")
        self.assertIsNotNone(result)
        self.assertEqual(result.classification, "urgent")
        self.assertIn("VIP", result.reason)

    def test_vip_email_returns_urgent(self):
        pf = self._pf()
        result = pf("meeting tomorrow", "mom@family.com", "Email")
        self.assertIsNotNone(result)
        self.assertEqual(result.classification, "urgent")

    def test_muted_tier_returns_noise(self):
        self.mock_vs.get_sender_tier.return_value = "muted"
        pf = self._pf()
        result = pf("buy now!", "spammer@junk.com", "Email")
        self.assertIsNotNone(result)
        self.assertEqual(result.classification, "noise")

    def test_upgraded_tier_returns_relevant(self):
        self.mock_vs.get_sender_tier.return_value = "upgraded"
        pf = self._pf()
        result = pf("checking in", "friend@test.com", "Email")
        self.assertIsNotNone(result)
        self.assertEqual(result.classification, "relevant")

    def test_downgraded_tier_returns_background(self):
        self.mock_vs.get_sender_tier.return_value = "downgraded"
        pf = self._pf()
        result = pf("hi there", "someone@test.com", "Email")
        self.assertIsNotNone(result)
        self.assertEqual(result.classification, "background")

    def test_automated_sender_email_returns_noise(self):
        pf = self._pf()
        result = pf("Your order has shipped", "noreply@amazon.com", "Email")
        self.assertIsNotNone(result)
        self.assertEqual(result.classification, "noise")

    def test_noise_subject_email_returns_noise(self):
        pf = self._pf()
        result = pf(
            "Order Confirmation #12345 - your order has been placed",
            "orders@shop.com",
            "Email",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.classification, "noise")

    def test_automated_sender_non_email_falls_through(self):
        """Automated sender patterns only apply to email channel."""
        pf = self._pf()
        result = pf("hello", "noreply@bot.com", "WhatsApp Group")
        self.assertIsNone(result)

    def test_unknown_sender_returns_none(self):
        pf = self._pf()
        result = pf("hey what's up", "random@person.com", "Email")
        self.assertIsNone(result)

    def test_hotel_reservation_email_noise(self):
        """Real-world: hotel reservation confirmation emails should be noise."""
        pf = self._pf()
        result = pf(
            "From: Marriott Bonvoy <no-reply@marriott.com>\nSubject: Reservation Confirmation\nYour reservation at...",
            "Marriott Bonvoy <no-reply@marriott.com>",
            "Email",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.classification, "noise")

    def test_ecobee_alert_noise(self):
        """Real-world: ecobee smart home alerts should be noise."""
        pf = self._pf()
        result = pf(
            "From: ecobee <notifications@ecobee.com>\nSubject: Your home report\nHere is your monthly energy summary",
            "ecobee <notifications@ecobee.com>",
            "Email",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.classification, "noise")

    def test_vanguard_statement_noise(self):
        """Real-world: Vanguard financial statements should be noise."""
        pf = self._pf()
        result = pf(
            "From: Vanguard <alerts@vanguard.com>\nSubject: Your account summary\nYour portfolio value as of...",
            "Vanguard <alerts@vanguard.com>",
            "Email",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.classification, "noise")

    def test_ups_tracking_noise(self):
        """Real-world: UPS delivery notifications should be noise."""
        pf = self._pf()
        result = pf(
            "From: UPS <notify@ups.com>\nSubject: UPS Delivery Update\nYour package has shipped and is on the way",
            "UPS <notify@ups.com>",
            "Email",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.classification, "noise")


# ---------------------------------------------------------------------------
# Channel prompt selection
# ---------------------------------------------------------------------------


class TestChannelPromptSelection(unittest.TestCase):
    def test_email_channel(self):
        from memory.triage import _get_channel_prompt, SYSTEM_PROMPT_EMAIL
        self.assertEqual(_get_channel_prompt("Email"), SYSTEM_PROMPT_EMAIL)
        self.assertEqual(_get_channel_prompt("Gmail"), SYSTEM_PROMPT_EMAIL)

    def test_imessage_channel(self):
        from memory.triage import _get_channel_prompt, SYSTEM_PROMPT_IMESSAGE
        self.assertEqual(_get_channel_prompt("iMessage"), SYSTEM_PROMPT_IMESSAGE)
        self.assertEqual(_get_channel_prompt("Messages"), SYSTEM_PROMPT_IMESSAGE)

    def test_group_fallback(self):
        from memory.triage import _get_channel_prompt, SYSTEM_PROMPT_GROUP
        self.assertEqual(_get_channel_prompt("Random Group"), SYSTEM_PROMPT_GROUP)
        self.assertEqual(_get_channel_prompt(""), SYSTEM_PROMPT_GROUP)


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------


class TestAutomatedSenderPatterns(unittest.TestCase):
    def test_noreply_matches(self):
        from memory.triage import AUTOMATED_SENDER_PATTERNS
        self.assertIsNotNone(AUTOMATED_SENDER_PATTERNS.search("noreply@company.com"))
        self.assertIsNotNone(AUTOMATED_SENDER_PATTERNS.search("no-reply@company.com"))
        self.assertIsNotNone(AUTOMATED_SENDER_PATTERNS.search("no_reply@company.com"))

    def test_newsletter_matches(self):
        from memory.triage import AUTOMATED_SENDER_PATTERNS
        self.assertIsNotNone(AUTOMATED_SENDER_PATTERNS.search("newsletter@news.com"))
        self.assertIsNotNone(AUTOMATED_SENDER_PATTERNS.search("newsletters@news.com"))

    def test_marketing_matches(self):
        from memory.triage import AUTOMATED_SENDER_PATTERNS
        self.assertIsNotNone(AUTOMATED_SENDER_PATTERNS.search("marketing@corp.com"))

    def test_alerts_and_notifications_match(self):
        from memory.triage import AUTOMATED_SENDER_PATTERNS
        self.assertIsNotNone(AUTOMATED_SENDER_PATTERNS.search("alerts@ecobee.com"))
        self.assertIsNotNone(AUTOMATED_SENDER_PATTERNS.search("notifications@vanguard.com"))
        self.assertIsNotNone(AUTOMATED_SENDER_PATTERNS.search("notify@ups.com"))

    def test_normal_sender_no_match(self):
        from memory.triage import AUTOMATED_SENDER_PATTERNS
        self.assertIsNone(AUTOMATED_SENDER_PATTERNS.search("john.smith@company.com"))
        self.assertIsNone(AUTOMATED_SENDER_PATTERNS.search("alice@personal.me"))


class TestNoiseSubjectPatterns(unittest.TestCase):
    def test_order_confirmation(self):
        from memory.triage import NOISE_SUBJECT_PATTERNS
        self.assertIsNotNone(NOISE_SUBJECT_PATTERNS.search("Order Confirmation #12345"))

    def test_password_reset(self):
        from memory.triage import NOISE_SUBJECT_PATTERNS
        self.assertIsNotNone(NOISE_SUBJECT_PATTERNS.search("Password Reset Request"))

    def test_verification_code(self):
        from memory.triage import NOISE_SUBJECT_PATTERNS
        self.assertIsNotNone(NOISE_SUBJECT_PATTERNS.search("Your verification code is 123456"))

    def test_shipping_update(self):
        from memory.triage import NOISE_SUBJECT_PATTERNS
        self.assertIsNotNone(NOISE_SUBJECT_PATTERNS.search("Shipping confirmation for your order"))

    def test_hotel_reservation(self):
        from memory.triage import NOISE_SUBJECT_PATTERNS
        self.assertIsNotNone(NOISE_SUBJECT_PATTERNS.search("Reservation Confirmation - Marriott"))
        self.assertIsNotNone(NOISE_SUBJECT_PATTERNS.search("Your booking confirmed at Hilton"))
        self.assertIsNotNone(NOISE_SUBJECT_PATTERNS.search("Check-in now available for your stay"))

    def test_package_tracking(self):
        from memory.triage import NOISE_SUBJECT_PATTERNS
        self.assertIsNotNone(NOISE_SUBJECT_PATTERNS.search("Your package has shipped"))
        self.assertIsNotNone(NOISE_SUBJECT_PATTERNS.search("UPS Delivery Update"))
        self.assertIsNotNone(NOISE_SUBJECT_PATTERNS.search("Your shipment has arrived"))

    def test_financial_notifications(self):
        from memory.triage import NOISE_SUBJECT_PATTERNS
        self.assertIsNotNone(NOISE_SUBJECT_PATTERNS.search("Your account summary for January"))
        self.assertIsNotNone(NOISE_SUBJECT_PATTERNS.search("Monthly statement is ready"))
        self.assertIsNotNone(NOISE_SUBJECT_PATTERNS.search("Portfolio update - Q4 2025"))

    def test_smart_home_alerts(self):
        from memory.triage import NOISE_SUBJECT_PATTERNS
        self.assertIsNotNone(NOISE_SUBJECT_PATTERNS.search("Your energy report for December"))
        self.assertIsNotNone(NOISE_SUBJECT_PATTERNS.search("Your home report is ready"))

    def test_normal_subject_no_match(self):
        from memory.triage import NOISE_SUBJECT_PATTERNS
        self.assertIsNone(NOISE_SUBJECT_PATTERNS.search("Meeting tomorrow at 3pm"))
        self.assertIsNone(NOISE_SUBJECT_PATTERNS.search("Hey Brian, quick question"))


# ---------------------------------------------------------------------------
# VectorStore sender tier integration (in-memory SQLite)
# ---------------------------------------------------------------------------


class TestSenderTierDB(unittest.TestCase):
    """Test sender tier CRUD against a real in-memory SQLite database."""

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row

        # Create just the tables we need (no vec extension required)
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sender_tiers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_pattern TEXT NOT NULL UNIQUE,
                tier TEXT NOT NULL DEFAULT 'normal',
                source TEXT DEFAULT 'manual',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS preference_signals (
                id TEXT PRIMARY KEY,
                signal_type TEXT,
                source TEXT,
                surfaced_summary TEXT,
                sender_pattern TEXT,
                owner_feedback TEXT,
                context TEXT,
                timestamp TEXT,
                created_at TEXT
            );
        """)

        # Build a minimal VectorStore with our test connection (skip __init__)
        import threading
        from memory.vectorstore import VectorStore
        self.vs = VectorStore.__new__(VectorStore)
        self.vs.conn = self.conn
        self.vs._write_lock = threading.Lock()

    def tearDown(self):
        self.conn.close()

    def test_upsert_and_get(self):
        self.vs.upsert_sender_tier("alice@test.com", "upgraded")
        self.assertEqual(self.vs.get_sender_tier("alice@test.com"), "upgraded")

    def test_upsert_overwrites(self):
        self.vs.upsert_sender_tier("bob@test.com", "upgraded")
        self.vs.upsert_sender_tier("bob@test.com", "muted")
        self.assertEqual(self.vs.get_sender_tier("bob@test.com"), "muted")

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.vs.get_sender_tier("nobody@test.com"))

    def test_case_insensitive(self):
        self.vs.upsert_sender_tier("Alice@Test.COM", "vip")
        self.assertEqual(self.vs.get_sender_tier("alice@test.com"), "vip")

    def test_get_sender_tiers_excludes_normal(self):
        self.vs.upsert_sender_tier("a@t.com", "upgraded")
        self.vs.upsert_sender_tier("b@t.com", "normal")
        self.vs.upsert_sender_tier("c@t.com", "muted")
        tiers = self.vs.get_sender_tiers()
        patterns = {t["sender_pattern"] for t in tiers}
        self.assertIn("a@t.com", patterns)
        self.assertIn("c@t.com", patterns)
        self.assertNotIn("b@t.com", patterns)

    def test_get_triage_context_signals_structure(self):
        self.vs.upsert_sender_tier("vip@test.com", "vip")
        signals = self.vs.get_triage_context_signals()
        self.assertIn("sender_tiers", signals)
        self.assertIn("dismissed_senders", signals)
        self.assertIsInstance(signals["sender_tiers"], list)
        self.assertIsInstance(signals["dismissed_senders"], list)

    def test_dismissed_senders_threshold(self):
        """Dismissed senders need 2+ dismissals to appear."""
        import uuid
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        # 1 dismissal — should NOT appear
        self.conn.execute(
            "INSERT INTO preference_signals (id, signal_type, sender_pattern, created_at) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), "dismissive_feedback", "once@test.com", now),
        )
        # 2 dismissals — SHOULD appear
        for _ in range(2):
            self.conn.execute(
                "INSERT INTO preference_signals (id, signal_type, sender_pattern, created_at) VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), "dismissive_feedback", "twice@test.com", now),
            )
        self.conn.commit()

        signals = self.vs.get_triage_context_signals()
        dismissed = {d["sender_pattern"] for d in signals["dismissed_senders"]}
        self.assertIn("twice@test.com", dismissed)
        self.assertNotIn("once@test.com", dismissed)


if __name__ == "__main__":
    unittest.main()
