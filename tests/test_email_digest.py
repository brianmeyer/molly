"""Tests for memory.email_digest module and automation wiring."""

import asyncio
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub heavyweight modules so importing automations.py doesn't fail.
# IMPORTANT: Clean up stubs after import to avoid polluting other tests.
# ---------------------------------------------------------------------------
_STUBS: dict[str, MagicMock] = {}
for _mod in (
    "numpy", "numpy.typing", "neo4j", "fastapi", "pydantic",
    "uvicorn", "memory.embeddings", "memory.processor",
    "memory.graph", "agent",
):
    if _mod not in sys.modules:
        _STUBS[_mod] = MagicMock()
if _STUBS:
    sys.modules.update(_STUBS)
    # Force the import that needs these stubs
    import automations  # noqa: F401
    import commitments  # noqa: F401
    # Clean up: remove stubs we injected so other test modules aren't polluted
    for _mod_name in _STUBS:
        if sys.modules.get(_mod_name) is _STUBS[_mod_name]:
            del sys.modules[_mod_name]


# ---------------------------------------------------------------------------
# Frozen UTC time helpers — avoids midnight edge cases
# ---------------------------------------------------------------------------
_FROZEN_UTC = datetime(2026, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
_FROZEN_DATE_STR = "2026-06-15"
_FROZEN_YESTERDAY_STR = "2026-06-14"


def _make_frozen_datetime_mock():
    """Create a datetime mock that returns _FROZEN_UTC from .now(tz)."""
    mock_dt = MagicMock(wraps=datetime)
    mock_dt.now.return_value = _FROZEN_UTC
    mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
    return mock_dt


# ---------------------------------------------------------------------------
# TestAppendDigestItem
# ---------------------------------------------------------------------------

class TestAppendDigestItem(unittest.TestCase):
    """Test that append_digest_item writes correct JSONL."""

    @patch("memory.email_digest.datetime", _make_frozen_datetime_mock())
    def test_creates_jsonl_file(self):
        from memory.email_digest import append_digest_item

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            with patch("memory.email_digest.QUEUE_DIR", queue_dir):
                append_digest_item(
                    msg_id="abc123",
                    sender="jane@example.com",
                    subject="Meeting Thursday",
                    snippet="Can we reschedule...",
                    classification="relevant",
                    score=0.72,
                    reason="Real person asking about schedule",
                    internal_ts_ms=1770831080000,
                )

            path = queue_dir / f"{_FROZEN_DATE_STR}.jsonl"
            self.assertTrue(path.exists(), f"Expected JSONL at {path}")

            lines = path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["msg_id"], "abc123")
            self.assertEqual(entry["sender"], "jane@example.com")
            self.assertEqual(entry["subject"], "Meeting Thursday")
            self.assertEqual(entry["classification"], "relevant")
            self.assertAlmostEqual(entry["score"], 0.72, places=2)
            self.assertEqual(entry["internal_ts_ms"], 1770831080000)

    @patch("memory.email_digest.datetime", _make_frozen_datetime_mock())
    def test_appends_multiple_entries(self):
        from memory.email_digest import append_digest_item

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            with patch("memory.email_digest.QUEUE_DIR", queue_dir):
                append_digest_item(
                    msg_id="a1", sender="alice@test.com", subject="Subj 1",
                    snippet="s1", classification="urgent", score=0.9,
                    reason="VIP", internal_ts_ms=1000,
                )
                append_digest_item(
                    msg_id="a2", sender="bob@test.com", subject="Subj 2",
                    snippet="s2", classification="background", score=0.3,
                    reason="Newsletter", internal_ts_ms=2000,
                )

            path = queue_dir / f"{_FROZEN_DATE_STR}.jsonl"
            lines = path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["msg_id"], "a1")
            self.assertEqual(json.loads(lines[1])["msg_id"], "a2")


# ---------------------------------------------------------------------------
# TestGetQueueItems
# ---------------------------------------------------------------------------

class TestGetQueueItems(unittest.TestCase):
    """Test reading queue items from JSONL files."""

    def test_reads_jsonl(self):
        from memory.email_digest import get_queue_items

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = queue_dir / f"{today}.jsonl"
            path.write_text(
                json.dumps({"msg_id": "a1", "classification": "urgent"}) + "\n"
                + json.dumps({"msg_id": "a2", "classification": "relevant"}) + "\n"
            )

            with patch("memory.email_digest.QUEUE_DIR", queue_dir):
                result = get_queue_items()

            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["msg_id"], "a1")
            self.assertEqual(result[1]["classification"], "relevant")

    def test_missing_file_returns_empty(self):
        from memory.email_digest import get_queue_items

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            with patch("memory.email_digest.QUEUE_DIR", queue_dir):
                result = get_queue_items("2020-01-01")

            self.assertEqual(result, [])

    def test_corrupt_lines_skipped(self):
        from memory.email_digest import get_queue_items

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = queue_dir / f"{today}.jsonl"
            path.write_text(
                '{"msg_id": "a1"}\n'
                '{corrupt json\n'
                '\n'
                '{"msg_id": "a2"}\n'
            )

            with patch("memory.email_digest.QUEUE_DIR", queue_dir):
                result = get_queue_items()

            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["msg_id"], "a1")
            self.assertEqual(result[1]["msg_id"], "a2")


# ---------------------------------------------------------------------------
# TestBuildDigest
# ---------------------------------------------------------------------------

class TestBuildDigest(unittest.TestCase):
    """Test digest formatting, consumption filtering, and empty case."""

    def _make_items(self, *classifications_and_ts):
        """Create test items: [(classification, ts_ms), ...]"""
        items = []
        for i, (cls, ts) in enumerate(classifications_and_ts):
            items.append({
                "msg_id": f"m{i}",
                "sender": f"person{i}@test.com",
                "subject": f"Subject {i}",
                "snippet": f"Snippet {i}",
                "classification": cls,
                "score": 0.5,
                "reason": "test",
                "internal_ts_ms": ts,
            })
        return items

    @patch("memory.email_digest.datetime", _make_frozen_datetime_mock())
    def test_no_items_returns_no_digest_items(self):
        from memory.email_digest import build_digest

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"
            with patch("memory.email_digest.QUEUE_DIR", queue_dir), \
                 patch("memory.email_digest.config.STATE_FILE", state_file):
                result = build_digest("noon")

        self.assertEqual(result, "NO_DIGEST_ITEMS")

    @patch("memory.email_digest.datetime", _make_frozen_datetime_mock())
    def test_formats_digest_with_sections(self):
        from memory.email_digest import build_digest

        items = self._make_items(
            ("urgent", 1000),
            ("relevant", 2000),
            ("background", 3000),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"

            # Write items to today's JSONL
            queue_dir.mkdir(parents=True, exist_ok=True)
            path = queue_dir / f"{_FROZEN_DATE_STR}.jsonl"
            path.write_text("\n".join(json.dumps(i) for i in items) + "\n")

            with patch("memory.email_digest.QUEUE_DIR", queue_dir), \
                 patch("memory.email_digest.config.STATE_FILE", state_file):
                result = build_digest("noon")

        self.assertIn("Email Digest (Noon)", result)
        self.assertIn("ALREADY NOTIFIED:", result)
        self.assertIn("(urgent)", result)
        self.assertIn("NEW ITEMS:", result)
        self.assertIn("(relevant)", result)
        self.assertIn("(background)", result)
        self.assertIn("2 new emails", result)
        self.assertIn("1 previously notified", result)

    @patch("memory.email_digest.datetime", _make_frozen_datetime_mock())
    def test_consumption_filters_already_seen(self):
        """Items with ts <= consumed_ts should be excluded."""
        from memory.email_digest import build_digest

        items = self._make_items(
            ("relevant", 1000),
            ("relevant", 2000),
            ("relevant", 3000),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"

            queue_dir.mkdir(parents=True, exist_ok=True)
            path = queue_dir / f"{_FROZEN_DATE_STR}.jsonl"
            path.write_text("\n".join(json.dumps(i) for i in items) + "\n")

            # Pre-set consumed ts to 2000 — only item at 3000 should appear
            state_file.write_text(json.dumps({
                "email_digest_consumed_ts_ms": {"noon": 2000}
            }))

            with patch("memory.email_digest.QUEUE_DIR", queue_dir), \
                 patch("memory.email_digest.config.STATE_FILE", state_file):
                result = build_digest("noon")

        self.assertIn("1 new email", result)
        self.assertNotIn("2 new", result)

    @patch("memory.email_digest.datetime", _make_frozen_datetime_mock())
    def test_advances_high_water_mark(self):
        """build_digest should update the consumed ts in state.json."""
        from memory.email_digest import build_digest

        items = self._make_items(("relevant", 5000),)

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"

            queue_dir.mkdir(parents=True, exist_ok=True)
            path = queue_dir / f"{_FROZEN_DATE_STR}.jsonl"
            path.write_text(json.dumps(items[0]) + "\n")

            with patch("memory.email_digest.QUEUE_DIR", queue_dir), \
                 patch("memory.email_digest.config.STATE_FILE", state_file):
                build_digest("afternoon")

            state = json.loads(state_file.read_text())
            self.assertEqual(
                state["email_digest_consumed_ts_ms"]["afternoon"], 5000
            )

    @patch("memory.email_digest.datetime", _make_frozen_datetime_mock())
    def test_reads_yesterday_for_midnight_boundary(self):
        """build_digest should include items from yesterday's JSONL."""
        from memory.email_digest import build_digest

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"
            queue_dir.mkdir(parents=True, exist_ok=True)

            # Write to yesterday's file only
            yesterday_path = queue_dir / f"{_FROZEN_YESTERDAY_STR}.jsonl"
            yesterday_path.write_text(json.dumps({
                "msg_id": "y1", "sender": "old@test.com",
                "subject": "Yesterday mail", "snippet": "...",
                "classification": "relevant", "score": 0.5,
                "reason": "test", "internal_ts_ms": 500,
            }) + "\n")

            with patch("memory.email_digest.QUEUE_DIR", queue_dir), \
                 patch("memory.email_digest.config.STATE_FILE", state_file):
                result = build_digest("morning")

        self.assertIn("Yesterday mail", result)

    @patch("memory.email_digest.datetime", _make_frozen_datetime_mock())
    def test_sender_display_name_extraction(self):
        """Digest should show display name, not full email."""
        from memory.email_digest import build_digest

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"
            queue_dir.mkdir(parents=True, exist_ok=True)

            path = queue_dir / f"{_FROZEN_DATE_STR}.jsonl"
            path.write_text(json.dumps({
                "msg_id": "d1",
                "sender": "Jane Doe <jane@example.com>",
                "subject": "Hello",
                "snippet": "...",
                "classification": "relevant",
                "score": 0.5,
                "reason": "test",
                "internal_ts_ms": 100,
            }) + "\n")

            with patch("memory.email_digest.QUEUE_DIR", queue_dir), \
                 patch("memory.email_digest.config.STATE_FILE", state_file):
                result = build_digest("morning")

        self.assertIn("Jane Doe", result)


# ---------------------------------------------------------------------------
# TestDirectActionDispatch
# ---------------------------------------------------------------------------

class TestDirectActionDispatch(unittest.TestCase):
    """Test _is_direct_action and _execute_direct_action on AutomationEngine."""

    def _make_engine(self):
        from automations import AutomationEngine
        molly = MagicMock()
        engine = AutomationEngine.__new__(AutomationEngine)
        engine.molly = molly
        return engine

    def test_is_direct_action_email_digest(self):
        engine = self._make_engine()
        self.assertTrue(engine._is_direct_action({"action": "email_digest"}))

    def test_is_not_direct_action_agent(self):
        engine = self._make_engine()
        self.assertFalse(engine._is_direct_action({"prompt": "do something"}))

    def test_is_not_direct_action_channel(self):
        engine = self._make_engine()
        self.assertFalse(engine._is_direct_action({"channel": "whatsapp"}))

    def test_is_not_direct_action_unknown(self):
        engine = self._make_engine()
        self.assertFalse(engine._is_direct_action({"action": "unknown_action"}))

    @patch("memory.email_digest.datetime", _make_frozen_datetime_mock())
    def test_execute_calls_build_digest(self):
        engine = self._make_engine()

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"

            # No items → should return NO_DIGEST_ITEMS
            with patch("memory.email_digest.QUEUE_DIR", queue_dir), \
                 patch("memory.email_digest.config.STATE_FILE", state_file):
                result = asyncio.run(engine._execute_direct_action(
                    {"action": "email_digest", "params": {"slot": "noon"}},
                    {}, {},
                ))

        self.assertEqual(result, "NO_DIGEST_ITEMS")

    def test_execute_unknown_action(self):
        engine = self._make_engine()
        result = asyncio.run(engine._execute_direct_action(
            {"action": "nonexistent"}, {}, {},
        ))
        self.assertIn("Unknown direct action", result)


# ---------------------------------------------------------------------------
# TestDigestDeliverySkip
# ---------------------------------------------------------------------------

class TestDigestDeliverySkip(unittest.TestCase):
    """Test _should_skip_digest_delivery."""

    def _make_engine(self):
        from automations import AutomationEngine
        engine = AutomationEngine.__new__(AutomationEngine)
        engine.molly = MagicMock()
        return engine

    def _make_automation(self, automation_id):
        from automations import Automation
        auto = Automation.__new__(Automation)
        auto.automation_id = automation_id
        return auto

    def test_skips_when_no_digest_items(self):
        engine = self._make_engine()
        auto = self._make_automation("email-digest-noon")
        outputs = {"build_digest": {"output": "NO_DIGEST_ITEMS"}}
        self.assertTrue(engine._should_skip_digest_delivery(auto, outputs, ""))

    def test_skips_when_no_digest_items_in_message(self):
        engine = self._make_engine()
        auto = self._make_automation("email-digest-morning")
        outputs = {}
        self.assertTrue(engine._should_skip_digest_delivery(
            auto, outputs, "NO_DIGEST_ITEMS"
        ))

    def test_does_not_skip_real_digest(self):
        engine = self._make_engine()
        auto = self._make_automation("email-digest-noon")
        outputs = {"build_digest": {"output": "Email Digest (Noon)\n\nNEW ITEMS:\n..."}}
        self.assertFalse(engine._should_skip_digest_delivery(auto, outputs, ""))

    def test_does_not_skip_non_digest_automation(self):
        engine = self._make_engine()
        auto = self._make_automation("email-triage")
        outputs = {"build_digest": {"output": "NO_DIGEST_ITEMS"}}
        self.assertFalse(engine._should_skip_digest_delivery(auto, outputs, ""))


# ---------------------------------------------------------------------------
# TestCleanupOldFiles
# ---------------------------------------------------------------------------

class TestCleanupOldFiles(unittest.TestCase):
    """Test cleanup_old_files removes old JSONL files."""

    def test_removes_old_keeps_recent(self):
        from memory.email_digest import cleanup_old_files

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            today = datetime.now(timezone.utc)

            # Old file (10 days ago)
            old_date = (today - timedelta(days=10)).strftime("%Y-%m-%d")
            old_path = queue_dir / f"{old_date}.jsonl"
            old_path.write_text('{"msg_id": "old"}\n')

            # Recent file (1 day ago)
            recent_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            recent_path = queue_dir / f"{recent_date}.jsonl"
            recent_path.write_text('{"msg_id": "recent"}\n')

            with patch("memory.email_digest.QUEUE_DIR", queue_dir):
                deleted = cleanup_old_files(keep_days=3)

            self.assertEqual(deleted, 1)
            self.assertFalse(old_path.exists())
            self.assertTrue(recent_path.exists())


# ---------------------------------------------------------------------------
# Source assertion tests (wiring checks)
# ---------------------------------------------------------------------------

class TestSourceAssertions(unittest.TestCase):
    """Verify email digest integration hooks are present in source files."""

    _PROJECT_ROOT = Path(__file__).parent.parent

    def _read_source(self, *parts: str) -> str:
        return self._PROJECT_ROOT.joinpath(*parts).read_text()

    def test_email_digest_module_exists(self):
        content = self._read_source("memory", "email_digest.py")
        self.assertIn("def append_digest_item", content)
        self.assertIn("def build_digest", content)
        self.assertIn("def get_queue_items", content)
        self.assertIn("def cleanup_old_files", content)

    def test_email_digest_uses_config_workspace(self):
        content = self._read_source("memory", "email_digest.py")
        self.assertIn("config.EMAIL_DIGEST_QUEUE_DIR", content)

    def test_config_has_email_digest_queue_dir(self):
        content = self._read_source("config.py")
        self.assertIn("EMAIL_DIGEST_QUEUE_DIR", content)

    def test_heartbeat_queues_for_digest(self):
        content = self._read_source("heartbeat.py")
        self.assertIn("append_digest_item", content)
        self.assertIn("email_digest", content)

    def test_automations_has_direct_action_dispatch(self):
        content = self._read_source("commitments.py")
        self.assertIn("_is_direct_action", content)
        self.assertIn("_execute_direct_action", content)
        self.assertIn("_DIRECT_ACTIONS", content)

    def test_automations_has_digest_delivery_skip(self):
        content = self._read_source("commitments.py")
        self.assertIn("_should_skip_digest_delivery", content)
        self.assertIn("NO_DIGEST_ITEMS", content)

    def test_maintenance_has_digest_queue_cleanup(self):
        content = self._read_source("monitoring", "jobs", "cleanup_jobs.py")
        self.assertIn("cleanup_old_files", content)
        self.assertIn("email_digest", content)


# ---------------------------------------------------------------------------
# TestInvalidSlotFallback (QA audit finding)
# ---------------------------------------------------------------------------

class TestInvalidSlotFallback(unittest.TestCase):
    """Verify invalid slot names fall back to 'morning' with a warning."""

    @patch("memory.email_digest.datetime", _make_frozen_datetime_mock())
    def test_invalid_slot_falls_back_to_morning(self):
        from memory.email_digest import build_digest

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"
            queue_dir.mkdir(parents=True, exist_ok=True)

            path = queue_dir / f"{_FROZEN_DATE_STR}.jsonl"
            path.write_text(json.dumps({
                "msg_id": "x1", "sender": "test@test.com",
                "subject": "Test", "snippet": "...",
                "classification": "relevant", "score": 0.5,
                "reason": "test", "internal_ts_ms": 100,
            }) + "\n")

            with patch("memory.email_digest.QUEUE_DIR", queue_dir), \
                 patch("memory.email_digest.config.STATE_FILE", state_file):
                result = build_digest("bogus_slot")

                self.assertIn("Email Digest (Morning)", result)
                # High-water mark should be written under "morning" key
                state = json.loads(state_file.read_text())
                self.assertIn("morning", state["email_digest_consumed_ts_ms"])


# ---------------------------------------------------------------------------
# TestStatePreservation (QA audit finding)
# ---------------------------------------------------------------------------

class TestStatePreservation(unittest.TestCase):
    """Verify build_digest preserves existing state.json keys."""

    @patch("memory.email_digest.datetime", _make_frozen_datetime_mock())
    def test_preserves_existing_state_keys(self):
        from memory.email_digest import build_digest

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"
            queue_dir.mkdir(parents=True, exist_ok=True)

            path = queue_dir / f"{_FROZEN_DATE_STR}.jsonl"
            path.write_text(json.dumps({
                "msg_id": "s1", "sender": "a@b.com",
                "subject": "S", "snippet": "...",
                "classification": "relevant", "score": 0.5,
                "reason": "t", "internal_ts_ms": 100,
            }) + "\n")

            # Pre-populate state with existing keys
            state_file.write_text(json.dumps({
                "email_heartbeat_hw": "2026-01-01",
                "imessage_mention_hw": 1234567890,
                "email_digest_consumed_ts_ms": {"noon": 50},
            }))

            with patch("memory.email_digest.QUEUE_DIR", queue_dir), \
                 patch("memory.email_digest.config.STATE_FILE", state_file):
                build_digest("afternoon")

            state = json.loads(state_file.read_text())
            # Existing keys must be preserved
            self.assertEqual(state["email_heartbeat_hw"], "2026-01-01")
            self.assertEqual(state["imessage_mention_hw"], 1234567890)
            # Existing slot must be preserved, new slot added
            self.assertEqual(state["email_digest_consumed_ts_ms"]["noon"], 50)
            self.assertEqual(state["email_digest_consumed_ts_ms"]["afternoon"], 100)


# ---------------------------------------------------------------------------
# TestSlotIndependence (QA audit finding)
# ---------------------------------------------------------------------------

class TestSlotIndependence(unittest.TestCase):
    """Verify that consuming in one slot does not affect another."""

    @patch("memory.email_digest.datetime", _make_frozen_datetime_mock())
    def test_slots_are_independent(self):
        from memory.email_digest import build_digest

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"
            queue_dir.mkdir(parents=True, exist_ok=True)

            path = queue_dir / f"{_FROZEN_DATE_STR}.jsonl"
            path.write_text(json.dumps({
                "msg_id": "i1", "sender": "a@b.com",
                "subject": "Independent", "snippet": "...",
                "classification": "relevant", "score": 0.5,
                "reason": "t", "internal_ts_ms": 500,
            }) + "\n")

            with patch("memory.email_digest.QUEUE_DIR", queue_dir), \
                 patch("memory.email_digest.config.STATE_FILE", state_file):
                # Noon consumes the item
                noon_result = build_digest("noon")
                self.assertIn("Independent", noon_result)

                # Afternoon should also see the same item (independent HW mark)
                afternoon_result = build_digest("afternoon")
                self.assertIn("Independent", afternoon_result)

            state = json.loads(state_file.read_text())
            self.assertEqual(state["email_digest_consumed_ts_ms"]["noon"], 500)
            self.assertEqual(state["email_digest_consumed_ts_ms"]["afternoon"], 500)


# ---------------------------------------------------------------------------
# TestSnippetTruncation (QA audit finding)
# ---------------------------------------------------------------------------

class TestSnippetTruncation(unittest.TestCase):
    """Verify snippet and reason are truncated in JSONL entries."""

    @patch("memory.email_digest.datetime", _make_frozen_datetime_mock())
    def test_snippet_and_reason_truncated(self):
        from memory.email_digest import append_digest_item

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            with patch("memory.email_digest.QUEUE_DIR", queue_dir):
                append_digest_item(
                    msg_id="t1", sender="a@b.com", subject="Long",
                    snippet="x" * 500, classification="relevant",
                    score=0.5, reason="y" * 300, internal_ts_ms=1,
                )

            path = queue_dir / f"{_FROZEN_DATE_STR}.jsonl"
            entry = json.loads(path.read_text().strip())
            self.assertLessEqual(len(entry["snippet"]), 300)
            self.assertLessEqual(len(entry["reason"]), 200)


# ---------------------------------------------------------------------------
# TestSenderDisplay (QA audit finding)
# ---------------------------------------------------------------------------

class TestSenderDisplay(unittest.TestCase):
    """Test _sender_display edge cases."""

    def test_bare_email(self):
        from memory.email_digest import _sender_display
        self.assertEqual(_sender_display("jane@example.com"), "jane@example.com")

    def test_name_with_angle_brackets(self):
        from memory.email_digest import _sender_display
        self.assertEqual(_sender_display("Jane Doe <jane@example.com>"), "Jane Doe")

    def test_empty_string(self):
        from memory.email_digest import _sender_display
        self.assertEqual(_sender_display(""), "")

    def test_angle_brackets_no_name(self):
        from memory.email_digest import _sender_display
        # "<jane@example.com>" — empty name before <
        self.assertEqual(_sender_display("<jane@example.com>"), "<jane@example.com>")

    def test_quoted_name(self):
        from memory.email_digest import _sender_display
        self.assertEqual(_sender_display('"Jane Doe" <jane@example.com>'), "Jane Doe")


# ---------------------------------------------------------------------------
# TestDirectActionWithItems (QA audit finding)
# ---------------------------------------------------------------------------

class TestDirectActionWithItems(unittest.TestCase):
    """Test _execute_direct_action with actual digest items."""

    @patch("memory.email_digest.datetime", _make_frozen_datetime_mock())
    def test_execute_returns_formatted_digest(self):
        from automations import AutomationEngine
        engine = AutomationEngine.__new__(AutomationEngine)
        engine.molly = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            state_file = Path(tmpdir) / "state.json"
            queue_dir.mkdir(parents=True, exist_ok=True)

            path = queue_dir / f"{_FROZEN_DATE_STR}.jsonl"
            path.write_text(json.dumps({
                "msg_id": "d1", "sender": "alice@test.com",
                "subject": "Budget Review", "snippet": "...",
                "classification": "relevant", "score": 0.6,
                "reason": "work", "internal_ts_ms": 999,
            }) + "\n")

            with patch("memory.email_digest.QUEUE_DIR", queue_dir), \
                 patch("memory.email_digest.config.STATE_FILE", state_file):
                result = asyncio.run(engine._execute_direct_action(
                    {"action": "email_digest", "params": {"slot": "noon"}},
                    {}, {},
                ))

        self.assertIn("Email Digest (Noon)", result)
        self.assertIn("Budget Review", result)


# ---------------------------------------------------------------------------
# TestCleanupBoundary (QA audit finding)
# ---------------------------------------------------------------------------

class TestCleanupBoundary(unittest.TestCase):
    """Test cleanup_old_files boundary: file exactly at cutoff is kept."""

    def test_file_exactly_at_cutoff_not_deleted(self):
        from memory.email_digest import cleanup_old_files

        with tempfile.TemporaryDirectory() as tmpdir:
            queue_dir = Path(tmpdir)
            today = datetime.now(timezone.utc)

            # File exactly at cutoff (3 days ago) — should NOT be deleted
            cutoff_date = (today - timedelta(days=3)).strftime("%Y-%m-%d")
            cutoff_path = queue_dir / f"{cutoff_date}.jsonl"
            cutoff_path.write_text('{"msg_id": "boundary"}\n')

            # File past cutoff (4 days ago) — should be deleted
            old_date = (today - timedelta(days=4)).strftime("%Y-%m-%d")
            old_path = queue_dir / f"{old_date}.jsonl"
            old_path.write_text('{"msg_id": "old"}\n')

            with patch("memory.email_digest.QUEUE_DIR", queue_dir):
                deleted = cleanup_old_files(keep_days=3)

            self.assertEqual(deleted, 1)
            self.assertTrue(cutoff_path.exists(), "File at cutoff should be kept")
            self.assertFalse(old_path.exists(), "File past cutoff should be deleted")


if __name__ == "__main__":
    unittest.main()
