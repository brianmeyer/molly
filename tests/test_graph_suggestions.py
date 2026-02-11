"""Tests for memory.graph_suggestions module and related integration wiring."""

import asyncio
import json
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Frozen UTC time helpers — avoids midnight edge cases (L5)
# ---------------------------------------------------------------------------
_FROZEN_UTC = datetime(2026, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
_FROZEN_DATE_STR = "2026-06-15"


def _make_frozen_datetime_mock():
    """Create a datetime mock that returns _FROZEN_UTC from .now(tz)."""
    mock_dt = MagicMock(wraps=datetime)
    mock_dt.now.return_value = _FROZEN_UTC
    mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
    return mock_dt


class TestLogRelationshipFallback(unittest.TestCase):
    """Test that log_relationship_fallback writes correct JSONL."""

    @patch("memory.graph_suggestions.datetime", _make_frozen_datetime_mock())
    def test_creates_jsonl_file(self):
        from memory.graph_suggestions import log_relationship_fallback

        with tempfile.TemporaryDirectory() as tmpdir:
            suggestions_dir = Path(tmpdir)
            with patch("memory.graph_suggestions.SUGGESTIONS_DIR", suggestions_dir):
                log_relationship_fallback(
                    head="Brian",
                    tail="Netflix",
                    original_type="subscribes to",
                    confidence=0.62,
                    context="Brian mentioned his Netflix subscription",
                )

            path = suggestions_dir / f"{_FROZEN_DATE_STR}.jsonl"
            self.assertTrue(path.exists(), f"Expected JSONL at {path}")

            lines = path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["type"], "relationship_fallback")
            self.assertEqual(entry["head"], "Brian")
            self.assertEqual(entry["tail"], "Netflix")
            self.assertEqual(entry["original_type"], "subscribes to")
            self.assertEqual(entry["fell_back_to"], "RELATED_TO")
            self.assertAlmostEqual(entry["confidence"], 0.62, places=2)
            self.assertIn("SUBSCRIBES_TO", entry["suggestion"])


class TestLogRepeatedRelatedTo(unittest.TestCase):
    """Test that log_repeated_related_to appends JSONL entries."""

    @patch("memory.graph_suggestions.datetime", _make_frozen_datetime_mock())
    def test_appends_multiple_entries(self):
        from memory.graph_suggestions import log_repeated_related_to

        with tempfile.TemporaryDirectory() as tmpdir:
            suggestions_dir = Path(tmpdir)
            with patch("memory.graph_suggestions.SUGGESTIONS_DIR", suggestions_dir):
                log_repeated_related_to("Alice", "ProjectX", 3)
                log_repeated_related_to("Bob", "CompanyY", 5)

            path = suggestions_dir / f"{_FROZEN_DATE_STR}.jsonl"
            lines = path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)

            entry1 = json.loads(lines[0])
            self.assertEqual(entry1["type"], "related_to_hotspot")
            self.assertEqual(entry1["head"], "Alice")
            self.assertEqual(entry1["tail"], "ProjectX")
            self.assertEqual(entry1["mention_count"], 3)

            entry2 = json.loads(lines[1])
            self.assertEqual(entry2["head"], "Bob")
            self.assertEqual(entry2["mention_count"], 5)


class TestGetSuggestions(unittest.TestCase):
    """Test reading suggestions from JSONL files."""

    def test_reads_jsonl(self):
        from memory.graph_suggestions import get_suggestions

        with tempfile.TemporaryDirectory() as tmpdir:
            suggestions_dir = Path(tmpdir)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = suggestions_dir / f"{today}.jsonl"
            path.write_text(
                json.dumps({"type": "relationship_fallback", "head": "A", "tail": "B"}) + "\n"
                + json.dumps({"type": "related_to_hotspot", "head": "C", "tail": "D"}) + "\n"
            )

            with patch("memory.graph_suggestions.SUGGESTIONS_DIR", suggestions_dir):
                result = get_suggestions()

            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["head"], "A")
            self.assertEqual(result[1]["type"], "related_to_hotspot")

    def test_missing_file_returns_empty(self):
        from memory.graph_suggestions import get_suggestions

        with tempfile.TemporaryDirectory() as tmpdir:
            suggestions_dir = Path(tmpdir)
            with patch("memory.graph_suggestions.SUGGESTIONS_DIR", suggestions_dir):
                result = get_suggestions("2020-01-01")

            self.assertEqual(result, [])


class TestGetRelatedToHotspots(unittest.TestCase):
    """Test Neo4j hotspot queries.

    Note (L6): These tests use plain dict mocks for Neo4j records. This is
    acceptable because get_related_to_hotspots() only calls dict(record) on
    the results, and iterating over a list of dicts works identically.
    """

    def test_returns_results_from_neo4j(self):
        """get_related_to_hotspots should execute Neo4j query and return records."""
        import memory.graph_suggestions as gs

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = [
            {"head": "Alice", "tail": "ProjectX", "mentions": 5, "contexts": []},
            {"head": "Bob", "tail": "CompanyY", "mentions": 3, "contexts": []},
        ]

        # Patch get_driver at the point of import inside get_related_to_hotspots
        with patch.dict("sys.modules", {"memory.graph": MagicMock(get_driver=MagicMock(return_value=mock_driver))}):
            result = gs.get_related_to_hotspots(min_mentions=3)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["head"], "Alice")
        self.assertEqual(result[1]["mentions"], 3)
        mock_session.run.assert_called_once()

    def test_neo4j_down_returns_empty(self):
        """get_related_to_hotspots should return [] when Neo4j is unavailable."""
        import memory.graph_suggestions as gs

        # Simulate Neo4j being down by making the import raise
        mock_graph = MagicMock()
        mock_graph.get_driver.side_effect = Exception("connection refused")
        with patch.dict("sys.modules", {"memory.graph": mock_graph}):
            result = gs.get_related_to_hotspots()

        self.assertEqual(result, [])

    def test_passes_min_mentions_parameter(self):
        """get_related_to_hotspots should pass min_mentions to the Cypher query."""
        import memory.graph_suggestions as gs

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = []

        with patch.dict("sys.modules", {"memory.graph": MagicMock(get_driver=MagicMock(return_value=mock_driver))}):
            gs.get_related_to_hotspots(min_mentions=10)

        call_kwargs = mock_session.run.call_args
        self.assertEqual(call_kwargs.kwargs.get("min_mentions"), 10)


class TestWriteObservation(unittest.TestCase):
    """Test foundry_adapter.write_observation."""

    def test_creates_jsonl_file(self):
        """write_observation should create a JSONL file with the observation."""
        from foundry_adapter import write_observation

        with tempfile.TemporaryDirectory() as tmpdir:
            obs_dir = Path(tmpdir)
            write_observation(
                tool_sequence=["search", "analyze", "report"],
                outcome="success",
                context="test context",
                observations_dir=obs_dir,
            )

            jsonl_files = list(obs_dir.glob("*.jsonl"))
            self.assertEqual(len(jsonl_files), 1)
            content = jsonl_files[0].read_text().strip()
            entry = json.loads(content)
            self.assertEqual(entry["tool_sequence"], ["search", "analyze", "report"])
            self.assertEqual(entry["outcome"], "success")
            self.assertIn("test context", entry["context"])

    def test_skips_short_sequences(self):
        """write_observation should not write when sequence < 3 steps."""
        from foundry_adapter import write_observation

        with tempfile.TemporaryDirectory() as tmpdir:
            obs_dir = Path(tmpdir)
            write_observation(
                tool_sequence=["search", "analyze"],
                outcome="success",
                observations_dir=obs_dir,
            )

            jsonl_files = list(obs_dir.glob("*.jsonl"))
            self.assertEqual(len(jsonl_files), 0)


class TestBuildSuggestionDigest(unittest.TestCase):
    """Test the nightly digest builder."""

    def test_digest_with_fallbacks(self):
        from memory.graph_suggestions import build_suggestion_digest

        with tempfile.TemporaryDirectory() as tmpdir:
            suggestions_dir = Path(tmpdir)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = suggestions_dir / f"{today}.jsonl"
            entries = [
                {"type": "relationship_fallback", "original_type": "subscribes to",
                 "head": "Brian", "tail": "Netflix"},
                {"type": "relationship_fallback", "original_type": "subscribes to",
                 "head": "Brian", "tail": "Spotify"},
                {"type": "related_to_hotspot", "head": "Alice", "tail": "ProjectX",
                 "mention_count": 5},
            ]
            path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

            with patch("memory.graph_suggestions.SUGGESTIONS_DIR", suggestions_dir), \
                 patch("memory.graph_suggestions.get_related_to_hotspots", return_value=[]):
                digest = build_suggestion_digest()

            self.assertIn("suggestion", digest.lower())
            self.assertIn("SUBSCRIBES_TO", digest)
            self.assertIn("2x", digest)
            # H2: Should show both items and events
            self.assertIn("events", digest.lower())

    def test_empty_returns_empty_string(self):
        from memory.graph_suggestions import build_suggestion_digest

        with tempfile.TemporaryDirectory() as tmpdir:
            suggestions_dir = Path(tmpdir)
            with patch("memory.graph_suggestions.SUGGESTIONS_DIR", suggestions_dir), \
                 patch("memory.graph_suggestions.get_related_to_hotspots", return_value=[]):
                digest = build_suggestion_digest()

            self.assertEqual(digest, "")


class TestBuildSuggestionDigestDeduplication(unittest.TestCase):
    """Test deduplication between JSONL and Neo4j hotspots in digest."""

    def test_neo4j_hotspots_deduplicated_against_jsonl(self):
        """Neo4j hotspots already in JSONL should not be listed twice."""
        from memory.graph_suggestions import build_suggestion_digest

        with tempfile.TemporaryDirectory() as tmpdir:
            suggestions_dir = Path(tmpdir)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = suggestions_dir / f"{today}.jsonl"
            entries = [
                {"type": "related_to_hotspot", "head": "Alice", "tail": "ProjectX",
                 "mention_count": 5},
            ]
            path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

            neo4j_hotspots = [
                {"head": "Alice", "tail": "ProjectX", "mentions": 5, "contexts": []},
                {"head": "Bob", "tail": "CompanyY", "mentions": 4, "contexts": []},
            ]

            with patch("memory.graph_suggestions.SUGGESTIONS_DIR", suggestions_dir), \
                 patch("memory.graph_suggestions.get_related_to_hotspots", return_value=neo4j_hotspots):
                digest = build_suggestion_digest()

            # Alice -> ProjectX should appear once (from JSONL), not twice
            self.assertEqual(digest.lower().count("alice"), 1)
            # Bob -> CompanyY should appear (Neo4j only)
            self.assertIn("Bob", digest)
            # Total should be 2 (1 JSONL hotspot + 1 Neo4j-only hotspot)
            self.assertIn("2 graph suggestion", digest)

    def test_case_insensitive_dedup(self):
        """Deduplication should be case-insensitive (M4)."""
        from memory.graph_suggestions import build_suggestion_digest

        with tempfile.TemporaryDirectory() as tmpdir:
            suggestions_dir = Path(tmpdir)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = suggestions_dir / f"{today}.jsonl"
            entries = [
                {"type": "related_to_hotspot", "head": "alice", "tail": "projectx",
                 "mention_count": 5},
            ]
            path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

            # Neo4j returns same entity with different casing
            neo4j_hotspots = [
                {"head": "Alice", "tail": "ProjectX", "mentions": 5, "contexts": []},
            ]

            with patch("memory.graph_suggestions.SUGGESTIONS_DIR", suggestions_dir), \
                 patch("memory.graph_suggestions.get_related_to_hotspots", return_value=neo4j_hotspots):
                digest = build_suggestion_digest()

            # Should deduplicate despite casing difference — only 1 item
            self.assertIn("1 graph suggestion", digest)


class TestMalformedJsonlHandling(unittest.TestCase):
    """Test that malformed JSONL lines are skipped gracefully."""

    def test_corrupt_lines_skipped(self):
        """get_suggestions should skip corrupt JSON lines without error."""
        from memory.graph_suggestions import get_suggestions

        with tempfile.TemporaryDirectory() as tmpdir:
            suggestions_dir = Path(tmpdir)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = suggestions_dir / f"{today}.jsonl"
            path.write_text(
                '{"type": "relationship_fallback", "head": "A", "tail": "B"}\n'
                '{corrupted json line\n'
                '\n'
                'not json at all\n'
                '{"type": "related_to_hotspot", "head": "C", "tail": "D"}\n'
            )

            with patch("memory.graph_suggestions.SUGGESTIONS_DIR", suggestions_dir):
                result = get_suggestions()

            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["head"], "A")
            self.assertEqual(result[1]["head"], "C")


# ---------------------------------------------------------------------------
# classify_local tests with text="" (F4 — exercises the M5 fix)
# ---------------------------------------------------------------------------

class TestClassifyLocalEmptyText(unittest.TestCase):
    """Verify classify_local handles text='' correctly (M5 contract)."""

    def test_empty_text_passes_prompt_unchanged(self):
        """With text='', the prompt should reach the model unchanged."""
        from memory.triage import classify_local

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "YES"}}],
        }

        prompt = "Is the user correcting the assistant? Respond YES or NO."
        with patch("memory.triage._load_model", return_value=mock_model):
            result = classify_local(prompt, "")

        # The prompt should be sent unchanged as the user message
        call_args = mock_model.create_chat_completion.call_args
        user_content = call_args.kwargs["messages"][1]["content"]
        self.assertEqual(user_content, prompt)
        self.assertEqual(result, "YES")

    def test_placeholder_tokens_in_prompt_replaced_with_empty(self):
        """If prompt contains literal {reply} or {text}, they are replaced with ''."""
        from memory.triage import classify_local

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "NO"}}],
        }

        # Simulate a prompt that happens to contain the placeholder token
        prompt = "The user said: '{reply}'. Is this a correction? YES or NO."
        with patch("memory.triage._load_model", return_value=mock_model):
            result = classify_local(prompt, "")

        call_args = mock_model.create_chat_completion.call_args
        user_content = call_args.kwargs["messages"][1]["content"]
        # {reply} should be replaced with "" since payload is ""
        self.assertEqual(user_content, "The user said: ''. Is this a correction? YES or NO.")
        self.assertEqual(result, "NO")


# ---------------------------------------------------------------------------
# Behavioral tests for LLM classification paths (L4)
#
# Importing main.py triggers a deep import chain (agent → processor →
# embeddings → numpy) that may not be available outside the venv. We
# mock heavyweight modules at sys.modules level before importing main.
# ---------------------------------------------------------------------------

def _import_main_safe():
    """Import main.py with heavyweight dependencies mocked out.

    NOTE: This stubs modules that main.py imports at the top level.
    If main.py adds new top-level imports of unavailable packages,
    add them to the stub list below. The cached module is reused
    across tests within the same process.
    """
    import sys
    _stubs = {}
    for mod in ("numpy", "numpy.typing", "neo4j", "fastapi", "pydantic",
                "uvicorn", "memory.embeddings", "memory.processor",
                "memory.graph", "agent"):
        if mod not in sys.modules:
            _stubs[mod] = MagicMock()
    with patch.dict("sys.modules", _stubs):
        import importlib
        # Force re-import if already cached without stubs
        if "main" in sys.modules:
            main_mod = sys.modules["main"]
        else:
            main_mod = importlib.import_module("main")
    return main_mod


class TestCorrectionDetection(unittest.TestCase):
    """Test _detect_and_log_correction behavioral logic with mocked LLM."""

    def _make_molly(self):
        """Create a minimal Molly instance with required attributes."""
        main_mod = _import_main_safe()
        Molly = main_mod.Molly
        molly = Molly.__new__(Molly)
        molly._last_responses = {}
        molly.registered_chats = {}
        molly.owner_jid = "owner@test"
        molly._recent_surfaces = []
        return molly

    def _run_async(self, coro):
        """Run a coroutine synchronously."""
        return asyncio.run(coro)

    def test_correction_detected_and_logged(self):
        """When keyword matches AND LLM says YES, log_correction should be called."""
        molly = self._make_molly()
        molly._last_responses["chat@test"] = ("I think Brian works at Google", time.time())

        mock_vs = MagicMock()
        mock_classify = AsyncMock(return_value="YES")
        mock_triage = MagicMock(classify_local_async=mock_classify)
        mock_retriever = MagicMock(get_vectorstore=MagicMock(return_value=mock_vs))
        with patch.dict("sys.modules", {
            "memory.triage": mock_triage,
            "memory.retriever": mock_retriever,
        }):
            self._run_async(
                molly._detect_and_log_correction("chat@test", "No, that's wrong — he works at Meta")
            )

        mock_vs.log_correction.assert_called_once()

    def test_correction_rejected_by_llm(self):
        """When keyword matches but LLM says NO, log_correction should NOT be called."""
        molly = self._make_molly()
        molly._last_responses["chat@test"] = ("The weather is nice today", time.time())

        mock_vs = MagicMock()
        mock_classify = AsyncMock(return_value="NO")
        mock_triage = MagicMock(classify_local_async=mock_classify)
        mock_retriever = MagicMock(get_vectorstore=MagicMock(return_value=mock_vs))
        with patch.dict("sys.modules", {
            "memory.triage": mock_triage,
            "memory.retriever": mock_retriever,
        }):
            self._run_async(
                molly._detect_and_log_correction("chat@test", "Actually, I agree with that")
            )

        mock_vs.log_correction.assert_not_called()

    def test_correction_skips_without_keyword(self):
        """When no correction keyword is found, LLM should not even be called."""
        molly = self._make_molly()
        molly._last_responses["chat@test"] = ("Hello there", time.time())

        mock_classify = AsyncMock()
        mock_triage = MagicMock(classify_local_async=mock_classify)
        with patch.dict("sys.modules", {"memory.triage": mock_triage}):
            self._run_async(
                molly._detect_and_log_correction("chat@test", "Thanks for the info")
            )

        mock_classify.assert_not_called()


# ---------------------------------------------------------------------------
# Source assertion tests (L3: DRY helper + integration-wiring checks)
# ---------------------------------------------------------------------------

class TestSourceAssertions(unittest.TestCase):
    """Verify integration hooks are present in source files.

    These are wiring checks — they confirm that modules are connected as
    designed (e.g., graph.py calls graph_suggestions hooks, maintenance.py
    includes digest steps). They are intentionally string-based and will
    break if hooks are renamed; update them alongside any refactoring.
    """

    _PROJECT_ROOT = Path(__file__).parent.parent

    def _read_source(self, *parts: str) -> str:
        """Read a source file relative to the project root."""
        return self._PROJECT_ROOT.joinpath(*parts).read_text()

    def test_graph_py_contains_fallback_hook(self):
        content = self._read_source("memory", "graph.py")
        self.assertIn("log_relationship_fallback", content)

    def test_graph_py_contains_repeated_hook(self):
        content = self._read_source("memory", "graph.py")
        self.assertIn("log_repeated_related_to", content)

    def test_maintenance_contains_graph_suggestions(self):
        content = self._read_source("maintenance.py")
        self.assertIn("Graph suggestions", content)
        self.assertIn("build_suggestion_digest", content)

    def test_maintenance_contains_neo4j_checkpoint(self):
        content = self._read_source("maintenance.py")
        self.assertIn("db.checkpoint()", content)
        self.assertIn("Neo4j checkpoint", content)
        # Should try modern syntax first, with legacy fallback (M3)
        self.assertIn("SHOW SERVER INFO", content)
        self.assertIn("dbms.components()", content)

    def test_maintenance_contains_operational_insights(self):
        content = self._read_source("maintenance.py")
        self.assertIn("Operational insights", content)
        self.assertIn("_compute_operational_insights", content)

    def test_maintenance_contains_foundry_scans(self):
        content = self._read_source("maintenance.py")
        self.assertIn("Foundry skill scan", content)
        self.assertIn("Tool gap scan", content)

    def test_main_contains_correction_detection(self):
        content = self._read_source("main.py")
        self.assertIn("_CORRECTION_KEYWORDS", content)
        self.assertIn("_detect_and_log_correction", content)
        self.assertIn("correction-detect:", content)
        self.assertIn("log_correction", content)

    def test_main_contains_llm_preference_classification(self):
        content = self._read_source("main.py")
        self.assertIn("classify_local_async", content)
        self.assertIn("_log_preference_signal_if_dismissive", content)
        # Regex fast-path should be removed
        self.assertNotIn("_EXPLICIT_PREFERENCE_PATTERNS", content)

    def test_foundry_adapter_contains_write_observation(self):
        content = self._read_source("foundry_adapter.py")
        self.assertIn("def write_observation", content)

    def test_foundry_adapter_uses_config_workspace(self):
        content = self._read_source("foundry_adapter.py")
        self.assertIn("config.WORKSPACE", content)
        # Should NOT have hardcoded user path
        self.assertNotIn("/Users/brianmeyer/", content)

    def test_agent_contains_foundry_write(self):
        content = self._read_source("agent.py")
        self.assertIn("write_observation", content)

    def test_graph_suggestions_module_exists(self):
        content = self._read_source("memory", "graph_suggestions.py")
        self.assertIn("def log_relationship_fallback", content)
        self.assertIn("def log_repeated_related_to", content)
        self.assertIn("def get_suggestions", content)
        self.assertIn("def get_related_to_hotspots", content)
        self.assertIn("def build_suggestion_digest", content)

    def test_graph_suggestions_uses_config_workspace(self):
        content = self._read_source("memory", "graph_suggestions.py")
        self.assertIn("config.WORKSPACE", content)

    def test_self_improve_has_public_wrappers(self):
        content = self._read_source("self_improve.py")
        self.assertIn("async def propose_skill_updates(self", content)
        self.assertIn("async def propose_tool_updates(self", content)

    def test_maintenance_uses_public_wrappers(self):
        content = self._read_source("maintenance.py")
        self.assertIn("propose_skill_updates(", content)
        self.assertIn("propose_tool_updates(", content)
        # Should NOT call the private versions directly
        self.assertNotIn("_propose_skill_updates_from_patterns", content)
        self.assertNotIn("_propose_tool_updates_from_failures", content)

    def test_maintenance_contains_correction_patterns(self):
        content = self._read_source("maintenance.py")
        self.assertIn("Correction patterns", content)
        self.assertIn("corrections", content)

    def test_graph_py_has_debug_logging(self):
        content = self._read_source("memory", "graph.py")
        self.assertIn("graph suggestion fallback logging failed", content)
        self.assertIn("graph suggestion hotspot logging failed", content)

    def test_maintenance_has_jsonl_cleanup(self):
        content = self._read_source("maintenance.py")
        self.assertIn("graph_suggestions", content)
        self.assertIn(".jsonl", content)
        self.assertIn("unlink()", content)


if __name__ == "__main__":
    unittest.main()
