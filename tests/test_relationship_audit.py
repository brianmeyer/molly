"""Tests for memory.relationship_audit and related integration wiring."""

import asyncio
import json
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
import importlib.util
import config


def _load_module_from_file(name: str, file_path: str):
    """Force-load a module from its real file path, bypassing sys.modules pollution."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Force-load from real files to avoid cross-test sys.modules pollution
_graph_mod = _load_module_from_file(
    "memory.graph", str(PROJECT_ROOT / "memory" / "graph.py")
)
VALID_REL_TYPES = _graph_mod.VALID_REL_TYPES

from memory.relationship_audit import (
    CONFLICTING_SYMMETRIC,
    DETERMINISTIC_RECLASSIFY,
    ENTITY_REL_COMPATIBILITY,
    _compatible,
    run_deterministic_audit,
    run_model_audit,
    run_relationship_audit,
)


# ---------------------------------------------------------------------------
# Compatibility matrix tests
# ---------------------------------------------------------------------------

class TestCompatibilityMatrix(unittest.TestCase):
    """Verify ENTITY_REL_COMPATIBILITY and DETERMINISTIC_RECLASSIFY are self-consistent."""

    def test_all_reclassify_targets_are_valid_rel_types(self):
        for key, target in DETERMINISTIC_RECLASSIFY.items():
            self.assertIn(
                target, VALID_REL_TYPES,
                f"DETERMINISTIC_RECLASSIFY {key} -> {target} not in VALID_REL_TYPES",
            )

    def test_all_compatibility_types_are_valid(self):
        for pair, types in ENTITY_REL_COMPATIBILITY.items():
            for t in types:
                self.assertIn(
                    t, VALID_REL_TYPES,
                    f"ENTITY_REL_COMPATIBILITY {pair} contains {t} not in VALID_REL_TYPES",
                )

    def test_conflicting_symmetric_types_are_valid(self):
        for type_a, type_b in CONFLICTING_SYMMETRIC:
            self.assertIn(type_a, VALID_REL_TYPES, f"{type_a} not in VALID_REL_TYPES")
            self.assertIn(type_b, VALID_REL_TYPES, f"{type_b} not in VALID_REL_TYPES")

    def test_related_to_in_all_compatibility_pairs(self):
        for pair, types in ENTITY_REL_COMPATIBILITY.items():
            self.assertIn(
                "RELATED_TO", types,
                f"ENTITY_REL_COMPATIBILITY {pair} missing RELATED_TO fallback",
            )

    def test_compatible_helper_known_pair(self):
        self.assertTrue(_compatible("Person", "Organization", "WORKS_AT"))

    def test_compatible_helper_mismatch(self):
        self.assertFalse(_compatible("Person", "Place", "WORKS_AT"))

    def test_compatible_helper_unknown_pair_only_related_to(self):
        self.assertTrue(_compatible("Concept", "Concept", "RELATED_TO"))
        self.assertFalse(_compatible("Concept", "Concept", "WORKS_AT"))

    def test_compatible_helper_reverse_pair_lookup(self):
        """(Organization, Person) should check (Person, Organization)."""
        self.assertTrue(_compatible("Organization", "Person", "WORKS_AT"))

    def test_org_org_pair_present(self):
        self.assertIn(("Organization", "Organization"), ENTITY_REL_COMPATIBILITY)
        self.assertTrue(_compatible("Organization", "Organization", "COLLABORATES_WITH"))

    def test_project_project_pair_present(self):
        self.assertIn(("Project", "Project"), ENTITY_REL_COMPATIBILITY)
        self.assertTrue(_compatible("Project", "Project", "DEPENDS_ON"))


# ---------------------------------------------------------------------------
# Deterministic audit tests (mocked Neo4j)
# ---------------------------------------------------------------------------

_MOD = "memory.relationship_audit"


def _mock_rels(rels: list[dict]) -> list[dict]:
    """Fill in default fields for test relationships."""
    defaults = {
        "head": "UNNAMED", "tail": "UNNAMED",
        "head_type": "Person", "tail_type": "Organization",
        "strength": 0.5, "mention_count": 2,
        "context_snippets": [], "audit_status": None,
        "first_mentioned": "2026-01-01", "last_mentioned": "2026-02-01",
    }
    return [{**defaults, **r} for r in rels]


_GS_MOD = "memory.graph_suggestions"


def _audit_stack(rels, dist=None, self_ref_count=0, suggestions=None, hotspots=None):
    """Return an ExitStack with all graph functions patched, plus mock dict for assertions."""
    if dist is None:
        dist = {r.get("rel_type", "RELATED_TO"): 1 for r in rels}
    mocks = {
        "get_relationships_for_audit": MagicMock(return_value=rels),
        "get_relationship_type_distribution": MagicMock(return_value=dist),
        "delete_self_referencing_rels": MagicMock(return_value=self_ref_count),
        "delete_specific_relationship": MagicMock(return_value=True),
        "set_relationship_audit_status": MagicMock(),
        "reclassify_relationship": MagicMock(),
    }
    stack = ExitStack()
    for name, mock in mocks.items():
        stack.enter_context(patch(f"{_MOD}.{name}", mock))
    # graph_suggestions are imported lazily inside run_deterministic_audit,
    # so we patch them on the graph_suggestions module itself
    stack.enter_context(
        patch(f"{_GS_MOD}.get_suggestions", return_value=suggestions or [])
    )
    stack.enter_context(
        patch(f"{_GS_MOD}.get_related_to_hotspots", return_value=hotspots or [])
    )
    return stack, mocks


class TestSelfRefDetection(unittest.TestCase):
    def test_self_refs_reported(self):
        stack, mocks = _audit_stack(_mock_rels([]), self_ref_count=3)
        with stack:
            result = run_deterministic_audit()

        check = next(c for c in result["checks"] if c["name"] == "self_refs")
        self.assertEqual(check["status"], "warn")
        self.assertIn("3", check["detail"])


class TestZeroStrengthZombies(unittest.TestCase):
    def test_single_mention_zombie_deleted(self):
        rels = _mock_rels([
            {"head": "A", "tail": "B", "rel_type": "KNOWS",
             "strength": 0.005, "mention_count": 1},
        ])
        stack, mocks = _audit_stack(rels)
        with stack:
            result = run_deterministic_audit()

        mocks["delete_specific_relationship"].assert_called_once()
        check = next(c for c in result["checks"] if c["name"] == "zero_strength_zombies")
        self.assertIn("Deleted 1", check["detail"])

    def test_multi_mention_zombie_quarantined(self):
        rels = _mock_rels([
            {"head": "A", "tail": "B", "rel_type": "KNOWS",
             "strength": 0.005, "mention_count": 3},
        ])
        stack, mocks = _audit_stack(rels)
        with stack:
            result = run_deterministic_audit()

        mocks["set_relationship_audit_status"].assert_called()
        self.assertGreater(result["quarantined"], 0)

    def test_strength_above_threshold_not_touched(self):
        rels = _mock_rels([
            {"head": "A", "tail": "B", "rel_type": "KNOWS",
             "head_type": "Person", "tail_type": "Person",
             "strength": 0.5, "mention_count": 1},
        ])
        stack, mocks = _audit_stack(rels)
        with stack:
            result = run_deterministic_audit()

        mocks["delete_specific_relationship"].assert_not_called()
        self.assertEqual(result["quarantined"], 0)

    def test_strength_exactly_001_treated_as_zombie(self):
        rels = _mock_rels([
            {"head": "A", "tail": "B", "rel_type": "KNOWS",
             "head_type": "Person", "tail_type": "Person",
             "strength": 0.01, "mention_count": 1},
        ])
        stack, mocks = _audit_stack(rels)
        with stack:
            result = run_deterministic_audit()

        mocks["delete_specific_relationship"].assert_called_once()

    def test_strength_just_above_001_not_zombie(self):
        rels = _mock_rels([
            {"head": "A", "tail": "B", "rel_type": "KNOWS",
             "head_type": "Person", "tail_type": "Person",
             "strength": 0.011, "mention_count": 1},
        ])
        stack, mocks = _audit_stack(rels)
        with stack:
            result = run_deterministic_audit()

        mocks["delete_specific_relationship"].assert_not_called()

    def test_zombie_delete_failure_not_counted(self):
        rels = _mock_rels([
            {"head": "A", "tail": "B", "rel_type": "KNOWS",
             "head_type": "Person", "tail_type": "Person",
             "strength": 0.005, "mention_count": 1},
        ])
        stack, mocks = _audit_stack(rels)
        mocks["delete_specific_relationship"].return_value = False
        with stack:
            result = run_deterministic_audit()

        check = next(c for c in result["checks"] if c["name"] == "zero_strength_zombies")
        self.assertIn("Deleted 0", check["detail"])


class TestTypeMismatchDetection(unittest.TestCase):
    def test_mismatch_auto_reclassified(self):
        rels = _mock_rels([
            {"head": "Sam", "tail": "Denver", "rel_type": "WORKS_AT",
             "head_type": "Person", "tail_type": "Place",
             "strength": 0.6, "mention_count": 2},
        ])
        stack, mocks = _audit_stack(rels)
        stack.enter_context(patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", True))
        stack.enter_context(patch(f"{_MOD}._log_auto_fix"))
        with stack:
            result = run_deterministic_audit()

        mocks["reclassify_relationship"].assert_called_once()
        call_args = mocks["reclassify_relationship"].call_args
        self.assertEqual(call_args[0][3], "LOCATED_IN")
        self.assertEqual(result["auto_fixes"], 1)

    def test_mismatch_quarantined_when_no_rule(self):
        rels = _mock_rels([
            {"head": "Sam", "tail": "Python", "rel_type": "PARENT_OF",
             "head_type": "Person", "tail_type": "Technology",
             "strength": 0.4, "mention_count": 1},
        ])
        stack, mocks = _audit_stack(rels)
        with stack:
            result = run_deterministic_audit()

        mocks["set_relationship_audit_status"].assert_called()
        self.assertGreater(result["quarantined"], 0)
        self.assertEqual(len(result["flagged"]), 1)
        self.assertEqual(result["flagged"][0]["reason"], "type_mismatch")

    def test_compatible_rel_not_flagged(self):
        rels = _mock_rels([
            {"head": "Sam", "tail": "Google", "rel_type": "WORKS_AT",
             "head_type": "Person", "tail_type": "Organization"},
        ])
        stack, mocks = _audit_stack(rels)
        with stack:
            result = run_deterministic_audit()

        check = next(c for c in result["checks"] if c["name"] == "type_mismatch")
        self.assertEqual(check["status"], "pass")

    def test_missing_entity_types_skipped(self):
        rels = _mock_rels([
            {"head": "X", "tail": "Y", "rel_type": "WORKS_AT",
             "head_type": None, "tail_type": "Organization"},
        ])
        stack, mocks = _audit_stack(rels)
        with stack:
            result = run_deterministic_audit()

        check = next(c for c in result["checks"] if c["name"] == "type_mismatch")
        self.assertEqual(check["status"], "pass")

    def test_reclassify_passes_first_mentioned(self):
        """Verify first_mentioned is forwarded to reclassify_relationship."""
        rels = _mock_rels([
            {"head": "Sam", "tail": "Denver", "rel_type": "WORKS_AT",
             "head_type": "Person", "tail_type": "Place",
             "strength": 0.6, "mention_count": 2,
             "first_mentioned": "2026-01-15"},
        ])
        stack, mocks = _audit_stack(rels)
        stack.enter_context(patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", True))
        stack.enter_context(patch(f"{_MOD}._log_auto_fix"))
        with stack:
            run_deterministic_audit()

        call_args = mocks["reclassify_relationship"].call_args
        self.assertEqual(call_args[0][7], "2026-01-15")


class TestContradictionDetection(unittest.TestCase):
    def test_multi_works_at_quarantines_weaker(self):
        rels = _mock_rels([
            {"head": "Sam", "tail": "Google", "rel_type": "WORKS_AT",
             "head_type": "Person", "tail_type": "Organization", "strength": 0.9},
            {"head": "Sam", "tail": "Meta", "rel_type": "WORKS_AT",
             "head_type": "Person", "tail_type": "Organization", "strength": 0.3},
        ])
        stack, mocks = _audit_stack(rels)
        with stack:
            result = run_deterministic_audit()

        check = next(c for c in result["checks"] if c["name"] == "contradictions")
        self.assertEqual(check["status"], "warn")
        flagged_tails = [f["tail"] for f in result["flagged"] if f.get("reason") == "multi_works_at"]
        self.assertIn("Meta", flagged_tails)

    def test_symmetric_conflict_detected(self):
        rels = _mock_rels([
            {"head": "Alice", "tail": "Bob", "rel_type": "MENTORS",
             "head_type": "Person", "tail_type": "Person", "strength": 0.8},
            {"head": "Bob", "tail": "Alice", "rel_type": "MENTORS",
             "head_type": "Person", "tail_type": "Person", "strength": 0.3},
        ])
        stack, mocks = _audit_stack(rels)
        with stack:
            result = run_deterministic_audit()

        check = next(c for c in result["checks"] if c["name"] == "contradictions")
        self.assertEqual(check["status"], "warn")

    def test_conflicting_pair_detected(self):
        """A->B MENTORS and A->B MENTORED_BY should quarantine weaker."""
        rels = _mock_rels([
            {"head": "Alice", "tail": "Bob", "rel_type": "MENTORS",
             "head_type": "Person", "tail_type": "Person", "strength": 0.8},
            {"head": "Alice", "tail": "Bob", "rel_type": "MENTORED_BY",
             "head_type": "Person", "tail_type": "Person", "strength": 0.3},
        ])
        stack, mocks = _audit_stack(rels)
        with stack:
            result = run_deterministic_audit()

        check = next(c for c in result["checks"] if c["name"] == "contradictions")
        self.assertEqual(check["status"], "warn")
        pair_flagged = [f for f in result["flagged"] if f.get("reason") == "conflicting_pair"]
        self.assertEqual(len(pair_flagged), 1)

    def test_edges_b_bidirectional_detected(self):
        """A->B MENTORED_BY and B->A MENTORED_BY should quarantine weaker."""
        rels = _mock_rels([
            {"head": "Alice", "tail": "Bob", "rel_type": "MENTORED_BY",
             "head_type": "Person", "tail_type": "Person", "strength": 0.7},
            {"head": "Bob", "tail": "Alice", "rel_type": "MENTORED_BY",
             "head_type": "Person", "tail_type": "Person", "strength": 0.2},
        ])
        stack, mocks = _audit_stack(rels)
        with stack:
            result = run_deterministic_audit()

        check = next(c for c in result["checks"] if c["name"] == "contradictions")
        self.assertEqual(check["status"], "warn")
        sym_flagged = [f for f in result["flagged"] if f.get("reason") == "symmetric_conflict"]
        self.assertEqual(len(sym_flagged), 1)


class TestStaleSnapshotProtection(unittest.TestCase):
    """Verify that edges mutated by earlier checks are skipped by later checks."""

    def test_zombie_deleted_edge_not_processed_by_check3(self):
        """A zombie with a type mismatch should only be deleted, not also reclassified."""
        rels = _mock_rels([
            {"head": "Sam", "tail": "Denver", "rel_type": "WORKS_AT",
             "head_type": "Person", "tail_type": "Place",
             "strength": 0.005, "mention_count": 1},
        ])
        stack, mocks = _audit_stack(rels)
        stack.enter_context(patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", True))
        stack.enter_context(patch(f"{_MOD}._log_auto_fix"))
        with stack:
            result = run_deterministic_audit()

        # Should be deleted by Check 2 (zombie)
        mocks["delete_specific_relationship"].assert_called_once()
        # Should NOT be reclassified by Check 3 (already mutated)
        mocks["reclassify_relationship"].assert_not_called()

    def test_reclassified_edge_not_double_processed(self):
        """A reclassified edge should not be flagged again by Check 5."""
        rels = _mock_rels([
            {"head": "Sam", "tail": "Denver", "rel_type": "WORKS_AT",
             "head_type": "Person", "tail_type": "Place",
             "strength": 0.2, "mention_count": 1},
        ])
        stack, mocks = _audit_stack(rels)
        stack.enter_context(patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", True))
        stack.enter_context(patch.object(config, "REL_AUDIT_LOW_CONFIDENCE_THRESHOLD", 0.35))
        stack.enter_context(patch(f"{_MOD}._log_auto_fix"))
        with stack:
            result = run_deterministic_audit()

        # Check 3 reclassifies it
        mocks["reclassify_relationship"].assert_called_once()
        # Check 5 should NOT flag it (mutated by Check 3)
        low_conf = [f for f in result["flagged"] if f.get("reason") == "low_confidence_single"]
        self.assertEqual(len(low_conf), 0)


class TestLowConfidenceSingleMention(unittest.TestCase):
    def test_low_confidence_flagged_not_deleted(self):
        rels = _mock_rels([
            {"head": "X", "tail": "Y", "rel_type": "KNOWS",
             "head_type": "Person", "tail_type": "Person",
             "strength": 0.2, "mention_count": 1},
        ])
        stack, mocks = _audit_stack(rels)
        stack.enter_context(patch.object(config, "REL_AUDIT_LOW_CONFIDENCE_THRESHOLD", 0.35))
        with stack:
            result = run_deterministic_audit()

        check = next(c for c in result["checks"] if c["name"] == "low_confidence_single")
        self.assertEqual(check["status"], "warn")
        self.assertEqual(len(result["flagged"]), 1)
        self.assertEqual(result["flagged"][0]["reason"], "low_confidence_single")
        # Should NOT be deleted
        mocks["delete_specific_relationship"].assert_not_called()


class TestRelatedToAccumulation(unittest.TestCase):
    def test_warns_on_threshold(self):
        rels = _mock_rels([
            {"head": "Sam", "tail": f"Thing{i}", "rel_type": "RELATED_TO",
             "head_type": "Person", "tail_type": "Concept"}
            for i in range(5)
        ])
        stack, mocks = _audit_stack(rels)
        stack.enter_context(patch.object(config, "REL_AUDIT_RELATED_TO_WARN_THRESHOLD", 3))
        with stack:
            result = run_deterministic_audit()

        check = next(c for c in result["checks"] if c["name"] == "related_to_accumulation")
        self.assertEqual(check["status"], "warn")
        self.assertIn("Sam", check["detail"])


class TestNewTypeMonitoring(unittest.TestCase):
    def test_warns_on_missing_types(self):
        rels = _mock_rels([
            {"head": "A", "tail": "B", "rel_type": "KNOWS",
             "head_type": "Person", "tail_type": "Person"},
        ])
        dist = {"KNOWS": 5}  # Only one type present
        stack, mocks = _audit_stack(rels, dist)
        with stack:
            result = run_deterministic_audit()

        check = next(c for c in result["checks"] if c["name"] == "new_type_monitoring")
        self.assertEqual(check["status"], "warn")
        self.assertIn("valid rel types with 0 extractions", check["detail"])


class TestCleanGraphPassesAll(unittest.TestCase):
    def test_clean_graph_passes(self):
        rels = _mock_rels([
            {"head": "Sam", "tail": "Google", "rel_type": "WORKS_AT",
             "head_type": "Person", "tail_type": "Organization", "strength": 0.8},
        ])
        dist = {t: 1 for t in VALID_REL_TYPES}
        stack, mocks = _audit_stack(rels, dist)
        with stack:
            result = run_deterministic_audit()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["auto_fixes"], 0)
        self.assertEqual(result["quarantined"], 0)
        self.assertEqual(len(result["flagged"]), 0)


# ---------------------------------------------------------------------------
# Graph suggestions integration tests
# ---------------------------------------------------------------------------

class TestGraphSuggestionsEnrichment(unittest.TestCase):
    def test_hotspots_added_to_flagged(self):
        rels = _mock_rels([])
        hotspots = [{"head": "Alice", "tail": "ProjectX", "mentions": 5, "contexts": []}]
        stack, mocks = _audit_stack(rels, hotspots=hotspots)
        with stack:
            result = run_deterministic_audit()

        hotspot_flagged = [f for f in result["flagged"] if f.get("reason") == "related_to_hotspot"]
        self.assertEqual(len(hotspot_flagged), 1)
        self.assertEqual(hotspot_flagged[0]["head"], "Alice")

    def test_fallback_hints_enrich_hotspots(self):
        rels = _mock_rels([])
        suggestions = [
            {"type": "relationship_fallback", "head": "Alice", "tail": "ProjectX",
             "original_type": "contributes to", "confidence": 0.6},
        ]
        hotspots = [{"head": "Alice", "tail": "ProjectX", "mentions": 5, "contexts": []}]
        stack, mocks = _audit_stack(rels, suggestions=suggestions, hotspots=hotspots)
        with stack:
            result = run_deterministic_audit()

        hotspot_flagged = [f for f in result["flagged"] if f.get("reason") == "related_to_hotspot"]
        self.assertEqual(len(hotspot_flagged), 1)
        self.assertEqual(hotspot_flagged[0].get("original_type_hint"), "contributes to")

    def test_suggestions_failure_degrades_gracefully(self):
        rels = _mock_rels([])
        stack, mocks = _audit_stack(rels)
        # Override the default patches with side_effect
        stack.enter_context(patch(f"{_GS_MOD}.get_suggestions", side_effect=Exception("boom")))
        stack.enter_context(patch(f"{_GS_MOD}.get_related_to_hotspots", side_effect=Exception("boom")))
        with stack:
            result = run_deterministic_audit()

        self.assertIn("status", result)


# ---------------------------------------------------------------------------
# Auto-fix safety tests
# ---------------------------------------------------------------------------

class TestAutoFixSafety(unittest.TestCase):
    def test_reclassify_preserves_fields(self):
        rels = _mock_rels([
            {"head": "Sam", "tail": "Denver", "rel_type": "WORKS_AT",
             "head_type": "Person", "tail_type": "Place",
             "strength": 0.75, "mention_count": 4,
             "context_snippets": ["lives in Denver"],
             "first_mentioned": "2026-01-15"},
        ])
        stack, mocks = _audit_stack(rels)
        stack.enter_context(patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", True))
        stack.enter_context(patch(f"{_MOD}._log_auto_fix"))
        with stack:
            run_deterministic_audit()

        call_args = mocks["reclassify_relationship"].call_args
        # Positional: head, tail, old_type, new_type, strength, mention_count,
        #             context_snippets, first_mentioned
        self.assertEqual(call_args[0][0], "Sam")
        self.assertEqual(call_args[0][1], "Denver")
        self.assertEqual(call_args[0][2], "WORKS_AT")
        self.assertEqual(call_args[0][3], "LOCATED_IN")
        self.assertAlmostEqual(call_args[0][4], 0.75)
        self.assertEqual(call_args[0][5], 4)
        self.assertEqual(call_args[0][6], ["lives in Denver"])
        self.assertEqual(call_args[0][7], "2026-01-15")

    def test_auto_fix_disabled_quarantines_instead(self):
        rels = _mock_rels([
            {"head": "Sam", "tail": "Denver", "rel_type": "WORKS_AT",
             "head_type": "Person", "tail_type": "Place"},
        ])
        stack, mocks = _audit_stack(rels)
        stack.enter_context(patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", False))
        with stack:
            result = run_deterministic_audit()

        mocks["reclassify_relationship"].assert_not_called()
        self.assertGreater(result["quarantined"], 0)

    def test_quarantined_rels_not_deleted(self):
        """Quarantined relationships should stay in the graph, not be deleted."""
        rels = _mock_rels([
            {"head": "Sam", "tail": "Python", "rel_type": "PARENT_OF",
             "head_type": "Person", "tail_type": "Technology",
             "strength": 0.4, "mention_count": 2},
        ])
        stack, mocks = _audit_stack(rels)
        with stack:
            run_deterministic_audit()

        # Should quarantine (set_audit_status) but NOT delete
        mocks["set_relationship_audit_status"].assert_called()
        mocks["delete_specific_relationship"].assert_not_called()


# ---------------------------------------------------------------------------
# Tier 2: run_model_audit tests
# ---------------------------------------------------------------------------

def _mock_httpx_module(response_content: str):
    """Create a fake httpx module with a mock AsyncClient that returns the given content."""
    import types
    mock_httpx = types.ModuleType("httpx")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": response_content}}],
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_httpx.AsyncClient = MagicMock(return_value=mock_client)
    return mock_httpx


_SAMPLE_FLAGGED = [
    {"head": "A", "tail": "B", "rel_type": "KNOWS",
     "head_type": "Person", "tail_type": "Person",
     "strength": 0.3, "mention_count": 1,
     "context_snippets": [], "first_mentioned": "2026-01-01",
     "reason": "low_confidence_single"},
]


class TestRunModelAudit(unittest.IsolatedAsyncioTestCase):
    async def test_empty_flagged_returns_pass(self):
        result = await run_model_audit([])
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["auto_fixes"], 0)

    async def test_missing_api_key_skips(self):
        with patch.object(config, "MOONSHOT_API_KEY", ""):
            result = await run_model_audit(_SAMPLE_FLAGGED)
        self.assertEqual(result["status"], "skipped")

    async def test_correct_high_sets_verified(self):
        response_json = json.dumps([
            {"index": 1, "verdict": "correct", "confidence": "high"},
        ])
        fake_httpx = _mock_httpx_module(response_json)

        with patch.object(config, "MOONSHOT_API_KEY", "test-key"), \
             patch.object(config, "MOONSHOT_BASE_URL", "https://api.example.com"), \
             patch.object(config, "REL_AUDIT_KIMI_MODEL", "test-model"), \
             patch.object(config, "REL_AUDIT_MAX_MODEL_BATCH", 30), \
             patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", True), \
             patch(f"{_MOD}.set_relationship_audit_status") as mock_set, \
             patch.dict(sys.modules, {"httpx": fake_httpx}):
            result = await run_model_audit(_SAMPLE_FLAGGED)

        self.assertEqual(result["status"], "pass")
        mock_set.assert_called_once_with("A", "B", "KNOWS", "verified")

    async def test_reclassify_high_triggers_auto_fix(self):
        response_json = json.dumps([
            {"index": 1, "verdict": "reclassify", "suggested_type": "COLLABORATES_WITH",
             "confidence": "high"},
        ])
        fake_httpx = _mock_httpx_module(response_json)

        with patch.object(config, "MOONSHOT_API_KEY", "test-key"), \
             patch.object(config, "MOONSHOT_BASE_URL", "https://api.example.com"), \
             patch.object(config, "REL_AUDIT_KIMI_MODEL", "test-model"), \
             patch.object(config, "REL_AUDIT_MAX_MODEL_BATCH", 30), \
             patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", True), \
             patch(f"{_MOD}.reclassify_relationship") as mock_reclass, \
             patch(f"{_MOD}._log_auto_fix"), \
             patch.dict(sys.modules, {"httpx": fake_httpx}):
            result = await run_model_audit(_SAMPLE_FLAGGED)

        self.assertEqual(result["auto_fixes"], 1)
        mock_reclass.assert_called_once()

    async def test_low_confidence_quarantines(self):
        response_json = json.dumps([
            {"index": 1, "verdict": "correct", "confidence": "low"},
        ])
        fake_httpx = _mock_httpx_module(response_json)

        with patch.object(config, "MOONSHOT_API_KEY", "test-key"), \
             patch.object(config, "MOONSHOT_BASE_URL", "https://api.example.com"), \
             patch.object(config, "REL_AUDIT_KIMI_MODEL", "test-model"), \
             patch.object(config, "REL_AUDIT_MAX_MODEL_BATCH", 30), \
             patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", True), \
             patch(f"{_MOD}.set_relationship_audit_status") as mock_set, \
             patch.dict(sys.modules, {"httpx": fake_httpx}):
            result = await run_model_audit(_SAMPLE_FLAGGED)

        self.assertEqual(result["quarantined"], 1)
        mock_set.assert_called_once_with("A", "B", "KNOWS", "quarantined")

    async def test_no_json_in_response_returns_error(self):
        fake_httpx = _mock_httpx_module("No JSON here, sorry.")

        with patch.object(config, "MOONSHOT_API_KEY", "test-key"), \
             patch.object(config, "MOONSHOT_BASE_URL", "https://api.example.com"), \
             patch.object(config, "REL_AUDIT_KIMI_MODEL", "test-model"), \
             patch.object(config, "REL_AUDIT_MAX_MODEL_BATCH", 30), \
             patch.dict(sys.modules, {"httpx": fake_httpx}):
            result = await run_model_audit(_SAMPLE_FLAGGED)

        self.assertEqual(result["status"], "error")


# ---------------------------------------------------------------------------
# Orchestrator: run_relationship_audit tests
# ---------------------------------------------------------------------------

class TestRunRelationshipAudit(unittest.IsolatedAsyncioTestCase):
    async def test_model_disabled_skips_tier2(self):
        det_result = {
            "status": "pass", "flagged": [{"head": "A", "tail": "B"}],
            "auto_fixes": 0, "quarantined": 0, "stats": {},
        }
        with patch(f"{_MOD}.run_deterministic_audit", return_value=det_result), \
             patch(f"{_MOD}.run_model_audit") as mock_model:
            result = await run_relationship_audit(model_enabled=False)

        mock_model.assert_not_called()
        self.assertEqual(result["status"], "pass")

    async def test_model_enabled_with_flagged_calls_tier2(self):
        det_result = {
            "status": "warn", "flagged": [{"head": "A", "tail": "B"}],
            "auto_fixes": 1, "quarantined": 0, "stats": {},
        }
        model_result = {"status": "pass", "auto_fixes": 2, "quarantined": 1, "verdicts": []}
        with patch(f"{_MOD}.run_deterministic_audit", return_value=det_result), \
             patch(f"{_MOD}.run_model_audit", new=AsyncMock(return_value=model_result)):
            result = await run_relationship_audit(model_enabled=True)

        self.assertEqual(result["auto_fixes_applied"], 3)  # 1 + 2
        self.assertEqual(result["quarantined_count"], 1)  # 0 + 1
        self.assertIn("model: pass", result["summary"])

    async def test_model_enabled_empty_flagged_skips_tier2(self):
        det_result = {
            "status": "pass", "flagged": [],
            "auto_fixes": 0, "quarantined": 0, "stats": {},
        }
        with patch(f"{_MOD}.run_deterministic_audit", return_value=det_result), \
             patch(f"{_MOD}.run_model_audit") as mock_model:
            result = await run_relationship_audit(model_enabled=True)

        mock_model.assert_not_called()


# ---------------------------------------------------------------------------
# Source assertion tests (wiring checks)
# ---------------------------------------------------------------------------

class TestRelationshipAuditSourceAssertions(unittest.TestCase):
    """Verify integration hooks are present in source files."""

    _PROJECT_ROOT = Path(__file__).parent.parent

    def _read_source(self, *parts: str) -> str:
        return self._PROJECT_ROOT.joinpath(*parts).read_text()

    def test_relationship_audit_module_exists(self):
        content = self._read_source("memory", "relationship_audit.py")
        self.assertIn("def run_deterministic_audit", content)
        self.assertIn("def run_model_audit", content)
        self.assertIn("def run_relationship_audit", content)
        self.assertIn("ENTITY_REL_COMPATIBILITY", content)
        self.assertIn("DETERMINISTIC_RECLASSIFY", content)

    def test_maintenance_contains_relationship_audit(self):
        content = self._read_source("monitoring", "maintenance.py")
        self.assertIn("Relationship audit", content)
        self.assertIn("run_relationship_audit", content)

    def test_self_improve_has_quarantine_filter(self):
        content = self._read_source("evolution", "skills.py")
        self.assertIn("audit_status", content)
        self.assertIn("quarantined", content)

    def test_maintenance_includes_relationship_audit_step(self):
        content = self._read_source("monitoring", "maintenance.py")
        self.assertIn("Relationship audit", content)
        self.assertIn("run_relationship_audit", content)

    def test_config_has_rel_audit_constants(self):
        self.assertTrue(hasattr(config, "REL_AUDIT_MODEL_ENABLED"))
        self.assertTrue(hasattr(config, "REL_AUDIT_KIMI_MODEL"))
        self.assertTrue(hasattr(config, "REL_AUDIT_LOW_CONFIDENCE_THRESHOLD"))
        self.assertTrue(hasattr(config, "REL_AUDIT_RELATED_TO_WARN_THRESHOLD"))
        self.assertTrue(hasattr(config, "REL_AUDIT_MAX_MODEL_BATCH"))
        self.assertTrue(hasattr(config, "REL_AUDIT_AUTO_FIX_ENABLED"))

    def test_graph_py_has_audit_helpers(self):
        content = self._read_source("memory", "graph.py")
        self.assertIn("def get_relationships_for_audit", content)
        self.assertIn("def get_relationship_type_distribution", content)
        self.assertIn("def set_relationship_audit_status", content)
        self.assertIn("def reclassify_relationship", content)
        self.assertIn("def delete_specific_relationship", content)

    def test_graph_py_resets_audit_status_on_remention(self):
        content = self._read_source("memory", "graph.py")
        self.assertIn("audit_status", content)

    def test_graph_py_preserves_quarantine_on_remention(self):
        content = self._read_source("memory", "graph.py")
        self.assertIn("quarantined", content)

    def test_graph_py_surfaces_audit_status_in_context(self):
        content = self._read_source("memory", "graph.py")
        self.assertIn("[unverified]", content)

    def test_relationship_audit_uses_graph_suggestions(self):
        content = self._read_source("memory", "relationship_audit.py")
        self.assertIn("get_suggestions", content)
        self.assertIn("get_related_to_hotspots", content)
        self.assertIn("original_type_hint", content)

    def test_maintenance_run_maintenance_calls_relationship_audit(self):
        """Verify run_maintenance() invokes the relationship audit step."""
        content = self._read_source("monitoring", "maintenance.py")
        self.assertIn("_step_relationship_audit", content)
        self.assertIn("from memory.relationship_audit import run_relationship_audit", content)

    def test_kimi_model_ids_use_k25(self):
        self.assertEqual(config.CONTRACT_AUDIT_KIMI_MODEL, "kimi-k2.5")
        self.assertEqual(config.REL_AUDIT_KIMI_MODEL, "kimi-k2.5")

    def test_reclassify_uses_explicit_transaction(self):
        content = self._read_source("memory", "graph.py")
        self.assertIn("begin_transaction", content)

    def test_reclassify_accepts_first_mentioned(self):
        content = self._read_source("memory", "graph.py")
        self.assertIn("first_mentioned: str | None = None", content)

    def test_graph_preserves_verified_on_remention(self):
        """Verified status should be preserved when a relationship is re-mentioned."""
        content = self._read_source("memory", "graph.py")
        self.assertIn("'verified'", content)
        # Ensure the CASE expression preserves both quarantined AND verified
        self.assertIn("IN ['quarantined', 'verified']", content)


# ---------------------------------------------------------------------------
# Round 2 audit fix tests
# ---------------------------------------------------------------------------

class TestEqualStrengthBidirectional(unittest.TestCase):
    """Two bidirectional edges with identical strength should quarantine exactly one."""

    def test_equal_strength_quarantines_exactly_one(self):
        rels = _mock_rels([
            {"head": "Alice", "tail": "Bob", "rel_type": "MENTORS",
             "head_type": "Person", "tail_type": "Person", "strength": 0.5},
            {"head": "Bob", "tail": "Alice", "rel_type": "MENTORS",
             "head_type": "Person", "tail_type": "Person", "strength": 0.5},
        ])
        stack, mocks = _audit_stack(rels)
        with stack:
            result = run_deterministic_audit()

        sym_flagged = [f for f in result["flagged"] if f.get("reason") == "symmetric_conflict"]
        self.assertEqual(len(sym_flagged), 1, "Exactly one edge should be quarantined")
        # Deterministic tiebreaker: lexicographically-later head gets quarantined
        self.assertEqual(sym_flagged[0]["head"], "Bob")


class TestReclassifyFailureResilience(unittest.TestCase):
    """Check 3 should fall back to quarantine if reclassify_relationship raises."""

    def test_reclassify_failure_falls_back_to_quarantine(self):
        rels = _mock_rels([
            {"head": "Sam", "tail": "Denver", "rel_type": "WORKS_AT",
             "head_type": "Person", "tail_type": "Place",
             "strength": 0.6, "mention_count": 2},
        ])
        stack, mocks = _audit_stack(rels)
        mocks["reclassify_relationship"].side_effect = RuntimeError("Neo4j down")
        stack.enter_context(patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", True))
        stack.enter_context(patch(f"{_MOD}._log_auto_fix"))
        with stack:
            result = run_deterministic_audit()

        # Should NOT count as auto-fix
        self.assertEqual(result["auto_fixes"], 0)
        # Should quarantine instead
        self.assertGreater(result["quarantined"], 0)
        mocks["set_relationship_audit_status"].assert_called()
        # Should still be flagged for review
        mismatch_flagged = [f for f in result["flagged"] if f.get("reason") == "type_mismatch"]
        self.assertEqual(len(mismatch_flagged), 1)


class TestHttpxImportError(unittest.IsolatedAsyncioTestCase):
    """run_model_audit should return 'skipped' when httpx is not importable."""

    async def test_httpx_unavailable_returns_skipped(self):
        # Setting sys.modules["httpx"] = None causes `import httpx` to raise
        # ImportError per PEP 302 ("import of httpx halted; None in sys.modules").
        with patch.object(config, "MOONSHOT_API_KEY", "test-key"), \
             patch.dict(sys.modules, {"httpx": None}):
            result = await run_model_audit(_SAMPLE_FLAGGED)

        self.assertEqual(result["status"], "skipped")


class TestCheck6SkipsMutatedEdges(unittest.TestCase):
    """RELATED_TO edges deleted by Check 2 should not inflate Check 6 counts."""

    def test_zombie_related_to_not_counted_in_accumulation(self):
        # 3 RELATED_TO edges from Sam, but one is a zombie (strength=0.005, mention_count=1)
        rels = _mock_rels([
            {"head": "Sam", "tail": "Thing0", "rel_type": "RELATED_TO",
             "head_type": "Person", "tail_type": "Concept",
             "strength": 0.005, "mention_count": 1},
            {"head": "Sam", "tail": "Thing1", "rel_type": "RELATED_TO",
             "head_type": "Person", "tail_type": "Concept",
             "strength": 0.5, "mention_count": 2},
            {"head": "Sam", "tail": "Thing2", "rel_type": "RELATED_TO",
             "head_type": "Person", "tail_type": "Concept",
             "strength": 0.5, "mention_count": 2},
        ])
        stack, mocks = _audit_stack(rels)
        stack.enter_context(patch.object(config, "REL_AUDIT_RELATED_TO_WARN_THRESHOLD", 3))
        with stack:
            result = run_deterministic_audit()

        # The zombie should be deleted by Check 2, leaving only 2 RELATED_TO edges for Sam
        # With threshold=3, Sam should NOT trigger the warning
        check = next(c for c in result["checks"] if c["name"] == "related_to_accumulation")
        self.assertNotIn("Sam", check["detail"])


class TestAlreadyQuarantinedSkipsReclassify(unittest.TestCase):
    """Edges quarantined on a previous night should NOT re-attempt reclassify."""

    def test_quarantined_edge_not_reclassified(self):
        """A type-mismatch edge already quarantined should be flagged but NOT reclassified."""
        rels = _mock_rels([
            {"head": "Sam", "tail": "Denver", "rel_type": "WORKS_AT",
             "head_type": "Person", "tail_type": "Place",
             "strength": 0.6, "mention_count": 2,
             "audit_status": "quarantined"},
        ])
        stack, mocks = _audit_stack(rels)
        stack.enter_context(patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", True))
        stack.enter_context(patch(f"{_MOD}._log_auto_fix"))
        with stack:
            result = run_deterministic_audit()

        # Should NOT attempt reclassify (already quarantined from previous run)
        mocks["reclassify_relationship"].assert_not_called()
        # Should still be flagged for review
        mismatch_flagged = [f for f in result["flagged"] if f.get("reason") == "type_mismatch"]
        self.assertEqual(len(mismatch_flagged), 1)
        # Should NOT re-count toward quarantined total (already quarantined)
        self.assertEqual(result["quarantined"], 0)
        # Should NOT re-call set_relationship_audit_status (already quarantined)
        mocks["set_relationship_audit_status"].assert_not_called()


class TestCheck5SkipsVerifiedEdges(unittest.TestCase):
    """Verified edges should NOT be re-flagged as low-confidence by Check 5."""

    def test_verified_edge_not_flagged_as_low_confidence(self):
        rels = _mock_rels([
            {"head": "X", "tail": "Y", "rel_type": "KNOWS",
             "head_type": "Person", "tail_type": "Person",
             "strength": 0.2, "mention_count": 1,
             "audit_status": "verified"},
        ])
        stack, mocks = _audit_stack(rels)
        stack.enter_context(patch.object(config, "REL_AUDIT_LOW_CONFIDENCE_THRESHOLD", 0.35))
        with stack:
            result = run_deterministic_audit()

        low_conf = [f for f in result["flagged"] if f.get("reason") == "low_confidence_single"]
        self.assertEqual(len(low_conf), 0, "Verified edge should not be flagged as low-confidence")

    def test_unverified_edge_still_flagged(self):
        """Non-verified edge with same stats should still be flagged."""
        rels = _mock_rels([
            {"head": "X", "tail": "Y", "rel_type": "KNOWS",
             "head_type": "Person", "tail_type": "Person",
             "strength": 0.2, "mention_count": 1,
             "audit_status": None},
        ])
        stack, mocks = _audit_stack(rels)
        stack.enter_context(patch.object(config, "REL_AUDIT_LOW_CONFIDENCE_THRESHOLD", 0.35))
        with stack:
            result = run_deterministic_audit()

        low_conf = [f for f in result["flagged"] if f.get("reason") == "low_confidence_single"]
        self.assertEqual(len(low_conf), 1, "Unverified edge should be flagged")


class TestTier2DeleteVerdict(unittest.IsolatedAsyncioTestCase):
    """Tier 2 'delete' verdict should actually delete the edge."""

    async def test_delete_high_confidence_deletes_edge(self):
        response_json = json.dumps([
            {"index": 1, "verdict": "delete", "confidence": "high"},
        ])
        fake_httpx = _mock_httpx_module(response_json)

        with patch.object(config, "MOONSHOT_API_KEY", "test-key"), \
             patch.object(config, "MOONSHOT_BASE_URL", "https://api.example.com"), \
             patch.object(config, "REL_AUDIT_KIMI_MODEL", "test-model"), \
             patch.object(config, "REL_AUDIT_MAX_MODEL_BATCH", 30), \
             patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", True), \
             patch(f"{_MOD}.delete_specific_relationship", return_value=True) as mock_del, \
             patch(f"{_MOD}.set_relationship_audit_status") as mock_set, \
             patch.dict(sys.modules, {"httpx": fake_httpx}):
            result = await run_model_audit(_SAMPLE_FLAGGED)

        # Should delete, not quarantine
        mock_del.assert_called_once_with("A", "B", "KNOWS")
        mock_set.assert_not_called()
        self.assertEqual(result["auto_fixes"], 1)
        self.assertEqual(result["quarantined"], 0)

    async def test_delete_low_confidence_quarantines(self):
        """Delete with low confidence should fall through to quarantine."""
        response_json = json.dumps([
            {"index": 1, "verdict": "delete", "confidence": "low"},
        ])
        fake_httpx = _mock_httpx_module(response_json)

        with patch.object(config, "MOONSHOT_API_KEY", "test-key"), \
             patch.object(config, "MOONSHOT_BASE_URL", "https://api.example.com"), \
             patch.object(config, "REL_AUDIT_KIMI_MODEL", "test-model"), \
             patch.object(config, "REL_AUDIT_MAX_MODEL_BATCH", 30), \
             patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", True), \
             patch(f"{_MOD}.delete_specific_relationship") as mock_del, \
             patch(f"{_MOD}.set_relationship_audit_status") as mock_set, \
             patch.dict(sys.modules, {"httpx": fake_httpx}):
            result = await run_model_audit(_SAMPLE_FLAGGED)

        # Should NOT delete — confidence too low
        mock_del.assert_not_called()
        # Should quarantine via the else branch
        mock_set.assert_called_once_with("A", "B", "KNOWS", "quarantined")
        self.assertEqual(result["quarantined"], 1)

    async def test_delete_failure_quarantines(self):
        """Delete that raises should fall back to quarantine."""
        response_json = json.dumps([
            {"index": 1, "verdict": "delete", "confidence": "high"},
        ])
        fake_httpx = _mock_httpx_module(response_json)

        with patch.object(config, "MOONSHOT_API_KEY", "test-key"), \
             patch.object(config, "MOONSHOT_BASE_URL", "https://api.example.com"), \
             patch.object(config, "REL_AUDIT_KIMI_MODEL", "test-model"), \
             patch.object(config, "REL_AUDIT_MAX_MODEL_BATCH", 30), \
             patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", True), \
             patch(f"{_MOD}.delete_specific_relationship", side_effect=RuntimeError("Neo4j down")) as mock_del, \
             patch(f"{_MOD}.set_relationship_audit_status") as mock_set, \
             patch.dict(sys.modules, {"httpx": fake_httpx}):
            result = await run_model_audit(_SAMPLE_FLAGGED)

        # Delete was attempted but failed
        mock_del.assert_called_once()
        # Should fall back to quarantine
        mock_set.assert_called_once_with("A", "B", "KNOWS", "quarantined")
        self.assertEqual(result["auto_fixes"], 0)
        self.assertEqual(result["quarantined"], 1)


class TestTier2ReclassifyFailure(unittest.IsolatedAsyncioTestCase):
    """Tier 2 reclassify failure should fall back to quarantine."""

    async def test_tier2_reclassify_failure_quarantines(self):
        response_json = json.dumps([
            {"index": 1, "verdict": "reclassify", "suggested_type": "COLLABORATES_WITH",
             "confidence": "high"},
        ])
        fake_httpx = _mock_httpx_module(response_json)

        with patch.object(config, "MOONSHOT_API_KEY", "test-key"), \
             patch.object(config, "MOONSHOT_BASE_URL", "https://api.example.com"), \
             patch.object(config, "REL_AUDIT_KIMI_MODEL", "test-model"), \
             patch.object(config, "REL_AUDIT_MAX_MODEL_BATCH", 30), \
             patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", True), \
             patch(f"{_MOD}.reclassify_relationship", side_effect=RuntimeError("Neo4j down")) as mock_reclass, \
             patch(f"{_MOD}._log_auto_fix"), \
             patch(f"{_MOD}.set_relationship_audit_status") as mock_set, \
             patch.dict(sys.modules, {"httpx": fake_httpx}):
            result = await run_model_audit(_SAMPLE_FLAGGED)

        # Reclassify was attempted but failed
        mock_reclass.assert_called_once()
        # Should NOT count as auto-fix
        self.assertEqual(result["auto_fixes"], 0)
        # Should fall back to quarantine
        self.assertEqual(result["quarantined"], 1)
        mock_set.assert_called_once_with("A", "B", "KNOWS", "quarantined")


class TestTier2DeleteReturnValue(unittest.IsolatedAsyncioTestCase):
    """Tier 2 delete should NOT count as auto_fix when delete returns False."""

    async def test_delete_returns_false_not_counted(self):
        response_json = json.dumps([
            {"index": 1, "verdict": "delete", "confidence": "high"},
        ])
        fake_httpx = _mock_httpx_module(response_json)

        with patch.object(config, "MOONSHOT_API_KEY", "test-key"), \
             patch.object(config, "MOONSHOT_BASE_URL", "https://api.example.com"), \
             patch.object(config, "REL_AUDIT_KIMI_MODEL", "test-model"), \
             patch.object(config, "REL_AUDIT_MAX_MODEL_BATCH", 30), \
             patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", True), \
             patch(f"{_MOD}.delete_specific_relationship", return_value=False) as mock_del, \
             patch(f"{_MOD}.set_relationship_audit_status") as mock_set, \
             patch.dict(sys.modules, {"httpx": fake_httpx}):
            result = await run_model_audit(_SAMPLE_FLAGGED)

        # Delete was attempted but returned False (edge not found)
        mock_del.assert_called_once()
        # Should NOT count as auto_fix
        self.assertEqual(result["auto_fixes"], 0)
        # Should fall through to quarantine
        mock_set.assert_called_once_with("A", "B", "KNOWS", "quarantined")
        self.assertEqual(result["quarantined"], 1)


class TestTier2ReclassifyAutoFixDisabled(unittest.IsolatedAsyncioTestCase):
    """Tier 2 reclassify with auto_fix disabled should quarantine."""

    async def test_reclassify_high_auto_fix_disabled_quarantines(self):
        response_json = json.dumps([
            {"index": 1, "verdict": "reclassify", "suggested_type": "COLLABORATES_WITH",
             "confidence": "high"},
        ])
        fake_httpx = _mock_httpx_module(response_json)

        with patch.object(config, "MOONSHOT_API_KEY", "test-key"), \
             patch.object(config, "MOONSHOT_BASE_URL", "https://api.example.com"), \
             patch.object(config, "REL_AUDIT_KIMI_MODEL", "test-model"), \
             patch.object(config, "REL_AUDIT_MAX_MODEL_BATCH", 30), \
             patch.object(config, "REL_AUDIT_AUTO_FIX_ENABLED", False), \
             patch(f"{_MOD}.reclassify_relationship") as mock_reclass, \
             patch(f"{_MOD}.set_relationship_audit_status") as mock_set, \
             patch.dict(sys.modules, {"httpx": fake_httpx}):
            result = await run_model_audit(_SAMPLE_FLAGGED)

        mock_reclass.assert_not_called()
        mock_set.assert_called_once_with("A", "B", "KNOWS", "quarantined")
        self.assertEqual(result["quarantined"], 1)
        self.assertEqual(result["auto_fixes"], 0)


class TestSelfRefTrackedInMutated(unittest.TestCase):
    """Self-referencing edges deleted by Check 1 should be tracked in _mutated."""

    def test_self_ref_not_processed_by_later_checks(self):
        """A self-ref edge should not be processed by Check 2 (zombie) after Check 1 deletes it."""
        rels = _mock_rels([
            {"head": "Sam", "tail": "Sam", "rel_type": "KNOWS",
             "head_type": "Person", "tail_type": "Person",
             "strength": 0.005, "mention_count": 1},
        ])
        stack, mocks = _audit_stack(rels, self_ref_count=1)
        with stack:
            result = run_deterministic_audit()

        # Check 1 reports the deletion
        check1 = next(c for c in result["checks"] if c["name"] == "self_refs")
        self.assertEqual(check1["status"], "warn")
        # Check 2 should NOT try to delete it again (already in _mutated)
        mocks["delete_specific_relationship"].assert_not_called()


class TestReclassifyUsesMerge(unittest.TestCase):
    """Verify reclassify_relationship uses MERGE to prevent duplicate edges."""

    _PROJECT_ROOT = Path(__file__).parent.parent

    def test_reclassify_uses_merge_not_create(self):
        content = self._PROJECT_ROOT.joinpath("memory", "graph.py").read_text()
        # Find the reclassify_relationship function body
        start = content.index("def reclassify_relationship")
        # Find the next function definition or end of file
        next_def = content.find("\ndef ", start + 1)
        body = content[start:next_def] if next_def > 0 else content[start:]
        self.assertIn("MERGE (h)-[r:", body, "reclassify should use MERGE, not CREATE")
        self.assertNotIn("CREATE (h)-[r:", body, "reclassify should NOT use CREATE")


if __name__ == "__main__":
    unittest.main()
