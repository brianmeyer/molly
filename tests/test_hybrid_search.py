"""Tests for Phase 5B hybrid search (FTS5 + vector).

Tests cover:
  - FTS5 table creation and backfill
  - FTS5 keyword search (exact match, partial match, empty)
  - Hybrid search score fusion (vector + BM25)
  - FTS5 kept in sync during store_chunk and store_chunks_batch
  - Edge cases (empty DB, no FTS results, no vector results)
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from memory.vectorstore import VectorStore, EMBEDDING_DIM


def _dummy_vec(seed: float = 0.0) -> list[float]:
    """Create a dummy embedding with a seed value for differentiation."""
    return [seed] * EMBEDDING_DIM


def _make_store(tmp_path: Path) -> VectorStore:
    """Create and initialize a fresh VectorStore at tmp_path."""
    vs = VectorStore(tmp_path)
    vs.initialize()
    return vs


class TestFTS5TableCreation(unittest.TestCase):
    """Test FTS5 virtual table is created on init."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)
        self.vs = _make_store(self.db_path)

    def tearDown(self):
        self.vs.close()
        os.unlink(self.db_path)

    def test_fts5_table_exists(self):
        """FTS5 table should be created during initialize()."""
        tables = self.vs.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchall()
        self.assertEqual(len(tables), 1)

    def test_fts5_empty_initially(self):
        count = self.vs.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        self.assertEqual(count, 0)


class TestFTS5Sync(unittest.TestCase):
    """Test FTS5 stays in sync with conversation_chunks."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)
        self.vs = _make_store(self.db_path)

    def tearDown(self):
        self.vs.close()
        os.unlink(self.db_path)

    def test_store_chunk_updates_fts(self):
        """store_chunk should insert into FTS5."""
        self.vs.store_chunk("Test User lives in City State", _dummy_vec(), source="whatsapp")
        fts_count = self.vs.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        self.assertEqual(fts_count, 1)

    def test_store_chunks_batch_updates_fts(self):
        """store_chunks_batch should insert all into FTS5."""
        chunks = [
            {"content": "Meeting at 3pm", "embedding": _dummy_vec(), "source": "whatsapp", "chat_jid": "x"},
            {"content": "Dinner at 7pm", "embedding": _dummy_vec(), "source": "email", "chat_jid": "y"},
            {"content": "Project deadline", "embedding": _dummy_vec(), "source": "email", "chat_jid": "z"},
        ]
        self.vs.store_chunks_batch(chunks)
        fts_count = self.vs.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        self.assertEqual(fts_count, 3)

    def test_fts_matches_chunks(self):
        """FTS row count should equal conversation_chunks count."""
        for i in range(5):
            self.vs.store_chunk(f"Test content {i}", _dummy_vec(), source="test")
        chunk_count = self.vs.chunk_count()
        fts_count = self.vs.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        self.assertEqual(fts_count, chunk_count)


class TestFTS5Backfill(unittest.TestCase):
    """Test FTS5 backfill from existing data."""

    def test_backfill_on_init(self):
        """If chunks exist but FTS is empty, init should backfill."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_path = Path(tmp.name)

        # First: create a store, add data, then drop FTS
        vs = _make_store(db_path)
        vs.store_chunk("Existing chunk one", _dummy_vec(), source="test")
        vs.store_chunk("Existing chunk two", _dummy_vec(), source="test")
        # Manually drop FTS to simulate pre-5B database
        vs.conn.execute("DROP TABLE chunks_fts")
        vs.conn.commit()
        vs.close()

        # Second: reinitialize — should detect missing FTS and rebuild
        vs2 = _make_store(db_path)
        fts_count = vs2.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        self.assertEqual(fts_count, 2, "FTS5 should have been backfilled")
        vs2.close()
        os.unlink(db_path)


class TestFTS5Search(unittest.TestCase):
    """Test BM25 keyword search."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)
        self.vs = _make_store(self.db_path)

        # Populate test data
        self.vs.store_chunk("Test User lives in City State", _dummy_vec(0.1), source="whatsapp")
        self.vs.store_chunk("Meeting with John at 3pm about the Q4 report", _dummy_vec(0.2), source="email")
        self.vs.store_chunk("Dinner reservation at Canlis for Saturday", _dummy_vec(0.3), source="imessage")
        self.vs.store_chunk("Project deadline moved to March 15", _dummy_vec(0.4), source="email")
        self.vs.store_chunk("Brian's calendar has a dentist appointment", _dummy_vec(0.5), source="whatsapp")

    def tearDown(self):
        self.vs.close()
        os.unlink(self.db_path)

    def test_exact_name_match(self):
        """Searching 'Brian Meyer' should find the exact content."""
        results = self.vs.search_fts("Brian Meyer", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("City State", results[0]["content"])

    def test_keyword_match(self):
        """Keyword search for 'meeting' should find relevant content."""
        results = self.vs.search_fts("meeting", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        found_meeting = any("Meeting" in r["content"] or "meeting" in r["content"] for r in results)
        self.assertTrue(found_meeting)

    def test_no_results(self):
        """Searching for non-existent term returns empty."""
        results = self.vs.search_fts("xyznonexistent", top_k=3)
        self.assertEqual(len(results), 0)

    def test_empty_query(self):
        results = self.vs.search_fts("", top_k=3)
        self.assertEqual(len(results), 0)

    def test_results_have_required_fields(self):
        results = self.vs.search_fts("Brian", top_k=3)
        self.assertGreater(len(results), 0)
        r = results[0]
        self.assertIn("id", r)
        self.assertIn("content", r)
        self.assertIn("source", r)
        self.assertIn("bm25_rank", r)
        self.assertIn("created_at", r)


class TestHybridSearch(unittest.TestCase):
    """Test hybrid search combining vector + BM25."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)
        self.vs = _make_store(self.db_path)

        # Use uniform dummy vectors (0-vectors) so all vector distances are equal
        self.vs.store_chunk("Test User lives in City State", _dummy_vec(), source="whatsapp")
        self.vs.store_chunk("Meeting with John at 3pm about the Q4 report", _dummy_vec(), source="email")
        self.vs.store_chunk("Dinner reservation at Canlis for Saturday", _dummy_vec(), source="imessage")
        self.vs.store_chunk("Project deadline moved to March 15", _dummy_vec(), source="email")
        self.vs.store_chunk("Brian's calendar has a dentist appointment", _dummy_vec(), source="whatsapp")

    def tearDown(self):
        self.vs.close()
        os.unlink(self.db_path)

    def test_hybrid_returns_results(self):
        results = self.vs.hybrid_search(
            query_text="meeting",
            query_embedding=_dummy_vec(),
            top_k=3,
        )
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)

    def test_hybrid_has_scores(self):
        results = self.vs.hybrid_search(
            query_text="Brian",
            query_embedding=_dummy_vec(),
            top_k=3,
        )
        self.assertGreater(len(results), 0)
        r = results[0]
        self.assertIn("vector_score", r)
        self.assertIn("bm25_score", r)
        self.assertIn("hybrid_score", r)
        self.assertGreaterEqual(r["hybrid_score"], 0.0)
        self.assertLessEqual(r["hybrid_score"], 1.0)

    def test_keyword_boost(self):
        """When vector scores are equal, BM25 should differentiate.

        With uniform vectors (all 0-vectors), vector similarity is equal
        for all chunks. BM25 should boost keyword-matching results.
        """
        results = self.vs.hybrid_search(
            query_text="Brian Meyer",
            query_embedding=_dummy_vec(),
            top_k=5,
        )
        # Results with BM25 match should have higher hybrid_score
        bm25_matched = [r for r in results if r["bm25_score"] > 0]
        bm25_unmatched = [r for r in results if r["bm25_score"] == 0]

        if bm25_matched and bm25_unmatched:
            max_matched = max(r["hybrid_score"] for r in bm25_matched)
            max_unmatched = max(r["hybrid_score"] for r in bm25_unmatched)
            self.assertGreater(max_matched, max_unmatched)

    def test_custom_weights(self):
        results_default = self.vs.hybrid_search(
            query_text="meeting",
            query_embedding=_dummy_vec(),
            top_k=3,
            vector_weight=0.7,
            bm25_weight=0.3,
        )
        results_keyword_heavy = self.vs.hybrid_search(
            query_text="meeting",
            query_embedding=_dummy_vec(),
            top_k=3,
            vector_weight=0.3,
            bm25_weight=0.7,
        )
        # Both should return results
        self.assertGreater(len(results_default), 0)
        self.assertGreater(len(results_keyword_heavy), 0)

    def test_empty_db_returns_empty(self):
        """Hybrid search on empty DB should return empty list."""
        tmp2 = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp2.close()
        vs2 = _make_store(Path(tmp2.name))
        results = vs2.hybrid_search(
            query_text="test",
            query_embedding=_dummy_vec(),
            top_k=3,
        )
        self.assertEqual(len(results), 0)
        vs2.close()
        os.unlink(tmp2.name)


class TestHybridSearchScoring(unittest.TestCase):
    """Test hybrid search score normalization and fusion."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)
        self.vs = _make_store(self.db_path)

    def tearDown(self):
        self.vs.close()
        os.unlink(self.db_path)

    def test_scores_normalized_0_1(self):
        """All scores should be in [0.0, 1.0]."""
        self.vs.store_chunk("Alpha bravo charlie", _dummy_vec(0.1))
        self.vs.store_chunk("Delta echo foxtrot", _dummy_vec(0.2))
        results = self.vs.hybrid_search(
            query_text="alpha",
            query_embedding=_dummy_vec(0.15),
            top_k=5,
        )
        for r in results:
            self.assertGreaterEqual(r["vector_score"], 0.0)
            self.assertLessEqual(r["vector_score"], 1.0)
            self.assertGreaterEqual(r["bm25_score"], 0.0)
            self.assertLessEqual(r["bm25_score"], 1.0)
            self.assertGreaterEqual(r["hybrid_score"], 0.0)
            self.assertLessEqual(r["hybrid_score"], 1.0)

    def test_fusion_formula(self):
        """Verify hybrid_score = vector_weight * vector_score + bm25_weight * bm25_score."""
        self.vs.store_chunk("Test fusion formula check", _dummy_vec())
        results = self.vs.hybrid_search(
            query_text="fusion",
            query_embedding=_dummy_vec(),
            top_k=1,
            vector_weight=0.6,
            bm25_weight=0.4,
        )
        if results:
            r = results[0]
            expected = 0.6 * r["vector_score"] + 0.4 * r["bm25_score"]
            self.assertAlmostEqual(r["hybrid_score"], expected, places=4)


if __name__ == "__main__":
    unittest.main()
