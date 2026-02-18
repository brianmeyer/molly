import asyncio
import json
import logging
import threading
import time
import config
from memory.embeddings import embed
from memory.vectorstore import VectorStore

log = logging.getLogger(__name__)

_vectorstore: VectorStore | None = None
_vectorstore_lock = threading.Lock()


def get_vectorstore() -> VectorStore:
    """Lazy-initialize the shared VectorStore singleton (thread-safe)."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    with _vectorstore_lock:
        if _vectorstore is None:
            vs = VectorStore(config.MOLLYGRAPH_PATH)
            vs.initialize()
            _vectorstore = vs
    return _vectorstore


async def retrieve_context(message: str, top_k: int = 5) -> str:
    """Retrieve relevant memory context for a message.

    Combines Layer 2 (semantic search) and Layer 3 (knowledge graph).
    Runs both retrievals concurrently in separate executor threads.
    Returns a formatted string to inject into the system prompt.
    Returns empty string if no relevant memories found.
    """
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()
    semantic_future = asyncio.wait_for(loop.run_in_executor(None, _retrieve_semantic, message, top_k), timeout=15.0)
    graph_future = asyncio.wait_for(loop.run_in_executor(None, _retrieve_graph, message), timeout=15.0)
    semantic_bundle, graph_result = await asyncio.gather(semantic_future, graph_future, return_exceptions=True)
    # Graceful degradation on timeout
    if isinstance(semantic_bundle, Exception):
        log.warning("Semantic retrieval failed: %s", semantic_bundle)
        semantic_bundle = {"context": "", "result_count": 0, "distances": [], "sources": []}
    if isinstance(graph_result, Exception):
        log.warning("Graph retrieval failed: %s", graph_result)
        graph_result = ""

    semantic_result = semantic_bundle["context"]
    result_count = semantic_bundle["result_count"]
    distances = semantic_bundle["distances"]
    sources = semantic_bundle["sources"]

    # Combine non-empty sections
    sections = [semantic_result, graph_result]
    combined = "\n\n".join(s for s in sections if s)
    latency_ms = int((time.monotonic() - t0) * 1000)
    if combined:
        log.debug("Retrieved memory context: %d chars in %dms", len(combined), latency_ms)

    # Log retrieval stats for observability
    _log_retrieval_stats(message, top_k, result_count, distances, sources, latency_ms)

    return combined


def _retrieve_semantic(message: str, top_k: int = 5) -> dict:
    """Layer 2+: Hybrid search combining vector similarity + BM25 keywords.

    Uses hybrid_search() (Phase 5B) when available, falling back to
    pure vector search if FTS5 is not yet set up.  Returns bundle with
    context + stats.
    """
    vs = get_vectorstore()
    empty = {"context": "", "result_count": 0, "distances": [], "sources": []}

    if vs.chunk_count() == 0:
        return empty

    try:
        query_vec = embed(message)
    except Exception:
        log.error("Embedding failed for retrieval", exc_info=True)
        return empty

    # Phase 5B: prefer hybrid search (vector + BM25), fall back to vector-only
    try:
        results = vs.hybrid_search(
            query_text=message,
            query_embedding=query_vec,
            top_k=top_k,
            vector_weight=0.7,
            bm25_weight=0.3,
        )
    except Exception:
        log.warning("Hybrid search unavailable, falling back to vector-only", exc_info=True)
        try:
            results = vs.search(query_vec, top_k=top_k)
        except Exception:
            log.error("Semantic memory retrieval failed", exc_info=True)
            return empty

    if not results:
        return empty

    # Filter by similarity threshold (cosine distance: similarity = 1 - distance).
    # Drop chunks below 0.35 cosine similarity — they add noise, not context.
    MIN_SIMILARITY = 0.35
    lines = ["<!-- Memory Context (hybrid search: semantic + BM25) -->"]
    lines.append("Relevant past conversations (most relevant first):\n")
    distances = []
    sources = []
    filtered_count = 0
    for r in results:
        dist = float(r.get("distance", 1.0))
        similarity = max(0.0, 1.0 - dist)
        if similarity < MIN_SIMILARITY:
            filtered_count += 1
            continue
        created = r["created_at"][:10]
        source = r["source"]
        lines.append(f"[{created}, {source}] {r['content']}")
        distances.append(dist)
        sources.append(source)

    if filtered_count:
        log.debug("Filtered %d/%d chunks below %.2f similarity threshold",
                   filtered_count, len(results), MIN_SIMILARITY)

    if len(distances) == 0:
        return empty

    context = "\n".join(lines)
    log.debug("Retrieved %d memory chunks (hybrid, %d filtered)", len(distances), filtered_count)
    return {
        "context": context,
        "result_count": len(distances),
        "distances": distances,
        "sources": sources,
    }


def _retrieve_graph(message: str) -> str:
    """Layer 3: Extract entities from message, query Neo4j for graph context."""
    try:
        from memory.extractor import extract_entities
        from memory.graph import query_entities_for_context

        # Quick entity extraction from the incoming message
        entities = extract_entities(message, threshold=0.3)

        if not entities:
            return ""

        entity_names = [e["text"] for e in entities]
        return query_entities_for_context(entity_names)

    except Exception:
        log.error("Graph memory retrieval failed", exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# Retrieval stats logging (for monitoring/agents/retrieval_quality.py)
# ---------------------------------------------------------------------------

_retrieval_stats_lock = threading.Lock()
_retrieval_stats_table_ensured = False


def _log_retrieval_stats(
    query_text: str,
    top_k: int,
    result_count: int,
    distances: list[float],
    sources: list[str],
    latency_ms: int,
) -> None:
    """Best-effort log of retrieval metrics to retrieval_stats table."""
    global _retrieval_stats_table_ensured
    conn = None
    try:
        import db_pool

        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        conn.execute("PRAGMA busy_timeout=5000")
        with _retrieval_stats_lock:
            if not _retrieval_stats_table_ensured:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS retrieval_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_text TEXT,
                        top_k INTEGER,
                        result_count INTEGER,
                        avg_similarity REAL,
                        max_similarity REAL,
                        latency_ms INTEGER,
                        sources TEXT,
                        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                    )"""
                )
                _retrieval_stats_table_ensured = True

        # Cosine distance: similarity = 1 - distance (range 0..1 for normalized vecs)
        similarities = [max(0.0, 1.0 - d) for d in distances] if distances else []
        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
        max_sim = max(similarities) if similarities else 0.0

        unique_sources = sorted(set(sources))

        conn.execute(
            """INSERT INTO retrieval_stats
               (query_text, top_k, result_count, avg_similarity, max_similarity, latency_ms, sources)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                query_text[:200],  # Truncate long queries
                top_k,
                result_count,
                round(avg_sim, 4),
                round(max_sim, 4),
                latency_ms,
                json.dumps(unique_sources),
            ),
        )
        conn.commit()
    except Exception:
        log.debug("Failed to log retrieval stats", exc_info=True)
    finally:
        if conn is not None:
            conn.close()
