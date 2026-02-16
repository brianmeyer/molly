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
    semantic_future = loop.run_in_executor(None, _retrieve_semantic, message, top_k)
    graph_future = loop.run_in_executor(None, _retrieve_graph, message)
    semantic_bundle, graph_result = await asyncio.gather(semantic_future, graph_future)

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
    """Layer 2: sqlite-vec semantic search. Returns bundle with context + stats."""
    vs = get_vectorstore()
    empty = {"context": "", "result_count": 0, "distances": [], "sources": []}

    if vs.chunk_count() == 0:
        return empty

    try:
        query_vec = embed(message)
        results = vs.search(query_vec, top_k=top_k)
    except Exception:
        log.error("Semantic memory retrieval failed", exc_info=True)
        return empty

    if not results:
        return empty

    lines = ["<!-- Memory Context (semantic search) -->"]
    lines.append("Relevant past conversations (most similar first):\n")
    distances = []
    sources = []
    for r in results:
        created = r["created_at"][:10]
        source = r["source"]
        lines.append(f"[{created}, {source}] {r['content']}")
        distances.append(float(r.get("distance", 0.0)))
        sources.append(source)

    context = "\n".join(lines)
    log.debug("Retrieved %d semantic memory chunks", len(results))
    return {
        "context": context,
        "result_count": len(results),
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

def _log_retrieval_stats(
    query_text: str,
    top_k: int,
    result_count: int,
    distances: list[float],
    sources: list[str],
    latency_ms: int,
) -> None:
    """Best-effort log of retrieval metrics to retrieval_stats table."""
    conn = None
    try:
        import db_pool

        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        if not _log_retrieval_stats._table_ensured:
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
            _log_retrieval_stats._table_ensured = True

        # Convert distances to similarity scores (1 - distance for cosine)
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


_log_retrieval_stats._table_ensured = False  # type: ignore[attr-defined]
