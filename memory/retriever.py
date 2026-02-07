import logging
from datetime import datetime

import config
from memory.embeddings import embed
from memory.vectorstore import VectorStore

log = logging.getLogger(__name__)

_vectorstore: VectorStore | None = None


def get_vectorstore() -> VectorStore:
    """Lazy-initialize the shared VectorStore singleton."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = VectorStore(config.MOLLYGRAPH_PATH)
        _vectorstore.initialize()
    return _vectorstore


def retrieve_context(message: str, top_k: int = 5) -> str:
    """Retrieve relevant memory context for a message.

    Combines Layer 2 (semantic search) and Layer 3 (knowledge graph).
    Returns a formatted string to inject into the system prompt.
    Returns empty string if no relevant memories found.
    """
    sections = []

    # Layer 2: Semantic search
    sections.append(_retrieve_semantic(message, top_k))

    # Layer 3: Knowledge graph
    sections.append(_retrieve_graph(message))

    # Combine non-empty sections
    combined = "\n\n".join(s for s in sections if s)
    if combined:
        log.debug("Retrieved memory context: %d chars", len(combined))
    return combined


def _retrieve_semantic(message: str, top_k: int = 5) -> str:
    """Layer 2: sqlite-vec semantic search."""
    vs = get_vectorstore()

    if vs.chunk_count() == 0:
        return ""

    try:
        query_vec = embed(message)
        results = vs.search(query_vec, top_k=top_k)
    except Exception:
        log.error("Semantic memory retrieval failed", exc_info=True)
        return ""

    if not results:
        return ""

    lines = ["<!-- Memory Context (semantic search) -->"]
    lines.append("Relevant past conversations (most similar first):\n")
    for r in results:
        created = r["created_at"][:10]
        source = r["source"]
        lines.append(f"[{created}, {source}] {r['content']}")

    context = "\n".join(lines)
    log.debug("Retrieved %d semantic memory chunks", len(results))
    return context


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
