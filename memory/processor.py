import asyncio
import logging
import re
from datetime import date

from memory.embeddings import embed, embed_batch
from memory.retriever import get_vectorstore

import config

log = logging.getLogger(__name__)

# Entities extracted from system prompts, pronouns, and generic noise.
# These should never be stored in the knowledge graph.
_ENTITY_BLOCKLIST = {
    # System artifacts
    "molly", "user", "heartbeat", "brian's approval", "approval",
    "context", "system", "assistant", "claude", "opus", "haiku", "sonnet",
    # Pronouns / generic words that GLiNER2 over-extracts
    "him", "her", "his", "them", "it", "its", "they", "we", "you", "i",
    "me", "my", "mine", "your", "yours", "he", "she",
    # Common noise
    "someone", "something", "nothing", "everything", "anyone",
}

# Minimum entity name length (single chars are always noise)
_MIN_ENTITY_LEN = 2


def _filter_entities(entities: list[dict]) -> list[dict]:
    """Remove noise entities that come from system prompts or are too generic."""
    filtered = []
    for ent in entities:
        name = ent.get("text", "").strip()
        if not name or len(name) < _MIN_ENTITY_LEN:
            continue
        if name.lower() in _ENTITY_BLOCKLIST:
            continue
        # Skip entities that contain newlines (parsing artifacts like "Rodrigo\nMolly")
        if "\n" in name:
            continue
        filtered.append(ent)
    return filtered


def _filter_relations(relations: list[dict]) -> list[dict]:
    """Remove relations involving blocked entities or self-references."""
    filtered = []
    for rel in relations:
        head = rel.get("head", "").strip()
        tail = rel.get("tail", "").strip()
        if not head or not tail:
            continue
        if head.lower() in _ENTITY_BLOCKLIST or tail.lower() in _ENTITY_BLOCKLIST:
            continue
        # Skip self-referencing relationships
        if head.lower() == tail.lower():
            continue
        filtered.append(rel)
    return filtered


async def embed_and_store(
    content: str,
    chat_jid: str,
    source: str = "whatsapp",
):
    """L2: Embed content and store in vectorstore.

    Used for both passive processing (all messages) and
    active conversations (combined user+assistant chunks).
    """
    try:
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(None, embed, content)
        vs = get_vectorstore()
        chunk_id = vs.store_chunk(
            content=content,
            embedding=vec,
            source=source,
            chat_jid=chat_jid,
        )
        log.debug("Stored chunk %s (%d chars)", chunk_id, len(content))
    except Exception:
        log.error("Embed/store failed", exc_info=True)


async def batch_embed_and_store(
    texts: list[str],
    chat_jid: str,
    source: str = "email",
) -> int:
    """L2: Batch embed and store multiple texts in a single transaction.

    Uses embed_batch() for a single model.encode() call instead of N individual
    calls. Returns count of chunks stored.
    """
    if not texts:
        return 0
    try:
        loop = asyncio.get_event_loop()
        vecs = await loop.run_in_executor(None, embed_batch, texts)
        vs = get_vectorstore()
        chunks = [
            {"content": text, "embedding": vec, "source": source, "chat_jid": chat_jid}
            for text, vec in zip(texts, vecs)
        ]
        chunk_ids = vs.store_chunks_batch(chunks)
        log.debug("Batch embedded+stored %d chunks", len(chunk_ids))
        return len(chunk_ids)
    except Exception:
        log.error("Batch embed/store failed", exc_info=True)
        return 0


async def extract_to_graph(
    content: str,
    chat_jid: str,
    source: str = "whatsapp",
):
    """L3: Full entity/relation extraction and Neo4j upsert."""
    try:
        from memory.extractor import extract
        from memory import graph

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, extract, content)
        entities = _filter_entities(result["entities"])
        relations = _filter_relations(result["relations"])

        if not entities:
            return

        # Build raw→canonical name mapping so relationships use the right names
        raw_to_canonical: dict[str, str] = {}
        entity_names = []
        for ent in entities:
            canonical = graph.upsert_entity(
                name=ent["text"],
                entity_type=ent["label"],
                confidence=ent["score"],
            )
            raw_to_canonical[ent["text"]] = canonical
            entity_names.append(canonical)

        for rel in relations:
            # Resolve raw extracted names to canonical graph names
            head = raw_to_canonical.get(rel["head"], rel["head"])
            tail = raw_to_canonical.get(rel["tail"], rel["tail"])
            graph.upsert_relationship(
                head_name=head,
                tail_name=tail,
                rel_type=rel["label"],
                confidence=rel["score"],
                context_snippet=content[:200],
            )

        graph.create_episode(
            content_preview=content,
            source=source,
            entity_names=list(set(entity_names)),
        )

        log.debug(
            "Graph updated: %d entities, %d relations (%dms)",
            len(entities), len(relations), result["latency_ms"],
        )
    except Exception:
        log.error("Graph extraction failed", exc_info=True)


async def process_conversation(
    user_msg: str,
    assistant_msg: str,
    chat_jid: str,
    source: str = "whatsapp",
):
    """Post-response processing: embed combined chunk, extract entities, update graph.

    Runs as an async task — non-blocking to the response path.
    Called from agent.py after Molly responds.
    """
    chunk_text = f"User: {user_msg}\nMolly: {assistant_msg}"
    await embed_and_store(chunk_text, chat_jid, source)
    await extract_to_graph(chunk_text, chat_jid, source)
    _append_daily_log(user_msg, assistant_msg, chat_jid)


def _append_daily_log(user_msg: str, assistant_msg: str, chat_jid: str):
    """Append a brief conversation summary to today's daily log.

    Writes to workspace/memory/YYYY-MM-DD.md so the identity stack
    picks it up on subsequent turns.
    """
    try:
        today = date.today().isoformat()
        log_path = config.WORKSPACE / "memory" / f"{today}.md"

        # Truncate for the log entry
        user_preview = user_msg[:150].replace("\n", " ")
        molly_preview = assistant_msg[:150].replace("\n", " ")
        chat_short = chat_jid.split("@")[0] if "@" in chat_jid else chat_jid

        entry = f"- [{chat_short}] **User:** {user_preview}"
        if len(user_msg) > 150:
            entry += "..."
        entry += f"\n  **Molly:** {molly_preview}"
        if len(assistant_msg) > 150:
            entry += "..."
        entry += "\n"

        if not log_path.exists():
            log_path.write_text(f"# Daily Log — {today}\n\n{entry}\n")
        else:
            with open(log_path, "a") as f:
                f.write(f"{entry}\n")

        log.debug("Appended to daily log: %s", log_path.name)
    except Exception:
        log.debug("Failed to write daily log", exc_info=True)
