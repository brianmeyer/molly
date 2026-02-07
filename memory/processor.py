import logging

from memory.embeddings import embed
from memory.retriever import get_vectorstore

log = logging.getLogger(__name__)


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
        vec = embed(content)
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


async def extract_to_graph(
    content: str,
    chat_jid: str,
    source: str = "whatsapp",
):
    """L3: Full entity/relation extraction and Neo4j upsert."""
    try:
        from memory.extractor import extract
        from memory import graph

        result = extract(content)
        entities = result["entities"]
        relations = result["relations"]

        if not entities:
            return

        entity_names = []
        for ent in entities:
            canonical = graph.upsert_entity(
                name=ent["text"],
                entity_type=ent["label"],
                confidence=ent["score"],
            )
            entity_names.append(canonical)

        for rel in relations:
            graph.upsert_relationship(
                head_name=rel["head"],
                tail_name=rel["tail"],
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
