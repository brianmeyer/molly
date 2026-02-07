import logging

from memory.embeddings import embed
from memory.retriever import get_vectorstore

log = logging.getLogger(__name__)


async def process_conversation(
    user_msg: str,
    assistant_msg: str,
    chat_jid: str,
    source: str = "whatsapp",
):
    """Post-response processing: embed, store, extract entities, update graph.

    Runs as an async task — non-blocking to the response path.
    """
    try:
        # Combine user + assistant into one chunk for context
        chunk_text = f"User: {user_msg}\nMolly: {assistant_msg}"

        # Layer 2: Embed and store in vectorstore
        vec = embed(chunk_text)

        vs = get_vectorstore()
        chunk_id = vs.store_chunk(
            content=chunk_text,
            embedding=vec,
            source=source,
            chat_jid=chat_jid,
        )

        log.debug("Stored conversation chunk %s (%d chars)", chunk_id, len(chunk_text))

    except Exception:
        log.error("Layer 2 post-processing failed", exc_info=True)

    # Layer 3: Entity extraction + Neo4j upsert (separate try so L2 failures don't block L3)
    try:
        from memory.extractor import extract
        from memory import graph

        result = extract(chunk_text)

        entities = result["entities"]
        relations = result["relations"]

        if not entities:
            return

        # Upsert entities into Neo4j
        entity_names = []
        for ent in entities:
            canonical = graph.upsert_entity(
                name=ent["text"],
                entity_type=ent["label"],
                confidence=ent["score"],
            )
            entity_names.append(canonical)

        # Upsert relationships
        for rel in relations:
            graph.upsert_relationship(
                head_name=rel["head"],
                tail_name=rel["tail"],
                rel_type=rel["label"],
                confidence=rel["score"],
                context_snippet=chunk_text[:200],
            )

        # Create episode linking conversation to entities
        graph.create_episode(
            content_preview=chunk_text,
            source=source,
            entity_names=list(set(entity_names)),
        )

        log.debug(
            "Graph updated: %d entities, %d relations (%dms)",
            len(entities), len(relations), result["latency_ms"],
        )

    except Exception:
        log.error("Layer 3 post-processing failed", exc_info=True)
