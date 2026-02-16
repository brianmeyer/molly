"""Graph Maintenance — strength decay, dedup, orphan cleanup, Neo4j checkpoint."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def run_strength_decay() -> str:
    """Recalculate entity strength via temporal decay."""
    from memory.graph import run_strength_decay

    count = await run_strength_decay()
    return f"{count} entities updated"


def run_dedup_sweep() -> str:
    """Run the deduplication sweep across entities."""
    from memory.dedup import run_dedup

    merged = int(run_dedup())
    return f"{merged} entities merged"


def run_orphan_cleanup() -> str:
    """Delete low-signal orphaned entities, self-referencing rels, and blocklisted entities."""
    from memory.graph import (
        delete_blocklisted_entities,
        delete_self_referencing_rels,
        get_driver,
    )
    from memory.processor import _ENTITY_BLOCKLIST

    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """MATCH (e:Entity)
               WHERE NOT (e)--()
                 AND e.mention_count <= 1
                 AND (e.strength IS NULL OR e.strength < 0.3)
               DELETE e
               RETURN count(e) AS deleted"""
        )
        orphans = int(result.single()["deleted"])

    self_refs = int(delete_self_referencing_rels())
    blocklisted = int(delete_blocklisted_entities(_ENTITY_BLOCKLIST))
    return f"orphans={orphans}, self_refs={self_refs}, blocklisted={blocklisted}"


def run_neo4j_checkpoint() -> str:
    """Run a Neo4j transaction log checkpoint (Enterprise edition only)."""
    from memory.graph import get_driver

    driver = get_driver()
    with driver.session() as neo_session:
        # Detect edition — db.checkpoint() is Enterprise-only
        try:
            comp = neo_session.run("SHOW SERVER INFO YIELD edition")
            comp_record = comp.single()
            edition = str(comp_record.get("edition", "")).lower() if comp_record else ""
        except Exception:
            comp = neo_session.run("CALL dbms.components()")
            comp_record = comp.single()
            edition = str(comp_record.get("edition", "")).lower() if comp_record else ""
        if "enterprise" in edition:
            neo_session.run("CALL db.checkpoint()")
            return "completed"
        return f"skipped ({edition or 'unknown'} edition)"
