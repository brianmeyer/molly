"""Entity Audit — Kimi 2.5 entity/relationship audit → GLiNER training pipeline.

Fully automatic pipeline. Kimi audits existing entities and relationships,
discovers missing ones, proposes new types, and feeds ALL corrections directly
into GLiNER training data with no approval step.

Pipeline phases:
  A. Audit existing entities and relationships
  B. Discover missing entities and relationships
  C. Type evolution (auto-adopt after 3 distinct nightly cycles)
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from utils import atomic_write

import config
import db_pool

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------

ENTITY_AUDIT_BATCH_SIZE = int(getattr(config, "ENTITY_AUDIT_BATCH_SIZE", 50))
ENTITY_AUDIT_REL_SAMPLE_SIZE = int(getattr(config, "ENTITY_AUDIT_REL_SAMPLE_SIZE", 100))
ENTITY_AUDIT_DISCOVERY_SAMPLE = int(getattr(config, "ENTITY_AUDIT_DISCOVERY_SAMPLE", 50))
TYPE_PROPOSAL_ADOPTION_THRESHOLD = int(getattr(config, "TYPE_PROPOSAL_ADOPTION_THRESHOLD", 3))
# Minimum confidence for destructive operations (delete, merge).
# Verdicts below this threshold are logged but not applied.
ENTITY_AUDIT_DESTRUCTIVE_MIN_CONFIDENCE = float(
    getattr(config, "ENTITY_AUDIT_DESTRUCTIVE_MIN_CONFIDENCE", 0.7)
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AuditVerdict:
    entity_name: str
    verdict: str  # correct | reclassify | merge | delete | split | propose_new_type
    new_type: str = ""
    merge_target: str = ""
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class RelAuditVerdict:
    source: str
    target: str
    rel_type: str
    verdict: str  # correct | reclassify | delete | propose_new_rel_type
    new_type: str = ""
    new_confidence: float = 0.0
    reasoning: str = ""
    proposed_type_name: str = ""


@dataclass
class EntityProposal:
    name: str
    entity_type: str
    context_snippet: str = ""
    reasoning: str = ""


@dataclass
class RelProposal:
    source: str
    target: str
    rel_type: str
    confidence: float = 0.0
    context_snippet: str = ""
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_entity_audit() -> dict[str, Any]:
    """Main entry point — runs the full Kimi entity audit pipeline.

    Returns audit summary dict with keys:
      entities_audited, rels_audited, entities_discovered, rels_discovered,
      types_proposed, types_adopted, gliner_examples_written, errors
    """
    _ensure_audit_tables()

    result: dict[str, Any] = {
        "entities_audited": 0,
        "rels_audited": 0,
        "entities_discovered": 0,
        "rels_discovered": 0,
        "types_proposed": 0,
        "types_adopted": 0,
        "gliner_examples_written": 0,
        "errors": [],
    }

    all_verdicts: list[AuditVerdict] = []
    all_rel_verdicts: list[RelAuditVerdict] = []
    all_entity_proposals: list[EntityProposal] = []
    all_rel_proposals: list[RelProposal] = []

    # --- Phase A: Audit existing entities and relationships ---
    try:
        entities = _sample_entities_for_audit(ENTITY_AUDIT_BATCH_SIZE)
        if entities:
            verdicts = await _kimi_audit_entities(entities)
            apply_result = _apply_audit_verdicts(verdicts)
            result["entities_audited"] = apply_result.get("processed", 0)
            all_verdicts.extend(verdicts)
    except Exception as exc:
        log.error("Entity audit Phase A (entities) failed: %s", exc, exc_info=True)
        result["errors"].append(f"entity_audit: {exc}")

    try:
        rel_verdicts = await _kimi_audit_relationships(ENTITY_AUDIT_REL_SAMPLE_SIZE)
        if rel_verdicts:
            rel_apply = _apply_rel_verdicts(rel_verdicts)
            result["rels_audited"] = rel_apply.get("processed", 0)
            all_rel_verdicts.extend(rel_verdicts)
    except Exception as exc:
        log.error("Entity audit Phase A (relationships) failed: %s", exc, exc_info=True)
        result["errors"].append(f"rel_audit: {exc}")

    # --- Phase B: Discover missing entities and relationships ---
    try:
        proposals = await _kimi_discover_missing_entities(ENTITY_AUDIT_DISCOVERY_SAMPLE)
        result["entities_discovered"] = len(proposals)
        all_entity_proposals.extend(proposals)
    except Exception as exc:
        log.error("Entity audit Phase B (missing entities) failed: %s", exc, exc_info=True)
        result["errors"].append(f"missing_entities: {exc}")

    try:
        rel_proposals = await _kimi_discover_missing_relationships(ENTITY_AUDIT_DISCOVERY_SAMPLE)
        result["rels_discovered"] = len(rel_proposals)
        all_rel_proposals.extend(rel_proposals)
    except Exception as exc:
        log.error("Entity audit Phase B (missing rels) failed: %s", exc, exc_info=True)
        result["errors"].append(f"missing_rels: {exc}")

    # --- Phase C: Type evolution pipeline ---
    try:
        new_type_proposals = _collect_type_proposals(
            all_verdicts, all_rel_verdicts, all_entity_proposals, all_rel_proposals
        )
        _tally_type_proposals(new_type_proposals)
        result["types_proposed"] = len(new_type_proposals)

        adoption_result = await _evaluate_type_adoption()
        result["types_adopted"] = adoption_result.get("adopted", 0)
    except Exception as exc:
        log.error("Entity audit Phase C (type evolution) failed: %s", exc, exc_info=True)
        result["errors"].append(f"type_evolution: {exc}")

    # --- Write GLiNER training data ---
    try:
        written = _write_gliner_training_labels(
            all_verdicts, all_rel_verdicts, all_entity_proposals, all_rel_proposals
        )
        result["gliner_examples_written"] = written
    except Exception as exc:
        log.error("GLiNER training data write failed: %s", exc, exc_info=True)
        result["errors"].append(f"gliner_write: {exc}")

    # --- Update metrics ---
    try:
        _update_audit_metrics(result)
    except Exception as exc:
        log.debug("Audit metrics update failed: %s", exc, exc_info=True)

    return result


# ---------------------------------------------------------------------------
# Phase A: Audit existing entities
# ---------------------------------------------------------------------------

def _sample_entities_for_audit(batch_size: int = 50) -> list[dict]:
    """Query Neo4j for entities with lowest audit_score or never-audited."""
    try:
        from memory.graph import get_driver

        driver = get_driver()
        with driver.session() as session:
            records = session.run(
                """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r]-()
                WITH e, count(r) AS rel_count
                RETURN e.name AS name,
                       e.entity_type AS entity_type,
                       e.strength AS strength,
                       e.mention_count AS mention_count,
                       e.last_mentioned AS last_mentioned,
                       COALESCE(e.audit_score, 0) AS audit_score,
                       rel_count
                ORDER BY e.audit_score ASC, rel_count DESC, e.mention_count DESC
                LIMIT $batch_size
                """,
                batch_size=batch_size,
            )
            return [dict(r) for r in records]
    except Exception as exc:
        log.error("Failed to sample entities for audit: %s", exc, exc_info=True)
        return []


async def _kimi_audit_entities(entities: list[dict]) -> list[AuditVerdict]:
    """Send batch to Kimi 2.5 for entity audit evaluation."""
    from memory.extractor import ENTITY_SCHEMA

    entity_types = list(ENTITY_SCHEMA.keys()) if ENTITY_SCHEMA else [
        "Person", "Technology", "Organization", "Project", "Place", "Concept"
    ]

    entity_summary = "\n".join(
        f"- {e['name']} (type={e.get('entity_type', '?')}, strength={e.get('strength', '?')}, "
        f"mentions={e.get('mention_count', '?')}, rels={e.get('rel_count', 0)})"
        for e in entities
    )

    prompt = f"""Audit these knowledge graph entities. For each entity, evaluate:
1. Is the entity type correct? Valid types: {', '.join(entity_types)}
2. Should it be merged with another entity listed here? (dedup candidate)
3. Should it be deleted? (noise, too generic, not a real entity)
4. Should the type be changed to an existing type? (reclassify)
5. Is the entity type TOO BROAD? Should we propose a new subtype? (e.g., "Event", "Skill", "Document")

Entities to audit:
{entity_summary}

Respond with a JSON array. Each element must have:
{{"entity_name": "...", "verdict": "correct|reclassify|merge|delete|propose_new_type",
  "new_type": "...", "merge_target": "...", "confidence": 0.85, "reasoning": "..."}}
Only output the JSON array, no other text."""

    raw = await _invoke_kimi(prompt, thinking=True)
    return _parse_entity_verdicts(raw)


def _apply_audit_verdicts(verdicts: list[AuditVerdict]) -> dict[str, int]:
    """Apply entity audit verdicts to Neo4j. Fully automatic — no approval step."""
    from memory.graph import get_driver

    stats = {"processed": 0, "reclassified": 0, "deleted": 0, "merged": 0}
    driver = get_driver()

    for v in verdicts:
        try:
            if v.verdict == "correct":
                _mark_entity_audited(driver, v.entity_name, v.confidence)
            elif v.verdict == "reclassify" and v.new_type:
                _reclassify_entity_type(driver, v.entity_name, v.new_type)
                stats["reclassified"] += 1
            elif v.verdict == "merge" and v.merge_target:
                if v.confidence < ENTITY_AUDIT_DESTRUCTIVE_MIN_CONFIDENCE:
                    log.info("Skipping low-confidence merge for %s (%.2f < %.2f)",
                             v.entity_name, v.confidence, ENTITY_AUDIT_DESTRUCTIVE_MIN_CONFIDENCE)
                else:
                    _merge_entity_pair(driver, v.entity_name, v.merge_target)
                    stats["merged"] += 1
            elif v.verdict == "delete":
                if v.confidence < ENTITY_AUDIT_DESTRUCTIVE_MIN_CONFIDENCE:
                    log.info("Skipping low-confidence delete for %s (%.2f < %.2f)",
                             v.entity_name, v.confidence, ENTITY_AUDIT_DESTRUCTIVE_MIN_CONFIDENCE)
                else:
                    _delete_entity(driver, v.entity_name)
                    stats["deleted"] += 1
            elif v.verdict == "propose_new_type":
                pass  # Handled in Phase C via _collect_type_proposals
            stats["processed"] += 1
            _log_audit_event(v.entity_name, "entity_audit", v.verdict,
                             v.entity_name, v.new_type, v.merge_target,
                             v.confidence, v.reasoning)
        except Exception as exc:
            log.debug("Failed to apply verdict for %s: %s", v.entity_name, exc)

    return stats


async def _kimi_audit_relationships(sample_size: int = 100) -> list[RelAuditVerdict]:
    """Sample relationships and have Kimi evaluate them."""
    from memory.graph import VALID_REL_TYPES, get_driver

    driver = get_driver()
    try:
        with driver.session() as session:
            records = session.run(
                """
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE r.audit_status IS NULL OR r.audit_status <> 'verified'
                RETURN a.name AS source, b.name AS target, type(r) AS rel_type,
                       a.entity_type AS source_type, b.entity_type AS target_type,
                       r.confidence AS confidence, r.context AS context
                ORDER BY COALESCE(r.confidence, 0) ASC
                LIMIT $n
                """,
                n=sample_size,
            )
            rels = [dict(r) for r in records]
    except Exception as exc:
        log.error("Failed to sample relationships: %s", exc, exc_info=True)
        return []

    if not rels:
        return []

    rel_summary = "\n".join(
        f"- {r['source']} ({r.get('source_type', '?')}) -[{r['rel_type']}]-> "
        f"{r['target']} ({r.get('target_type', '?')}), conf={r.get('confidence', '?')}"
        for r in rels
    )

    prompt = f"""Audit these knowledge graph relationships. Valid relationship types: {', '.join(sorted(VALID_REL_TYPES))}

For each relationship, evaluate:
1. Is the type correct given the entity types?
2. Should it be reclassified to a different existing type?
3. Should it be deleted? (noise, hallucinated)
4. Is a NEW relationship type needed that doesn't exist yet?

Relationships to audit:
{rel_summary}

Respond with a JSON array. Each element must have:
{{"source": "...", "target": "...", "rel_type": "...",
  "verdict": "correct|reclassify|delete|propose_new_rel_type",
  "new_type": "...", "new_confidence": 0.0-1.0, "reasoning": "...",
  "proposed_type_name": "..."}}
Only output the JSON array, no other text."""

    raw = await _invoke_kimi(prompt, thinking=True)
    return _parse_rel_verdicts(raw)


def _apply_rel_verdicts(verdicts: list[RelAuditVerdict]) -> dict[str, int]:
    """Apply relationship verdicts to Neo4j."""
    from memory.graph import VALID_REL_TYPES, get_driver, reclassify_relationship, set_relationship_audit_status

    stats = {"processed": 0, "reclassified": 0, "deleted": 0}
    driver = get_driver()

    for v in verdicts:
        try:
            if v.verdict == "correct":
                set_relationship_audit_status(v.source, v.target, v.rel_type, "verified")
            elif v.verdict == "reclassify" and v.new_type and v.new_type in VALID_REL_TYPES:
                reclassify_relationship(
                    v.source, v.target, v.rel_type, v.new_type,
                    strength=v.new_confidence or 0.5, mention_count=1,
                )
                stats["reclassified"] += 1
            elif v.verdict == "delete":
                if v.new_confidence < ENTITY_AUDIT_DESTRUCTIVE_MIN_CONFIDENCE:
                    log.info("Skipping low-confidence rel delete %s->%s[%s] (%.2f < %.2f)",
                             v.source, v.target, v.rel_type,
                             v.new_confidence, ENTITY_AUDIT_DESTRUCTIVE_MIN_CONFIDENCE)
                else:
                    from memory.graph import delete_specific_relationship
                    delete_specific_relationship(v.source, v.target, v.rel_type)
                    stats["deleted"] += 1
            elif v.verdict == "propose_new_rel_type":
                pass  # Handled in Phase C
            stats["processed"] += 1
            _log_audit_event(f"{v.source}->{v.target}", "rel_audit", v.verdict,
                             v.rel_type, v.new_type, "", v.new_confidence, v.reasoning)
        except Exception as exc:
            log.debug("Failed to apply rel verdict %s->%s: %s", v.source, v.target, exc)

    return stats


# ---------------------------------------------------------------------------
# Phase B: Discover missing entities and relationships
# ---------------------------------------------------------------------------

async def _kimi_discover_missing_entities(sample_size: int = 50) -> list[EntityProposal]:
    """Sample recent chunks with low entity counts and ask Kimi what was missed."""
    rows: list = []
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        rows = conn.execute(
            """
            SELECT c.content, c.source, c.created_at,
                   (SELECT COUNT(*) FROM entity_mentions em WHERE em.chunk_id = c.id) AS entity_count
            FROM conversation_chunks c
            WHERE c.created_at > ?
            ORDER BY entity_count ASC, c.created_at DESC
            LIMIT ?
            """,
            ((datetime.now(timezone.utc) - timedelta(days=7)).isoformat(), sample_size),
        ).fetchall()
    except Exception:
        # entity_mentions table might not exist — fall back to simpler query
        try:
            if conn is not None:
                conn.close()
                conn = None
            conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
            rows = conn.execute(
                """
                SELECT content, source, created_at, 0 AS entity_count
                FROM conversation_chunks
                WHERE created_at > ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                ((datetime.now(timezone.utc) - timedelta(days=7)).isoformat(), sample_size),
            ).fetchall()
        except Exception as exc:
            log.error("Failed to sample chunks for discovery: %s", exc, exc_info=True)
            return []
    finally:
        if conn is not None:
            conn.close()

    if not rows:
        return []

    chunks_text = "\n---\n".join(
        f"[{row[1] or 'unknown'}] {(row[0] or '')[:500]}" for row in rows[:20]
    )

    prompt = f"""Given these conversation excerpts, identify entities that SHOULD have been extracted but were missed.
Focus on: people's names, companies, technologies, projects, places, and concepts.

Conversation excerpts:
{chunks_text}

Respond with a JSON array. Each element must have:
{{"name": "...", "entity_type": "Person|Technology|Organization|Project|Place|Concept",
  "context_snippet": "...", "reasoning": "..."}}
Only output the JSON array, no other text."""

    raw = await _invoke_kimi(prompt, thinking=False)
    proposals = _parse_entity_proposals(raw)

    # Auto-create discovered entities (use async upsert_entity, not sync variant)
    for p in proposals:
        try:
            from memory.graph import upsert_entity
            await upsert_entity(p.name, p.entity_type, confidence=0.7)
            _log_audit_event(p.name, "missing_entity", "discovered",
                             "", p.entity_type, "", 0.7, p.reasoning)
        except Exception as exc:
            log.debug("Failed to create discovered entity %s: %s", p.name, exc)

    return proposals


async def _kimi_discover_missing_relationships(sample_size: int = 50) -> list[RelProposal]:
    """Sample entity pairs that co-occur but have no relationship."""
    from memory.graph import VALID_REL_TYPES, get_driver

    try:
        driver = get_driver()
        with driver.session() as session:
            records = session.run(
                """
                MATCH (a:Entity), (b:Entity)
                WHERE a.name < b.name
                  AND NOT (a)--(b)
                  AND a.last_mentioned IS NOT NULL
                  AND b.last_mentioned IS NOT NULL
                RETURN a.name AS source, b.name AS target,
                       a.entity_type AS source_type, b.entity_type AS target_type
                ORDER BY a.mention_count + b.mention_count DESC
                LIMIT $n
                """,
                n=sample_size,
            )
            pairs = [dict(r) for r in records]
    except Exception as exc:
        log.error("Failed to sample disconnected pairs: %s", exc, exc_info=True)
        return []

    if not pairs:
        return []

    pairs_text = "\n".join(
        f"- {p['source']} ({p.get('source_type', '?')}) <-> {p['target']} ({p.get('target_type', '?')})"
        for p in pairs[:30]
    )

    prompt = f"""These entity pairs appear in conversation but have NO relationship in the knowledge graph.
Valid relationship types: {', '.join(sorted(VALID_REL_TYPES))}

Pairs:
{pairs_text}

For each pair where a relationship likely exists, provide it. Skip pairs with no clear relationship.
Respond with a JSON array. Each element must have:
{{"source": "...", "target": "...", "rel_type": "...",
  "confidence": 0.85, "context_snippet": "...", "reasoning": "..."}}
Only output the JSON array, no other text."""

    raw = await _invoke_kimi(prompt, thinking=False)
    proposals = _parse_rel_proposals(raw)

    # Auto-create discovered relationships (only if rel_type is valid)
    for p in proposals:
        if p.rel_type in VALID_REL_TYPES:
            try:
                from memory.graph import upsert_relationship
                await upsert_relationship(p.source, p.target, p.rel_type, p.confidence, p.context_snippet)
                _log_audit_event(f"{p.source}->{p.target}", "missing_rel", "discovered",
                                 "", p.rel_type, "", p.confidence, p.reasoning)
            except Exception as exc:
                log.debug("Failed to create discovered rel %s->%s: %s", p.source, p.target, exc)

    return proposals


# ---------------------------------------------------------------------------
# Phase C: Type evolution pipeline
# ---------------------------------------------------------------------------

def _collect_type_proposals(
    entity_verdicts: list[AuditVerdict],
    rel_verdicts: list[RelAuditVerdict],
    entity_proposals: list[EntityProposal],
    rel_proposals: list[RelProposal],
) -> list[dict]:
    """Collect all new type proposals from this cycle's Kimi outputs."""
    proposals = []
    for v in entity_verdicts:
        if v.verdict == "propose_new_type" and v.new_type:
            proposals.append({
                "type_category": "entity",
                "proposed_name": v.new_type,
                "proposed_by": "kimi_entity_audit",
                "context": v.reasoning,
            })
    for v in rel_verdicts:
        if v.verdict == "propose_new_rel_type" and v.proposed_type_name:
            proposals.append({
                "type_category": "relationship",
                "proposed_name": v.proposed_type_name,
                "proposed_by": "kimi_rel_audit",
                "context": v.reasoning,
            })
    return proposals


def _tally_type_proposals(new_proposals: list[dict]) -> None:
    """Tally proposals — only count once per distinct nightly cycle."""
    today_str = date.today().isoformat()
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        _ensure_type_proposal_tables(conn)

        for p in new_proposals:
            cat = p["type_category"]
            name = p["proposed_name"].strip().upper()
            context = p.get("context", "")
            proposed_by = p.get("proposed_by", "kimi_audit")

            row = conn.execute(
                "SELECT id, distinct_cycle_dates, context_examples, occurrence_count "
                "FROM type_proposals WHERE type_category = ? AND proposed_name = ?",
                (cat, name),
            ).fetchone()

            if row:
                row_id, dates_json, examples_json, count = row
                dates = json.loads(dates_json) if dates_json else []
                examples = json.loads(examples_json) if examples_json else []
                if today_str not in dates:
                    dates.append(today_str)
                    count += 1
                examples.append(context[:500])
                conn.execute(
                    "UPDATE type_proposals SET distinct_cycle_dates = ?, "
                    "context_examples = ?, occurrence_count = ? WHERE id = ?",
                    (json.dumps(dates), json.dumps(examples[-10:]), count, row_id),
                )
            else:
                conn.execute(
                    """INSERT INTO type_proposals
                       (type_category, proposed_name, proposed_by, context_examples,
                        distinct_cycle_dates, occurrence_count, status)
                       VALUES (?, ?, ?, ?, ?, 1, 'proposed')""",
                    (cat, name, proposed_by, json.dumps([context[:500]]),
                     json.dumps([today_str])),
                )

        conn.commit()
    except Exception as exc:
        log.error("Failed to tally type proposals: %s", exc, exc_info=True)
    finally:
        if conn is not None:
            conn.close()


async def _evaluate_type_adoption() -> dict[str, int]:
    """Evaluate and auto-adopt types that meet the threshold."""
    result = {"adopted": 0, "retroactive": 0}
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        _ensure_type_proposal_tables(conn)

        candidates = conn.execute(
            "SELECT id, type_category, proposed_name, context_examples "
            "FROM type_proposals "
            "WHERE occurrence_count >= ? AND status = 'proposed'",
            (TYPE_PROPOSAL_ADOPTION_THRESHOLD,),
        ).fetchall()

        for row_id, category, name, examples_json in candidates:
            try:
                if category == "entity":
                    _adopt_entity_type(name)
                elif category == "relationship":
                    retro_count = _adopt_relationship_type(name)
                    result["retroactive"] += retro_count

                conn.execute(
                    "UPDATE type_proposals SET status = 'adopted', adopted_at = ? WHERE id = ?",
                    (datetime.now(timezone.utc).isoformat(), row_id),
                )
                _log_schema_evolution(conn, category, name)
                result["adopted"] += 1
            except Exception as exc:
                log.error("Failed to adopt type %s/%s: %s", category, name, exc, exc_info=True)

        conn.commit()
    except Exception as exc:
        log.error("Type adoption evaluation failed: %s", exc, exc_info=True)
    finally:
        if conn is not None:
            conn.close()

    return result


def _adopt_entity_type(type_name: str) -> None:
    """Add a new entity type to the entity schema config."""
    schema_path = config.WORKSPACE / "config" / "entity_schema.yaml"
    schema_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if schema_path.exists():
        try:
            import yaml
            existing = yaml.safe_load(schema_path.read_text()) or {}
        except Exception:
            pass

    if type_name not in existing:
        existing[type_name] = {
            "description": f"Auto-adopted entity type from Kimi audit ({type_name})",
            "threshold": 0.45,
        }
        try:
            import yaml
            atomic_write(schema_path, yaml.dump(existing, default_flow_style=False))
        except ImportError:
            # Fall back to JSON if yaml not available
            atomic_write(schema_path.with_suffix(".json"), json.dumps(existing, indent=2))

    log.info("Adopted new entity type: %s", type_name)


def _adopt_relationship_type(type_name: str) -> int:
    """Add a new relationship type to the relation schema and retroactively reclassify."""
    # Update relation_schema.yaml
    schema_path = config.WORKSPACE / "config" / "relation_schema.yaml"
    schema_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if schema_path.exists():
        try:
            import yaml
            existing = yaml.safe_load(schema_path.read_text()) or {}
        except Exception:
            pass

    if type_name not in existing:
        existing[type_name] = {
            "description": f"Auto-adopted relationship type from Kimi audit",
            "threshold": 0.4,
        }
        try:
            import yaml
            atomic_write(schema_path, yaml.dump(existing, default_flow_style=False))
        except ImportError:
            atomic_write(schema_path.with_suffix(".json"), json.dumps(existing, indent=2))

    # Add to VALID_REL_TYPES at runtime (thread-safe via graph lock)
    try:
        from memory.graph import VALID_REL_TYPES, _GRAPH_SYNC_WRITE_LOCK
        with _GRAPH_SYNC_WRITE_LOCK:
            VALID_REL_TYPES.add(type_name)
    except Exception:
        pass

    # Retroactive reclassification
    retro_count = _retroactive_reclassify(type_name)
    log.info("Adopted new relationship type: %s (retroactively reclassified %d)", type_name, retro_count)
    return retro_count


def _retroactive_reclassify(adopted_type: str) -> int:
    """Scan RELATED_TO relationships that might match the newly adopted type."""
    # This is a best-effort heuristic — we look for context clues
    try:
        from memory.graph import get_driver, reclassify_relationship

        driver = get_driver()
        type_lower = adopted_type.lower().replace("_", " ")
        with driver.session() as session:
            records = session.run(
                """
                MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
                WHERE r.context IS NOT NULL AND toLower(r.context) CONTAINS $hint
                RETURN a.name AS source, b.name AS target,
                       r.strength AS strength, r.mention_count AS mention_count
                LIMIT 50
                """,
                hint=type_lower,
            )
            matches = [dict(r) for r in records]

        count = 0
        for m in matches:
            try:
                reclassify_relationship(
                    m["source"], m["target"], "RELATED_TO", adopted_type,
                    strength=float(m.get("strength") or 0.5),
                    mention_count=int(m.get("mention_count") or 1),
                )
                count += 1
            except Exception:
                pass
        return count
    except Exception as exc:
        log.debug("Retroactive reclassification failed: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# GLiNER training data writer
# ---------------------------------------------------------------------------

def _write_gliner_training_labels(
    entity_verdicts: list[AuditVerdict],
    rel_verdicts: list[RelAuditVerdict],
    entity_proposals: list[EntityProposal],
    rel_proposals: list[RelProposal],
) -> int:
    """Convert all Kimi outputs into GLiNER-format JSONL training examples.

    Writes to WORKSPACE/memory/gliner_training/ — the same directory that
    SelfImprovementEngine._gliner_training_dir() reads from, so that the
    GLiNER nightly cycle in Step 10 picks up Kimi-generated labels from Step 6.
    """
    training_dir = config.WORKSPACE / "memory" / "gliner_training"
    training_dir.mkdir(parents=True, exist_ok=True)
    output_path = training_dir / f"{date.today().isoformat()}.jsonl"

    count = 0
    lines: list[str] = []

    # Entity verdicts → entity labels
    for v in entity_verdicts:
        if v.verdict in ("correct", "reclassify") and v.entity_name:
            label = v.new_type if v.verdict == "reclassify" and v.new_type else ""
            if not label:
                continue
            example = {
                "text": v.reasoning[:500] if v.reasoning else v.entity_name,
                "entities": [{"text": v.entity_name, "label": label}],
                "source": "kimi_entity_audit",
                "verdict": v.verdict,
            }
            lines.append(json.dumps(example, ensure_ascii=False))
            count += 1

    # Discovered entities → highest-value training examples
    for p in entity_proposals:
        example = {
            "text": p.context_snippet[:500] if p.context_snippet else p.name,
            "entities": [{"text": p.name, "label": p.entity_type}],
            "source": "kimi_discovery",
            "verdict": "discovered",
        }
        lines.append(json.dumps(example, ensure_ascii=False))
        count += 1

    # Relationship verdicts → relationship labels
    for v in rel_verdicts:
        if v.verdict in ("correct", "reclassify"):
            rel_label = v.new_type if v.verdict == "reclassify" and v.new_type else v.rel_type
            example = {
                "text": v.reasoning[:500] if v.reasoning else f"{v.source} {v.target}",
                "relations": [{
                    "head": v.source, "tail": v.target, "label": rel_label,
                }],
                "source": "kimi_rel_audit",
                "verdict": v.verdict,
            }
            lines.append(json.dumps(example, ensure_ascii=False))
            count += 1

    # Discovered relationships
    for p in rel_proposals:
        example = {
            "text": p.context_snippet[:500] if p.context_snippet else f"{p.source} {p.target}",
            "relations": [{
                "head": p.source, "tail": p.target, "label": p.rel_type,
            }],
            "source": "kimi_discovery",
            "verdict": "discovered",
        }
        lines.append(json.dumps(example, ensure_ascii=False))
        count += 1

    if lines:
        import fcntl
        fd = os.open(str(output_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, ("\n".join(lines) + "\n").encode())
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    return count


# ---------------------------------------------------------------------------
# Graph operations helpers
# ---------------------------------------------------------------------------

def _mark_entity_audited(driver, entity_name: str, confidence: float) -> None:
    with driver.session() as session:
        session.run(
            "MATCH (e:Entity {name: $name}) "
            "SET e.audit_score = $score, e.last_audited = $now",
            name=entity_name,
            score=min(1.0, confidence + 0.1),
            now=datetime.now(timezone.utc).isoformat(),
        )


def _reclassify_entity_type(driver, entity_name: str, new_type: str) -> None:
    with driver.session() as session:
        session.run(
            "MATCH (e:Entity {name: $name}) SET e.entity_type = $new_type",
            name=entity_name, new_type=new_type,
        )


_APOC_AVAILABLE: bool | None = None


def _has_apoc(session) -> bool:
    """Check whether the APOC plugin is available (cached after first call)."""
    global _APOC_AVAILABLE
    if _APOC_AVAILABLE is not None:
        return _APOC_AVAILABLE
    try:
        session.run("RETURN apoc.version() AS v").consume()
        _APOC_AVAILABLE = True
    except Exception:
        _APOC_AVAILABLE = False
        log.info("APOC not available — using native Cypher fallback for entity merges")
    return _APOC_AVAILABLE


def _merge_entity_pair(driver, source_name: str, target_name: str) -> None:
    """Merge source entity into target — transfer relationships, then delete source.

    Uses APOC when available (preserves dynamic relationship types), otherwise
    falls back to native Cypher that also preserves the original relationship type
    by using a per-type parameterised CREATE query.
    """
    with driver.session() as session:
        use_apoc = _has_apoc(session)

        # Materialise result sets as lists BEFORE iterating, otherwise
        # subsequent session.run() calls inside the loop exhaust the cursor.
        outgoing = list(session.run(
            """
            MATCH (s:Entity {name: $source})-[r]->(other)
            WHERE other.name <> $target
            RETURN other.name AS other_name, type(r) AS rtype, properties(r) AS props
            """,
            source=source_name, target=target_name,
        ))
        incoming = list(session.run(
            """
            MATCH (other)-[r]->(s:Entity {name: $source})
            WHERE other.name <> $target
            RETURN other.name AS other_name, type(r) AS rtype, properties(r) AS props
            """,
            source=source_name, target=target_name,
        ))

        # Transfer outgoing relationships
        for rec in outgoing:
            rtype = rec["rtype"]
            props = rec["props"] or {}
            if use_apoc:
                session.run(
                    """
                    MATCH (t:Entity {name: $target}), (other:Entity {name: $other_name})
                    CALL apoc.create.relationship(t, $rtype, $props, other) YIELD rel
                    RETURN rel
                    """,
                    target=target_name, other_name=rec["other_name"],
                    rtype=rtype, props=props,
                )
            else:
                # Native Cypher: use the original type via string concatenation
                # in the query so relationship type is preserved (Cypher does not
                # support parameterised relationship types without APOC).
                _create_typed_relationship(
                    session, target_name, rec["other_name"], rtype, props,
                )

        # Transfer incoming relationships
        for rec in incoming:
            rtype = rec["rtype"]
            props = rec["props"] or {}
            if use_apoc:
                session.run(
                    """
                    MATCH (other:Entity {name: $other_name}), (t:Entity {name: $target})
                    CALL apoc.create.relationship(other, $rtype, $props, t) YIELD rel
                    RETURN rel
                    """,
                    target=target_name, other_name=rec["other_name"],
                    rtype=rtype, props=props,
                )
            else:
                _create_typed_relationship(
                    session, rec["other_name"], target_name, rtype, props,
                )

        # Delete old relationships from source
        session.run(
            "MATCH (s:Entity {name: $source})-[r]-() DELETE r",
            source=source_name,
        )
        # Add source name as alias on target
        session.run(
            "MATCH (t:Entity {name: $target}) "
            "SET t.aliases = COALESCE(t.aliases, []) + $alias, "
            "    t.mention_count = COALESCE(t.mention_count, 0) + 1",
            target=target_name, alias=source_name,
        )
        # Delete source entity
        session.run(
            "MATCH (s:Entity {name: $source}) DELETE s",
            source=source_name,
        )


def _create_typed_relationship(
    session, from_name: str, to_name: str, rel_type: str,
    props: dict,
) -> None:
    """Create a relationship from_name -> to_name preserving the original type.

    Callers control direction by swapping from_name/to_name arguments.
    Cypher doesn't allow parameterised relationship types, so we validate the
    type against VALID_REL_TYPES and interpolate it directly into the query.
    Unknown types fall back to RELATED_TO with original_type stored in props.
    """
    from memory.graph import VALID_REL_TYPES

    # Sanitise — only allow known types to prevent injection
    if rel_type in VALID_REL_TYPES:
        safe_type = rel_type
    else:
        safe_type = "RELATED_TO"
        props = {**props, "original_type": rel_type}

    query = (
        f"MATCH (a:Entity {{name: $from_name}}), (b:Entity {{name: $to_name}}) "
        f"CREATE (a)-[r:{safe_type}]->(b) SET r = $props RETURN r"
    )
    session.run(query, from_name=from_name, to_name=to_name, props=props)


def _delete_entity(driver, entity_name: str) -> None:
    with driver.session() as session:
        session.run(
            "MATCH (e:Entity {name: $name}) DETACH DELETE e",
            name=entity_name,
        )


# ---------------------------------------------------------------------------
# Kimi invocation
# ---------------------------------------------------------------------------

async def _invoke_kimi(prompt: str, thinking: bool = False) -> str:
    """Invoke Kimi 2.5 via the tools/kimi module's internal API client."""
    import httpx

    api_key = getattr(config, "MOONSHOT_API_KEY", "")
    if not api_key:
        raise RuntimeError("MOONSHOT_API_KEY not configured")

    model = getattr(config, "CONTRACT_AUDIT_KIMI_MODEL", "kimi-k2-0711")
    base_url = getattr(config, "MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")

    messages = [
        {"role": "system", "content": "You are a knowledge graph auditor. Respond with valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
    }
    if thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": 8192}

    async with httpx.AsyncClient(timeout=300) as client:
        for attempt in range(3):
            try:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as exc:
                if attempt == 2:
                    raise
                import asyncio
                await asyncio.sleep(5 * (attempt + 1))
                log.debug("Kimi retry %d: %s", attempt + 1, exc)

    return ""


# ---------------------------------------------------------------------------
# JSON response parsers (defensive)
# ---------------------------------------------------------------------------

def _parse_entity_verdicts(raw: str) -> list[AuditVerdict]:
    try:
        data = json.loads(_extract_json_array(raw))
        return [
            AuditVerdict(
                entity_name=str(item.get("entity_name", "")),
                verdict=str(item.get("verdict", "correct")),
                new_type=str(item.get("new_type", "")),
                merge_target=str(item.get("merge_target", "")),
                confidence=float(item.get("confidence", 0)),
                reasoning=str(item.get("reasoning", "")),
            )
            for item in data if isinstance(item, dict) and item.get("entity_name")
        ]
    except Exception as exc:
        log.debug("Failed to parse entity verdicts: %s", exc)
        return []


def _parse_rel_verdicts(raw: str) -> list[RelAuditVerdict]:
    try:
        data = json.loads(_extract_json_array(raw))
        return [
            RelAuditVerdict(
                source=str(item.get("source", "")),
                target=str(item.get("target", "")),
                rel_type=str(item.get("rel_type", "")),
                verdict=str(item.get("verdict", "correct")),
                new_type=str(item.get("new_type", "")),
                new_confidence=float(item.get("new_confidence", 0)),
                reasoning=str(item.get("reasoning", "")),
                proposed_type_name=str(item.get("proposed_type_name", "")),
            )
            for item in data if isinstance(item, dict) and item.get("source")
        ]
    except Exception as exc:
        log.debug("Failed to parse rel verdicts: %s", exc)
        return []


def _parse_entity_proposals(raw: str) -> list[EntityProposal]:
    try:
        data = json.loads(_extract_json_array(raw))
        return [
            EntityProposal(
                name=str(item.get("name", "")),
                entity_type=str(item.get("entity_type", "Concept")),
                context_snippet=str(item.get("context_snippet", "")),
                reasoning=str(item.get("reasoning", "")),
            )
            for item in data if isinstance(item, dict) and item.get("name")
        ]
    except Exception as exc:
        log.debug("Failed to parse entity proposals: %s", exc)
        return []


def _parse_rel_proposals(raw: str) -> list[RelProposal]:
    try:
        data = json.loads(_extract_json_array(raw))
        return [
            RelProposal(
                source=str(item.get("source", "")),
                target=str(item.get("target", "")),
                rel_type=str(item.get("rel_type", "RELATED_TO")),
                confidence=float(item.get("confidence", 0.5)),
                context_snippet=str(item.get("context_snippet", "")),
                reasoning=str(item.get("reasoning", "")),
            )
            for item in data if isinstance(item, dict) and item.get("source")
        ]
    except Exception as exc:
        log.debug("Failed to parse rel proposals: %s", exc)
        return []


def _extract_json_array(text: str) -> str:
    """Extract the first JSON array from text that may contain markdown fences."""
    text = text.strip()
    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    # Find the array
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return "[]"


# ---------------------------------------------------------------------------
# Database table management
# ---------------------------------------------------------------------------

def _ensure_audit_tables() -> None:
    """Ensure all audit-related tables exist in mollygraph.db."""
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS entity_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_name TEXT,
                audit_phase TEXT,
                verdict TEXT,
                old_type TEXT,
                new_type TEXT,
                merge_target TEXT,
                confidence REAL,
                reasoning TEXT,
                gliner_examples_written INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );

            CREATE TABLE IF NOT EXISTS type_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type_category TEXT,
                proposed_name TEXT,
                proposed_by TEXT,
                context_examples TEXT,
                distinct_cycle_dates TEXT,
                occurrence_count INTEGER DEFAULT 1,
                status TEXT DEFAULT 'proposed',
                adopted_at TEXT,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                UNIQUE(type_category, proposed_name)
            );

            CREATE TABLE IF NOT EXISTS schema_evolution_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                change_type TEXT,
                type_name TEXT,
                trigger TEXT,
                details TEXT,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );
        """)
    except Exception as exc:
        log.error("Failed to ensure audit tables: %s", exc)
    finally:
        if conn is not None:
            conn.close()


def _ensure_type_proposal_tables(conn: sqlite3.Connection) -> None:
    """Ensure type_proposals table exists (idempotent)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS type_proposals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type_category TEXT,
            proposed_name TEXT,
            proposed_by TEXT,
            context_examples TEXT,
            distinct_cycle_dates TEXT,
            occurrence_count INTEGER DEFAULT 1,
            status TEXT DEFAULT 'proposed',
            adopted_at TEXT,
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            UNIQUE(type_category, proposed_name)
        );
    """)


def _log_audit_event(
    entity_name: str, phase: str, verdict: str,
    old_type: str, new_type: str, merge_target: str,
    confidence: float, reasoning: str,
) -> None:
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        conn.execute(
            """INSERT INTO entity_audit_log
               (entity_name, audit_phase, verdict, old_type, new_type,
                merge_target, confidence, reasoning)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (entity_name, phase, verdict, old_type, new_type,
             merge_target, confidence, reasoning[:1000]),
        )
        conn.commit()
    except Exception:
        log.debug("Failed to log audit event for %s", entity_name, exc_info=True)
    finally:
        if conn is not None:
            conn.close()


def _log_schema_evolution(conn: sqlite3.Connection, category: str, type_name: str) -> None:
    change_type = f"add_{category}_type"
    conn.execute(
        "INSERT INTO schema_evolution_log (change_type, type_name, trigger, details) "
        "VALUES (?, ?, 'kimi_audit_3x_threshold', ?)",
        (change_type, type_name, f"Auto-adopted after {TYPE_PROPOSAL_ADOPTION_THRESHOLD} distinct cycles"),
    )


def _update_audit_metrics(result: dict[str, Any]) -> None:
    """Write summary metrics to entity_audit_log."""
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        conn.execute(
            """INSERT INTO entity_audit_log
               (entity_name, audit_phase, verdict, reasoning)
               VALUES ('_summary', 'nightly_run', 'summary', ?)""",
            (json.dumps(result, default=str),),
        )
        conn.commit()
    except Exception:
        log.debug("Failed to write audit metrics summary", exc_info=True)
    finally:
        if conn is not None:
            conn.close()
