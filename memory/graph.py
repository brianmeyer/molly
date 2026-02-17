import asyncio
import logging
import math
import re
import threading
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from neo4j import GraphDatabase

import config
from utils import track_latency


class GraphUnavailableError(RuntimeError):
    """Raised when Neo4j is not reachable — callers should degrade gracefully."""

log = logging.getLogger(__name__)

_driver = None
_driver_lock = threading.Lock()
_GRAPH_WRITE_LOCK: asyncio.Lock | None = None
_GRAPH_WRITE_LOCK_LOOP_ID: int | None = None
# threading.Lock for sync callers (ThreadPoolExecutor, direct calls).
# asyncio.Lock only serializes async callers on the event loop; sync wrappers
# called via asyncio.to_thread() bypass it entirely.
_GRAPH_SYNC_WRITE_LOCK = threading.Lock()

VALID_REL_TYPES = {
    "WORKS_ON", "WORKS_AT", "KNOWS", "USES", "LOCATED_IN",
    "DISCUSSED_WITH", "INTERESTED_IN", "CREATED", "MANAGES",
    "DEPENDS_ON", "RELATED_TO",
    "CLASSMATE_OF", "STUDIED_AT", "ALUMNI_OF",
    "MENTORS", "MENTORED_BY", "REPORTS_TO", "COLLABORATES_WITH",
    "CONTACT_OF",
    "CUSTOMER_OF", "ATTENDS", "PARENT_OF", "CHILD_OF", "RECEIVED_FROM",
}


def _get_graph_write_lock() -> asyncio.Lock:
    global _GRAPH_WRITE_LOCK, _GRAPH_WRITE_LOCK_LOOP_ID
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    if _GRAPH_WRITE_LOCK is None or _GRAPH_WRITE_LOCK_LOOP_ID != loop_id:
        _GRAPH_WRITE_LOCK = asyncio.Lock()
        _GRAPH_WRITE_LOCK_LOOP_ID = loop_id
    return _GRAPH_WRITE_LOCK


# --- Driver management ---


def get_driver():
    """Lazy-initialize the Neo4j driver and create indexes (thread-safe).

    Raises GraphUnavailableError if Neo4j is unreachable, allowing callers
    to degrade gracefully via their existing except-Exception handlers.
    """
    global _driver
    if _driver is not None:
        return _driver
    with _driver_lock:
        if _driver is None:
            try:
                drv = GraphDatabase.driver(
                    config.NEO4J_URI,
                    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
                )
                with drv.session() as session:
                    session.run(
                        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)"
                    )
                    session.run(
                        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)"
                    )
                    session.run(
                        "CREATE INDEX episode_id IF NOT EXISTS FOR (ep:Episode) ON (ep.id)"
                    )
                    session.run(
                        "CREATE INDEX entity_phone IF NOT EXISTS FOR (e:Entity) ON (e.phone)"
                    )
                _driver = drv
                log.info("Neo4j driver initialized at %s", config.NEO4J_URI)
            except Exception:
                log.warning("Neo4j unavailable — graph layer disabled", exc_info=True)
                raise GraphUnavailableError("Neo4j is not reachable")
    return _driver


def close():
    global _driver
    with _driver_lock:
        if _driver:
            _driver.close()
            _driver = None
            log.info("Neo4j driver closed")


# --- Strength / decay ---


def recency_score(days_since: float) -> float:
    """Exponential decay with ~23-day half-life."""
    return math.exp(-0.03 * days_since)


def strength_score(mentions: int, days_since: float, boost: float = 0) -> float:
    return (mentions * recency_score(days_since)) + boost


# --- Deduplication helpers ---


def _normalize(name: str) -> str:
    return name.strip().lower()


def _fuzzy_ratio(a: str, b: str) -> float:
    """SequenceMatcher ratio (0-1). Approximates Jaro-Winkler well enough."""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


@track_latency("neo4j")
def find_matching_entity(
    name: str,
    entity_type: str,
    fuzzy_threshold: float = 0.85,
) -> str | None:
    """Find an existing entity by exact or fuzzy match. Returns entity name or None."""
    driver = get_driver()
    normalized = _normalize(name)

    with driver.session() as session:
        # 1. Exact match on name
        result = session.run(
            "MATCH (e:Entity) WHERE toLower(e.name) = $name RETURN e.name AS name",
            name=normalized,
        )
        record = result.single()
        if record:
            return record["name"]

        # 2. Exact match on aliases
        result = session.run(
            """MATCH (e:Entity)
               WHERE ANY(a IN e.aliases WHERE toLower(a) = $name)
               RETURN e.name AS name""",
            name=normalized,
        )
        record = result.single()
        if record:
            return record["name"]

        # 3. Fuzzy match (same entity_type, ratio >= threshold)
        result = session.run(
            "MATCH (e:Entity {entity_type: $etype}) RETURN e.name AS name",
            etype=entity_type,
        )
        for record in result:
            existing_name = record["name"]
            if _fuzzy_ratio(name, existing_name) >= fuzzy_threshold:
                return existing_name

    return None


# --- Entity CRUD ---


def _upsert_entity_sync(
    name: str,
    entity_type: str,
    confidence: float,
) -> str:
    """Create or merge an entity. Returns the canonical entity name."""
    driver = get_driver()
    now = datetime.now(timezone.utc).isoformat()

    # Check for existing match
    existing = find_matching_entity(name, entity_type)

    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        if existing:
            # Merge into existing entity
            session.run(
                """MATCH (e:Entity) WHERE e.name = $existing_name
                   SET e.mention_count = e.mention_count + 1,
                       e.last_mentioned = $now,
                       e.confidence = CASE
                           WHEN $confidence > e.confidence THEN $confidence
                           ELSE e.confidence END,
                       e.aliases = CASE
                           WHEN NOT toLower($alias) IN [x IN e.aliases | toLower(x)]
                                AND toLower($alias) <> toLower(e.name)
                           THEN e.aliases + $alias
                           ELSE e.aliases END""",
                existing_name=existing,
                now=now,
                confidence=confidence,
                alias=name.strip(),
            )
            return existing
        else:
            # Create new entity
            session.run(
                """CREATE (e:Entity {
                       name: $name,
                       entity_type: $entity_type,
                       mention_count: 1,
                       first_mentioned: $now,
                       last_mentioned: $now,
                       strength: 1.0,
                       confidence: $confidence,
                       aliases: [],
                       summary: ''
                   })""",
                name=name.strip(),
                entity_type=entity_type,
                now=now,
                confidence=confidence,
            )
            return name.strip()


@track_latency("neo4j")
def upsert_entity_sync(
    name: str,
    entity_type: str,
    confidence: float,
) -> str:
    return _upsert_entity_sync(name, entity_type, confidence)


@track_latency("neo4j")
async def upsert_entity(
    name: str,
    entity_type: str,
    confidence: float,
) -> str:
    async with _get_graph_write_lock():
        return await asyncio.to_thread(_upsert_entity_sync, name, entity_type, confidence)


_SAFE_PROPERTY_KEY = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def set_entity_properties(name: str, properties: dict) -> None:
    """Set properties on an existing entity node."""
    if not properties:
        return
    driver = get_driver()
    # Validate property keys to prevent Cypher injection.
    for k in properties:
        if not _SAFE_PROPERTY_KEY.match(k):
            raise ValueError(f"Unsafe Cypher property key: {k!r}")
    set_clauses = ", ".join(f"e.{k} = ${k}" for k in properties)
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        session.run(
            f"MATCH (e:Entity {{name: $name}}) SET {set_clauses}",
            name=name,
            **properties,
        )


def _upsert_relationship_sync(
    head_name: str,
    tail_name: str,
    rel_type: str,
    confidence: float,
    context_snippet: str = "",
) -> None:
    """Create or update a relationship between two entities."""
    driver = get_driver()
    now = datetime.now(timezone.utc).isoformat()

    # Sanitize relationship type against whitelist
    label = rel_type.strip().upper().replace(" ", "_")
    if label not in VALID_REL_TYPES:
        try:
            from memory.graph_suggestions import log_relationship_fallback
            log_relationship_fallback(head_name, tail_name, rel_type, confidence, context_snippet)
        except Exception:
            log.debug("graph suggestion fallback logging failed", exc_info=True)
        label = "RELATED_TO"

    snippet = context_snippet[:200]

    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        # Dynamic relationship type requires a workaround since Cypher
        # doesn't allow parameterized relationship types.
        # Validated against whitelist above, so f-string is safe.
        #
        # audit_status lifecycle on re-mention:
        #   'quarantined' → preserved (edge stays flagged until manually reviewed)
        #   'verified'    → preserved (model-confirmed edges stay trusted)
        #   'auto_fixed'  → reset to null (re-mention provides new evidence;
        #                    if the fix was correct, the new type is compatible
        #                    and won't be re-flagged)
        result = session.run(
            f"""MATCH (h:Entity {{name: $head}})
                MATCH (t:Entity {{name: $tail}})
                MERGE (h)-[r:{label}]->(t)
                ON CREATE SET
                    r.strength = $confidence,
                    r.mention_count = 1,
                    r.first_mentioned = $now,
                    r.last_mentioned = $now,
                    r.context_snippets = [$snippet]
                ON MATCH SET
                    r.mention_count = r.mention_count + 1,
                    r.last_mentioned = $now,
                    r.strength = CASE
                        WHEN $confidence > r.strength THEN $confidence
                        ELSE r.strength END,
                    r.context_snippets = CASE
                        WHEN size(r.context_snippets) >= 3
                        THEN r.context_snippets[1..] + [$snippet]
                        ELSE r.context_snippets + [$snippet]
                    END,
                    r.audit_status = CASE WHEN r.audit_status IN ['quarantined', 'verified'] THEN r.audit_status ELSE null END
                RETURN r.mention_count AS mention_count""",
            head=head_name,
            tail=tail_name,
            now=now,
            confidence=confidence,
            snippet=snippet,
        )

        # Log RELATED_TO hotspots when mention count crosses threshold
        if label == "RELATED_TO":
            try:
                record = result.single()
                mention_count = record["mention_count"] if record else 0
                if mention_count == 3:  # Fires once at threshold; nightly get_related_to_hotspots() catches persistent hotspots
                    from memory.graph_suggestions import log_repeated_related_to
                    log_repeated_related_to(head_name, tail_name, mention_count)
            except Exception:
                log.debug("graph suggestion hotspot logging failed", exc_info=True)


@track_latency("neo4j")
def upsert_relationship_sync(
    head_name: str,
    tail_name: str,
    rel_type: str,
    confidence: float,
    context_snippet: str = "",
) -> None:
    _upsert_relationship_sync(
        head_name=head_name,
        tail_name=tail_name,
        rel_type=rel_type,
        confidence=confidence,
        context_snippet=context_snippet,
    )


@track_latency("neo4j")
async def upsert_relationship(
    head_name: str,
    tail_name: str,
    rel_type: str,
    confidence: float,
    context_snippet: str = "",
) -> None:
    async with _get_graph_write_lock():
        await asyncio.to_thread(
            _upsert_relationship_sync,
            head_name,
            tail_name,
            rel_type,
            confidence,
            context_snippet,
        )


def _create_episode_sync(
    content_preview: str,
    source: str,
    entity_names: list[str],
) -> str:
    """Create an Episode node linked to its extracted entities."""
    driver = get_driver()
    episode_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        session.run(
            """CREATE (ep:Episode {
                   id: $id,
                   content_preview: $preview,
                   created_at: $now,
                   source: $source,
                   entities_extracted: $entities
               })""",
            id=episode_id,
            preview=content_preview[:300],
            now=now,
            source=source,
            entities=entity_names,
        )

        # Link episode to entities
        for name in entity_names:
            session.run(
                """MATCH (ep:Episode {id: $eid})
                   MATCH (e:Entity {name: $name})
                   MERGE (ep)-[:MENTIONS]->(e)""",
                eid=episode_id,
                name=name,
            )

    return episode_id


@track_latency("neo4j")
def create_episode_sync(
    content_preview: str,
    source: str,
    entity_names: list[str],
) -> str:
    return _create_episode_sync(content_preview, source, entity_names)


@track_latency("neo4j")
async def create_episode(
    content_preview: str,
    source: str,
    entity_names: list[str],
) -> str:
    async with _get_graph_write_lock():
        return await asyncio.to_thread(_create_episode_sync, content_preview, source, entity_names)


# --- Retrieval ---


@track_latency("neo4j")
def query_entity(name: str) -> dict[str, Any] | None:
    """Look up an entity and its relationships for system prompt injection."""
    driver = get_driver()
    normalized = _normalize(name)

    with driver.session() as session:
        # Find entity
        result = session.run(
            """MATCH (e:Entity)
               WHERE toLower(e.name) = $name
                  OR ANY(a IN e.aliases WHERE toLower(a) = $name)
               RETURN e""",
            name=normalized,
        )
        record = result.single()
        if not record:
            return None

        raw = dict(record["e"])
        # Strip PII properties that shouldn't flow into LLM context
        _PII_FIELDS = {"phone", "email"}
        entity = {k: v for k, v in raw.items() if k not in _PII_FIELDS}

        # Get outgoing relationships
        rels_out = session.run(
            """MATCH (e:Entity)-[r]->(t:Entity)
               WHERE toLower(e.name) = $name
                  OR ANY(a IN e.aliases WHERE toLower(a) = $name)
               RETURN type(r) AS rel_type, properties(r) AS props, t.name AS target""",
            name=normalized,
        )
        # Get incoming relationships
        rels_in = session.run(
            """MATCH (s:Entity)-[r]->(e:Entity)
               WHERE toLower(e.name) = $name
                  OR ANY(a IN e.aliases WHERE toLower(a) = $name)
               RETURN type(r) AS rel_type, properties(r) AS props, s.name AS source""",
            name=normalized,
        )

        relationships = []
        for rec in rels_out:
            relationships.append({
                "type": rec["rel_type"],
                "target": rec["target"],
                "direction": "outgoing",
                **rec["props"],
            })
        for rec in rels_in:
            relationships.append({
                "type": rec["rel_type"],
                "source": rec["source"],
                "direction": "incoming",
                **rec["props"],
            })

        entity["relationships"] = relationships
        return entity


@track_latency("neo4j")
def query_entities_for_context(entity_names: list[str]) -> str:
    """Query Neo4j for multiple entities and format for system prompt injection.

    Uses batched Cypher queries with UNWIND to minimize round-trips.
    Returns formatted string or empty string if no graph context found.
    """
    if not entity_names:
        return ""

    driver = get_driver()
    normalized_names = [_normalize(name) for name in entity_names]

    with driver.session() as session:
        # Batch query: find all matching entities in one round-trip
        entity_result = session.run(
            """UNWIND $names AS lookup_name
               MATCH (e:Entity)
               WHERE toLower(e.name) = lookup_name
                  OR ANY(a IN e.aliases WHERE toLower(a) = lookup_name)
               RETURN DISTINCT e.name AS name, e.entity_type AS entity_type,
                      e.mention_count AS mention_count,
                      e.last_mentioned AS last_mentioned""",
            names=normalized_names,
        )
        entities_by_name: dict[str, dict] = {}
        for rec in entity_result:
            name = rec["name"]
            if name not in entities_by_name:
                entities_by_name[name] = {
                    "name": name,
                    "entity_type": rec["entity_type"],
                    "mention_count": rec["mention_count"],
                    "last_mentioned": rec["last_mentioned"] or "",
                    "relationships": [],
                }

        if not entities_by_name:
            return ""

        found_names = list(entities_by_name.keys())

        # Batch query: all outgoing relationships for found entities
        rels_out = session.run(
            """UNWIND $names AS ename
               MATCH (e:Entity {name: ename})-[r]->(t:Entity)
               RETURN e.name AS entity_name, type(r) AS rel_type, t.name AS target,
                      r.audit_status AS audit_status""",
            names=found_names,
        )
        for rec in rels_out:
            ent = entities_by_name.get(rec["entity_name"])
            if ent:
                ent["relationships"].append({
                    "type": rec["rel_type"],
                    "target": rec["target"],
                    "direction": "outgoing",
                    "audit_status": rec["audit_status"],
                })

        # Batch query: all incoming relationships for found entities
        rels_in = session.run(
            """UNWIND $names AS ename
               MATCH (s:Entity)-[r]->(e:Entity {name: ename})
               RETURN e.name AS entity_name, type(r) AS rel_type, s.name AS source,
                      r.audit_status AS audit_status""",
            names=found_names,
        )
        for rec in rels_in:
            ent = entities_by_name.get(rec["entity_name"])
            if ent:
                ent["relationships"].append({
                    "type": rec["rel_type"],
                    "source": rec["source"],
                    "direction": "incoming",
                    "audit_status": rec["audit_status"],
                })

    # Format output
    results = list(entities_by_name.values())
    lines = ["<!-- Memory Context (knowledge graph) -->"]
    lines.append("Known entities and relationships:\n")

    for ent in results:
        etype = ent.get("entity_type", "Unknown")
        mentions = ent.get("mention_count", 0)
        last = str(ent.get("last_mentioned", ""))[:10]
        lines.append(f"- **{ent['name']}** ({etype}, {mentions} mentions, last: {last})")

        for rel in ent.get("relationships", []):
            rtype = rel["type"].replace("_", " ").lower()
            flag = " [unverified]" if rel.get("audit_status") == "quarantined" else ""
            if rel["direction"] == "outgoing":
                lines.append(f"  → {rtype} {rel['target']}{flag}")
            else:
                lines.append(f"  ← {rel['source']} {rtype}{flag}")

    return "\n".join(lines)


# --- Deletion (for /forget) ---


@track_latency("neo4j")
def delete_entity(name: str) -> bool:
    """Delete an entity and all its relationships. Returns True if found."""
    driver = get_driver()
    normalized = _normalize(name)

    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        result = session.run(
            """MATCH (e:Entity)
               WHERE toLower(e.name) = $name
                  OR ANY(a IN e.aliases WHERE toLower(a) = $name)
               DETACH DELETE e
               RETURN count(e) AS deleted""",
            name=normalized,
        )
        record = result.single()
        return record and record["deleted"] > 0


# --- Maintenance ---


def _run_strength_decay_sync() -> int:
    """Recalculate strength for all entities based on mentions * recency decay.

    Returns the number of entities updated.
    """
    driver = get_driver()
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE e.last_mentioned IS NOT NULL
            WITH e,
                 e.mention_count AS mentions,
                 duration.between(datetime(e.last_mentioned), datetime()).days AS days_since
            SET e.strength = mentions * exp(-0.03 * days_since)
            RETURN count(e) AS updated
            """
        )
        updated = result.single()["updated"]
        log.info("Strength decay: updated %d entities", updated)
        return updated


def run_strength_decay_sync() -> int:
    return _run_strength_decay_sync()


async def run_strength_decay() -> int:
    async with _get_graph_write_lock():
        return await asyncio.to_thread(_run_strength_decay_sync)


def _delete_orphan_entities_sync() -> int:
    """Delete entities with zero relationships (no MENTIONS from episodes either counts).

    Returns the number of entities deleted.
    """
    driver = get_driver()
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE NOT (e)--()
            DELETE e
            RETURN count(e) AS deleted
            """
        )
        deleted = result.single()["deleted"]
        log.info("Deleted %d orphan entities", deleted)
        return deleted


def delete_orphan_entities_sync() -> int:
    return _delete_orphan_entities_sync()


async def delete_orphan_entities() -> int:
    async with _get_graph_write_lock():
        return await asyncio.to_thread(_delete_orphan_entities_sync)


def delete_self_referencing_rels() -> int:
    """Delete relationships where head and tail are the same entity.

    Returns the number of relationships deleted.
    """
    driver = get_driver()
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)-[r]->(e)
            DELETE r
            RETURN count(r) AS deleted
            """
        )
        deleted = result.single()["deleted"]
        log.info("Deleted %d self-referencing relationships", deleted)
        return deleted


def delete_blocklisted_entities(blocklist: set[str]) -> int:
    """Delete entities whose names match the blocklist (case-insensitive).

    Returns the number of entities deleted.
    """
    driver = get_driver()
    names_lower = [n.lower() for n in blocklist]
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE toLower(e.name) IN $names
            DETACH DELETE e
            RETURN count(e) AS deleted
            """,
            names=names_lower,
        )
        deleted = result.single()["deleted"]
        log.info("Deleted %d blocklisted entities", deleted)
        return deleted


# --- Audit helpers ---


def get_relationships_for_audit(limit: int = 500) -> list[dict]:
    """Return relationships with full context for audit (ordered by strength ASC)."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (h:Entity)-[r]->(t:Entity)
            RETURN h.name AS head, h.entity_type AS head_type,
                   t.name AS tail, t.entity_type AS tail_type,
                   type(r) AS rel_type,
                   r.strength AS strength,
                   r.mention_count AS mention_count,
                   r.context_snippets AS context_snippets,
                   r.audit_status AS audit_status,
                   r.first_mentioned AS first_mentioned,
                   r.last_mentioned AS last_mentioned
            ORDER BY r.strength ASC
            LIMIT $limit
            """,
            limit=limit,
        )
        return [dict(rec) for rec in result]


def get_relationship_type_distribution() -> dict[str, int]:
    """Return count of relationships by type."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS cnt
            ORDER BY cnt DESC
            """
        )
        return {rec["rel_type"]: rec["cnt"] for rec in result}


def set_relationship_audit_status(
    head: str, tail: str, rel_type: str, status: str,
) -> None:
    """Set audit_status on a specific edge.

    Status: 'verified' | 'quarantined' | 'auto_fixed'
    """
    driver = get_driver()
    if rel_type not in VALID_REL_TYPES:
        return
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        session.run(
            f"""MATCH (h:Entity {{name: $head}})-[r:{rel_type}]->(t:Entity {{name: $tail}})
                SET r.audit_status = $status""",
            head=head,
            tail=tail,
            status=status,
        )


def reclassify_relationship(
    head: str,
    tail: str,
    old_type: str,
    new_type: str,
    strength: float,
    mention_count: int,
    context_snippets: list[str] | None = None,
    first_mentioned: str | None = None,
) -> None:
    """Delete old rel, merge new one with corrected type + audit_status='auto_fixed'.

    Uses MERGE (not CREATE) for the new edge so that repeated reclassifications
    of the same entity pair merge into an existing edge rather than creating
    duplicates.  Preserves strength, mention_count, context_snippets, and
    timestamps.  Uses an explicit transaction so delete + merge are atomic.
    """
    driver = get_driver()
    if old_type not in VALID_REL_TYPES or new_type not in VALID_REL_TYPES:
        return
    now = datetime.now(timezone.utc).isoformat()
    snippets = context_snippets or []
    first_ts = first_mentioned or now
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        with session.begin_transaction() as tx:
            tx.run(
                f"""MATCH (h:Entity {{name: $head}})-[r:{old_type}]->(t:Entity {{name: $tail}})
                    DELETE r""",
                head=head,
                tail=tail,
            )
            tx.run(
                f"""MATCH (h:Entity {{name: $head}})
                    MATCH (t:Entity {{name: $tail}})
                    MERGE (h)-[r:{new_type}]->(t)
                    ON CREATE SET
                        r.strength = $strength,
                        r.mention_count = $mention_count,
                        r.context_snippets = $snippets,
                        r.first_mentioned = $first_mentioned,
                        r.last_mentioned = $now,
                        r.audit_status = 'auto_fixed'
                    ON MATCH SET
                        r.strength = CASE WHEN $strength > r.strength THEN $strength ELSE r.strength END,
                        r.mention_count = r.mention_count + $mention_count,
                        r.context_snippets = CASE
                            WHEN size(r.context_snippets) >= 3
                            THEN r.context_snippets[1..] + $snippets
                            ELSE r.context_snippets + $snippets
                        END,
                        r.last_mentioned = $now,
                        r.audit_status = 'auto_fixed'""",
                head=head,
                tail=tail,
                strength=strength,
                mention_count=mention_count,
                snippets=snippets,
                first_mentioned=first_ts,
                now=now,
            )
            tx.commit()


def delete_specific_relationship(head: str, tail: str, rel_type: str) -> bool:
    """Delete a specific edge. Returns True if deleted."""
    driver = get_driver()
    if rel_type not in VALID_REL_TYPES:
        return False
    with _GRAPH_SYNC_WRITE_LOCK, driver.session() as session:
        result = session.run(
            f"""MATCH (h:Entity {{name: $head}})-[r:{rel_type}]->(t:Entity {{name: $tail}})
                DELETE r
                RETURN count(r) AS deleted""",
            head=head,
            tail=tail,
        )
        record = result.single()
        return record is not None and record["deleted"] > 0


# --- Stats ---


def entity_count() -> int:
    driver = get_driver()
    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN count(e) AS c")
        return result.single()["c"]


def relationship_count() -> int:
    driver = get_driver()
    with driver.session() as session:
        result = session.run("MATCH ()-[r]->() RETURN count(r) AS c")
        return result.single()["c"]


@track_latency("neo4j")
def get_graph_summary() -> dict[str, Any]:
    """Return overall graph stats: counts, top entities by connections, most recent."""
    driver = get_driver()
    with driver.session() as session:
        e_count = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
        r_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

        # Top 10 entities by connection count (in + out)
        top_connected = session.run(
            """MATCH (e:Entity)
               OPTIONAL MATCH (e)-[r]-()
               WITH e, count(r) AS connections
               ORDER BY connections DESC
               LIMIT 10
               RETURN e.name AS name, e.entity_type AS type,
                      e.mention_count AS mentions, connections""",
        )
        top = [dict(rec) for rec in top_connected]

        # 5 most recently added entities
        recent = session.run(
            """MATCH (e:Entity)
               WHERE e.first_mentioned IS NOT NULL
               RETURN e.name AS name, e.entity_type AS type,
                      e.first_mentioned AS added
               ORDER BY e.first_mentioned DESC
               LIMIT 5""",
        )
        recent_list = [dict(rec) for rec in recent]

    return {
        "entity_count": e_count,
        "relationship_count": r_count,
        "top_connected": top,
        "recent": recent_list,
    }


@track_latency("neo4j")
def get_top_entities(limit: int = 20) -> list[dict[str, Any]]:
    """Return top entities ordered by strength (mentions * recency).

    Used by triage to build context about what Brian tracks.
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE e.strength IS NOT NULL
            RETURN e.name AS name, e.entity_type AS type,
                   e.strength AS strength, e.mention_count AS mentions
            ORDER BY e.strength DESC
            LIMIT $limit
            """,
            limit=limit,
        )
        return [dict(record) for record in result]
