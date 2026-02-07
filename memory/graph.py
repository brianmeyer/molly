import logging
import math
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from neo4j import GraphDatabase

import config

log = logging.getLogger(__name__)

_driver = None

VALID_REL_TYPES = {
    "WORKS_ON", "WORKS_AT", "KNOWS", "USES", "LOCATED_IN",
    "DISCUSSED_WITH", "INTERESTED_IN", "CREATED", "MANAGES",
    "DEPENDS_ON", "RELATED_TO",
    "CLASSMATE_OF", "STUDIED_AT", "ALUMNI_OF",
    "MENTORS", "MENTORED_BY", "REPORTS_TO", "COLLABORATES_WITH",
}


# --- Driver management ---


def get_driver():
    """Lazy-initialize the Neo4j driver and create indexes."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
        with _driver.session() as session:
            session.run(
                "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)"
            )
            session.run(
                "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)"
            )
            session.run(
                "CREATE INDEX episode_id IF NOT EXISTS FOR (ep:Episode) ON (ep.id)"
            )
        log.info("Neo4j driver initialized at %s", config.NEO4J_URI)
    return _driver


def close():
    global _driver
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


def upsert_entity(
    name: str,
    entity_type: str,
    confidence: float,
) -> str:
    """Create or merge an entity. Returns the canonical entity name."""
    driver = get_driver()
    now = datetime.now(timezone.utc).isoformat()

    # Check for existing match
    existing = find_matching_entity(name, entity_type)

    with driver.session() as session:
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


def upsert_relationship(
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
        label = "RELATED_TO"

    snippet = context_snippet[:200]

    with driver.session() as session:
        # Dynamic relationship type requires a workaround since Cypher
        # doesn't allow parameterized relationship types.
        # Validated against whitelist above, so f-string is safe.
        session.run(
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
                    END""",
            head=head_name,
            tail=tail_name,
            now=now,
            confidence=confidence,
            snippet=snippet,
        )


def create_episode(
    content_preview: str,
    source: str,
    entity_names: list[str],
) -> str:
    """Create an Episode node linked to its extracted entities."""
    driver = get_driver()
    episode_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with driver.session() as session:
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


# --- Retrieval ---


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

        entity = dict(record["e"])

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


def query_entities_for_context(entity_names: list[str]) -> str:
    """Query Neo4j for multiple entities and format for system prompt injection.

    Returns formatted string or empty string if no graph context found.
    """
    if not entity_names:
        return ""

    results = []
    seen = set()

    for name in entity_names:
        entity = query_entity(name)
        if entity and entity["name"] not in seen:
            seen.add(entity["name"])
            results.append(entity)

    if not results:
        return ""

    lines = ["<!-- Memory Context (knowledge graph) -->"]
    lines.append("Known entities and relationships:\n")

    for ent in results:
        etype = ent.get("entity_type", "Unknown")
        mentions = ent.get("mention_count", 0)
        last = ent.get("last_mentioned", "")[:10]
        lines.append(f"- **{ent['name']}** ({etype}, {mentions} mentions, last: {last})")

        for rel in ent.get("relationships", []):
            rtype = rel["type"].replace("_", " ").lower()
            if rel["direction"] == "outgoing":
                lines.append(f"  → {rtype} {rel['target']}")
            else:
                lines.append(f"  ← {rel['source']} {rtype}")

    return "\n".join(lines)


# --- Deletion (for /forget) ---


def delete_entity(name: str) -> bool:
    """Delete an entity and all its relationships. Returns True if found."""
    driver = get_driver()
    normalized = _normalize(name)

    with driver.session() as session:
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
