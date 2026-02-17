"""Graph and memory operations service for SelfImprovementEngine.

Methods for Neo4j knowledge graph maintenance:
- Entity consolidation via fuzzy matching
- Stale entity detection
- Contradiction detection (e.g., multiple WORKS_AT)
- Community detection (entity type distribution)
- Memory markdown generation from graph snapshots
- User fact extraction from message history
- Memory optimization orchestration (nightly)
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import config
import db_pool

log = logging.getLogger(__name__)



class GraphOpsService:
    """Graph/memory operations service.

    Receives explicit ``OwnerCommsService`` dependency instead of calling
    ``self._log_improvement_event()`` through MRO.
    """

    def __init__(self, ctx, comms):
        from evolution.context import EngineContext
        from evolution.owner_comms import OwnerCommsService
        self.ctx: EngineContext = ctx
        self.comms: OwnerCommsService = comms

    def run_memory_optimization_sync(self) -> dict[str, Any]:
        from memory.graph import get_driver, run_strength_decay_sync

        decayed = run_strength_decay_sync()
        consolidated = self.consolidate_entities()
        stale = self.stale_entities(days=60)
        contradictions = self.detect_contradictions()
        community = self.community_detection()

        rel_decay = 0
        try:
            driver = get_driver()
            with driver.session() as session:
                rec = session.run(
                    """
                    MATCH ()-[r]->()
                    WHERE r.last_mentioned IS NOT NULL
                    WITH r, duration.between(datetime(r.last_mentioned), datetime()).days AS age_days
                    SET r.strength = CASE
                        WHEN r.strength IS NULL THEN exp(-0.02 * age_days)
                        ELSE r.strength * exp(-0.02 * age_days)
                    END
                    RETURN count(r) AS c
                    """
                ).single()
                rel_decay = int(rec["c"]) if rec else 0
        except Exception:
            log.debug("Relationship decay failed", exc_info=True)

        results = {
            "strength_decay": decayed,
            "entity_consolidations": consolidated,
            "stale_entities": len(stale),
            "contradictions": len(contradictions),
            "contradiction_samples": contradictions[:10],
            "relationship_decay": rel_decay,
            "communities": community,
        }

        self.comms.log_improvement_event(
            event_type="maintenance",
            category="memory",
            title="Nightly memory optimization",
            payload=json.dumps(results, ensure_ascii=True),
            status="completed",
        )
        return results

    def consolidate_entities(self) -> int:
        from difflib import SequenceMatcher
        from memory.graph import get_driver

        driver = get_driver()
        with driver.session() as session:
            rows = session.run(
                "MATCH (e:Entity) RETURN e.name AS name, e.entity_type AS t ORDER BY e.entity_type, e.name"
            )
            entities = [dict(r) for r in rows]

        by_type: dict[str, list[str]] = {}
        for row in entities:
            by_type.setdefault(row.get("t") or "Unknown", []).append(row.get("name") or "")

        merged = 0
        for etype, names in by_type.items():
            for idx, left in enumerate(names):
                if not left:
                    continue
                for right in names[idx + 1:]:
                    if not right:
                        continue
                    score = SequenceMatcher(None, left.lower(), right.lower()).ratio()
                    if score < 0.9 or left.lower() == right.lower():
                        continue
                    keep, drop = sorted([left, right], key=lambda s: (len(s), s))
                    with driver.session() as session:
                        session.run(
                            """
                            MATCH (keep:Entity {name: $keep}), (drop:Entity {name: $drop})
                            SET keep.mention_count = coalesce(keep.mention_count, 0) + coalesce(drop.mention_count, 0),
                                keep.aliases = coalesce(keep.aliases, []) + coalesce(drop.aliases, []) + [$drop]
                            WITH keep, drop
                            DETACH DELETE drop
                            """,
                            keep=keep,
                            drop=drop,
                        )
                    merged += 1
                    if merged >= 30:
                        return merged
        return merged

    def stale_entities(self, days: int = 60) -> list[dict[str, Any]]:
        from memory.graph import get_driver

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        driver = get_driver()
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (e:Entity)
                WHERE e.last_mentioned IS NOT NULL AND e.last_mentioned < $cutoff
                RETURN e.name AS name, e.entity_type AS type, e.last_mentioned AS last_mentioned
                ORDER BY e.last_mentioned ASC
                LIMIT 100
                """,
                cutoff=cutoff,
            )
            return [dict(r) for r in rows]

    def detect_contradictions(self) -> list[dict[str, Any]]:
        from memory.graph import get_driver

        driver = get_driver()
        contradictions: list[dict[str, Any]] = []
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (p:Entity {entity_type: 'Person'})-[:WORKS_AT]->(o:Entity)
                WITH p, collect(DISTINCT o.name) AS orgs
                WHERE size(orgs) > 1
                RETURN p.name AS person, orgs
                LIMIT 25
                """
            )
            for row in rows:
                contradictions.append(
                    {"entity": row["person"], "type": "WORKS_AT", "values": list(row["orgs"])}
                )
        return contradictions

    def community_detection(self) -> dict[str, int]:
        from memory.graph import get_driver

        driver = get_driver()
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (e:Entity)
                RETURN e.entity_type AS type, count(e) AS c
                ORDER BY c DESC
                """
            )
            return {str(r["type"] or "Unknown"): int(r["c"]) for r in rows}

    def build_memory_md_from_graph(self, limit: int = 100) -> str:
        from memory.graph import get_driver

        driver = get_driver()
        sections: dict[str, list[str]] = {}
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (e:Entity)
                WHERE e.strength IS NOT NULL
                RETURN e.name AS name, e.entity_type AS type, e.strength AS strength, e.mention_count AS mentions
                ORDER BY e.strength DESC
                LIMIT $limit
                """,
                limit=limit,
            )
            for row in rows:
                etype = str(row["type"] or "Unknown")
                sections.setdefault(etype, []).append(
                    f"- {row['name']} (strength={float(row['strength']):.2f}, mentions={int(row['mentions'] or 0)})"
                )

        today = date.today().isoformat()
        lines = [f"# MEMORY Snapshot ({today})", ""]
        ordered = ["Person", "Project", "Organization", "Technology", "Concept", "Place"]
        seen = set()
        for key in ordered:
            values = sections.get(key)
            if not values:
                continue
            seen.add(key)
            lines.append(f"## {key}s")
            lines.extend(values)
            lines.append("")
        for key, values in sections.items():
            if key in seen:
                continue
            lines.append(f"## {key}")
            lines.extend(values)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def extract_user_facts(self, days: int = 30) -> list[str]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        owner_ids = set(config.OWNER_IDS)
        if not owner_ids:
            return []
        conn = db_pool.sqlite_connect(str(config.DATABASE_PATH))
        placeholders = ",".join("?" for _ in owner_ids)
        query = (
            f"SELECT content FROM messages WHERE timestamp > ? AND sender IN ({placeholders}) "
            "ORDER BY timestamp DESC LIMIT 400"
        )
        rows = conn.execute(query, (cutoff, *owner_ids)).fetchall()
        conn.close()
        facts: set[str] = set()
        patterns = [
            re.compile(r"\bI (?:prefer|like|love|hate)\b[^.!\n]{0,120}", re.IGNORECASE),
            re.compile(r"\bI work (?:at|for)\b[^.!\n]{0,120}", re.IGNORECASE),
            re.compile(r"\bI live in\b[^.!\n]{0,120}", re.IGNORECASE),
            re.compile(r"\bmy (?:favorite|preferred)\b[^.!\n]{0,120}", re.IGNORECASE),
        ]
        for (content,) in rows:
            text = str(content or "").strip()
            if not text:
                continue
            for pattern in patterns:
                for match in pattern.findall(text):
                    fact = re.sub(r"\s+", " ", match).strip(" .")
                    if len(fact) >= 8:
                        facts.add(fact)
        return sorted(facts)
