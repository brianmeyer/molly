"""One-time migration: Fix misclassified educational relationships in Neo4j.

Issues found:
- Three duplicate Kellogg entities: "Kellogg EMBA program", "Kellogg EMBA", "Kellogg"
- Sam --WORKS_AT/WORKS_ON--> Kellogg EMBA program (should be CLASSMATE_OF)
- Brian --WORKS_AT--> Kellogg EMBA (should be ALUMNI_OF — he graduated Dec 2025)
- Delphine --USES--> Kellogg (should be CLASSMATE_OF)
- Sushetra has no Kellogg relationship at all
- Molly --CREATED--> Kellogg EMBA program (wrong)
- User --INTERESTED_IN--> Kellogg EMBA program (wrong — User entity is redundant with Brian)
- Self-referencing relationships: Sam→Sam, Delphine→Delphine

Run: cd ~/molly && source .venv/bin/activate && python migrations/001_fix_educational_relationships.py
"""

import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from memory.graph import get_driver

CANONICAL_NAME = "Kellogg EMBA"
MERGE_NAMES = ["Kellogg EMBA program", "Kellogg"]
NOW = datetime.now(timezone.utc).isoformat()

changes = []


def log_change(action: str, detail: str):
    changes.append(f"- {action}: {detail}")
    print(f"  {action}: {detail}")


def run_migration():
    driver = get_driver()

    with driver.session() as session:
        # ------------------------------------------------------------------
        # Step 1: Merge Kellogg entities into "Kellogg EMBA"
        # ------------------------------------------------------------------
        print("\n=== Step 1: Merge Kellogg entities ===")

        # Ensure canonical entity exists
        result = session.run(
            "MATCH (e:Entity {name: $name}) RETURN e",
            name=CANONICAL_NAME,
        )
        if not result.single():
            print(f"  Canonical entity '{CANONICAL_NAME}' not found — creating")
            session.run(
                """CREATE (e:Entity {
                       name: $name, entity_type: 'Organization',
                       mention_count: 0, first_mentioned: $now,
                       last_mentioned: $now, strength: 1.0,
                       confidence: 0.8, aliases: [], summary: ''
                   })""",
                name=CANONICAL_NAME, now=NOW,
            )
            log_change("CREATE", f"Created canonical entity '{CANONICAL_NAME}'")

        for old_name in MERGE_NAMES:
            result = session.run(
                "MATCH (e:Entity {name: $name}) RETURN e.mention_count AS mc, e.aliases AS aliases",
                name=old_name,
            )
            record = result.single()
            if not record:
                print(f"  '{old_name}' not found, skipping")
                continue

            old_mentions = record["mc"] or 0
            old_aliases = record["aliases"] or []

            # Transfer mention count and add as alias
            session.run(
                """MATCH (canonical:Entity {name: $canonical})
                   SET canonical.mention_count = canonical.mention_count + $mc,
                       canonical.aliases = CASE
                           WHEN NOT $alias IN canonical.aliases
                           THEN canonical.aliases + $alias
                           ELSE canonical.aliases END""",
                canonical=CANONICAL_NAME, mc=old_mentions, alias=old_name,
            )

            # Transfer any additional aliases
            for alias in old_aliases:
                session.run(
                    """MATCH (canonical:Entity {name: $canonical})
                       SET canonical.aliases = CASE
                           WHEN NOT $alias IN canonical.aliases
                           THEN canonical.aliases + $alias
                           ELSE canonical.aliases END""",
                    canonical=CANONICAL_NAME, alias=alias,
                )

            # Move all incoming relationships from old to canonical
            # (except MENTIONS from Episodes — we'll re-link those)
            result = session.run(
                """MATCH (source)-[r]->(old:Entity {name: $old})
                   WHERE NOT type(r) = 'MENTIONS'
                   RETURN source.name AS src, type(r) AS rtype,
                          properties(r) AS props, id(r) AS rid""",
                old=old_name,
            )
            for rec in result:
                # We'll handle these relationships in later steps
                log_change("NOTED", f"Incoming rel {rec['src']} --[{rec['rtype']}]--> {old_name}")

            # Move all outgoing relationships from old to canonical
            result = session.run(
                """MATCH (old:Entity {name: $old})-[r]->(target)
                   RETURN target.name AS tgt, type(r) AS rtype,
                          properties(r) AS props""",
                old=old_name,
            )
            for rec in result:
                rtype = rec["rtype"]
                tgt = rec["tgt"]
                props = rec["props"]

                # Check if canonical already has this relationship
                existing = session.run(
                    f"""MATCH (c:Entity {{name: $canonical}})-[r:{rtype}]->(t:Entity {{name: $target}})
                        RETURN r""",
                    canonical=CANONICAL_NAME, target=tgt,
                )
                if not existing.single():
                    # Create the relationship on canonical
                    session.run(
                        f"""MATCH (c:Entity {{name: $canonical}})
                            MATCH (t:Entity {{name: $target}})
                            CREATE (c)-[r:{rtype}]->(t)
                            SET r = $props""",
                        canonical=CANONICAL_NAME, target=tgt, props=props,
                    )
                    log_change("MOVE_REL", f"{CANONICAL_NAME} --[{rtype}]--> {tgt} (from {old_name})")

            # Re-link Episode MENTIONS
            session.run(
                """MATCH (ep:Episode)-[r:MENTIONS]->(old:Entity {name: $old})
                   MATCH (canonical:Entity {name: $canonical})
                   MERGE (ep)-[:MENTIONS]->(canonical)
                   DELETE r""",
                old=old_name, canonical=CANONICAL_NAME,
            )

            # Delete old entity and all remaining relationships
            session.run(
                "MATCH (e:Entity {name: $name}) DETACH DELETE e",
                name=old_name,
            )
            log_change("MERGE", f"Merged '{old_name}' into '{CANONICAL_NAME}'")

        # ------------------------------------------------------------------
        # Step 2: Delete self-referencing relationships
        # ------------------------------------------------------------------
        print("\n=== Step 2: Delete self-referencing relationships ===")

        result = session.run(
            """MATCH (e:Entity)-[r]->(e)
               RETURN e.name AS name, type(r) AS rtype"""
        )
        self_refs = list(result)
        for rec in self_refs:
            log_change("DELETE_SELF_REF", f"{rec['name']} --[{rec['rtype']}]--> {rec['name']}")

        session.run("MATCH (e:Entity)-[r]->(e) DELETE r")

        # ------------------------------------------------------------------
        # Step 3: Delete wrong relationships to Kellogg EMBA
        # ------------------------------------------------------------------
        print("\n=== Step 3: Fix relationships to Kellogg EMBA ===")

        # Delete Sam --WORKS_AT--> Kellogg EMBA
        result = session.run(
            """MATCH (s:Entity {name: 'Sam'})-[r:WORKS_AT]->(k:Entity {name: $kellogg})
               DELETE r RETURN count(r) AS c""",
            kellogg=CANONICAL_NAME,
        )
        if result.single()["c"] > 0:
            log_change("DELETE", f"Sam --[WORKS_AT]--> {CANONICAL_NAME}")

        # Delete Sam --WORKS_ON--> Kellogg EMBA
        result = session.run(
            """MATCH (s:Entity {name: 'Sam'})-[r:WORKS_ON]->(k:Entity {name: $kellogg})
               DELETE r RETURN count(r) AS c""",
            kellogg=CANONICAL_NAME,
        )
        if result.single()["c"] > 0:
            log_change("DELETE", f"Sam --[WORKS_ON]--> {CANONICAL_NAME}")

        # Delete Delphine --USES--> Kellogg EMBA
        result = session.run(
            """MATCH (d:Entity {name: 'Delphine'})-[r:USES]->(k:Entity {name: $kellogg})
               DELETE r RETURN count(r) AS c""",
            kellogg=CANONICAL_NAME,
        )
        if result.single()["c"] > 0:
            log_change("DELETE", f"Delphine --[USES]--> {CANONICAL_NAME}")

        # Delete Brian --WORKS_AT--> Kellogg EMBA
        result = session.run(
            """MATCH (b:Entity {name: 'Brian'})-[r:WORKS_AT]->(k:Entity {name: $kellogg})
               DELETE r RETURN count(r) AS c""",
            kellogg=CANONICAL_NAME,
        )
        if result.single()["c"] > 0:
            log_change("DELETE", f"Brian --[WORKS_AT]--> {CANONICAL_NAME}")

        # Delete Molly --CREATED--> Kellogg EMBA
        result = session.run(
            """MATCH (m:Entity {name: 'Molly'})-[r:CREATED]->(k:Entity {name: $kellogg})
               DELETE r RETURN count(r) AS c""",
            kellogg=CANONICAL_NAME,
        )
        if result.single()["c"] > 0:
            log_change("DELETE", f"Molly --[CREATED]--> {CANONICAL_NAME}")

        # Delete User --INTERESTED_IN--> Kellogg EMBA
        result = session.run(
            """MATCH (u:Entity {name: 'User'})-[r:INTERESTED_IN]->(k:Entity {name: $kellogg})
               DELETE r RETURN count(r) AS c""",
            kellogg=CANONICAL_NAME,
        )
        if result.single()["c"] > 0:
            log_change("DELETE", f"User --[INTERESTED_IN]--> {CANONICAL_NAME}")

        # ------------------------------------------------------------------
        # Step 4: Create correct relationships
        # ------------------------------------------------------------------
        print("\n=== Step 4: Create correct relationships ===")

        snippet = "Classmate from Kellogg EMBA program in Miami (migrated)"

        # Brian --ALUMNI_OF--> Kellogg EMBA (graduated Dec 2025)
        session.run(
            f"""MATCH (b:Entity {{name: 'Brian'}})
                MATCH (k:Entity {{name: $kellogg}})
                MERGE (b)-[r:ALUMNI_OF]->(k)
                ON CREATE SET r.strength = 0.8, r.mention_count = 1,
                    r.first_mentioned = $now, r.last_mentioned = $now,
                    r.context_snippets = [$snippet]
                ON MATCH SET r.mention_count = r.mention_count + 1,
                    r.last_mentioned = $now""",
            kellogg=CANONICAL_NAME, now=NOW, snippet=snippet,
        )
        log_change("CREATE", f"Brian --[ALUMNI_OF]--> {CANONICAL_NAME}")

        # Sam, Sushetra, Delphine --CLASSMATE_OF--> Kellogg EMBA
        for person in ["Sam", "Sushetra", "Delphine"]:
            result = session.run(
                "MATCH (e:Entity {name: $name}) RETURN e", name=person,
            )
            if not result.single():
                print(f"  Warning: entity '{person}' not found")
                continue

            session.run(
                f"""MATCH (p:Entity {{name: $person}})
                    MATCH (k:Entity {{name: $kellogg}})
                    MERGE (p)-[r:CLASSMATE_OF]->(k)
                    ON CREATE SET r.strength = 0.8, r.mention_count = 1,
                        r.first_mentioned = $now, r.last_mentioned = $now,
                        r.context_snippets = [$snippet]
                    ON MATCH SET r.mention_count = r.mention_count + 1,
                        r.last_mentioned = $now""",
                person=person, kellogg=CANONICAL_NAME, now=NOW, snippet=snippet,
            )
            log_change("CREATE", f"{person} --[CLASSMATE_OF]--> {CANONICAL_NAME}")

        # ------------------------------------------------------------------
        # Step 5: Catch-all — fix any remaining WORKS_AT/USES to educational entities
        # ------------------------------------------------------------------
        print("\n=== Step 5: Scan for other educational misclassifications ===")

        result = session.run(
            """MATCH (p:Entity)-[r]->(e:Entity)
               WHERE e.entity_type = 'Organization'
                 AND (toLower(e.name) CONTAINS 'university' OR
                      toLower(e.name) CONTAINS 'college' OR
                      toLower(e.name) CONTAINS 'school' OR
                      toLower(e.name) CONTAINS 'institute' OR
                      toLower(e.name) CONTAINS 'mba' OR
                      toLower(e.name) CONTAINS 'emba' OR
                      toLower(e.name) CONTAINS 'program')
                 AND type(r) IN ['WORKS_AT', 'USES', 'WORKS_ON']
               RETURN p.name AS src, type(r) AS rtype, e.name AS tgt,
                      r.context_snippets AS snippets"""
        )
        remaining = list(result)
        if remaining:
            for rec in remaining:
                log_change("REMAINING", f"{rec['src']} --[{rec['rtype']}]--> {rec['tgt']} (needs manual review)")
        else:
            print("  No remaining misclassifications found")

    # ------------------------------------------------------------------
    # Step 6: Verify final state
    # ------------------------------------------------------------------
    print("\n=== Verification ===")
    with driver.session() as session:
        result = session.run(
            """MATCH (p:Entity)-[r]->(k:Entity {name: $kellogg})
               RETURN p.name AS src, type(r) AS rtype, k.name AS tgt""",
            kellogg=CANONICAL_NAME,
        )
        print(f"  Relationships to {CANONICAL_NAME}:")
        for rec in result:
            print(f"    {rec['src']} --[{rec['rtype']}]--> {rec['tgt']}")

        result = session.run(
            "MATCH (e:Entity {name: $name}) RETURN e.aliases AS aliases, e.mention_count AS mc",
            name=CANONICAL_NAME,
        )
        rec = result.single()
        if rec:
            print(f"  Aliases: {rec['aliases']}")
            print(f"  Mention count: {rec['mc']}")

    # ------------------------------------------------------------------
    # Write to daily maintenance log
    # ------------------------------------------------------------------
    today = date.today().isoformat()
    log_path = config.WORKSPACE / "memory" / "maintenance" / f"{today}.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_content = f"## Migration: Fix Educational Relationships\n\n"
    log_content += f"**Run at:** {NOW}\n\n"
    log_content += "### Changes\n\n"
    log_content += "\n".join(changes) + "\n"

    # Append to existing maintenance log or create new one
    if log_path.exists():
        with open(log_path, "a") as f:
            f.write("\n\n" + log_content)
    else:
        log_path.write_text(log_content)

    print(f"\n  Migration log written to {log_path}")
    print(f"  Total changes: {len(changes)}")


if __name__ == "__main__":
    print("Running migration: Fix educational relationships")
    print("=" * 60)
    run_migration()
    print("\nDone.")
