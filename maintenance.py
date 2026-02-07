import logging
import os
import shutil
import subprocess
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import config

log = logging.getLogger(__name__)

MAINTENANCE_DIR = config.WORKSPACE / "memory" / "maintenance"
HEALTH_LOG_PATH = MAINTENANCE_DIR / "health-log.md"
MAINTENANCE_HOUR = 23  # 11 PM


def should_run_maintenance(last_run: datetime | None) -> bool:
    """Check if nightly maintenance is due (once per day at 11 PM)."""
    now = datetime.now()
    if now.hour != MAINTENANCE_HOUR:
        return False
    if last_run is None:
        return True
    # Run once per calendar day
    return last_run.date() < now.date()


# ---------------------------------------------------------------------------
# Health check (programmatic, fast)
# ---------------------------------------------------------------------------

def _count_new_chunks_since(last_check_date: str | None) -> tuple[int, int]:
    """Return (total_chunks, new_since_last_check)."""
    from memory.retriever import get_vectorstore

    vs = get_vectorstore()
    total = vs.chunk_count()

    if not last_check_date:
        return total, total

    cursor = vs.conn.execute(
        "SELECT COUNT(*) FROM conversation_chunks WHERE created_at > ?",
        (last_check_date,),
    )
    new = cursor.fetchone()[0]
    return total, new


def _graph_stats(last_check_date: str | None) -> dict:
    """Get entity/relationship counts and orphans from Neo4j."""
    from memory.graph import get_driver

    driver = get_driver()
    stats = {"total_entities": 0, "new_entities": 0, "total_rels": 0, "orphans": 0}

    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN count(e) AS c")
        stats["total_entities"] = result.single()["c"]

        result = session.run("MATCH ()-[r]->() RETURN count(r) AS c")
        stats["total_rels"] = result.single()["c"]

        # Orphaned entities (no relationships at all)
        result = session.run(
            "MATCH (e:Entity) WHERE NOT (e)--() RETURN count(e) AS c"
        )
        stats["orphans"] = result.single()["c"]

        if last_check_date:
            result = session.run(
                "MATCH (e:Entity) WHERE e.first_mentioned > $since RETURN count(e) AS c",
                since=last_check_date,
            )
            stats["new_entities"] = result.single()["c"]
        else:
            stats["new_entities"] = stats["total_entities"]

    return stats


def _extraction_stats() -> dict:
    """Get extraction success/latency from operational tables."""
    from memory.retriever import get_vectorstore

    vs = get_vectorstore()
    stats = {"success_rate": "N/A", "avg_latency_ms": "N/A"}

    try:
        # Check if tool_calls table has extraction records
        cursor = vs.conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS ok, "
            "AVG(latency_ms) AS avg_lat "
            "FROM tool_calls WHERE tool_name = 'extraction'"
        )
        row = cursor.fetchone()
        if row and row[0] > 0:
            total, ok, avg_lat = row[0], row[1], row[2]
            stats["success_rate"] = f"{ok}/{total} ({ok / total * 100:.0f}%)"
            stats["avg_latency_ms"] = f"{avg_lat:.0f}ms" if avg_lat else "N/A"
    except Exception:
        pass

    return stats


def _check_model_status(model_name: str) -> str:
    """Check if a model module can be imported / is loaded."""
    try:
        if model_name == "embedding":
            from memory.embeddings import _model
            return "loaded" if _model is not None else "not loaded (lazy)"
        elif model_name == "extractor":
            from memory.extractor import _model
            return "loaded" if _model is not None else "not loaded (lazy)"
    except Exception as e:
        return f"error: {e}"
    return "unknown"


def _check_neo4j() -> str:
    """Check Neo4j connection."""
    try:
        from memory.graph import get_driver
        driver = get_driver()
        with driver.session() as session:
            session.run("RETURN 1")
        return "connected"
    except Exception as e:
        return f"error: {e}"


def _check_sqlite() -> str:
    """Run SQLite integrity check on mollygraph.db."""
    try:
        from memory.retriever import get_vectorstore
        vs = get_vectorstore()
        cursor = vs.conn.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        return result  # "ok" if healthy
    except Exception as e:
        return f"error: {e}"


def _disk_usage() -> str:
    """Get disk usage of store/ directory."""
    store = config.STORE_DIR
    if not store.exists():
        return "0 B"
    total = sum(f.stat().st_size for f in store.rglob("*") if f.is_file())
    for unit in ["B", "KB", "MB", "GB"]:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def _ram_snapshot() -> str:
    """Get current process RSS memory usage."""
    try:
        import resource
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports in bytes, Linux in KB
        import platform
        if platform.system() == "Darwin":
            rss_mb = rss_bytes / (1024 * 1024)
        else:
            rss_mb = rss_bytes / 1024
        return f"{rss_mb:.0f} MB"
    except Exception:
        return "unavailable"


def _get_last_check_date() -> str | None:
    """Parse the most recent check date from the health log."""
    if not HEALTH_LOG_PATH.exists():
        return None
    try:
        content = HEALTH_LOG_PATH.read_text()
        # Find most recent "## Health Check: YYYY-MM-DD" header
        for line in reversed(content.splitlines()):
            if line.startswith("## Health Check: "):
                return line.split(": ", 1)[1].strip()[:10] + "T00:00:00"
    except Exception:
        pass
    return None


def _prune_health_log():
    """Remove entries older than 7 days from the health log."""
    if not HEALTH_LOG_PATH.exists():
        return

    content = HEALTH_LOG_PATH.read_text()
    lines = content.splitlines()
    cutoff = (date.today() - timedelta(days=7)).isoformat()

    kept = []
    skip = False
    for line in lines:
        if line.startswith("## Health Check: "):
            check_date = line.split(": ", 1)[1].strip()[:10]
            skip = check_date < cutoff
        if not skip:
            kept.append(line)

    new_content = "\n".join(kept)
    if new_content != content:
        HEALTH_LOG_PATH.write_text(new_content)
        log.debug("Pruned health log entries older than %s", cutoff)


def run_health_check() -> str:
    """Run the health check and return the report as a markdown string."""
    now = datetime.now(timezone.utc)
    last_check = _get_last_check_date()

    total_chunks, new_chunks = _count_new_chunks_since(last_check)
    graph = _graph_stats(last_check)
    ext = _extraction_stats()

    report = f"""## Health Check: {now.strftime('%Y-%m-%d %H:%M UTC')}

| Metric | Value |
|--------|-------|
| Total conversation chunks | {total_chunks} |
| New chunks since last check | {new_chunks} |
| Total entities | {graph['total_entities']} |
| New entities since last check | {graph['new_entities']} |
| Total relationships | {graph['total_rels']} |
| Orphaned entities (no rels) | {graph['orphans']} |
| Extraction success rate | {ext['success_rate']} |
| Avg extraction latency | {ext['avg_latency_ms']} |
| Embedding model status | {_check_model_status('embedding')} |
| Extractor model status | {_check_model_status('extractor')} |
| Neo4j connection | {_check_neo4j()} |
| SQLite integrity | {_check_sqlite()} |
| Disk usage (store/) | {_disk_usage()} |
| RAM snapshot | {_ram_snapshot()} |

"""
    return report


def write_health_check():
    """Run the health check, append to log, prune old entries."""
    MAINTENANCE_DIR.mkdir(parents=True, exist_ok=True)

    report = run_health_check()

    # Append to rolling log
    with open(HEALTH_LOG_PATH, "a") as f:
        f.write(report)

    # Prune entries older than 7 days
    _prune_health_log()

    log.info("Health check written to %s", HEALTH_LOG_PATH)


# ---------------------------------------------------------------------------
# Opus maintenance turn (agent-based reasoning session)
# ---------------------------------------------------------------------------

MAINTENANCE_SYSTEM_PROMPT = """\
You are Molly's maintenance agent. This is a nightly maintenance session at 11 PM.

You have full access to Molly's workspace at {workspace}. Your job is to analyze
the day's activity and maintain the knowledge systems.

Tasks (work through each one):

1. **Strength decay**: Read entity and relationship data from Neo4j. Recompute
   strength scores using: strength = mentions * exp(-0.03 * days_since_last_mention).
   Update entities and relationships with decayed scores.

2. **Deduplication sweep**: Look for duplicate entities (same name variants, acronyms,
   typos). Merge them: combine mention_counts, union aliases, keep earliest first_mentioned.

3. **Orphan cleanup**: Find entities with zero relationships and low mention counts
   (mention_count <= 1, strength < 0.3). Delete them.

4. **Community detection**: Group entities into clusters (Work, Side Projects, Family,
   Social, etc.) based on relationship density. Add a `community` property to entities.

5. **Contradiction detection**: Find conflicting relationships (e.g., person WORKS_AT
   two different companies with overlapping valid_from/valid_until). Write findings to
   {workspace}/memory/maintenance/{today}.md.

6. **Prune stale daily logs**: Archive daily logs older than 30 days from
   {workspace}/memory/ to {workspace}/memory/archive/.

7. **Update MEMORY.md**: Consolidate today's insights — new entities learned, key
   relationships discovered, patterns noticed. Append a dated section to MEMORY.md.

8. **Write maintenance report**: Write a summary of everything you did to
   {workspace}/memory/maintenance/{today}.md.

9. **Relationship type review**: Review all WORKS_AT and USES relationships where the
   target entity is an educational institution (universities, MBA programs, schools).
   Check context_snippets for classmate/student/alumni/cohort/program indicators and
   reclassify to CLASSMATE_OF, STUDIED_AT, or ALUMNI_OF as appropriate. USES should
   never apply to a school — reclassify those too.

10. **Semantic consistency check**: Review relationship types that don't make semantic
    sense (e.g., USES with a Person, WORKS_AT with a Concept, MANAGES a Place). Propose
    or execute reclassifications to the correct relationship type.

11. **Shared institution relationships**: When multiple people appear in the same
    institutional context (same school, same program, same cohort), check if they should
    share a CLASSMATE_OF relationship. Create missing CLASSMATE_OF links between people
    who both have STUDIED_AT or CLASSMATE_OF relationships with the same institution.

Use Neo4j Cypher queries via Bash (cypher-shell) for graph operations.
Be thorough but concise. This runs unattended.
"""


async def run_maintenance():
    """Run the full nightly maintenance cycle."""
    from agent import handle_message

    today = date.today().isoformat()
    log.info("Starting nightly maintenance for %s", today)

    # Step 1: Programmatic health check
    try:
        write_health_check()
    except Exception:
        log.error("Health check failed", exc_info=True)

    # Step 2: Opus agent maintenance turn
    try:
        system_prompt = MAINTENANCE_SYSTEM_PROMPT.format(
            workspace=config.WORKSPACE,
            today=today,
        )

        from claude_agent_sdk import ClaudeAgentOptions, query, AssistantMessage, TextBlock, ResultMessage
        from tools.calendar import calendar_server
        from tools.contacts import contacts_server
        from tools.gmail import gmail_server
        from tools.imessage import imessage_server

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            model="opus",
            permission_mode="bypassPermissions",
            allowed_tools=config.ALLOWED_TOOLS,
            mcp_servers={
                "google-calendar": calendar_server,
                "gmail": gmail_server,
                "apple-contacts": contacts_server,
                "imessage": imessage_server,
            },
            cwd=str(config.WORKSPACE),
        )

        async def _maintenance_prompt(text: str):
            yield {
                "type": "user",
                "session_id": "",
                "message": {"role": "user", "content": text},
                "parent_tool_use_id": None,
            }

        prompt_text = (
            f"Run nightly maintenance for {today}. "
            "Work through all maintenance tasks. Be thorough."
        )

        response_text = ""
        async for message in query(prompt=_maintenance_prompt(prompt_text), options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text
            elif isinstance(message, ResultMessage):
                if message.is_error:
                    log.error("Maintenance agent error: %s", message.result)

        log.info("Maintenance completed: %d chars output", len(response_text))

    except Exception:
        log.error("Opus maintenance turn failed", exc_info=True)
