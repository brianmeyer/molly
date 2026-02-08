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
# Programmatic maintenance tasks (direct Python, no Agent SDK tools)
# ---------------------------------------------------------------------------


def _run_strength_decay() -> int:
    """Task 1: Recalculate strength for all entities."""
    from memory.graph import run_strength_decay
    return run_strength_decay()


def _run_dedup_sweep() -> int:
    """Task 2: Find and merge duplicate entities with fuzzy matching."""
    from memory.graph import get_driver
    from difflib import SequenceMatcher

    driver = get_driver()
    merged = 0

    with driver.session() as session:
        # Get all entities grouped by type
        result = session.run(
            "MATCH (e:Entity) RETURN e.name AS name, e.entity_type AS type "
            "ORDER BY e.entity_type, e.name"
        )
        entities = [dict(r) for r in result]

    # Group by type
    by_type: dict[str, list[str]] = {}
    for ent in entities:
        by_type.setdefault(ent["type"], []).append(ent["name"])

    for etype, names in by_type.items():
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                ratio = SequenceMatcher(
                    None, name_a.lower(), name_b.lower()
                ).ratio()
                if ratio >= 0.85 and name_a != name_b:
                    # Merge name_b into name_a (keep first alphabetically)
                    keep, drop = sorted([name_a, name_b])
                    with driver.session() as session:
                        session.run(
                            """MATCH (keep:Entity {name: $keep}), (drop:Entity {name: $drop})
                               SET keep.mention_count = keep.mention_count + drop.mention_count,
                                   keep.aliases = keep.aliases + drop.aliases + [$drop_name],
                                   keep.first_mentioned = CASE
                                       WHEN keep.first_mentioned < drop.first_mentioned
                                       THEN keep.first_mentioned ELSE drop.first_mentioned END
                               WITH keep, drop
                               CALL {
                                   WITH keep, drop
                                   MATCH (drop)-[r]->()
                                   RETURN collect(r) AS rels
                               }
                               DETACH DELETE drop""",
                            keep=keep, drop=drop, drop_name=drop,
                        )
                    merged += 1
                    log.debug("Merged '%s' into '%s'", drop, keep)

    return merged


def _run_orphan_cleanup() -> int:
    """Task 3: Delete low-value orphan entities."""
    from memory.graph import get_driver

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
        deleted = result.single()["deleted"]

    log.info("Orphan cleanup: deleted %d entities", deleted)
    return deleted


def _run_self_ref_cleanup() -> int:
    """Task 3b: Delete self-referencing relationships."""
    from memory.graph import delete_self_referencing_rels
    return delete_self_referencing_rels()


def _run_blocklist_cleanup() -> int:
    """Task 3c: Delete blocklisted entities from graph."""
    from memory.graph import delete_blocklisted_entities
    from memory.processor import _ENTITY_BLOCKLIST
    return delete_blocklisted_entities(_ENTITY_BLOCKLIST)


def _prune_daily_logs() -> int:
    """Task 6: Archive daily logs older than 30 days."""
    memory_dir = config.WORKSPACE / "memory"
    archive_dir = memory_dir / "archive"
    cutoff = (date.today() - timedelta(days=30)).isoformat()
    moved = 0

    for path in memory_dir.glob("????-??-??.md"):
        if path.stem < cutoff:
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(archive_dir / path.name))
            moved += 1

    if moved:
        log.info("Archived %d daily logs older than %s", moved, cutoff)
    return moved


def _build_maintenance_report(results: dict) -> str:
    """Build a markdown maintenance report from task results."""
    today = date.today().isoformat()
    lines = [f"# Maintenance Report — {today}\n"]

    lines.append("## Task Results\n")
    lines.append("| Task | Result |")
    lines.append("|------|--------|")
    for task, result in results.items():
        lines.append(f"| {task} | {result} |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Opus analysis pass (text-only, no tools)
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """\
You are Molly's maintenance analyst. Review the maintenance report and graph data below.
Produce a brief analysis with:

1. **Summary**: Key observations about the knowledge graph health.
2. **New insights**: Any patterns, clusters, or notable entities from today.
3. **MEMORY.md update**: A dated section to append to MEMORY.md with the most important
   facts learned today. Format: `## {date}` followed by bullet points. Only include
   genuinely important, durable facts — skip noise and ephemeral details.

Be concise. Output ONLY the MEMORY.md section (starting with ## {date}).
"""


async def _run_opus_analysis(report: str, graph_summary: str, today: str) -> str:
    """Run a text-only Claude query for analysis — no tools, no permissions needed."""
    from claude_agent_sdk import (
        ClaudeAgentOptions, query, AssistantMessage, TextBlock, ResultMessage,
    )

    prompt_text = (
        f"Today is {today}.\n\n"
        f"## Maintenance Report\n{report}\n\n"
        f"## Graph Summary\n{graph_summary}\n\n"
        "Based on this data, produce the MEMORY.md update section."
    )

    options = ClaudeAgentOptions(
        system_prompt=ANALYSIS_SYSTEM_PROMPT.format(date=today),
        model="sonnet",
        allowed_tools=[],  # No tools — text-only analysis
        cwd=str(config.WORKSPACE),
    )

    async def _prompt():
        yield {
            "type": "user",
            "session_id": "",
            "message": {"role": "user", "content": prompt_text},
            "parent_tool_use_id": None,
        }

    response_text = ""
    async for message in query(prompt=_prompt(), options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_text += block.text
        elif isinstance(message, ResultMessage):
            if message.is_error:
                log.error("Maintenance analysis error: %s", message.result)

    return response_text


async def run_maintenance(molly=None):
    """Run the full nightly maintenance cycle using direct Python calls.

    No Agent SDK tools, no bypassPermissions. All graph operations use
    the Neo4j driver directly. File I/O is direct Python. The only Claude
    call is a text-only analysis pass with no tools available.

    Args:
        molly: Optional Molly instance for sending summary to owner DM.
    """
    today = date.today().isoformat()
    log.info("Starting nightly maintenance for %s", today)

    results: dict[str, str] = {}
    improver = None

    # Step 1: Programmatic health check
    try:
        write_health_check()
        results["Health check"] = "completed"
    except Exception:
        log.error("Health check failed", exc_info=True)
        results["Health check"] = "failed"

    # Step 2: Strength decay
    try:
        updated = _run_strength_decay()
        results["Strength decay"] = f"{updated} entities updated"
    except Exception:
        log.error("Strength decay failed", exc_info=True)
        results["Strength decay"] = "failed"

    # Step 3: Deduplication sweep
    try:
        merged = _run_dedup_sweep()
        results["Deduplication"] = f"{merged} entities merged"
    except Exception:
        log.error("Dedup sweep failed", exc_info=True)
        results["Deduplication"] = "failed"

    # Step 4: Orphan cleanup
    try:
        deleted = _run_orphan_cleanup()
        self_refs = _run_self_ref_cleanup()
        blocked = _run_blocklist_cleanup()
        results["Orphan cleanup"] = (
            f"{deleted} orphans, {self_refs} self-refs, {blocked} blocklisted"
        )
    except Exception:
        log.error("Orphan cleanup failed", exc_info=True)
        results["Orphan cleanup"] = "failed"

    # Step 4b: Phase 7 memory optimization loop
    try:
        from self_improve import SelfImprovementEngine

        improver = getattr(molly, "self_improvement", None) if molly else None
        if improver is None:
            improver = SelfImprovementEngine(molly=molly)
            await improver.initialize()
        mem_opt = await improver.run_memory_optimization()
        results["Memory optimization"] = (
            f"consolidated={mem_opt.get('entity_consolidations', 0)}, "
            f"stale={mem_opt.get('stale_entities', 0)}, "
            f"contradictions={mem_opt.get('contradictions', 0)}"
        )
    except Exception:
        log.error("Memory optimization failed", exc_info=True)
        results["Memory optimization"] = "failed"

    # Step 5: Prune stale daily logs
    try:
        moved = _prune_daily_logs()
        results["Daily log pruning"] = f"{moved} logs archived"
    except Exception:
        log.error("Daily log pruning failed", exc_info=True)
        results["Daily log pruning"] = "failed"

    # Step 6: Write maintenance report
    report = _build_maintenance_report(results)
    report_path = MAINTENANCE_DIR / f"{today}.md"
    MAINTENANCE_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    log.info("Maintenance report written to %s", report_path)

    # Step 7: Opus analysis pass (text-only, no tools)
    try:
        from memory.graph import get_graph_summary
        summary = get_graph_summary()
        graph_text = (
            f"Entities: {summary['entity_count']}, "
            f"Relationships: {summary['relationship_count']}\n"
            f"Top connected: {summary['top_connected']}\n"
            f"Recent: {summary['recent']}"
        )

        analysis = await _run_opus_analysis(report, graph_text, today)

        if analysis.strip():
            # Append to MEMORY.md
            memory_path = config.WORKSPACE / "MEMORY.md"
            if memory_path.exists():
                existing = memory_path.read_text()
                memory_path.write_text(existing.rstrip() + "\n\n" + analysis.strip() + "\n")
            else:
                memory_path.write_text(analysis.strip() + "\n")

            # Also append analysis to the report
            with open(report_path, "a") as f:
                f.write(f"\n## Analysis\n\n{analysis}\n")

            log.info("MEMORY.md updated with maintenance analysis")
        else:
            log.warning("Opus analysis returned empty response")

    except Exception:
        log.error("Opus analysis pass failed", exc_info=True)

    # Step 8: GLiNER training accumulation + conditional fine-tune trigger
    try:
        if improver is None:
            from self_improve import SelfImprovementEngine

            improver = getattr(molly, "self_improvement", None) if molly else None
            if improver is None:
                improver = SelfImprovementEngine(molly=molly)
                await improver.initialize()

        gliner_cycle = await improver.run_gliner_nightly_cycle()
        loop_status = str(gliner_cycle.get("status", "unknown"))
        if loop_status in {"insufficient_examples", "cooldown_active"}:
            results["GLiNER loop"] = str(gliner_cycle.get("message", loop_status))
        else:
            pipeline = gliner_cycle.get("pipeline", {}) if isinstance(gliner_cycle, dict) else {}
            pipeline_status = str(pipeline.get("status", loop_status))
            if pipeline_status == "deployed":
                improvement = float(pipeline.get("benchmark", {}).get("improvement", 0.0) or 0.0)
                results["GLiNER loop"] = f"deployed ({improvement:+.2%} F1)"
            elif pipeline_status:
                results["GLiNER loop"] = pipeline_status
            else:
                results["GLiNER loop"] = loop_status
    except Exception:
        log.error("GLiNER closed-loop run failed", exc_info=True)
        results["GLiNER loop"] = "failed"

    # Step 9: Health Doctor daily run (after maintenance completes)
    try:
        from health import get_health_doctor

        doctor = get_health_doctor(molly=molly)
        doctor.run_daily()
        results["Health Doctor"] = "completed"
    except Exception:
        log.error("Health Doctor run failed", exc_info=True)
        results["Health Doctor"] = "failed"

    # Step 10: Send brief summary to owner DM
    if molly and molly.wa:
        try:
            owner_jid = molly._get_owner_dm_jid()
            if owner_jid:
                from memory.graph import entity_count, relationship_count
                e_count = entity_count()
                r_count = relationship_count()

                summary_parts = [f"Nightly maintenance done ({today})."]
                for task_name, result in results.items():
                    summary_parts.append(f"{task_name}: {result}")
                summary_parts.append(f"Graph: {e_count} entities, {r_count} relationships.")

                summary_msg = "\n".join(summary_parts)
                # Cap at ~100 words
                words = summary_msg.split()
                if len(words) > 100:
                    summary_msg = " ".join(words[:100]) + "..."

                molly._track_send(molly.wa.send_message(owner_jid, summary_msg))
        except Exception:
            log.error("Failed to send maintenance summary", exc_info=True)

    log.info("Nightly maintenance completed for %s", today)
