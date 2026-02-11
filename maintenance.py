import asyncio
import logging
import os
import shutil
import sqlite3
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import config
from contract_audit import run_contract_audits
from health_remediation import route_health_signal
from memory.issue_registry import (
    append_issue_event,
    ensure_issue_registry_tables,
    should_notify,
    upsert_issue,
)

log = logging.getLogger(__name__)

MAINTENANCE_DIR = config.WORKSPACE / "memory" / "maintenance"
HEALTH_LOG_PATH = MAINTENANCE_DIR / "health-log.md"
MAINTENANCE_HOUR = 23  # 11 PM
MAINTENANCE_NOTIFY_COOLDOWN_HOURS = 24
MAINTENANCE_STEP_PREFIX = "maintenance."


@dataclass
class MaintenanceRunState:
    status: str = "idle"  # queued | running | success | partial | failed
    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    last_error: str = ""
    queued_requests: int = 0
    failed_steps: list[str] = field(default_factory=list)
    results: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_error": self.last_error,
            "queued_requests": int(self.queued_requests),
            "failed_steps": list(self.failed_steps),
            "results": dict(self.results),
        }


_MAINTENANCE_LOCK: asyncio.Lock | None = None
_MAINTENANCE_LOCK_LOOP_ID: int | None = None
_RUN_STATE = MaintenanceRunState()


def get_maintenance_run_state() -> dict[str, Any]:
    return _RUN_STATE.as_dict()


def _get_maintenance_lock() -> asyncio.Lock:
    """Return a loop-safe singleton lock for single-flight maintenance runs."""
    global _MAINTENANCE_LOCK, _MAINTENANCE_LOCK_LOOP_ID
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    if _MAINTENANCE_LOCK is None or _MAINTENANCE_LOCK_LOOP_ID != loop_id:
        _MAINTENANCE_LOCK = asyncio.Lock()
        _MAINTENANCE_LOCK_LOOP_ID = loop_id
    return _MAINTENANCE_LOCK


def should_run_maintenance(last_run: datetime | None) -> bool:
    """Check if nightly maintenance is due.

    Normal window: once per day at/after 11 PM local.
    Catch-up: if maintenance was missed for at least one full day, run on next opportunity.
    """
    now = datetime.now()
    if last_run is None:
        return now.hour >= MAINTENANCE_HOUR

    if last_run.date() >= now.date():
        return False

    if now.hour >= MAINTENANCE_HOUR:
        return True

    missed_days = (now.date() - last_run.date()).days
    return missed_days > 1


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


def _compute_operational_insights() -> dict:
    """Compute 24h tool success rates, flag failing tools, find unused skills."""
    from memory.retriever import get_vectorstore

    vs = get_vectorstore()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

    # Tool success rates (last 24h)
    cursor = vs.conn.execute(
        """SELECT tool_name,
                  COUNT(*) AS total,
                  SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successes
           FROM tool_calls
           WHERE created_at > ?
             AND tool_name NOT LIKE 'routing:%'
             AND tool_name NOT LIKE 'approval:%'
           GROUP BY tool_name
           ORDER BY total DESC""",
        (cutoff,),
    )
    failing_tools: list[str] = []
    tool_count = 0
    for row in cursor.fetchall():
        tool_count += 1
        total = row[1]
        successes = row[2]
        rate = successes / total if total > 0 else 0
        if total >= 3 and rate < 0.9:
            failing_tools.append(f"{row[0]} ({successes}/{total}={rate:.0%})")

    # Unused skills (7+ days since last execution)
    stale_cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    cursor = vs.conn.execute(
        "SELECT DISTINCT skill_name FROM skill_executions WHERE created_at > ?",
        (stale_cutoff,),
    )
    recent_skills = {row[0] for row in cursor.fetchall()}
    cursor = vs.conn.execute("SELECT DISTINCT skill_name FROM skill_executions")
    all_skills = {row[0] for row in cursor.fetchall()}
    unused = sorted(all_skills - recent_skills)

    return {
        "tool_count_24h": tool_count,
        "failing_tools": failing_tools,
        "unused_skills": unused,
    }


def _run_strength_decay() -> int:
    """Task 1: Recalculate strength for all entities."""
    from memory.graph import run_strength_decay
    return run_strength_decay()


def _run_dedup_sweep() -> int:
    """Task 2: Find and merge duplicate entities with the shared dedup engine."""
    from memory.dedup import find_near_duplicates
    from memory.graph import get_driver

    driver = get_driver()
    merged = 0
    already_merged: set[str] = set()

    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)
            RETURN e.name AS name, coalesce(e.entity_type, 'unknown') AS type
            ORDER BY e.entity_type, e.name
            """
        )
        entities = [dict(row) for row in result if row.get("name")]

    by_type: dict[str, list[str]] = {}
    for ent in entities:
        by_type.setdefault(str(ent.get("type") or "unknown"), []).append(str(ent["name"]))

    for _etype, names in by_type.items():
        for pair in find_near_duplicates(names):
            left = str(pair.left)
            right = str(pair.right)
            if left in already_merged or right in already_merged:
                continue
            keep, drop = sorted((left, right))
            with driver.session() as session:
                session.run(
                    """
                    MATCH (keep:Entity {name: $keep}), (drop:Entity {name: $drop})
                    SET keep.mention_count = coalesce(keep.mention_count, 0) + coalesce(drop.mention_count, 0),
                        keep.aliases = reduce(
                            acc = coalesce(keep.aliases, []),
                            alias IN coalesce(drop.aliases, []) + [$drop_name] |
                            CASE
                                WHEN toLower(alias) IN [x IN acc | toLower(x)] THEN acc
                                ELSE acc + alias
                            END
                        ),
                        keep.first_mentioned = CASE
                            WHEN keep.first_mentioned IS NULL THEN drop.first_mentioned
                            WHEN drop.first_mentioned IS NULL THEN keep.first_mentioned
                            WHEN keep.first_mentioned <= drop.first_mentioned THEN keep.first_mentioned
                            ELSE drop.first_mentioned
                        END,
                        keep.last_mentioned = CASE
                            WHEN keep.last_mentioned IS NULL THEN drop.last_mentioned
                            WHEN drop.last_mentioned IS NULL THEN keep.last_mentioned
                            WHEN keep.last_mentioned >= drop.last_mentioned THEN keep.last_mentioned
                            ELSE drop.last_mentioned
                        END,
                        keep.confidence = CASE
                            WHEN coalesce(drop.confidence, 0.0) > coalesce(keep.confidence, 0.0)
                            THEN drop.confidence
                            ELSE keep.confidence
                        END
                    DETACH DELETE drop
                    """,
                    keep=keep,
                    drop=drop,
                    drop_name=drop,
                )
            already_merged.add(drop)
            merged += 1
            log.debug("Merged '%s' into '%s' via shared dedup sweep", drop, keep)

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
    archived = 0
    deleted = 0

    for path in memory_dir.glob("????-??-??.md"):
        if path.stem < cutoff:
            try:
                archive_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(archive_dir / path.name))
                archived += 1
            except OSError:
                log.debug("Failed to archive %s", path, exc_info=True)

    # Cleanup JSONL files in graph_suggestions/ older than 30 days
    gs_dir = config.WORKSPACE / "memory" / "graph_suggestions"
    if gs_dir.is_dir():
        for path in gs_dir.glob("????-??-??.jsonl"):
            if path.stem < cutoff:
                try:
                    path.unlink()
                    deleted += 1
                except OSError:
                    log.debug("Failed to delete %s", path, exc_info=True)

    # Cleanup JSONL files in foundry/observations/ older than 30 days
    fo_dir = config.WORKSPACE / "foundry" / "observations"
    if fo_dir.is_dir():
        for path in fo_dir.glob("????-??-??.jsonl"):
            if path.stem < cutoff:
                try:
                    path.unlink()
                    deleted += 1
                except OSError:
                    log.debug("Failed to delete %s", path, exc_info=True)

    parts = []
    if archived:
        parts.append(f"archived {archived} daily log(s)")
    if deleted:
        parts.append(f"deleted {deleted} JSONL file(s)")
    if parts:
        log.info("%s older than %s", ", ".join(parts).capitalize(), cutoff)
    return archived + deleted


def _slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text or ""))
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return collapsed or "unknown"


def _latest_scheduled_weekly_date(now_local: datetime) -> date:
    weekday_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    target_day = config.WEEKLY_ASSESSMENT_DAY.strip().lower()
    target_idx = weekday_map.get(target_day, 6)
    delta = (now_local.weekday() - target_idx) % 7
    return (now_local - timedelta(days=delta)).date()


def _parse_iso_date(value: str) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _weekly_assessment_due_or_overdue(improver: Any, now_local: datetime) -> bool:
    state = getattr(improver, "_state", {}) or {}
    last_date = _parse_iso_date(str(state.get("last_weekly_assessment", "")))
    target_date = _latest_scheduled_weekly_date(now_local)

    if last_date and last_date >= target_date:
        return False

    if now_local.date() > target_date:
        return True

    return now_local.hour >= int(config.WEEKLY_ASSESSMENT_HOUR)


def _issue_last_notified_at(conn: sqlite3.Connection, fingerprint: str) -> str | None:
    row = conn.execute(
        """
        SELECT created_at
        FROM maintenance_issue_events
        WHERE issue_fingerprint = ? AND event_type = 'notified'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (fingerprint,),
    ).fetchone()
    if not row:
        return None
    return str(row[0] or "")


def _severity_from_result(result_text: str) -> str:
    lowered = str(result_text or "").strip().lower()
    if lowered == "failed":
        return "red"
    if lowered in {"skipped", "not due"} or lowered.startswith("skipped"):
        return "yellow"
    return "green"


def _record_maintenance_issues(
    results: dict[str, str],
    run_status: str,
) -> tuple[int, int]:
    synced = 0
    notified = 0
    try:
        conn = sqlite3.connect(str(config.MOLLYGRAPH_PATH))
        try:
            ensure_issue_registry_tables(conn)
            now = datetime.now(timezone.utc).isoformat()
            for task_name, result in results.items():
                check_id = f"{MAINTENANCE_STEP_PREFIX}{_slugify(task_name)}"
                severity = _severity_from_result(result)
                issue = upsert_issue(
                    conn,
                    check_id=check_id,
                    severity=severity,
                    detail=str(result),
                    source="maintenance",
                    observed_at=now,
                )
                synced += 1
                if severity not in {"yellow", "red"}:
                    continue
                last_notified = _issue_last_notified_at(conn, issue["fingerprint"])
                if should_notify(
                    fingerprint=issue["fingerprint"],
                    cooldown_hours=MAINTENANCE_NOTIFY_COOLDOWN_HOURS,
                    last_notified_at=last_notified,
                    severity_changed=bool(issue.get("severity_changed", False)),
                ):
                    plan = route_health_signal(check_id, severity)
                    append_issue_event(
                        conn,
                        issue_fingerprint=issue["fingerprint"],
                        event_type="notified",
                        created_at=now,
                        payload={
                            "task": task_name,
                            "result": result,
                            "run_status": run_status,
                            "action": plan.action,
                            "suggested_action": plan.suggested_action,
                        },
                    )
                    notified += 1
            conn.commit()
        finally:
            conn.close()
    except Exception:
        log.debug("Maintenance issue registry sync failed", exc_info=True)
    return synced, notified


def _build_maintenance_report(
    results: dict[str, str],
    *,
    run_status: str = "",
    failed_steps: list[str] | None = None,
) -> str:
    """Build a markdown maintenance report from task results."""
    today = date.today().isoformat()
    lines = [f"# Maintenance Report — {today}\n"]

    if run_status:
        lines.append("## Run Status\n")
        lines.append(f"- Status: {run_status}")
        if failed_steps:
            lines.append(f"- Failed steps: {', '.join(failed_steps)}")
        lines.append("")

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


async def run_maintenance(molly=None) -> dict[str, Any]:
    """Run the full nightly maintenance cycle using direct Python calls."""

    lock = _get_maintenance_lock()
    if lock.locked():
        _RUN_STATE.queued_requests += 1
        log.info(
            "Maintenance already running; skipping overlapping request (queued=%d)",
            _RUN_STATE.queued_requests,
        )
        return {
            "status": "queued",
            "run_id": _RUN_STATE.run_id,
            "queued_requests": _RUN_STATE.queued_requests,
        }

    async with lock:
        today = date.today().isoformat()
        run_id = uuid.uuid4().hex
        report_path = MAINTENANCE_DIR / f"{today}.md"
        results: dict[str, str] = {}
        failed_steps: list[str] = []
        analysis_text = ""
        improver = None
        weekly_due = False
        weekly_result = "not evaluated"

        def _record_step(step_name: str, result: str, *, failed: bool = False) -> None:
            results[step_name] = str(result)
            _RUN_STATE.results[step_name] = str(result)
            if failed and step_name not in failed_steps:
                failed_steps.append(step_name)
                _RUN_STATE.failed_steps = list(failed_steps)

        def _derive_run_status() -> str:
            if not results:
                return "failed"
            if not failed_steps:
                return "success"
            if len(failed_steps) >= len(results):
                return "failed"
            return "partial"

        async def _ensure_improver():
            nonlocal improver
            if improver is not None:
                return improver
            from self_improve import SelfImprovementEngine

            improver = getattr(molly, "self_improvement", None) if molly else None
            if improver is None:
                improver = SelfImprovementEngine(molly=molly)
                await improver.initialize()
            return improver

        _RUN_STATE.status = "running"
        _RUN_STATE.run_id = run_id
        _RUN_STATE.started_at = datetime.now(timezone.utc).isoformat()
        _RUN_STATE.finished_at = ""
        _RUN_STATE.last_error = ""
        _RUN_STATE.failed_steps = []
        _RUN_STATE.results = {}
        _RUN_STATE.queued_requests = 0
        MAINTENANCE_DIR.mkdir(parents=True, exist_ok=True)

        log.info("Starting nightly maintenance run_id=%s for %s", run_id, today)

        try:
            # Step 1: Programmatic health check
            try:
                write_health_check()
                _record_step("Health check", "completed")
            except Exception:
                log.error("Health check failed", exc_info=True)
                _record_step("Health check", "failed", failed=True)

            # Step 2: Strength decay
            try:
                updated = _run_strength_decay()
                _record_step("Strength decay", f"{updated} entities updated")
            except Exception:
                log.error("Strength decay failed", exc_info=True)
                _record_step("Strength decay", "failed", failed=True)

            # Step 3: Deduplication sweep
            try:
                merged = _run_dedup_sweep()
                _record_step("Deduplication", f"{merged} entities merged")
            except Exception:
                log.error("Dedup sweep failed", exc_info=True)
                _record_step("Deduplication", "failed", failed=True)

            # Step 4: Orphan cleanup
            try:
                deleted = _run_orphan_cleanup()
                self_refs = _run_self_ref_cleanup()
                blocked = _run_blocklist_cleanup()
                _record_step(
                    "Orphan cleanup",
                    f"{deleted} orphans, {self_refs} self-refs, {blocked} blocklisted",
                )
            except Exception:
                log.error("Orphan cleanup failed", exc_info=True)
                _record_step("Orphan cleanup", "failed", failed=True)

            # Step 4b: Relationship quality audit
            try:
                from memory.relationship_audit import run_relationship_audit
                rel_audit = await run_relationship_audit(
                    model_enabled=config.REL_AUDIT_MODEL_ENABLED,
                    molly=molly,
                )
                ra_auto = rel_audit.get("auto_fixes_applied", 0)
                ra_quar = rel_audit.get("quarantined_count", 0)
                ra_status = rel_audit.get("deterministic_result", {}).get("status", "pass")
                _record_step(
                    "Relationship audit",
                    f"{ra_auto} auto-fixed, {ra_quar} quarantined ({ra_status})",
                    failed=ra_status == "fail",
                )
            except Exception:
                log.error("Relationship quality audit failed", exc_info=True)
                _record_step("Relationship audit", "failed", failed=True)

            # Step 4.5: Neo4j transaction log checkpoint
            try:
                from memory.graph import get_driver

                driver = get_driver()
                with driver.session() as neo_session:
                    # Detect edition — db.checkpoint() is Enterprise-only
                    # Try modern syntax first (Neo4j 5.x+), fall back to legacy
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
                        _record_step("Neo4j checkpoint", "completed")
                    else:
                        _record_step("Neo4j checkpoint", f"skipped ({edition or 'unknown'} edition)")
            except Exception:
                log.error("Neo4j checkpoint failed", exc_info=True)
                _record_step("Neo4j checkpoint", "failed", failed=True)

            # Step 5: Phase 7 memory optimization loop
            try:
                improver = await _ensure_improver()
                mem_opt = await improver.run_memory_optimization()
                _record_step(
                    "Memory optimization",
                    (
                        f"consolidated={mem_opt.get('entity_consolidations', 0)}, "
                        f"stale={mem_opt.get('stale_entities', 0)}, "
                        f"contradictions={mem_opt.get('contradictions', 0)}"
                    ),
                )
            except Exception:
                log.error("Memory optimization failed", exc_info=True)
                _record_step("Memory optimization", "failed", failed=True)

            # Step 6: Prune stale daily logs
            try:
                moved = _prune_daily_logs()
                _record_step("Daily log pruning", f"{moved} logs archived")
            except Exception:
                log.error("Daily log pruning failed", exc_info=True)
                _record_step("Daily log pruning", "failed", failed=True)

            # Step 7: GLiNER training accumulation + conditional fine-tune trigger
            try:
                improver = await _ensure_improver()
                gliner_cycle = await improver.run_gliner_nightly_cycle()
                loop_status = str(gliner_cycle.get("status", "unknown"))
                if loop_status in {"insufficient_examples", "cooldown_active"}:
                    _record_step("GLiNER loop", str(gliner_cycle.get("message", loop_status)))
                else:
                    pipeline = (
                        gliner_cycle.get("pipeline", {})
                        if isinstance(gliner_cycle, dict)
                        else {}
                    )
                    pipeline_status = str(pipeline.get("status", loop_status))
                    if pipeline_status == "deployed":
                        improvement = float(
                            pipeline.get("benchmark", {}).get("improvement", 0.0) or 0.0
                        )
                        _record_step("GLiNER loop", f"deployed ({improvement:+.2%} F1)")
                    elif pipeline_status:
                        _record_step("GLiNER loop", pipeline_status)
                    else:
                        _record_step("GLiNER loop", loop_status)
            except Exception:
                log.error("GLiNER closed-loop run failed", exc_info=True)
                _record_step("GLiNER loop", "failed", failed=True)

            # Step 7.5: Operational insights (24h tool success rates, unused skills)
            try:
                insights = _compute_operational_insights()
                parts = [f"{insights['tool_count_24h']} tools active"]
                if insights["failing_tools"]:
                    parts.append(f"failing: {', '.join(insights['failing_tools'][:5])}")
                if insights["unused_skills"]:
                    parts.append(f"unused skills (7d): {', '.join(insights['unused_skills'][:5])}")
                _record_step("Operational insights", "; ".join(parts))
            except Exception:
                log.error("Operational insights failed", exc_info=True)
                _record_step("Operational insights", "failed", failed=True)

            # Step 7.6: Foundry skill scan (nightly pattern detection)
            try:
                improver = await _ensure_improver()
                from foundry_adapter import load_foundry_sequence_signals

                signals = load_foundry_sequence_signals(days=7)
                patterns = [
                    {
                        "steps": list(sig.steps),
                        "count": sig.count,
                        "confidence": sig.success_rate,
                        "name": key,
                        "steps_text": key,
                    }
                    for key, sig in signals.items()
                    if sig.count >= 3
                ]
                if patterns:
                    skill_result = await improver.propose_skill_updates(patterns)
                    _record_step("Foundry skill scan", str(skill_result.get("status", "no candidates")))
                else:
                    _record_step("Foundry skill scan", "no qualifying patterns")
            except Exception:
                log.error("Foundry skill scan failed", exc_info=True)
                _record_step("Foundry skill scan", "failed", failed=True)

            # Step 7.7: Tool gap scan (nightly failure analysis)
            try:
                improver = await _ensure_improver()
                gap_result = await improver.propose_tool_updates(
                    days=7, min_failures=5,
                )
                _record_step("Tool gap scan", str(gap_result.get("status", "no gaps")))
            except Exception:
                log.error("Tool gap scan failed", exc_info=True)
                _record_step("Tool gap scan", "failed", failed=True)

            # Step 7.8: Correction pattern analysis
            try:
                from memory.retriever import get_vectorstore
                vs = get_vectorstore()
                cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

                cursor = vs.conn.execute(
                    "SELECT pattern, COUNT(*) as cnt FROM corrections "
                    "WHERE created_at > ? GROUP BY pattern ORDER BY cnt DESC LIMIT 10",
                    (cutoff_24h,),
                )
                pattern_rows = cursor.fetchall()

                cursor = vs.conn.execute(
                    "SELECT molly_output, user_correction, pattern FROM corrections "
                    "WHERE created_at > ? ORDER BY created_at DESC LIMIT 5",
                    (cutoff_24h,),
                )
                example_rows = cursor.fetchall()

                total_corrections = sum(row[1] for row in pattern_rows)
                if total_corrections == 0:
                    _record_step("Correction patterns", "0 corrections in last 24h")
                else:
                    parts = [f"{total_corrections} correction(s) in last 24h"]
                    for row in pattern_rows[:5]:
                        parts.append(f"  '{row[0]}': {row[1]}x")
                    if example_rows:
                        parts.append("Recent examples:")
                        for ex in example_rows[:3]:
                            molly_out = (ex[0] or "")[:80]
                            user_corr = (ex[1] or "")[:80]
                            parts.append(f"  Molly: {molly_out}... -> User: {user_corr}...")
                    _record_step("Correction patterns", "\n".join(parts))
            except Exception:
                log.error("Correction pattern analysis failed", exc_info=True)
                _record_step("Correction patterns", "failed", failed=True)

            # Step 8: Weekly assessment catch-up if due/overdue.
            try:
                improver = await _ensure_improver()
                try:
                    now_local = datetime.now(ZoneInfo(config.TIMEZONE))
                except Exception:
                    now_local = datetime.now(timezone.utc)
                weekly_due = _weekly_assessment_due_or_overdue(improver, now_local)
                if weekly_due:
                    weekly_path = await improver.run_weekly_assessment()
                    weekly_name = Path(str(weekly_path)).name
                    weekly_result = f"generated {weekly_name}"
                    _record_step("Weekly assessment", weekly_result)
                else:
                    weekly_result = "not due"
                    _record_step("Weekly assessment", weekly_result)
            except Exception:
                log.error("Weekly assessment catch-up failed", exc_info=True)
                weekly_result = "failed"
                _record_step("Weekly assessment", "failed", failed=True)

            # Step 9: Health Doctor daily run (after maintenance completes)
            try:
                from health import get_health_doctor

                doctor = get_health_doctor(molly=molly)
                doctor.run_daily()
                _record_step("Health Doctor", "completed")
            except Exception:
                log.error("Health Doctor run failed", exc_info=True)
                _record_step("Health Doctor", "failed", failed=True)

            # Step 9.5: Graph suggestions digest
            try:
                from memory.graph_suggestions import build_suggestion_digest

                digest = build_suggestion_digest()
                if digest:
                    _record_step("Graph suggestions", digest[:500])
                else:
                    _record_step("Graph suggestions", "no suggestions today")
            except Exception:
                log.error("Graph suggestions digest failed", exc_info=True)
                _record_step("Graph suggestions", "failed", failed=True)

            # Step 10: Contract audits (deterministic first, model second)
            try:
                audit_bundle = await run_contract_audits(
                    today=today,
                    task_results=results,
                    weekly_due=weekly_due,
                    weekly_result=weekly_result,
                    maintenance_dir=MAINTENANCE_DIR,
                    health_dir=config.HEALTH_REPORT_DIR,
                )

                nightly_det = dict(audit_bundle.get("nightly_deterministic", {}))
                weekly_det = dict(audit_bundle.get("weekly_deterministic", {}))
                nightly_model = dict(audit_bundle.get("nightly_model", {}))
                weekly_model = dict(audit_bundle.get("weekly_model", {}))
                artifacts = dict(audit_bundle.get("artifacts", {}))

                nightly_det_status = str(nightly_det.get("status", "pass"))
                weekly_det_status = str(weekly_det.get("status", "pass"))
                _record_step(
                    "Contract audit nightly (deterministic)",
                    str(nightly_det.get("summary", "pass")),
                    failed=nightly_det_status == "fail",
                )
                _record_step(
                    "Contract audit weekly (deterministic)",
                    str(weekly_det.get("summary", "pass")),
                    failed=weekly_det_status == "fail",
                )

                nightly_model_status = str(nightly_model.get("status", "disabled")).strip().lower()
                weekly_model_status = str(weekly_model.get("status", "disabled")).strip().lower()
                model_blocking = bool(config.CONTRACT_AUDIT_LLM_BLOCKING)

                _record_step(
                    "Contract audit nightly (model)",
                    str(nightly_model.get("summary", "disabled by config")),
                    failed=model_blocking and nightly_model_status in {"error", "unavailable"},
                )
                _record_step(
                    "Contract audit weekly (model)",
                    str(weekly_model.get("summary", "disabled by config")),
                    failed=model_blocking and weekly_model_status in {"error", "unavailable"},
                )

                artifact_error = str(artifacts.get("error", "")).strip()
                if artifact_error:
                    _record_step("Contract audit artifacts", f"write error: {artifact_error}")
                else:
                    _record_step(
                        "Contract audit artifacts",
                        (
                            f"maintenance={Path(str(artifacts.get('maintenance', '-'))).name}, "
                            f"health={Path(str(artifacts.get('health', '-'))).name}"
                        ),
                    )
            except Exception:
                log.error("Contract audit pass failed", exc_info=True)
                _record_step("Contract audit nightly (deterministic)", "failed", failed=True)
                _record_step("Contract audit weekly (deterministic)", "failed", failed=True)
                if config.CONTRACT_AUDIT_LLM_BLOCKING:
                    _record_step("Contract audit nightly (model)", "failed", failed=True)
                    _record_step("Contract audit weekly (model)", "failed", failed=True)
                else:
                    _record_step("Contract audit nightly (model)", "error (report-only)")
                    _record_step("Contract audit weekly (model)", "error (report-only)")
                _record_step("Contract audit artifacts", "unavailable")

            # Step 11: Opus analysis pass (text-only, no tools)
            try:
                from memory.graph import get_graph_summary

                summary = get_graph_summary()
                graph_text = (
                    f"Entities: {summary['entity_count']}, "
                    f"Relationships: {summary['relationship_count']}\n"
                    f"Top connected: {summary['top_connected']}\n"
                    f"Recent: {summary['recent']}"
                )
                pre_analysis_report = _build_maintenance_report(
                    results,
                    run_status=_derive_run_status(),
                    failed_steps=failed_steps,
                )
                analysis_text = await _run_opus_analysis(
                    pre_analysis_report,
                    graph_text,
                    today,
                )
                if analysis_text.strip():
                    memory_path = config.WORKSPACE / "MEMORY.md"
                    if memory_path.exists():
                        existing = memory_path.read_text()
                        memory_path.write_text(
                            existing.rstrip() + "\n\n" + analysis_text.strip() + "\n"
                        )
                    else:
                        memory_path.write_text(analysis_text.strip() + "\n")
                    _record_step("Analysis", "MEMORY.md updated")
                else:
                    _record_step("Analysis", "empty response")
            except Exception:
                log.error("Opus analysis pass failed", exc_info=True)
                _record_step("Analysis", "failed", failed=True)

            run_status = _derive_run_status()
            synced, notified = _record_maintenance_issues(results, run_status)
            _record_step("Issue registry", f"{synced} synced, {notified} notified")
            run_status = _derive_run_status()

            report = _build_maintenance_report(
                results,
                run_status=run_status,
                failed_steps=failed_steps,
            )
            if analysis_text.strip():
                report = report.rstrip() + f"\n\n## Analysis\n\n{analysis_text.strip()}\n"
            report_path.write_text(report)
            log.info("Maintenance report written to %s", report_path)

            # Step 12: Send brief summary to owner DM
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
                        summary_parts.append(
                            f"Graph: {e_count} entities, {r_count} relationships."
                        )

                        summary_msg = "\n".join(summary_parts)
                        words = summary_msg.split()
                        if len(words) > 100:
                            summary_msg = " ".join(words[:100]) + "..."

                        molly._track_send(molly.wa.send_message(owner_jid, summary_msg))
                except Exception:
                    log.error("Failed to send maintenance summary", exc_info=True)

            _RUN_STATE.status = run_status
            _RUN_STATE.last_error = ""
            log.info(
                "Nightly maintenance completed run_id=%s status=%s",
                run_id,
                run_status,
            )
            return _RUN_STATE.as_dict()
        except Exception as exc:
            log.error("Nightly maintenance aborted run_id=%s", run_id, exc_info=True)
            _RUN_STATE.status = "failed"
            _RUN_STATE.last_error = str(exc)
            return _RUN_STATE.as_dict()
        finally:
            _RUN_STATE.finished_at = datetime.now(timezone.utc).isoformat()
            _RUN_STATE.failed_steps = list(failed_steps)
            _RUN_STATE.results = dict(results)
