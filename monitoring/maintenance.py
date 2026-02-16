"""Nightly maintenance orchestrator — thin coordinator delegating to monitoring.jobs.*"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import config
from contract_audit import run_contract_audits
from monitoring._base import _now_local
from monitoring.health import get_health_doctor
from utils import atomic_write, atomic_write_json, load_json, normalize_timestamp

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAINTENANCE_DIR = config.WORKSPACE / "memory" / "maintenance"
HEALTH_LOG_PATH = MAINTENANCE_DIR / "health-log.md"
MAINTENANCE_HOUR = 23


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------

@dataclass
class MaintenanceRunState:
    status: str = "idle"
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


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _maintenance_checkpoint_scope() -> str:
    try:
        return str(MAINTENANCE_DIR.resolve())
    except Exception:
        return str(MAINTENANCE_DIR)


def _load_checkpoint(run_date: str) -> set[int]:
    payload = load_json(config.MAINTENANCE_STATE_FILE, {})
    if not isinstance(payload, dict):
        return set()
    if str(payload.get("run_date", "")) != run_date:
        return set()
    if str(payload.get("maintenance_dir", "")) != _maintenance_checkpoint_scope():
        return set()
    completed: set[int] = set()
    for value in payload.get("completed_steps", []):
        try:
            completed.add(int(value))
        except (TypeError, ValueError):
            continue
    return completed


def _save_checkpoint(run_date: str, completed_steps: set[int]) -> None:
    atomic_write_json(
        config.MAINTENANCE_STATE_FILE,
        {
            "run_date": run_date,
            "completed_steps": sorted(completed_steps),
            "maintenance_dir": _maintenance_checkpoint_scope(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _finalize_step(run_date: str, completed_steps: set[int], step_no: int) -> None:
    completed_steps.add(int(step_no))
    _save_checkpoint(run_date, completed_steps)


def _clear_checkpoint() -> None:
    atomic_write_json(
        config.MAINTENANCE_STATE_FILE,
        {
            "run_date": "",
            "completed_steps": [],
            "maintenance_dir": _maintenance_checkpoint_scope(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Lock management
# ---------------------------------------------------------------------------

def _get_maintenance_lock() -> asyncio.Lock:
    global _MAINTENANCE_LOCK, _MAINTENANCE_LOCK_LOOP_ID
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    if _MAINTENANCE_LOCK is None or _MAINTENANCE_LOCK_LOOP_ID != loop_id:
        _MAINTENANCE_LOCK = asyncio.Lock()
        _MAINTENANCE_LOCK_LOOP_ID = loop_id
    return _MAINTENANCE_LOCK


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def should_run_maintenance(last_run: datetime | None) -> bool:
    now = _now_local()
    if last_run is None:
        return now.hour >= MAINTENANCE_HOUR
    # Make last_run naive-comparable in local time if it's aware
    if last_run.tzinfo is not None:
        last_run = last_run.astimezone(ZoneInfo(config.TIMEZONE))
    if last_run.date() >= now.date():
        return False
    if now.hour >= MAINTENANCE_HOUR:
        return True
    return (now.date() - last_run.date()).days > 1


# ---------------------------------------------------------------------------
# Health check step (step 1)
# ---------------------------------------------------------------------------

def run_health_check() -> str:
    return (
        f"## Health Check: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        "| Status | ok |\n\n"
    )


def write_health_check() -> None:
    MAINTENANCE_DIR.mkdir(parents=True, exist_ok=True)
    existing = ""
    if HEALTH_LOG_PATH.exists():
        try:
            existing = HEALTH_LOG_PATH.read_text()
        except Exception:
            existing = ""
    atomic_write(HEALTH_LOG_PATH, existing + run_health_check())


# ---------------------------------------------------------------------------
# Thin wrappers delegating to monitoring.jobs.*
# These names are kept for backward compat (test_audit_fixes.py reads source).
# ---------------------------------------------------------------------------

async def _run_strength_decay() -> str:
    from monitoring.jobs.graph_maintenance import run_strength_decay
    return await run_strength_decay()


def _run_dedup_sweep() -> str:
    from monitoring.jobs.graph_maintenance import run_dedup_sweep
    return run_dedup_sweep()


def _run_orphan_cleanup() -> str:
    from monitoring.jobs.graph_maintenance import run_orphan_cleanup
    return run_orphan_cleanup()


def _run_self_ref_cleanup() -> int:
    from memory.graph import delete_self_referencing_rels
    return int(delete_self_referencing_rels())


def _run_blocklist_cleanup() -> int:
    from memory.graph import delete_blocklisted_entities
    from memory.processor import _ENTITY_BLOCKLIST
    return int(delete_blocklisted_entities(_ENTITY_BLOCKLIST))


def _prune_daily_logs() -> str:
    from monitoring.jobs.cleanup_jobs import prune_daily_logs
    return prune_daily_logs()


async def _run_opus_analysis(report: str, graph_summary: str, today: str) -> str:
    from monitoring.jobs.analysis_jobs import run_opus_analysis
    return await run_opus_analysis(report, graph_summary, today)


# ---------------------------------------------------------------------------
# Report building + WhatsApp summary
# ---------------------------------------------------------------------------

def _build_maintenance_report(
    results: dict[str, str],
    *,
    run_status: str = "",
    failed_steps: list[str] | None = None,
) -> str:
    today = date.today().isoformat()
    lines = [f"# Maintenance Report — {today}\n"]
    if run_status:
        lines += ["## Run Status\n", f"- Status: {run_status}"]
        if failed_steps:
            lines.append(f"- Failed steps: {', '.join(failed_steps)}")
        lines.append("")

    lines += ["## Task Results\n", "| Task | Result |", "|------|--------|"]
    for task, result in results.items():
        lines.append(f"| {task} | {result} |")
    lines.append("")
    return "\n".join(lines)


def _send_summary_to_owner(molly, summary_text: str) -> bool:
    if not molly:
        return False
    wa = getattr(molly, "wa", None)
    owner_getter = getattr(molly, "_get_owner_dm_jid", None)
    if wa is None or not callable(owner_getter):
        return False
    owner_jid = owner_getter()
    if not owner_jid:
        return False
    try:
        send_result = wa.send_message(owner_jid, summary_text)
        tracker = getattr(molly, "_track_send", None)
        if callable(tracker):
            tracker(send_result)
        return bool(send_result)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main maintenance orchestrator — full 22-step sequence
# ---------------------------------------------------------------------------

async def run_maintenance(molly=None) -> dict[str, Any]:
    lock = _get_maintenance_lock()
    # In asyncio's cooperative model, lock.locked() and the subsequent
    # `async with lock` are effectively atomic because no other coroutine
    # can run between them (no await in between).  If the lock is held,
    # return "queued" immediately.
    if lock.locked():
        _RUN_STATE.queued_requests += 1
        return {
            "status": "queued",
            "run_id": _RUN_STATE.run_id,
            "queued_requests": _RUN_STATE.queued_requests,
        }

    async with lock:
        today = date.today().isoformat()
        completed_steps = _load_checkpoint(today)
        run_id = uuid.uuid4().hex
        report_path = MAINTENANCE_DIR / f"{today}.md"
        pending_summary_path = MAINTENANCE_DIR / "pending_summary.txt"

        results: dict[str, str] = {}
        failed_steps: list[str] = []
        analysis_text = ""
        whatsapp_summary = ""

        _RUN_STATE.status = "running"
        _RUN_STATE.run_id = run_id
        _RUN_STATE.started_at = datetime.now(timezone.utc).isoformat()
        _RUN_STATE.finished_at = ""
        _RUN_STATE.last_error = ""
        _RUN_STATE.failed_steps = []
        _RUN_STATE.results = {}
        _RUN_STATE.queued_requests = 0

        MAINTENANCE_DIR.mkdir(parents=True, exist_ok=True)

        improver = None
        weekly_due = False
        weekly_result = "not evaluated"

        async def _ensure_improver():
            nonlocal improver
            if improver is not None:
                return improver
            improver = getattr(molly, "self_improvement", None) if molly else None
            if improver is None:
                from self_improve import SelfImprovementEngine
                improver = SelfImprovementEngine(molly=molly)
                await improver.initialize()
            return improver

        def _record(name: str, result: str, *, failed: bool = False) -> None:
            results[name] = str(result)
            _RUN_STATE.results[name] = str(result)
            if failed and name not in failed_steps:
                failed_steps.append(name)
                _RUN_STATE.failed_steps = list(failed_steps)

        def _is_done(step_no: int, name: str) -> bool:
            if step_no in completed_steps:
                _record(name, "skipped (checkpoint resume)")
                return True
            return False

        def _final(step_no: int) -> None:
            _finalize_step(today, completed_steps, step_no)

        def _status() -> str:
            if not results:
                return "failed"
            if not failed_steps:
                return "success"
            return "failed" if len(failed_steps) >= len(results) else "partial"

        async def _run_step(step_no: int, name: str, fn) -> None:
            if _is_done(step_no, name):
                return
            try:
                value = fn()
                if asyncio.iscoroutine(value):
                    value = await value
                if isinstance(value, tuple) and len(value) == 2:
                    text, failed = str(value[0]), bool(value[1])
                else:
                    text, failed = str(value), False
            except Exception:
                text, failed = "failed", True
            _record(name, text, failed=failed)
            _final(step_no)

        # -- Individual step functions ----------------------------------------

        async def _step_relationship_audit() -> tuple[str, bool]:
            from memory.relationship_audit import run_relationship_audit
            rel_audit = await run_relationship_audit(
                model_enabled=config.REL_AUDIT_MODEL_ENABLED,
                molly=molly,
            )
            ra_auto = rel_audit.get("auto_fixes_applied", 0)
            ra_quar = rel_audit.get("quarantined_count", 0)
            ra_status = rel_audit.get("deterministic_result", {}).get("status", "pass")
            return f"{ra_auto} auto-fixed, {ra_quar} quarantined ({ra_status})", ra_status == "fail"

        async def _step_strength_decay() -> str:
            return await _run_strength_decay()

        async def _step_entity_audit() -> str:
            from monitoring.jobs.entity_audit import run_entity_audit
            result = await run_entity_audit()
            audited = result.get("entities_audited", 0)
            rels = result.get("rels_audited", 0)
            gliner = result.get("gliner_examples_written", 0)
            adopted = result.get("types_adopted", 0)
            parts = [f"entities={audited}", f"rels={rels}", f"gliner_labels={gliner}"]
            if adopted:
                parts.append(f"types_adopted={adopted}")
            return ", ".join(parts)

        async def _step_neo4j_checkpoint() -> str:
            from monitoring.jobs.graph_maintenance import run_neo4j_checkpoint
            return run_neo4j_checkpoint()

        async def _step_memory_optimization() -> str:
            from monitoring.jobs.self_improve_jobs import run_memory_optimization
            return await run_memory_optimization(await _ensure_improver())

        async def _step_gliner_loop() -> str:
            from monitoring.jobs.self_improve_jobs import run_gliner_loop
            return await run_gliner_loop(await _ensure_improver())

        async def _step_weekly_assessment() -> str:
            nonlocal weekly_due, weekly_result
            from monitoring.jobs.self_improve_jobs import run_weekly_assessment
            now_local = datetime.now(ZoneInfo(config.TIMEZONE))
            ran, desc = await run_weekly_assessment(await _ensure_improver(), now_local)
            weekly_due = ran
            weekly_result = desc
            return desc

        async def _step_operational_insights() -> str:
            from monitoring.jobs.analysis_jobs import compute_operational_insights
            insights = compute_operational_insights()
            parts = [f"tools_24h={insights['tool_count_24h']}"]
            if insights["failing_tools"]:
                parts.append(f"failing={', '.join(insights['failing_tools'])}")
            if insights["unused_skills"]:
                parts.append(f"unused_skills={len(insights['unused_skills'])}")
            return ", ".join(parts)

        async def _step_foundry_skill_scan() -> str:
            from monitoring.jobs.learning_jobs import run_foundry_skill_scan
            return await run_foundry_skill_scan(await _ensure_improver())

        async def _step_tool_gap_scan() -> str:
            from monitoring.jobs.learning_jobs import run_tool_gap_scan
            return await run_tool_gap_scan(await _ensure_improver())

        async def _step_correction_patterns() -> str:
            from monitoring.jobs.learning_jobs import run_correction_patterns
            return run_correction_patterns()

        async def _step_graph_suggestions() -> str:
            from monitoring.jobs.analysis_jobs import run_graph_suggestions_digest
            return run_graph_suggestions_digest()

        async def _step_issue_registry() -> str:
            from monitoring.jobs.audit_jobs import record_maintenance_issues
            synced, notified = record_maintenance_issues(results, _status())
            return f"synced={synced}, notified={notified}"

        async def _step_contract_audits() -> None:
            if _is_done(18, "Contract audit nightly (deterministic)"):
                for skipped_name in (
                    "Contract audit weekly (deterministic)",
                    "Contract audit nightly (model)",
                    "Contract audit weekly (model)",
                    "Contract audit artifacts",
                ):
                    _record(skipped_name, "skipped (checkpoint resume)")
                return

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

                _record(
                    "Contract audit nightly (deterministic)",
                    str(nightly_det.get("summary", "pass")),
                    failed=str(nightly_det.get("status", "pass")) == "fail",
                )
                _record(
                    "Contract audit weekly (deterministic)",
                    str(weekly_det.get("summary", "pass")),
                    failed=str(weekly_det.get("status", "pass")) == "fail",
                )

                model_blocking = bool(config.CONTRACT_AUDIT_LLM_BLOCKING)
                nightly_model_status = str(nightly_model.get("status", "disabled")).strip().lower()
                weekly_model_status = str(weekly_model.get("status", "disabled")).strip().lower()
                _record(
                    "Contract audit nightly (model)",
                    str(nightly_model.get("summary", "disabled by config")),
                    failed=model_blocking and nightly_model_status in {"error", "unavailable"},
                )
                _record(
                    "Contract audit weekly (model)",
                    str(weekly_model.get("summary", "disabled by config")),
                    failed=model_blocking and weekly_model_status in {"error", "unavailable"},
                )

                artifact_error = str(artifacts.get("error", "")).strip()
                if artifact_error:
                    _record("Contract audit artifacts", f"write error: {artifact_error}")
                else:
                    _record(
                        "Contract audit artifacts",
                        (
                            f"maintenance={Path(str(artifacts.get('maintenance', '-'))).name}, "
                            f"health={Path(str(artifacts.get('health', '-'))).name}"
                        ),
                    )
            except Exception:
                _record("Contract audit nightly (deterministic)", "failed", failed=True)
                _record("Contract audit weekly (deterministic)", "failed", failed=True)
                if config.CONTRACT_AUDIT_LLM_BLOCKING:
                    _record("Contract audit nightly (model)", "failed", failed=True)
                    _record("Contract audit weekly (model)", "failed", failed=True)
                else:
                    _record("Contract audit nightly (model)", "error (report-only)")
                    _record("Contract audit weekly (model)", "error (report-only)")
                _record("Contract audit artifacts", "unavailable")
            finally:
                _final(18)

        # -- Execute full sequence --------------------------------------------

        try:
            # Step 1: Health check
            await _run_step(1, "Health check", lambda: (write_health_check(), "completed")[1])

            # Step 2: Strength decay
            await _run_step(2, "Strength decay", _step_strength_decay)

            # Step 3: Deduplication
            await _run_step(3, "Deduplication", lambda: _run_dedup_sweep())

            # Step 4: Orphan cleanup
            await _run_step(
                4,
                "Orphan cleanup",
                lambda: _run_orphan_cleanup(),
            )

            # Step 5: Relationship audit
            await _run_step(5, "Relationship audit", _step_relationship_audit)

            # Step 6: Kimi entity audit (NEW — before GLiNER so training data is available)
            await _run_step(6, "Entity audit", _step_entity_audit)

            # Step 7: Neo4j checkpoint (NEW)
            await _run_step(7, "Neo4j checkpoint", _step_neo4j_checkpoint)

            # Step 8: Memory optimization
            await _run_step(8, "Memory optimization", _step_memory_optimization)

            # Step 9: Daily log pruning (enhanced with foundry/observations)
            await _run_step(9, "Daily log pruning", lambda: _prune_daily_logs())

            # Step 10: GLiNER loop (now has Kimi-generated training data from step 6)
            await _run_step(10, "GLiNER loop", _step_gliner_loop)

            # Step 11: Operational insights (NEW)
            await _run_step(11, "Operational insights", _step_operational_insights)

            # Step 12: Foundry skill scan (NEW)
            await _run_step(12, "Foundry skill scan", _step_foundry_skill_scan)

            # Step 13: Tool gap scan (NEW)
            await _run_step(13, "Tool gap scan", _step_tool_gap_scan)

            # Step 14: Correction patterns (NEW)
            await _run_step(14, "Correction patterns", _step_correction_patterns)

            # Step 15: Weekly assessment
            await _run_step(15, "Weekly assessment", _step_weekly_assessment)

            # Step 16: Health Doctor
            await _run_step(16, "Health Doctor", lambda: (get_health_doctor(molly=molly).run_daily(), "completed")[1])

            # Step 17: Graph suggestions (NEW)
            await _run_step(17, "Graph suggestions", _step_graph_suggestions)

            # Step 18: Contract audits
            await _step_contract_audits()

            # Step 19: Analysis (FULL Opus analysis, not stub)
            # Persist analysis/whatsapp text to a file so steps 21/22 can
            # recover them after a checkpoint resume (where the in-memory
            # analysis_text and whatsapp_summary variables are empty).
            analysis_cache_path = MAINTENANCE_DIR / f"{today}_analysis.json"
            if not _is_done(19, "Analysis"):
                try:
                    from memory.graph import get_graph_summary

                    summary = get_graph_summary()
                    graph_text = (
                        f"Entities: {summary['entity_count']}, Relationships: {summary['relationship_count']}\n"
                        f"Top connected: {summary['top_connected']}\nRecent: {summary['recent']}"
                    )
                    analysis = await _run_opus_analysis(
                        _build_maintenance_report(results, run_status=_status(), failed_steps=failed_steps),
                        graph_text,
                        today,
                    )
                    if "---WHATSAPP---" in analysis:
                        analysis_text, whatsapp_summary = [
                            part.strip() for part in analysis.split("---WHATSAPP---", 1)
                        ]
                    else:
                        analysis_text = analysis.strip()
                        whatsapp_summary = analysis.strip()
                    if analysis_text:
                        memory_path = config.WORKSPACE / "MEMORY.md"
                        existing = memory_path.read_text() if memory_path.exists() else ""
                        atomic_write(memory_path, (existing.rstrip() + "\n\n" + analysis_text + "\n").lstrip())
                        _record("Analysis", "MEMORY.md updated")
                    else:
                        _record("Analysis", "empty response")
                    # Cache analysis text for checkpoint resume recovery
                    try:
                        import json as _json
                        atomic_write_json(analysis_cache_path, {
                            "analysis_text": analysis_text,
                            "whatsapp_summary": whatsapp_summary,
                        })
                    except Exception:
                        pass
                except Exception:
                    _record("Analysis", "failed", failed=True)
                finally:
                    _final(19)
            else:
                # Recover analysis text from cache after checkpoint resume
                try:
                    cached = load_json(analysis_cache_path, {})
                    analysis_text = str(cached.get("analysis_text", ""))
                    whatsapp_summary = str(cached.get("whatsapp_summary", ""))
                except Exception:
                    pass

            # Step 20: Issue registry sync (NEW)
            await _run_step(20, "Issue registry", _step_issue_registry)

            # Step 21: Report
            if not _is_done(21, "Report"):
                report = _build_maintenance_report(results, run_status=_status(), failed_steps=failed_steps)
                if analysis_text.strip():
                    report = report.rstrip() + f"\n\n## Analysis\n\n{analysis_text.strip()}\n"
                atomic_write(report_path, report)
                _record("Report", "written")
                _final(21)

            # Step 22: Summary
            if not _is_done(22, "Summary"):
                try:
                    from memory.graph import entity_count, relationship_count

                    if whatsapp_summary:
                        message = f"*\U0001f9e0 Nightly Maintenance — {today}*\n\n{whatsapp_summary}"
                    else:
                        message = (
                            f"*\u2699\ufe0f Maintenance complete — {today}*\n"
                            f"Graph: {entity_count()} entities, {relationship_count()} relationships."
                        )
                        if failed_steps:
                            message += f"\n\u26a0\ufe0f {len(failed_steps)} step(s) failed: " + ", ".join(failed_steps)
                    atomic_write(pending_summary_path, message[:2000])
                    _record("Summary", "saved for morning delivery")
                except Exception:
                    _record("Summary", "failed", failed=True)
                finally:
                    _final(22)

            _RUN_STATE.status = _status()
            _RUN_STATE.last_error = ""
            _RUN_STATE.finished_at = datetime.now(timezone.utc).isoformat()
            _RUN_STATE.failed_steps = list(failed_steps)
            _RUN_STATE.results = dict(results)
            _clear_checkpoint()
            return _RUN_STATE.as_dict()

        except Exception as exc:
            _RUN_STATE.status = "failed"
            _RUN_STATE.last_error = str(exc)
            _RUN_STATE.finished_at = datetime.now(timezone.utc).isoformat()
            _RUN_STATE.failed_steps = list(failed_steps)
            _RUN_STATE.results = dict(results)
            return _RUN_STATE.as_dict()
