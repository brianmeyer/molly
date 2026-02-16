import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    from croniter import croniter
except Exception:  # pragma: no cover - optional dependency guard
    croniter = None

import automations_legacy
import config
from agent import handle_message
from memory.email_digest import build_digest
from utils import atomic_write_json, load_json

log = logging.getLogger(__name__)


REQUIRED_GATEWAY_STEPS = (
    "Morning packet",
    "Noon digest",
    "Evening wrap",
)


def validate_step_results(
    task_results: dict[str, str],
    required_steps: tuple[str, ...] = REQUIRED_GATEWAY_STEPS,
) -> dict[str, Any]:
    """Small deterministic step validator extracted from contract audits."""

    missing = [step for step in required_steps if step not in task_results]
    failed = [
        step
        for step, status in task_results.items()
        if str(status).strip().lower() in {"failed", "error"}
    ]

    status = "pass"
    if missing or failed:
        status = "warn"

    return {
        "status": status,
        "missing_steps": missing,
        "failed_steps": failed,
        "summary": (
            f"missing={len(missing)}, failed={len(failed)}, total={len(task_results)}"
        ),
    }


@dataclass
class GatewayJob:
    automation_id: str
    name: str
    cron_expr: str
    timezone_name: str
    handler: str
    enabled: bool = True


class GatewayEngine:
    """Python-first automation gateway with cron scheduling + webhooks.

    This replaces the direct YAML runtime in ``main.py`` while keeping a legacy
    engine mounted for compatibility-only flows (commitment extraction/reporting
    and proposal mining).
    """

    def __init__(self, molly):
        self.molly = molly
        self._legacy = automations_legacy.AutomationEngine(molly)
        self._state_path = config.AUTOMATIONS_STATE_FILE
        self._state_lock = asyncio.Lock()
        self._tick_lock = asyncio.Lock()
        self._initialized = False
        self._last_tick_at: datetime | None = None
        self._running: set[str] = set()
        self._jobs = self._build_jobs()
        self._state: dict[str, Any] = {
            "jobs": {},
            "daily_message_counts": {},
            "validation": {},
        }

    def _build_jobs(self) -> dict[str, GatewayJob]:
        return {
            "morning-packet": GatewayJob(
                automation_id="morning-packet",
                name="Morning Packet",
                cron_expr="0 7 * * 1-5",
                timezone_name=config.TIMEZONE,
                handler="morning_packet",
                enabled=True,
            ),
            "noon-digest": GatewayJob(
                automation_id="noon-digest",
                name="Noon Digest",
                cron_expr="0 12 * * 1-5",
                timezone_name=config.TIMEZONE,
                handler="noon_digest",
                enabled=True,
            ),
            "evening-wrap": GatewayJob(
                automation_id="evening-wrap",
                name="Evening Wrap",
                cron_expr="0 18 * * 1-5",
                timezone_name=config.TIMEZONE,
                handler="evening_wrap",
                enabled=True,
            ),
        }

    async def initialize(self):
        if self._initialized:
            return
        await self._legacy.initialize()
        await self._load_state()
        self._ensure_job_state()
        self._initialized = True
        log.info("Gateway initialized with %d hardcoded jobs", len(self._jobs))

    def _ensure_job_state(self):
        jobs_state = self._state.setdefault("jobs", {})
        for job_id, job in self._jobs.items():
            row = jobs_state.setdefault(job_id, {})
            row.setdefault("name", job.name)
            row.setdefault("enabled", bool(job.enabled))
            row.setdefault("last_run", "")
            row.setdefault("last_status", "never")
            row.setdefault("last_result_preview", "")
            row.setdefault("next_run", "")

    async def tick(self):
        await self.initialize()

        now_utc = datetime.now(timezone.utc)
        if self._last_tick_at is not None:
            elapsed = (now_utc - self._last_tick_at).total_seconds()
            if elapsed < config.AUTOMATION_TICK_INTERVAL:
                return

        if self._tick_lock.locked():
            return

        async with self._tick_lock:
            self._last_tick_at = now_utc
            for job_id, job in self._jobs.items():
                if job_id in self._running:
                    continue
                if not await self._is_job_enabled(job_id, job):
                    continue
                if not await self._job_due(job_id, job, now_utc):
                    continue
                self._schedule_job(job_id, job, now_utc)

    async def on_message(self, message_data: dict):
        """Delegate message-triggered commitment extraction to legacy engine."""

        await self.initialize()
        await self._legacy.on_message(message_data)

    async def on_webhook(self, event: dict):
        await self.initialize()
        job_id = str(event.get("id", "")).strip()
        payload = event.get("payload", {})
        if not job_id:
            return {"ok": False, "error": "missing webhook id"}
        return await self.handle_webhook(job_id, payload if isinstance(payload, dict) else {})

    async def handle_webhook(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        await self.initialize()
        normalized = str(job_id or "").strip()
        if normalized == "gateway-status":
            return {"ok": True, "status": await self.gateway_status()}

        # Supported webhooks map to hardcoded gateway routines.
        if normalized in {"morning-packet", "noon-digest", "evening-wrap"}:
            job = self._jobs.get(normalized)
            if not job:
                return {"ok": False, "error": f"unknown webhook id: {normalized}"}
            result = await self._execute_job(job, datetime.now(timezone.utc), payload=payload)
            return {"ok": True, "job": normalized, "result": result}

        return {"ok": False, "error": f"unknown webhook id: {normalized}"}

    async def _is_job_enabled(self, job_id: str, job: GatewayJob) -> bool:
        async with self._state_lock:
            jobs_state = self._state.setdefault("jobs", {})
            row = jobs_state.setdefault(job_id, {})
            return bool(row.get("enabled", job.enabled))

    async def _job_due(self, job_id: str, job: GatewayJob, now_utc: datetime) -> bool:
        if croniter is None:
            return False

        tz = ZoneInfo(job.timezone_name)
        now_local = now_utc.astimezone(tz).replace(second=0, microsecond=0)

        async with self._state_lock:
            jobs_state = self._state.setdefault("jobs", {})
            row = jobs_state.setdefault(job_id, {})
            next_run_raw = str(row.get("next_run", "")).strip()
            if next_run_raw:
                try:
                    next_run = datetime.fromisoformat(next_run_raw)
                except ValueError:
                    next_run = None
            else:
                next_run = None

            if next_run is None:
                # Seed the first run target using current local time.
                next_run = croniter(job.cron_expr, now_local - timedelta(minutes=1)).get_next(datetime)
                row["next_run"] = next_run.isoformat()
                await self._save_state_locked()
                return False

            return now_local >= next_run

    def _schedule_job(self, job_id: str, job: GatewayJob, now_utc: datetime):
        self._running.add(job_id)
        task = asyncio.create_task(
            self._run_job(job_id, job, now_utc),
            name=f"gateway:{job_id}",
        )

        def _done(done_task: asyncio.Task):
            self._running.discard(job_id)
            if done_task.cancelled():
                return
            exc = done_task.exception()
            if exc:
                log.error("Gateway job failed: %s", job_id, exc_info=exc)

        task.add_done_callback(_done)

    async def _run_job(self, job_id: str, job: GatewayJob, now_utc: datetime):
        try:
            result = await self._execute_job(job, now_utc)
            status = "success"
            preview = (result or "")[:500]
        except Exception as exc:  # pragma: no cover - operational guard
            status = "failed"
            preview = str(exc)
            log.error("Gateway job execution failed: %s", job_id, exc_info=True)

        tz = ZoneInfo(job.timezone_name)
        now_local = now_utc.astimezone(tz).replace(second=0, microsecond=0)
        next_run = ""
        if croniter is not None:
            next_run = croniter(job.cron_expr, now_local).get_next(datetime).isoformat()

        async with self._state_lock:
            row = self._state.setdefault("jobs", {}).setdefault(job_id, {})
            row["name"] = job.name
            row["last_run"] = now_utc.isoformat()
            row["last_status"] = status
            row["last_result_preview"] = preview
            row["next_run"] = next_run
            self._refresh_validation_locked()
            await self._save_state_locked()

    async def _execute_job(
        self,
        job: GatewayJob,
        now_utc: datetime,
        *,
        payload: dict[str, Any] | None = None,
    ) -> str:
        handler = getattr(self, f"_job_{job.handler}", None)
        if handler is None:
            raise RuntimeError(f"Unknown gateway handler: {job.handler}")
        return await handler(now_utc=now_utc, payload=payload or {})

    async def _job_morning_packet(self, now_utc: datetime, payload: dict[str, Any]) -> str:
        parts: list[str] = []

        pending_summary = config.WORKSPACE / "memory" / "maintenance" / "pending_summary.txt"
        if pending_summary.exists():
            text = pending_summary.read_text().strip()
            if text:
                parts.append(text)
                try:
                    pending_summary.unlink()
                except Exception:
                    log.debug("Could not delete pending summary after read", exc_info=True)

        digest = build_digest("morning")
        if "NO_DIGEST_ITEMS" not in digest:
            parts.append(digest)

        briefing_prompt = (
            "Create a concise morning briefing under 250 words. "
            "Include urgent actions, calendar highlights, and top follow-ups."
        )
        briefing = await self._agent_summary(briefing_prompt)
        if briefing:
            parts.append(briefing)

        text = "\n\n".join(p for p in parts if p.strip())
        if not text:
            text = "Morning packet: no new maintenance, email, or briefing items."

        await self._send_owner_message(text, now_utc)
        return text

    async def _job_noon_digest(self, now_utc: datetime, payload: dict[str, Any]) -> str:
        digest = build_digest("noon")
        if "NO_DIGEST_ITEMS" in digest:
            return "No noon digest items"
        await self._send_owner_message(digest, now_utc)
        return digest

    async def _job_evening_wrap(self, now_utc: datetime, payload: dict[str, Any]) -> str:
        prompt = (
            "Build an end-of-day wrap for Brian with completed items, open loops, "
            "and tomorrow priorities. Keep it under 220 words."
        )
        wrap = await self._agent_summary(prompt)
        if not wrap:
            wrap = "End-of-day wrap: no additional updates to report."

        evening_digest = build_digest("evening")
        if "NO_DIGEST_ITEMS" not in evening_digest:
            wrap = f"{wrap}\n\n{evening_digest}"

        await self._send_owner_message(wrap, now_utc)
        return wrap

    async def _agent_summary(self, prompt: str) -> str:
        owner_jid = self.molly._get_owner_dm_jid() if self.molly else ""
        if not owner_jid:
            return ""

        session_key = "gateway:summary"
        session_id = self.molly.sessions.get(session_key)
        response, new_session_id = await handle_message(
            prompt,
            owner_jid,
            session_id,
            approval_manager=getattr(self.molly, "approvals", None),
            molly_instance=self.molly,
            source="gateway",
        )
        if new_session_id:
            self.molly.sessions[session_key] = new_session_id
            self.molly.save_sessions()
        return (response or "").strip()

    async def _send_owner_message(self, text: str, now_utc: datetime):
        owner_jid = self.molly._get_owner_dm_jid() if self.molly else ""
        if not owner_jid or not getattr(self.molly, "wa", None):
            return

        if not await self._can_send_daily_message(now_utc):
            log.info("Gateway daily cap reached; skipping owner message")
            return

        self.molly._track_send(self.molly.wa.send_message(owner_jid, text))
        await self._mark_daily_message_sent(now_utc)

    async def _can_send_daily_message(self, now_utc: datetime) -> bool:
        local_day = now_utc.astimezone(ZoneInfo(config.TIMEZONE)).date().isoformat()
        async with self._state_lock:
            counts = self._state.setdefault("daily_message_counts", {})
            count = int(counts.get(local_day, 0) or 0)
            return count < 5

    async def _mark_daily_message_sent(self, now_utc: datetime):
        local_day = now_utc.astimezone(ZoneInfo(config.TIMEZONE)).date().isoformat()
        async with self._state_lock:
            counts = self._state.setdefault("daily_message_counts", {})
            counts[local_day] = int(counts.get(local_day, 0) or 0) + 1
            # Keep the last 14 days only.
            keep_days = sorted(counts.keys())[-14:]
            self._state["daily_message_counts"] = {k: counts[k] for k in keep_days}
            await self._save_state_locked()

    async def get_status_rows(self) -> list[dict[str, Any]]:
        await self.initialize()
        rows: list[dict[str, Any]] = []
        async with self._state_lock:
            jobs_state = dict(self._state.get("jobs", {}))

        for job_id, job in self._jobs.items():
            row = jobs_state.get(job_id, {})
            rows.append(
                {
                    "id": job_id,
                    "name": job.name,
                    "enabled": bool(row.get("enabled", job.enabled)),
                    "last_run": str(row.get("last_run", "-") or "-"),
                    "last_status": str(row.get("last_status", "never") or "never"),
                    "next_run": str(row.get("next_run", "-") or "-"),
                    "trigger_type": "schedule",
                }
            )

        return sorted(rows, key=lambda item: item["name"].lower())

    async def status_report(self) -> str:
        rows = await self.get_status_rows()
        lines = [f"Gateway automations ({len(rows)})", ""]
        for row in rows:
            enabled = "enabled" if row["enabled"] else "disabled"
            lines.append(f"- {row['name']} ({row['id']}) [{enabled}]")
            lines.append(f"  Trigger: {row['trigger_type']}")
            lines.append(f"  Last run: {row['last_run']} ({row['last_status']})")
            lines.append(f"  Next run: {row['next_run']}")
        validation = await self._validation_snapshot()
        lines.append("")
        lines.append(f"Validation: {validation.get('status', 'pass')} ({validation.get('summary', '')})")
        return "\n".join(lines)

    async def status_summary(self) -> dict[str, Any]:
        rows = await self.get_status_rows()
        enabled = sum(1 for row in rows if row["enabled"])
        last_runs = [row["last_run"] for row in rows if row["last_run"] and row["last_run"] != "-"]
        return {
            "loaded": len(rows),
            "enabled": enabled,
            "last_run": max(last_runs) if last_runs else None,
        }

    async def commitments_report(self) -> str:
        # Keep the proven commitment reporting implementation during migration.
        return await self._legacy.commitments_report()

    async def gateway_status(self) -> dict[str, Any]:
        rows = await self.get_status_rows()
        validation = await self._validation_snapshot()
        return {
            "jobs": rows,
            "validation": validation,
        }

    async def _validation_snapshot(self) -> dict[str, Any]:
        async with self._state_lock:
            return dict(self._state.get("validation", {}))

    def _refresh_validation_locked(self):
        jobs = self._state.get("jobs", {})
        step_results = {
            "Morning packet": str(jobs.get("morning-packet", {}).get("last_status", "missing")),
            "Noon digest": str(jobs.get("noon-digest", {}).get("last_status", "missing")),
            "Evening wrap": str(jobs.get("evening-wrap", {}).get("last_status", "missing")),
        }
        self._state["validation"] = validate_step_results(step_results)

    async def _load_state(self):
        data = load_json(self._state_path, {})
        if not isinstance(data, dict):
            data = {}
        data.setdefault("jobs", {})
        data.setdefault("daily_message_counts", {})
        data.setdefault("validation", {})
        self._state = data

    async def _save_state_locked(self):
        def _write():
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self._state_path, self._state)

        await asyncio.to_thread(_write)


def propose_automation(operational_logs: list[dict]) -> str | None:
    """Compatibility proposal helper used by self_improve.py.

    We keep legacy mining output during migration so proposal quality does not
    regress while the gateway scheduler is replacing YAML execution paths.
    """

    return automations_legacy.propose_automation(operational_logs)


def attach_gateway_routes(app, molly) -> None:
    """Attach gateway webhook/status endpoints to FastAPI app."""

    try:
        from fastapi import Body
    except Exception:  # pragma: no cover - FastAPI optional in some contexts
        return

    @app.get("/gateway/status")
    async def gateway_status_endpoint():
        engine = getattr(molly, "automations", None)
        if engine is None:
            return {"ok": False, "error": "gateway not initialized"}
        if hasattr(engine, "gateway_status"):
            return {"ok": True, "status": await engine.gateway_status()}
        return {"ok": False, "error": "gateway status unavailable"}

    @app.post("/webhook/{job_id}")
    async def gateway_webhook_endpoint(job_id: str, payload: dict[str, Any] = Body(default={})):  # type: ignore[misc]
        engine = getattr(molly, "automations", None)
        if engine is None:
            return {"ok": False, "error": "gateway not initialized"}
        if hasattr(engine, "handle_webhook"):
            return await engine.handle_webhook(job_id, payload or {})
        return {"ok": False, "error": "webhook handler unavailable"}


__all__ = [
    "GatewayEngine",
    "propose_automation",
    "validate_step_results",
    "attach_gateway_routes",
]
