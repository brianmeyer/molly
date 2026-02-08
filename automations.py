import asyncio
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

import config
from agent import handle_message
from automation_triggers import BaseTrigger, create_trigger

log = logging.getLogger(__name__)

_STEP_OUTPUT_RE = re.compile(r"\{([a-zA-Z0-9_-]+)\.output\}")
_TRIGGER_RE = re.compile(r"\{trigger\.([a-zA-Z0-9_.-]+)\}")
_SIMPLE_RENDER_TOKEN_RE = re.compile(r"\{([a-zA-Z0-9_.-]+)\}")


def _parse_time_hhmm(value: str, fallback: str) -> time:
    raw = (value or fallback).strip()
    try:
        hour, minute = raw.split(":")
        return time(hour=int(hour), minute=int(minute))
    except Exception:
        hour, minute = fallback.split(":")
        return time(hour=int(hour), minute=int(minute))


def _parse_duration_seconds(value: Any, default_seconds: int = 0) -> int:
    if value is None:
        return default_seconds
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, str):
        value = value.strip().lower()
        match = re.fullmatch(r"(\d+)\s*([smhd]?)", value)
        if not match:
            return default_seconds
        amount = int(match.group(1))
        unit = match.group(2) or "s"
        mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        return amount * mult
    return default_seconds


def _iso_to_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _render_path(payload: dict, path_expr: str, default: str = "") -> str:
    current: Any = payload
    for part in path_expr.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    if current is None:
        return default
    return str(current)


@dataclass
class Automation:
    automation_id: str
    name: str
    path: Path
    enabled: bool
    version: int
    trigger_cfg: dict
    trigger: BaseTrigger
    conditions: list[dict]
    pipeline: list[dict]
    min_interval_s: int


class AutomationEngine:
    def __init__(self, molly):
        self.molly = molly
        self.automations_dir = config.AUTOMATIONS_DIR
        self.state_path = config.AUTOMATIONS_STATE_FILE
        self._state: dict[str, Any] = {"automations": {}}
        self._automations: dict[str, Automation] = {}
        self._state_lock = asyncio.Lock()
        self._tick_lock = asyncio.Lock()
        self._last_tick_at: datetime | None = None
        self._running: set[str] = set()
        self._tasks: set[asyncio.Task] = set()
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return
        await asyncio.to_thread(self.automations_dir.mkdir, parents=True, exist_ok=True)
        await self._load_state()
        await self.load_automations()
        self._initialized = True
        log.info("Automation engine initialized with %d automations", len(self._automations))

    async def load_automations(self):
        def _read_all() -> list[tuple[Path, dict]]:
            rows: list[tuple[Path, dict]] = []
            for path in sorted(self.automations_dir.glob("*.yaml")):
                try:
                    data = yaml.safe_load(path.read_text()) or {}
                    if isinstance(data, dict):
                        rows.append((path, data))
                except Exception:
                    log.error("Failed to parse automation YAML: %s", path, exc_info=True)
            return rows

        rows = await asyncio.to_thread(_read_all)
        loaded: dict[str, Automation] = {}
        for path, raw in rows:
            trigger_cfg = raw.get("trigger") or {}
            if not isinstance(trigger_cfg, dict):
                log.warning("Skipping automation with invalid trigger: %s", path)
                continue

            try:
                trigger = create_trigger(trigger_cfg)
            except Exception:
                log.warning("Skipping automation with unsupported trigger: %s", path, exc_info=True)
                continue

            automation_id = str(raw.get("id", path.stem)).strip() or path.stem
            conditions = raw.get("conditions") or []
            if not isinstance(conditions, list):
                conditions = []
            pipeline = raw.get("pipeline") or []
            if not isinstance(pipeline, list):
                pipeline = []

            loaded[automation_id] = Automation(
                automation_id=automation_id,
                name=str(raw.get("name", automation_id)).strip() or automation_id,
                path=path,
                enabled=bool(raw.get("enabled", True)),
                version=int(raw.get("version", 1)),
                trigger_cfg=trigger_cfg,
                trigger=trigger,
                conditions=[c for c in conditions if isinstance(c, dict)],
                pipeline=[p for p in pipeline if isinstance(p, dict)],
                min_interval_s=_parse_duration_seconds(raw.get("min_interval"), default_seconds=0),
            )

        self._automations = loaded

    async def tick(self):
        await self.initialize()
        now = datetime.now(timezone.utc)

        if self._last_tick_at:
            elapsed = (now - self._last_tick_at).total_seconds()
            if elapsed < config.AUTOMATION_TICK_INTERVAL:
                return

        if self._tick_lock.locked():
            return

        async with self._tick_lock:
            self._last_tick_at = now
            context = await self._build_base_context(now=now)
            for automation in self._automations.values():
                trigger_type = str(automation.trigger_cfg.get("type", "")).lower()
                if trigger_type in {"message", "commitment", "webhook"}:
                    continue
                await self._evaluate_and_schedule(automation, context)

    async def on_message(self, message_data: dict):
        await self.initialize()
        now = datetime.now(timezone.utc)
        context = await self._build_base_context(
            now=now,
            message=message_data,
        )
        for automation in self._automations.values():
            trigger_type = str(automation.trigger_cfg.get("type", "")).lower()
            if trigger_type not in {"message", "commitment"}:
                continue
            await self._evaluate_and_schedule(automation, context)

    async def on_webhook(self, event: dict):
        await self.initialize()
        now = datetime.now(timezone.utc)
        context = await self._build_base_context(now=now, webhook_event=event)
        for automation in self._automations.values():
            trigger_type = str(automation.trigger_cfg.get("type", "")).lower()
            if trigger_type != "webhook":
                continue
            await self._evaluate_and_schedule(automation, context)

    async def _evaluate_and_schedule(self, automation: Automation, context: dict):
        if not automation.enabled:
            return
        if automation.automation_id in self._running:
            return

        try:
            should_fire = await automation.trigger.should_fire(context)
        except Exception:
            log.error("Trigger evaluation failed for %s", automation.automation_id, exc_info=True)
            return

        if not should_fire:
            return

        run_context = dict(context)
        run_context["trigger_payload"] = dict(automation.trigger.last_payload)
        run_context["trigger_type"] = str(automation.trigger_cfg.get("type", "")).lower()

        if not await self._passes_conditions(automation, run_context):
            return
        if not await self._can_run(automation, run_context):
            return

        self._schedule_execution(automation, run_context)

    def _schedule_execution(self, automation: Automation, run_context: dict):
        self._running.add(automation.automation_id)
        task = asyncio.create_task(
            self._run_automation(automation, run_context),
            name=f"automation:{automation.automation_id}",
        )
        self._tasks.add(task)

        def _done(done_task: asyncio.Task):
            self._running.discard(automation.automation_id)
            self._tasks.discard(done_task)
            if done_task.cancelled():
                return
            exc = done_task.exception()
            if exc:
                log.error("Automation task failed: %s", automation.automation_id, exc_info=exc)

        task.add_done_callback(_done)

    async def _run_automation(self, automation: Automation, run_context: dict):
        started_at = datetime.now(timezone.utc)
        status = "success"
        error_message = ""
        outputs: dict[str, dict[str, str]] = {}
        payload_hash = self._payload_hash(run_context.get("trigger_payload") or {})
        not_duplicate_keys: dict[str, str] = {}
        run_result_preview = ""

        try:
            for idx, raw_step in enumerate(automation.pipeline):
                step_name = str(raw_step.get("step", f"step_{idx+1}"))
                step = self._render_value(raw_step, outputs, run_context)
                output = ""

                if "channel" in step:
                    output = await self._execute_channel_step(automation, step, outputs, run_context)
                else:
                    output = await self._execute_agent_step(automation, step, outputs, run_context)

                outputs[step_name] = {"output": output}
                if output:
                    run_result_preview = output[:500]
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            log.error("Automation %s failed", automation.automation_id, exc_info=True)

        ended_at = datetime.now(timezone.utc)
        duration_ms = int((ended_at - started_at).total_seconds() * 1000)

        for cond in automation.conditions:
            if str(cond.get("type", "")).lower() != "not_duplicate":
                continue
            key_template = str(cond.get("key", "")).strip()
            if not key_template:
                continue
            not_duplicate_keys[key_template] = self._render_string(
                key_template, outputs, run_context
            )

        async with self._state_lock:
            state = self._state.setdefault("automations", {}).setdefault(automation.automation_id, {})
            state["name"] = automation.name
            state["last_run"] = ended_at.isoformat()
            state["last_status"] = status
            state["last_error"] = error_message
            state["last_duration_ms"] = duration_ms
            state["last_payload_hash"] = payload_hash
            state["last_result_preview"] = run_result_preview
            state["last_trigger_type"] = run_context.get("trigger_type", "")
            state["last_not_duplicate_keys"] = not_duplicate_keys
            state["run_count"] = int(state.get("run_count", 0)) + 1
            await self._save_state_locked()

        log.info(
            "Automation %s finished: %s (%dms)",
            automation.automation_id,
            status,
            duration_ms,
        )

    async def _execute_agent_step(
        self,
        automation: Automation,
        step: dict,
        outputs: dict[str, dict[str, str]],
        run_context: dict,
    ) -> str:
        owner_jid = run_context.get("owner_jid", "")
        if not owner_jid:
            raise RuntimeError("No owner chat available for automation execution")

        step_name = str(step.get("step", "step"))
        agent_tier = str(step.get("agent", "worker")).strip().lower() or "worker"
        prompt = str(step.get("prompt", "")).strip()
        action = str(step.get("action", "")).strip()
        params = step.get("params", {})
        inputs = step.get("inputs", [])
        if not isinstance(inputs, list):
            inputs = []

        context_blocks = []
        for input_name in inputs:
            key = str(input_name)
            value = outputs.get(key, {}).get("output", "")
            if value:
                context_blocks.append(f"[{key}]\n{value}")

        instructions = [
            f"You are executing automation '{automation.name}' step '{step_name}'.",
            f"Delegate this step to the '{agent_tier}' sub-agent via the Task tool.",
            "Keep the response focused on this step's result.",
        ]

        if action:
            action_block = json.dumps(params, indent=2, default=str)
            instructions.append(
                f"Run action/tool '{action}' with these parameters:\n{action_block}"
            )
        if prompt:
            instructions.append(f"Step instructions:\n{prompt}")
        if context_blocks:
            instructions.append("Previous step outputs:\n" + "\n\n".join(context_blocks))

        message = "\n\n".join(instructions)
        session_key = f"automation:{automation.automation_id}:{step_name}"
        session_id = self.molly.sessions.get(session_key)

        response, new_session_id = await handle_message(
            user_message=message,
            chat_id=owner_jid,
            session_id=session_id,
            approval_manager=getattr(self.molly, "approvals", None),
            molly_instance=self.molly,
            source="automation",
        )

        if new_session_id:
            self.molly.sessions[session_key] = new_session_id
            self.molly.save_sessions()

        return (response or "").strip()

    async def _execute_channel_step(
        self,
        automation: Automation,
        step: dict,
        outputs: dict[str, dict[str, str]],
        run_context: dict,
    ) -> str:
        channel = str(step.get("channel", "")).strip().lower()
        if channel != "whatsapp":
            return f"Unsupported channel: {channel}"

        to = str(step.get("to", "owner")).strip().lower()
        if to != "owner":
            return "Only owner delivery is supported"

        owner_jid = run_context.get("owner_jid", "")
        if not owner_jid:
            return "No owner chat available"
        if not getattr(self.molly, "wa", None):
            return "WhatsApp client unavailable"

        message = str(step.get("message", "")).strip()
        if not message:
            if outputs:
                last_key = list(outputs.keys())[-1]
                message = outputs[last_key].get("output", "")

        if not message:
            return "Nothing to deliver"

        self.molly._track_send(self.molly.wa.send_message(owner_jid, message))
        return "Delivered via WhatsApp"

    async def _build_base_context(
        self,
        now: datetime,
        message: dict | None = None,
        webhook_event: dict | None = None,
    ) -> dict:
        owner_jid = self.molly._get_owner_dm_jid() if self.molly else ""
        msg_count_today = await self._message_count_today()
        owner_online = await self._owner_online(30)

        is_owner = False
        if message:
            sender = str(message.get("sender_jid", "")).split("@")[0]
            is_owner = sender in config.OWNER_IDS

        return {
            "now": now,
            "owner_jid": owner_jid,
            "message": message or {},
            "is_owner": is_owner,
            "webhook_event": webhook_event or {},
            "message_count_today": msg_count_today,
            "owner_online": owner_online,
        }

    async def _passes_conditions(self, automation: Automation, run_context: dict) -> bool:
        for cond in automation.conditions:
            cond_type = str(cond.get("type", "")).strip().lower()
            if not cond_type:
                continue
            ok = await self._evaluate_condition(cond_type, cond, automation, run_context)
            if not ok:
                return False
        return True

    async def _evaluate_condition(
        self,
        cond_type: str,
        cond: dict,
        automation: Automation,
        run_context: dict,
    ) -> bool:
        now = run_context["now"]
        state = self._state.get("automations", {}).get(automation.automation_id, {})
        trigger_payload = run_context.get("trigger_payload", {})

        if cond_type == "not_quiet_hours":
            if not self._is_quiet_hours(now, cond):
                return True
            return self._quiet_hours_bypass_allowed(cond, trigger_payload)

        if cond_type == "owner_has_events":
            return await self._owner_has_events(now)

        if cond_type == "owner_online":
            minutes = int(cond.get("minutes", 30))
            return await self._owner_online(minutes)

        if cond_type == "not_duplicate":
            cooldown_s = _parse_duration_seconds(cond.get("cooldown"), default_seconds=0)
            cooldown_minutes = int(cond.get("cooldown_minutes", 0))
            if cooldown_minutes > 0 and cooldown_s == 0:
                cooldown_s = cooldown_minutes * 60

            last_run = _iso_to_datetime(str(state.get("last_run", "")))
            if last_run and cooldown_s > 0:
                elapsed = (now - last_run).total_seconds()
                if elapsed < cooldown_s:
                    return False

            key_template = str(cond.get("key", "")).strip()
            if key_template:
                key = self._render_string(key_template, {}, run_context)
                prev_map = state.get("last_not_duplicate_keys", {})
                prev = prev_map.get(key_template, "")
                if prev and prev == key and cooldown_s > 0 and last_run:
                    elapsed = (now - last_run).total_seconds()
                    if elapsed < cooldown_s:
                        return False
            return True

        if cond_type == "day_of_week":
            allowed = cond.get("days", [])
            if isinstance(allowed, str):
                allowed = [allowed]
            allowed = [str(d).strip().lower()[:3] for d in allowed]
            if not allowed:
                return True
            weekday = now.astimezone(ZoneInfo(config.TIMEZONE)).strftime("%a").lower()[:3]
            return weekday in allowed

        if cond_type == "custom":
            expr = str(cond.get("expression", cond.get("check", ""))).strip()
            if not expr:
                return True
            safe_ctx = {
                "message_count_today": run_context.get("message_count_today", 0),
                "owner_online": run_context.get("owner_online", False),
                "trigger": trigger_payload,
                "now_hour": now.hour,
                "weekday": now.weekday(),
            }
            try:
                return bool(eval(expr, {"__builtins__": {}}, safe_ctx))
            except Exception:
                log.warning("Custom condition failed: %s", expr, exc_info=True)
                return False

        log.warning("Unknown automation condition type: %s", cond_type)
        return False

    async def _can_run(self, automation: Automation, run_context: dict) -> bool:
        state = self._state.get("automations", {}).get(automation.automation_id, {})
        now = run_context["now"]
        last_run = _iso_to_datetime(str(state.get("last_run", "")))
        min_interval = automation.min_interval_s

        if last_run and min_interval > 0:
            if (now - last_run).total_seconds() < min_interval:
                return False

        payload_hash = self._payload_hash(run_context.get("trigger_payload", {}))
        previous_hash = str(state.get("last_payload_hash", ""))
        if payload_hash and previous_hash and payload_hash == previous_hash:
            fallback_window = max(min_interval, 300)
            if last_run and (now - last_run).total_seconds() < fallback_window:
                return False

        return True

    def _is_quiet_hours(self, now: datetime, cond: dict) -> bool:
        tz_name = str(cond.get("timezone", config.QUIET_HOURS_TIMEZONE))
        tz = ZoneInfo(tz_name)
        start = _parse_time_hhmm(str(cond.get("start", config.QUIET_HOURS_START)), config.QUIET_HOURS_START)
        end = _parse_time_hhmm(str(cond.get("end", config.QUIET_HOURS_END)), config.QUIET_HOURS_END)

        local_now = now.astimezone(tz).time()
        if start < end:
            return start <= local_now < end
        return local_now >= start or local_now < end

    def _quiet_hours_bypass_allowed(self, cond: dict, trigger_payload: dict) -> bool:
        vip_bypass = bool(cond.get("vip_bypass", config.QUIET_HOURS_VIP_BYPASS))
        urgent_bypass = bool(cond.get("urgent_bypass", config.QUIET_HOURS_URGENT_BYPASS))

        if urgent_bypass and bool(trigger_payload.get("urgent", False)):
            return True
        if vip_bypass and self._trigger_from_vip(trigger_payload):
            return True
        return False

    def _trigger_from_vip(self, trigger_payload: dict) -> bool:
        vip_contacts = config.VIP_CONTACTS or []
        if not vip_contacts:
            return False

        searchable = json.dumps(trigger_payload, default=str).lower()
        for vip in vip_contacts:
            for key in ("email", "phone", "name"):
                value = str(vip.get(key, "")).strip().lower()
                if value and value in searchable:
                    return True
        return False

    async def _owner_has_events(self, now: datetime) -> bool:
        from tools.google_auth import get_calendar_service

        tz = ZoneInfo(config.TIMEZONE)
        local_now = now.astimezone(tz)
        start_local = datetime.combine(local_now.date(), time.min, tzinfo=tz)
        end_local = datetime.combine(local_now.date(), time.max, tzinfo=tz)

        def _run() -> bool:
            service = get_calendar_service()
            result = (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=start_local.astimezone(timezone.utc).isoformat(),
                    timeMax=end_local.astimezone(timezone.utc).isoformat(),
                    singleEvents=True,
                    maxResults=10,
                    orderBy="startTime",
                )
                .execute()
            )
            return bool(result.get("items", []))

        try:
            return await asyncio.to_thread(_run)
        except Exception:
            log.debug("owner_has_events check failed", exc_info=True)
            return False

    async def _owner_online(self, minutes: int) -> bool:
        owner_ids = set(config.OWNER_IDS)
        candidates = set(owner_ids)
        candidates.update({f"{oid}@s.whatsapp.net" for oid in owner_ids})

        if not candidates:
            return False

        since = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()

        def _run() -> bool:
            conn = sqlite3.connect(str(config.DATABASE_PATH))
            try:
                placeholders = ",".join("?" for _ in candidates)
                sql = (
                    f"SELECT COUNT(*) FROM messages "
                    f"WHERE sender IN ({placeholders}) AND timestamp > ?"
                )
                params = list(candidates) + [since]
                cursor = conn.execute(sql, params)
                return int(cursor.fetchone()[0]) > 0
            finally:
                conn.close()

        try:
            return await asyncio.to_thread(_run)
        except Exception:
            log.debug("owner_online check failed", exc_info=True)
            return False

    async def _message_count_today(self) -> int:
        today_iso = date.today().isoformat()

        def _run() -> int:
            conn = sqlite3.connect(str(config.DATABASE_PATH))
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM messages WHERE timestamp > ?", (today_iso,))
                return int(cursor.fetchone()[0])
            finally:
                conn.close()

        try:
            return await asyncio.to_thread(_run)
        except Exception:
            log.debug("message_count_today check failed", exc_info=True)
            return 0

    async def _load_state(self):
        if not self.state_path.exists():
            self._state = {"automations": {}}
            await self._save_state()
            return

        def _run() -> dict:
            try:
                return json.loads(self.state_path.read_text())
            except Exception:
                return {"automations": {}}

        data = await asyncio.to_thread(_run)
        if not isinstance(data, dict):
            data = {"automations": {}}
        data.setdefault("automations", {})
        self._state = data

    async def _save_state(self):
        async with self._state_lock:
            await self._save_state_locked()

    async def _save_state_locked(self):
        payload = json.dumps(self._state, indent=2, default=str)

        def _write():
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(payload)

        try:
            await asyncio.to_thread(_write)
        except Exception as exc:
            # Keep engine operational even if state path is temporarily unwritable.
            log.warning("Failed to write automation state %s: %s", self.state_path, exc)

    def _payload_hash(self, payload: dict) -> str:
        if not payload:
            return ""
        try:
            return json.dumps(payload, sort_keys=True, default=str)
        except Exception:
            return str(payload)

    def _render_value(self, value: Any, outputs: dict, run_context: dict) -> Any:
        if isinstance(value, str):
            return self._render_string(value, outputs, run_context)
        if isinstance(value, list):
            return [self._render_value(v, outputs, run_context) for v in value]
        if isinstance(value, dict):
            return {k: self._render_value(v, outputs, run_context) for k, v in value.items()}
        return value

    def _render_string(self, text: str, outputs: dict, run_context: dict) -> str:
        trigger_payload = run_context.get("trigger_payload", {})
        now = run_context.get("now") or datetime.now(timezone.utc)
        simple_values = {
            "date": now.date().isoformat(),
            "datetime": now.isoformat(),
            **{k: v.get("output", "") for k, v in outputs.items()},
        }

        def _token_repl(match: re.Match) -> str:
            token = match.group(1)

            if token in simple_values:
                return str(simple_values[token])

            step_match = _STEP_OUTPUT_RE.fullmatch(match.group(0))
            if step_match:
                return outputs.get(step_match.group(1), {}).get("output", "")

            trigger_match = _TRIGGER_RE.fullmatch(match.group(0))
            if trigger_match:
                return _render_path(trigger_payload, trigger_match.group(1), "")

            return match.group(0)

        return _SIMPLE_RENDER_TOKEN_RE.sub(_token_repl, text)

    async def get_status_rows(self) -> list[dict]:
        await self.initialize()
        now = datetime.now(timezone.utc)
        rows = []
        for automation in sorted(self._automations.values(), key=lambda a: a.name.lower()):
            state = self._state.get("automations", {}).get(automation.automation_id, {})
            next_run_dt = await automation.trigger.next_fire_time({"now": now})
            next_run = next_run_dt.isoformat() if next_run_dt else "-"
            rows.append(
                {
                    "id": automation.automation_id,
                    "name": automation.name,
                    "enabled": automation.enabled,
                    "last_run": state.get("last_run", "-"),
                    "last_status": state.get("last_status", "-"),
                    "next_run": next_run,
                    "trigger_type": automation.trigger_cfg.get("type", ""),
                }
            )
        return rows

    async def status_report(self) -> str:
        rows = await self.get_status_rows()
        if not rows:
            return "No automations found in workspace/automations/."

        lines = [f"Automations ({len(rows)})", ""]
        for row in rows:
            enabled = "enabled" if row["enabled"] else "disabled"
            lines.append(f"- {row['name']} ({row['id']}) [{enabled}]")
            lines.append(f"  Trigger: {row['trigger_type']}")
            lines.append(f"  Last run: {row['last_run']} ({row['last_status']})")
            lines.append(f"  Next run: {row['next_run']}")
        return "\n".join(lines)

    async def status_summary(self) -> dict:
        rows = await self.get_status_rows()
        if not rows:
            return {
                "loaded": 0,
                "enabled": 0,
                "last_run": None,
            }

        enabled = sum(1 for row in rows if row["enabled"])
        last_runs = [r["last_run"] for r in rows if r["last_run"] and r["last_run"] != "-"]
        last_run = max(last_runs) if last_runs else None
        return {"loaded": len(rows), "enabled": enabled, "last_run": last_run}


def propose_automation(operational_logs: list[dict]) -> str | None:
    """Phase 7 stub: generate a skeleton automation proposal from logs.

    This is intentionally lightweight for now. Nightly maintenance can call this
    and decide whether to forward the proposal to Brian for approval.
    """
    if not operational_logs:
        return None

    skeleton = {
        "name": "Proposed Automation",
        "enabled": False,
        "version": 1,
        "trigger": {
            "type": "schedule",
            "cron": "0 9 * * 1-5",
            "timezone": config.TIMEZONE,
        },
        "conditions": [
            {"type": "not_quiet_hours"},
        ],
        "pipeline": [
            {
                "step": "analyze_pattern",
                "agent": "analyst",
                "prompt": "Analyze recent operational patterns and produce an actionable summary.",
            },
            {
                "step": "deliver",
                "channel": "whatsapp",
                "to": "owner",
                "message": "{analyze_pattern.output}",
            },
        ],
        "meta": {
            "generated_from_log_count": len(operational_logs),
            "note": "Stub proposal. Phase 7 will implement full pattern mining logic.",
        },
    }
    return yaml.safe_dump(skeleton, sort_keys=False)
