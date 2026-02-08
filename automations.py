import asyncio
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import yaml

import config
from agent import handle_message
from automation_triggers import BaseTrigger, create_trigger

log = logging.getLogger(__name__)

_STEP_OUTPUT_RE = re.compile(r"\{([a-zA-Z0-9_-]+)\.output\}")
_TRIGGER_RE = re.compile(r"\{trigger\.([a-zA-Z0-9_.-]+)\}")
_SIMPLE_RENDER_TOKEN_RE = re.compile(r"\{([a-zA-Z0-9_.-]+)\}")
_SEQUENCE_SPLIT_RE = re.compile(r"\s*->\s*")
_WORD_TOKEN_RE = re.compile(r"[a-z0-9]+")

_AUTOMATION_PROPOSAL_AUTO_ENABLE_THRESHOLD = 0.88
_PROPOSAL_MIN_OCCURRENCES = 3
_PROPOSAL_MAX_PATTERNS = 3

_COMMITMENT_AUTOMATION_ID = "commitment-tracker"
_COMMITMENT_REMINDERS_LIST = "Molly"
_COMMITMENT_SYNC_INTERVAL_S = 300
_COMMITMENT_RECORD_LIMIT = 300

_COMMITMENT_TITLE_PATTERNS = (
    re.compile(r"\bremind me to\s+(.+)$", flags=re.IGNORECASE),
    re.compile(r"\bi['’]?ll\s+(.+)$", flags=re.IGNORECASE),
    re.compile(r"\bi\s+(?:will|can|need to|have to|must)\s+(.+)$", flags=re.IGNORECASE),
    re.compile(r"\bfollow up(?:\s+with|\s+on)?\s+(.+)$", flags=re.IGNORECASE),
)

_COMMITMENT_TRAILING_DUE_RE = re.compile(
    r"\s+\b(?:"
    r"today|tomorrow|tonight|"
    r"this\s+(?:morning|afternoon|evening)|"
    r"next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"by\s+(?:today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{4}-\d{2}-\d{2})|"
    r"on\s+(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{4}-\d{2}-\d{2})"
    r")(?:\b.*)?$",
    flags=re.IGNORECASE,
)

_DUE_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b", flags=re.IGNORECASE)
_DUE_TIME_12H_RE = re.compile(
    r"\b(?:at|by)?\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
    flags=re.IGNORECASE,
)
_DUE_TIME_24H_RE = re.compile(r"\b(?:at|by)\s*(\d{1,2}):(\d{2})\b", flags=re.IGNORECASE)
_DUE_WEEKDAY_RE = re.compile(
    r"\b(next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    flags=re.IGNORECASE,
)

_WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_COMMITMENT_STOPWORDS = {
    "a", "about", "an", "and", "at", "by", "for", "i", "me", "my", "of", "on", "please", "the", "to", "with",
}

_OPERATIONAL_LOG_SCHEMA_DOC = {
    "tool_name": "Primary action/tool identifier for each event.",
    "created_at": "ISO-8601 event timestamp used for sequencing and recurrence detection.",
    "success": "Boolean or int success flag used to measure reliability of repeated workflows.",
    "latency_ms": "Optional duration signal used as evidence context.",
    "error_message": "Failure detail; non-empty values reduce confidence.",
    "parameters": "Optional JSON payload sampled to contextualize inferred workflows.",
}


def _clean_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _normalize_commitment_title(value: str) -> str:
    tokens = _WORD_TOKEN_RE.findall((value or "").lower())
    if not tokens:
        return ""
    filtered = [token for token in tokens if token not in _COMMITMENT_STOPWORDS]
    if not filtered:
        filtered = tokens
    return " ".join(filtered)


def _titles_similar(left: str, right: str) -> bool:
    norm_left = _normalize_commitment_title(left)
    norm_right = _normalize_commitment_title(right)
    if not norm_left or not norm_right:
        return False
    if norm_left == norm_right:
        return True
    if norm_left in norm_right or norm_right in norm_left:
        shorter = min(len(norm_left), len(norm_right))
        if shorter >= 8:
            return True

    left_tokens = set(norm_left.split())
    right_tokens = set(norm_right.split())
    if left_tokens and right_tokens:
        overlap = len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))
        if overlap >= 0.75 and min(len(left_tokens), len(right_tokens)) >= 2:
            return True

    return SequenceMatcher(None, norm_left, norm_right).ratio() >= 0.88


def _parse_payload_timestamp(value: Any, fallback: datetime) -> datetime:
    if value is None:
        return fallback

    if isinstance(value, (int, float)):
        raw_num = float(value)
    else:
        raw = str(value).strip()
        if not raw:
            return fallback
        if raw.isdigit():
            raw_num = float(raw)
        else:
            maybe_dt = _iso_to_datetime(raw.replace("Z", "+00:00"))
            if maybe_dt is None:
                return fallback
            if maybe_dt.tzinfo is None:
                maybe_dt = maybe_dt.replace(tzinfo=timezone.utc)
            return maybe_dt.astimezone(timezone.utc)

    if raw_num > 1e18:
        raw_num /= 1_000_000_000.0
    elif raw_num > 1e14:
        raw_num /= 1_000_000.0
    elif raw_num > 1e11:
        raw_num /= 1_000.0

    try:
        return datetime.fromtimestamp(raw_num, tz=timezone.utc)
    except Exception:
        return fallback


def _extract_commitment_title(message_text: str) -> str:
    stripped = (message_text or "").strip()
    if not stripped:
        return "Untitled commitment"

    first_line = next((line.strip() for line in stripped.splitlines() if line.strip()), stripped)
    candidate = first_line

    for pattern in _COMMITMENT_TITLE_PATTERNS:
        match = pattern.search(first_line)
        if match:
            candidate = match.group(1).strip()
            break

    candidate = re.split(r"[.!?]\s", candidate, maxsplit=1)[0].strip()
    candidate = _COMMITMENT_TRAILING_DUE_RE.sub("", candidate).strip(" ,;:-")
    candidate = _clean_spaces(candidate)
    if not candidate:
        candidate = _clean_spaces(first_line)
    if len(candidate) > 140:
        candidate = candidate[:137].rstrip() + "..."
    return candidate or "Untitled commitment"


def _extract_due_time_components(message_text: str) -> tuple[int, int] | None:
    text = (message_text or "").lower()

    match_12h = _DUE_TIME_12H_RE.search(text)
    if match_12h:
        hour = int(match_12h.group(1))
        minute = int(match_12h.group(2) or 0)
        ampm = (match_12h.group(3) or "").lower()
        hour %= 12
        if ampm == "pm":
            hour += 12
        return hour, minute

    match_24h = _DUE_TIME_24H_RE.search(text)
    if match_24h:
        hour = int(match_24h.group(1))
        minute = int(match_24h.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute
    return None


def _extract_due_datetime(message_text: str, now_utc: datetime) -> datetime | None:
    text = (message_text or "").strip()
    if not text:
        return None

    tz = ZoneInfo(config.TIMEZONE)
    now_local = now_utc.astimezone(tz)
    lowered = text.lower()

    due_date = None
    date_match = _DUE_ISO_DATE_RE.search(lowered)
    if date_match:
        try:
            due_date = datetime.fromisoformat(date_match.group(1)).date()
        except ValueError:
            due_date = None

    if due_date is None:
        if "tomorrow" in lowered:
            due_date = now_local.date() + timedelta(days=1)
        elif any(token in lowered for token in ("today", "tonight", "this morning", "this afternoon", "this evening")):
            due_date = now_local.date()
        else:
            weekday_match = _DUE_WEEKDAY_RE.search(lowered)
            if weekday_match:
                target = _WEEKDAY_INDEX[weekday_match.group(2).lower()]
                days_ahead = (target - now_local.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                due_date = now_local.date() + timedelta(days=days_ahead)

    explicit_time = _extract_due_time_components(lowered)
    if explicit_time:
        hour, minute = explicit_time
    else:
        if "tonight" in lowered:
            hour, minute = 19, 0
        elif "afternoon" in lowered:
            hour, minute = 15, 0
        elif "evening" in lowered:
            hour, minute = 18, 0
        else:
            hour, minute = 9, 0

    if due_date is None and explicit_time:
        today_candidate = datetime.combine(now_local.date(), time(hour=hour, minute=minute), tzinfo=tz)
        due_date = now_local.date() if today_candidate > now_local else now_local.date() + timedelta(days=1)

    if due_date is None:
        return None

    due_local = datetime.combine(due_date, time(hour=hour, minute=minute), tzinfo=tz)
    return due_local.astimezone(timezone.utc)


def _format_due_display(iso_value: str) -> str:
    parsed = _iso_to_datetime((iso_value or "").replace("Z", "+00:00"))
    if parsed is None:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    local = parsed.astimezone(ZoneInfo(config.TIMEZONE))
    return local.strftime("%Y-%m-%d %I:%M %p %Z")


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


def _coerce_non_negative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed >= 0 else default


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
        self._last_commitment_sync_at: datetime | None = None
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
            automation_id = str(raw.get("id", path.stem)).strip() or path.stem
            trigger_cfg = raw.get("trigger") or {}
            if not isinstance(trigger_cfg, dict):
                log.warning("Skipping automation with invalid trigger: %s", path)
                continue

            runtime_trigger_cfg = dict(trigger_cfg)
            runtime_trigger_cfg["_automation_id"] = automation_id
            runtime_trigger_cfg["_state_path"] = str(self.state_path)

            try:
                trigger = create_trigger(runtime_trigger_cfg)
            except Exception:
                log.warning("Skipping automation with unsupported trigger: %s", path, exc_info=True)
                continue

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
            await self._maybe_sync_commitment_status(now)
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
            if status == "success":
                self._update_email_trigger_high_water_locked(
                    automation=automation,
                    run_context=run_context,
                    state=state,
                )
            await self._save_state_locked()

        if status == "success":
            try:
                await self._run_post_execution_hooks(
                    automation=automation,
                    run_context=run_context,
                    outputs=outputs,
                    ended_at=ended_at,
                )
            except Exception:
                log.error(
                    "Post-execution hook failed for automation %s",
                    automation.automation_id,
                    exc_info=True,
                )

        log.info(
            "Automation %s finished: %s (%dms)",
            automation.automation_id,
            status,
            duration_ms,
        )

    def _update_email_trigger_high_water_locked(
        self,
        automation: Automation,
        run_context: dict,
        state: dict,
    ) -> None:
        if str(run_context.get("trigger_type", "")).strip().lower() != "email":
            return

        payload = run_context.get("trigger_payload", {})
        if not isinstance(payload, dict):
            return

        candidate = payload.get("_email_trigger_state", {})
        if not isinstance(candidate, dict):
            return

        candidate_ts_ms = _coerce_non_negative_int(candidate.get("high_water_internal_ts_ms"), 0)
        candidate_ids_raw = candidate.get("high_water_ids", [])
        candidate_ids: list[str] = []
        if isinstance(candidate_ids_raw, list):
            candidate_ids = [
                str(value).strip()
                for value in candidate_ids_raw
                if str(value).strip()
            ]

        fallback_id = str(candidate.get("high_water_message_id", "")).strip()
        if fallback_id and fallback_id not in candidate_ids:
            candidate_ids.append(fallback_id)

        if candidate_ts_ms <= 0 and not candidate_ids:
            return

        trigger_state = state.setdefault("trigger_state", {})
        if not isinstance(trigger_state, dict):
            trigger_state = {}
            state["trigger_state"] = trigger_state
        email_state = trigger_state.setdefault("email", {})
        if not isinstance(email_state, dict):
            email_state = {}
            trigger_state["email"] = email_state

        current_ts_ms = _coerce_non_negative_int(email_state.get("high_water_internal_ts_ms"), 0)
        current_ids_raw = email_state.get("high_water_ids", [])
        current_ids = [
            str(value).strip()
            for value in (current_ids_raw if isinstance(current_ids_raw, list) else [])
            if str(value).strip()
        ]

        if candidate_ts_ms < current_ts_ms:
            return

        if candidate_ts_ms > current_ts_ms:
            merged_ids = candidate_ids
        else:
            merged_ids = list(dict.fromkeys([*current_ids, *candidate_ids]))

        if not merged_ids and fallback_id:
            merged_ids = [fallback_id]

        if candidate_ts_ms > 0:
            email_state["high_water_internal_ts_ms"] = candidate_ts_ms
            email_state["high_water_unix_s"] = candidate_ts_ms // 1000
        email_state["high_water_ids"] = merged_ids[-50:]
        email_state["high_water_message_id"] = (
            merged_ids[-1] if merged_ids else str(email_state.get("high_water_message_id", ""))
        )
        email_state["last_query"] = str(payload.get("query", ""))
        email_state["updated_at"] = datetime.now(timezone.utc).isoformat()

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
        if automation.automation_id == "email-triage" and step_name == "classify":
            instructions.append(
                "Output rules:\n"
                "- If there are no urgent and no important emails, return exactly: NO_ACTIONABLE_ITEMS\n"
                "- Otherwise return only urgent and important items in a concise report.\n"
                "- Exclude FYI-only items from the report text."
            )

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

        if self._should_skip_email_triage_delivery(automation, outputs, message):
            log.info("- Email triage: no actionable items, skipping report")
            return "Skipped delivery: no actionable items"

        self.molly._track_send(self.molly.wa.send_message(owner_jid, message))
        return "Delivered via WhatsApp"

    def _should_skip_email_triage_delivery(
        self,
        automation: Automation,
        outputs: dict[str, dict[str, str]],
        message: str,
    ) -> bool:
        if automation.automation_id != "email-triage":
            return False

        classify_output = str(outputs.get("classify", {}).get("output", "")).strip()
        source_text = classify_output or (message or "").strip()
        return not self._email_triage_has_actionable_items(source_text)

    def _email_triage_has_actionable_items(self, text: str) -> bool:
        content = (text or "").strip()
        if not content:
            return False

        lowered = content.lower()
        if "no_actionable_items" in lowered:
            return False

        # Structured payload support if the classifier returns JSON.
        try:
            payload = json.loads(content)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            urgent = payload.get("urgent", [])
            important = payload.get("important", [])
            if isinstance(urgent, list) and isinstance(important, list):
                return bool(urgent or important)

        urgent_count_match = re.search(r"\burgent_count\s*:\s*(\d+)\b", lowered)
        important_count_match = re.search(r"\bimportant_count\s*:\s*(\d+)\b", lowered)
        if urgent_count_match and important_count_match:
            return (
                int(urgent_count_match.group(1)) + int(important_count_match.group(1))
            ) > 0

        no_urgent = bool(
            re.search(r"\burgent\b[^\n]{0,40}\b(?:none|no\b|0\b|n/?a)\b", lowered)
            or re.search(r"\b(?:no|none|0)\b[^\n]{0,40}\burgent\b", lowered)
        )
        no_important = bool(
            re.search(r"\bimportant\b[^\n]{0,40}\b(?:none|no\b|0\b|n/?a)\b", lowered)
            or re.search(r"\b(?:no|none|0)\b[^\n]{0,40}\bimportant\b", lowered)
        )
        if no_urgent and no_important:
            return False

        if (
            "all duplicates from prior runs" in lowered
            and ("urgent" not in lowered or no_urgent)
            and ("important" not in lowered or no_important)
        ):
            return False

        return True

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

    async def _run_post_execution_hooks(
        self,
        automation: Automation,
        run_context: dict,
        outputs: dict[str, dict[str, str]],
        ended_at: datetime,
    ):
        if automation.automation_id != _COMMITMENT_AUTOMATION_ID:
            return
        await self._handle_commitment_tracker_hook(run_context, outputs, ended_at)

    async def _handle_commitment_tracker_hook(
        self,
        run_context: dict,
        outputs: dict[str, dict[str, str]],
        ended_at: datetime,
    ):
        extract_output = str(outputs.get("extract_commitments", {}).get("output", "")).strip()
        if not extract_output:
            return
        if "NO_COMMITMENT" in extract_output.upper():
            return

        trigger_payload = run_context.get("trigger_payload", {})
        if not isinstance(trigger_payload, dict):
            return

        message_text = str(trigger_payload.get("message_text", "")).strip()
        if not message_text:
            return

        now_utc = ended_at if ended_at.tzinfo else ended_at.replace(tzinfo=timezone.utc)
        title = _extract_commitment_title(message_text)
        due_dt = _extract_due_datetime(message_text, now_utc)
        due_iso = due_dt.isoformat() if due_dt else ""

        source_dt = _parse_payload_timestamp(trigger_payload.get("timestamp"), fallback=now_utc)
        source_date = source_dt.astimezone(ZoneInfo(config.TIMEZONE)).date().isoformat()
        source_note = f"From WhatsApp conversation on {source_date}"

        reminders = await self._list_molly_reminders(include_completed=True)
        reminder_match = self._find_matching_reminder(reminders, title)

        record_id, should_create_reminder = await self._upsert_commitment_record(
            title=title,
            raw_text=message_text,
            due_iso=due_iso,
            source_note=source_note,
            trigger_payload=trigger_payload,
            recorded_at=now_utc,
            reminder_match=reminder_match,
        )
        if not record_id:
            return

        if not should_create_reminder:
            return

        created = await self._create_commitment_reminder(
            title=title,
            due_dt=due_dt,
            source_note=source_note,
            owner_jid=str(run_context.get("owner_jid", "")),
        )
        if not created:
            return

        await self._attach_reminder_to_commitment(record_id, created, now_utc)

    async def _maybe_sync_commitment_status(self, now_utc: datetime, force: bool = False):
        if not force and self._last_commitment_sync_at is not None:
            elapsed = (now_utc - self._last_commitment_sync_at).total_seconds()
            if elapsed < _COMMITMENT_SYNC_INTERVAL_S:
                return
        self._last_commitment_sync_at = now_utc
        await self._sync_commitment_status(now_utc)

    async def _sync_commitment_status(self, now_utc: datetime):
        reminders = await self._list_molly_reminders(include_completed=True)
        if not reminders:
            return

        reminders_by_id = {
            str(row.get("id", "")).strip(): row
            for row in reminders
            if str(row.get("id", "")).strip()
        }
        now_iso = now_utc.isoformat()

        async with self._state_lock:
            tracker_state, commitments = self._get_commitment_state_locked()
            if not commitments:
                return

            changed = False
            completion_events = tracker_state.setdefault("completion_events", [])
            if not isinstance(completion_events, list):
                completion_events = []
                tracker_state["completion_events"] = completion_events

            for record in commitments:
                if not isinstance(record, dict):
                    continue

                reminder = None
                reminder_id = str(record.get("reminder_id", "")).strip()
                if reminder_id and reminder_id in reminders_by_id:
                    reminder = reminders_by_id[reminder_id]
                elif not reminder_id:
                    reminder = self._find_matching_reminder(reminders, str(record.get("title", "")))

                if reminder is None:
                    continue

                was_completed = str(record.get("status", "")).lower() == "completed"
                changed |= self._apply_reminder_to_record(record, reminder, now_iso)
                is_completed = str(record.get("status", "")).lower() == "completed"

                if is_completed and not was_completed:
                    title = str(record.get("title", "(untitled)"))
                    completed_at = str(record.get("completed_at", now_iso))
                    completion_events.append({"title": title, "completed_at": completed_at})
                    log.info("Commitment completed: %s", title)
                    changed = True

            if len(completion_events) > 200:
                tracker_state["completion_events"] = completion_events[-200:]

            if changed:
                await self._save_state_locked()

    async def _list_molly_reminders(self, include_completed: bool) -> list[dict]:
        try:
            from tools.reminders import list_reminders

            rows = await asyncio.to_thread(
                list_reminders,
                _COMMITMENT_REMINDERS_LIST,
                include_completed,
            )
            return [row for row in rows if isinstance(row, dict)]
        except Exception:
            log.debug("Failed to read Apple reminders", exc_info=True)
            return []

    async def _create_commitment_reminder(
        self,
        title: str,
        due_dt: datetime | None,
        source_note: str,
        owner_jid: str,
    ) -> dict | None:
        approval_mgr = getattr(self.molly, "approvals", None) if self.molly else None
        if not approval_mgr or not owner_jid:
            log.info(
                "Skipping reminder creation for '%s' because no approval channel is available",
                title,
            )
            return None

        tool_input = {
            "operation": "create",
            "list": _COMMITMENT_REMINDERS_LIST,
            "title": title,
            "notes": source_note,
        }
        if due_dt:
            tool_input["due_at"] = due_dt.isoformat()

        try:
            decision = await approval_mgr.request_tool_approval(
                "reminders",
                tool_input,
                owner_jid,
                self.molly,
            )
        except Exception:
            log.debug("Reminder approval request failed", exc_info=True)
            return None
        if decision is not True:
            log.info("Reminder creation not approved for commitment '%s'", title)
            return None

        try:
            from tools.reminders import create_reminder

            return await asyncio.to_thread(
                create_reminder,
                title,
                source_note,
                due_dt,
                _COMMITMENT_REMINDERS_LIST,
            )
        except Exception:
            log.error("Failed creating Apple reminder for commitment", exc_info=True)
            return None

    def _find_matching_reminder(self, reminders: list[dict], title: str) -> dict | None:
        for reminder in reminders:
            reminder_title = str(reminder.get("title", "")).strip()
            if reminder_title and _titles_similar(title, reminder_title):
                return reminder
        return None

    async def _upsert_commitment_record(
        self,
        title: str,
        raw_text: str,
        due_iso: str,
        source_note: str,
        trigger_payload: dict,
        recorded_at: datetime,
        reminder_match: dict | None,
    ) -> tuple[str, bool]:
        now_iso = recorded_at.isoformat()
        async with self._state_lock:
            tracker_state, commitments = self._get_commitment_state_locked()

            existing: dict | None = None
            for row in reversed(commitments):
                if not isinstance(row, dict):
                    continue
                if str(row.get("status", "open")).lower() == "completed":
                    continue
                if _titles_similar(str(row.get("title", "")), title):
                    existing = row
                    break

            if existing is not None:
                existing["updated_at"] = now_iso
                existing["last_seen_at"] = now_iso
                existing["source_note"] = source_note
                existing["last_trigger_payload"] = dict(trigger_payload)
                if due_iso and not str(existing.get("due_at", "")).strip():
                    existing["due_at"] = due_iso
                if reminder_match is not None:
                    self._apply_reminder_to_record(existing, reminder_match, now_iso)

                tracker_state["commitments"] = commitments[-_COMMITMENT_RECORD_LIMIT:]
                await self._save_state_locked()
                should_create = not str(existing.get("reminder_id", "")).strip() and reminder_match is None
                return str(existing.get("id", "")), should_create

            record_id = f"cmt-{recorded_at.strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
            record = {
                "id": record_id,
                "title": title,
                "raw_text": raw_text,
                "due_at": due_iso,
                "source_note": source_note,
                "created_at": now_iso,
                "updated_at": now_iso,
                "last_seen_at": now_iso,
                "status": "open",
                "last_trigger_payload": dict(trigger_payload),
                "reminder_list": _COMMITMENT_REMINDERS_LIST,
                "reminder_id": "",
                "reminder_title": "",
                "reminder_completed": False,
                "completed_at": "",
            }
            if reminder_match is not None:
                self._apply_reminder_to_record(record, reminder_match, now_iso)

            commitments.append(record)
            if len(commitments) > _COMMITMENT_RECORD_LIMIT:
                tracker_state["commitments"] = commitments[-_COMMITMENT_RECORD_LIMIT:]
            else:
                tracker_state["commitments"] = commitments

            await self._save_state_locked()
            return record_id, reminder_match is None

    async def _attach_reminder_to_commitment(
        self,
        commitment_id: str,
        reminder: dict,
        recorded_at: datetime,
    ):
        now_iso = recorded_at.isoformat()
        async with self._state_lock:
            tracker_state, commitments = self._get_commitment_state_locked()
            changed = False
            for row in commitments:
                if not isinstance(row, dict):
                    continue
                if str(row.get("id", "")) != commitment_id:
                    continue
                changed |= self._apply_reminder_to_record(row, reminder, now_iso)
                row["updated_at"] = now_iso
                changed = True
                break
            if changed:
                tracker_state["commitments"] = commitments[-_COMMITMENT_RECORD_LIMIT:]
                await self._save_state_locked()

    def _get_commitment_state_locked(self) -> tuple[dict, list[dict]]:
        tracker_state = self._state.setdefault("automations", {}).setdefault(_COMMITMENT_AUTOMATION_ID, {})
        commitments_raw = tracker_state.get("commitments", [])
        if not isinstance(commitments_raw, list):
            commitments_raw = []
        commitments = [row for row in commitments_raw if isinstance(row, dict)]
        tracker_state["commitments"] = commitments
        return tracker_state, commitments

    def _apply_reminder_to_record(self, record: dict, reminder: dict, now_iso: str) -> bool:
        changed = False
        reminder_id = str(reminder.get("id", "")).strip()
        reminder_title = str(reminder.get("title", "")).strip()
        due_at = str(reminder.get("due_at", "")).strip()
        completed = bool(reminder.get("completed", False))
        completed_at = str(reminder.get("completed_at", "")).strip()

        if reminder_id and reminder_id != str(record.get("reminder_id", "")):
            record["reminder_id"] = reminder_id
            changed = True
        if reminder_title and reminder_title != str(record.get("reminder_title", "")):
            record["reminder_title"] = reminder_title
            changed = True
        if due_at and due_at != str(record.get("due_at", "")):
            record["due_at"] = due_at
            changed = True

        if bool(record.get("reminder_completed", False)) != completed:
            record["reminder_completed"] = completed
            changed = True

        if completed:
            if str(record.get("status", "")).lower() != "completed":
                record["status"] = "completed"
                changed = True
            completion_value = completed_at or now_iso
            if completion_value != str(record.get("completed_at", "")):
                record["completed_at"] = completion_value
                changed = True
        else:
            if str(record.get("status", "")).lower() == "completed":
                record["status"] = "open"
                changed = True
            if str(record.get("completed_at", "")):
                record["completed_at"] = ""
                changed = True

        if changed:
            record["updated_at"] = now_iso
        return changed

    async def commitments_report(self) -> str:
        await self.initialize()
        now_utc = datetime.now(timezone.utc)
        await self._maybe_sync_commitment_status(now_utc, force=True)
        reminders = await self._list_molly_reminders(include_completed=False)

        async with self._state_lock:
            _, commitments = self._get_commitment_state_locked()

        active_commitments = [
            row
            for row in commitments
            if isinstance(row, dict) and str(row.get("status", "open")).lower() != "completed"
        ]
        active_commitments = sorted(active_commitments, key=lambda row: str(row.get("created_at", "")))

        lines = [f"Commitments ({len(active_commitments)} internal active, {len(reminders)} Apple active)", ""]
        lines.append("Internal tracking:")
        if not active_commitments:
            lines.append("- No active internal commitments.")
        else:
            for row in active_commitments[:20]:
                title = str(row.get("title", "(untitled)"))
                lines.append(f"- {title}")
                due_label = _format_due_display(str(row.get("due_at", "")))
                if due_label:
                    lines.append(f"  Due: {due_label}")
                source_note = str(row.get("source_note", "")).strip()
                if source_note:
                    lines.append(f"  Source: {source_note}")
                reminder_title = str(row.get("reminder_title", "")).strip()
                if reminder_title:
                    lines.append(f"  Reminder: {reminder_title}")

        lines.append("")
        lines.append(f"Apple Reminders list '{_COMMITMENT_REMINDERS_LIST}':")
        if not reminders:
            lines.append("- No active reminders.")
        else:
            for reminder in reminders[:20]:
                title = str(reminder.get("title", "(untitled)")).strip()
                due_label = _format_due_display(str(reminder.get("due_at", "")))
                if due_label:
                    lines.append(f"- {title} (due {due_label})")
                else:
                    lines.append(f"- {title}")

        if len(active_commitments) > 20 or len(reminders) > 20:
            lines.append("")
            lines.append("Showing first 20 items.")
        return "\n".join(lines)

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


def _safe_iso_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    raw = value.strip()
    if not raw or not raw.startswith("{"):
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _tokenize(text: str) -> set[str]:
    return set(_WORD_TOKEN_RE.findall((text or "").lower()))


def _format_hhmm(minutes_since_midnight: int) -> str:
    minutes_since_midnight %= 24 * 60
    hour = minutes_since_midnight // 60
    minute = minutes_since_midnight % 60
    return f"{hour:02d}:{minute:02d}"


def _percentile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        return 0
    idx = int((len(sorted_values) - 1) * pct)
    return sorted_values[max(0, min(idx, len(sorted_values) - 1))]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _infer_schedule(occurrence_times: list[datetime]) -> dict[str, Any]:
    if not occurrence_times:
        return {
            "cron": "0 9 * * 1-5",
            "day_names": [],
            "window_start": "08:00",
            "window_end": "10:00",
        }

    tz = ZoneInfo(config.TIMEZONE)
    local_times = [ts.astimezone(tz) for ts in occurrence_times]

    minute_of_day = [lt.hour * 60 + lt.minute for lt in local_times]
    minute_counts = Counter(minute_of_day)
    center = minute_counts.most_common(1)[0][0]
    rounded_minute = int(round((center % 60) / 5.0) * 5)
    hour = center // 60
    if rounded_minute >= 60:
        rounded_minute = 0
        hour = (hour + 1) % 24

    weekday_counts = Counter(lt.weekday() for lt in local_times)  # Monday=0
    total = len(local_times)
    active_weekdays = sorted(
        day for day, count in weekday_counts.items()
        if count >= 2 and (count / total) >= 0.2
    )

    cron_days = "*"
    if active_weekdays:
        if active_weekdays == [0, 1, 2, 3, 4]:
            cron_days = "1-5"
        else:
            # cron weekday: Sunday=0, Monday=1, ..., Saturday=6
            to_cron = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "0"}
            cron_days = ",".join(to_cron[d] for d in active_weekdays)

    sorted_minutes = sorted(minute_of_day)
    p25 = _percentile(sorted_minutes, 0.25)
    p75 = _percentile(sorted_minutes, 0.75)
    half_window = max(30, int((p75 - p25) / 2) + 15)
    window_start = _format_hhmm(center - half_window)
    window_end = _format_hhmm(center + half_window)

    day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    day_labels = [day_names[d] for d in active_weekdays]

    return {
        "cron": f"{rounded_minute} {hour} * * {cron_days}",
        "day_names": day_labels,
        "window_start": window_start,
        "window_end": window_end,
    }


def _confidence_score(
    count: int,
    sequence_len: int,
    total_events: int,
    intervals_m: list[float],
    success_ratio: float,
) -> float:
    freq_score = min(1.0, count / 8.0)
    density_score = min(1.0, (count * sequence_len) / max(1, total_events))

    if len(intervals_m) >= 2:
        mean = sum(intervals_m) / len(intervals_m)
        if mean > 0:
            variance = sum((x - mean) ** 2 for x in intervals_m) / len(intervals_m)
            cv = (variance ** 0.5) / mean
            regularity_score = 1.0 / (1.0 + cv)
        else:
            regularity_score = 0.0
    elif len(intervals_m) == 1:
        regularity_score = 0.75
    else:
        regularity_score = 0.5

    score = (
        (0.38 * freq_score)
        + (0.28 * regularity_score)
        + (0.18 * max(0.0, min(1.0, success_ratio)))
        + (0.16 * density_score)
    )
    return round(max(0.05, min(0.99, score)), 3)


def _normalize_operational_events(operational_logs: list[dict]) -> tuple[list[dict], list[str]]:
    events: list[dict] = []
    schema_fields: set[str] = set()

    for idx, row in enumerate(operational_logs):
        if not isinstance(row, dict):
            continue
        schema_fields.update(str(k) for k in row.keys())

        tool_name = str(
            row.get("tool_name")
            or row.get("tool")
            or row.get("action")
            or row.get("step")
            or ""
        ).strip()
        if not tool_name or tool_name.startswith("approval:"):
            continue

        created_at = _safe_iso_datetime(
            row.get("created_at") or row.get("timestamp") or row.get("time")
        )
        if created_at is None:
            continue

        success_raw = row.get("success")
        if isinstance(success_raw, bool):
            success = success_raw
        elif isinstance(success_raw, (int, float)):
            success = bool(success_raw)
        elif isinstance(success_raw, str):
            success = success_raw.strip().lower() not in {"0", "false", "failed", "error", "no"}
        else:
            success = not bool(str(row.get("error_message", "")).strip())

        events.append(
            {
                "index": idx,
                "tool_name": tool_name,
                "created_at": created_at,
                "created_at_iso": created_at.isoformat(),
                "success": success,
                "latency_ms": _safe_int(row.get("latency_ms"), default=0),
                "error_message": str(row.get("error_message", "")).strip(),
                "parameters": _safe_dict(row.get("parameters")),
            }
        )

    events.sort(key=lambda r: r["created_at"])
    return events, sorted(schema_fields)


def _mine_repeated_sequences(events: list[dict], min_occurrences: int) -> list[dict]:
    if len(events) < 4:
        return []

    sequence_hits: dict[tuple[str, ...], list[int]] = defaultdict(list)
    max_len = min(4, len(events))
    for start in range(len(events)):
        for size in range(2, max_len + 1):
            end = start + size
            if end > len(events):
                break
            seq = tuple(events[i]["tool_name"] for i in range(start, end))
            if len(set(seq)) == 1:
                continue
            sequence_hits[seq].append(start)

    patterns: list[dict] = []
    for sequence, starts in sequence_hits.items():
        # Prevent overlapping windows from inflating confidence.
        selected: list[int] = []
        last_end = -1
        for s in sorted(starts):
            if s <= last_end:
                continue
            selected.append(s)
            last_end = s + len(sequence) - 1

        if len(selected) < min_occurrences:
            continue

        start_times = [events[s]["created_at"] for s in selected]
        interval_minutes = [
            (start_times[i] - start_times[i - 1]).total_seconds() / 60.0
            for i in range(1, len(start_times))
        ]
        median_interval = _median(interval_minutes) if interval_minutes else 0.0

        step_successes = []
        for s in selected:
            for idx in range(s, s + len(sequence)):
                step_successes.append(1.0 if events[idx]["success"] else 0.0)
        success_ratio = (sum(step_successes) / len(step_successes)) if step_successes else 0.0

        inferred_schedule = _infer_schedule(start_times)
        confidence = _confidence_score(
            count=len(selected),
            sequence_len=len(sequence),
            total_events=len(events),
            intervals_m=interval_minutes,
            success_ratio=success_ratio,
        )

        sample_events = []
        for s in selected[:3]:
            event = events[s]
            sample_events.append(
                {
                    "tool_name": event["tool_name"],
                    "created_at": event["created_at_iso"],
                    "success": event["success"],
                    "latency_ms": event["latency_ms"],
                    "parameters": event["parameters"],
                }
            )

        patterns.append(
            {
                "sequence": list(sequence),
                "sequence_key": " -> ".join(sequence),
                "occurrence_count": len(selected),
                "sequence_len": len(sequence),
                "first_seen": start_times[0].isoformat(),
                "last_seen": start_times[-1].isoformat(),
                "median_interval_minutes": round(median_interval, 1) if median_interval else None,
                "success_ratio": round(success_ratio, 3),
                "confidence": confidence,
                "schedule": inferred_schedule,
                "sample_events": sample_events,
            }
        )

    return sorted(
        patterns,
        key=lambda p: (p["confidence"], p["occurrence_count"], p["sequence_len"]),
        reverse=True,
    )


def _coerce_pattern_summaries(operational_logs: list[dict]) -> list[dict]:
    patterns: list[dict] = []
    for row in operational_logs:
        if not isinstance(row, dict):
            continue
        steps = str(row.get("steps", "")).strip()
        if not steps:
            continue
        sequence = [s.strip() for s in _SEQUENCE_SPLIT_RE.split(steps) if s.strip()]
        if len(sequence) < 2:
            continue
        count = max(1, _safe_int(row.get("count"), default=1))
        confidence = float(row.get("confidence", 0.0) or 0.0)
        confidence = round(max(0.05, min(0.99, confidence)), 3)
        patterns.append(
            {
                "sequence": sequence,
                "sequence_key": " -> ".join(sequence),
                "occurrence_count": count,
                "sequence_len": len(sequence),
                "first_seen": "",
                "last_seen": "",
                "median_interval_minutes": None,
                "success_ratio": 0.0,
                "confidence": confidence,
                "schedule": {
                    "cron": "0 9 * * 1-5",
                    "day_names": ["mon", "tue", "wed", "thu", "fri"],
                    "window_start": "08:30",
                    "window_end": "09:30",
                },
                "sample_events": [],
            }
        )
    return sorted(
        patterns,
        key=lambda p: (p["confidence"], p["occurrence_count"], p["sequence_len"]),
        reverse=True,
    )


def _load_existing_automation_signatures() -> list[dict]:
    signatures: list[dict] = []
    root = config.AUTOMATIONS_DIR
    try:
        paths = sorted(root.glob("*.yaml"))
    except Exception:
        return signatures

    for path in paths:
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", path.stem))
        automation_id = str(raw.get("id", path.stem))
        trigger = raw.get("trigger") or {}
        cron = str(trigger.get("cron", "")).strip() if isinstance(trigger, dict) else ""
        pipeline = raw.get("pipeline") or []

        text_parts = [name, automation_id]
        if isinstance(pipeline, list):
            for step in pipeline:
                if not isinstance(step, dict):
                    continue
                text_parts.append(str(step.get("action", "")))
                text_parts.append(str(step.get("step", "")))
                text_parts.append(str(step.get("agent", "")))

        signatures.append(
            {
                "path": str(path),
                "name": name,
                "id": automation_id,
                "cron": cron,
                "tokens": _tokenize(" ".join(text_parts)),
            }
        )
    return signatures


def _pattern_overlaps_existing(pattern: dict, existing: list[dict]) -> tuple[bool, str]:
    sequence = [str(s) for s in pattern.get("sequence", [])]
    seq_tokens = _tokenize(" ".join(sequence))
    if not seq_tokens:
        return False, ""

    pattern_cron = str((pattern.get("schedule") or {}).get("cron", "")).strip()
    for sig in existing:
        existing_tokens = sig.get("tokens") or set()
        if not existing_tokens:
            continue
        overlap = len(seq_tokens & existing_tokens) / max(1, len(seq_tokens))
        cron_match = bool(pattern_cron and pattern_cron == sig.get("cron", ""))
        if overlap >= 0.6 or (cron_match and overlap >= 0.4):
            return True, f"overlap={overlap:.2f} with {sig.get('id', '')}"
    return False, ""


def _humanize_tool(tool_name: str) -> str:
    cleaned = tool_name.replace("routing:subagent_start:", "subagent ")
    cleaned = cleaned.replace(":", " ").replace("_", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.title() if cleaned else "Workflow"


def _sequence_slug(sequence: list[str]) -> str:
    raw = "-".join(sequence[:3]).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    return slug[:64] or "proposal"


def propose_automation(operational_logs: list[dict]) -> str | None:
    """Generate a log-driven automation proposal YAML from operational traces."""
    if not operational_logs:
        return None

    min_occurrences = max(
        _PROPOSAL_MIN_OCCURRENCES,
        int(getattr(config, "AUTOMATION_MIN_PATTERN_COUNT", _PROPOSAL_MIN_OCCURRENCES)),
    )
    detected_events, detected_schema_fields = _normalize_operational_events(operational_logs)
    mined_patterns = _mine_repeated_sequences(detected_events, min_occurrences=min_occurrences)
    source_mode = "operational_logs"

    if not mined_patterns:
        # Backward compatibility for callers still passing aggregated pattern summaries.
        mined_patterns = _coerce_pattern_summaries(operational_logs)
        source_mode = "pattern_summaries"

    if not mined_patterns:
        return None

    existing = _load_existing_automation_signatures()
    selected_patterns: list[dict] = []
    dedupe_notes: list[str] = []
    for pattern in mined_patterns:
        overlaps, reason = _pattern_overlaps_existing(pattern, existing)
        if overlaps:
            dedupe_notes.append(f"Skipped {pattern.get('sequence_key', '')}: {reason}")
            continue
        selected_patterns.append(pattern)
        if len(selected_patterns) >= _PROPOSAL_MAX_PATTERNS:
            break

    if not selected_patterns:
        return None

    primary = selected_patterns[0]
    primary_schedule = primary.get("schedule") or {}
    primary_sequence = [str(t) for t in primary.get("sequence", [])]
    if len(primary_sequence) < 2:
        return None

    confidence = float(primary.get("confidence", 0.0) or 0.0)
    auto_enable_threshold = _AUTOMATION_PROPOSAL_AUTO_ENABLE_THRESHOLD
    enabled = confidence >= auto_enable_threshold

    median_interval_minutes = float(primary.get("median_interval_minutes") or 0.0)
    if median_interval_minutes > 0:
        min_interval_s = int(max(600, median_interval_minutes * 60 * 0.6))
    else:
        min_interval_s = 3600

    day_names = [str(d) for d in primary_schedule.get("day_names", []) if str(d).strip()]
    conditions: list[dict] = [{"type": "not_quiet_hours"}]
    if day_names and len(day_names) < 7:
        conditions.append({"type": "day_of_week", "days": day_names})
    conditions.append(
        {
            "type": "not_duplicate",
            "key": f"{_sequence_slug(primary_sequence)}:{{date}}",
            "cooldown": f"{max(30, int(min_interval_s / 60))}m",
        }
    )

    top_pattern_lines = []
    for idx, pattern in enumerate(selected_patterns, start=1):
        top_pattern_lines.append(
            (
                f"{idx}. {pattern.get('sequence_key', '')} "
                f"(occurrences={pattern.get('occurrence_count', 0)}, "
                f"confidence={pattern.get('confidence', 0.0)})"
            )
        )

    primary_human_start = _humanize_tool(primary_sequence[0])
    primary_human_end = _humanize_tool(primary_sequence[-1])
    proposal_name = f"{primary_human_start} to {primary_human_end} Routine"

    rationale = (
        "Proposed because operational logs show a repeated workflow sequence with stable timing "
        "and high recurrence count."
    )
    pipeline_prompt = (
        "Execute the primary recurring workflow observed in operational logs.\n"
        f"Primary sequence: {primary.get('sequence_key', '')}\n"
        f"Schedule window (local {config.TIMEZONE}): "
        f"{primary_schedule.get('window_start', '')}-{primary_schedule.get('window_end', '')}\n"
        "Supporting patterns:\n"
        + "\n".join(top_pattern_lines)
        + "\n"
        "If exact inputs are missing, produce a concise operator-ready checklist and next actions."
    )

    deliver_message = (
        "Automation run for pattern "
        f"'{primary.get('sequence_key', '')}' "
        "(confidence "
        f"{confidence:.2f}"
        ")\n"
        "{execute_pattern.output}"
    )

    evidence = {
        "top_pattern": {
            "sequence": primary_sequence,
            "sequence_key": primary.get("sequence_key", ""),
            "occurrence_count": primary.get("occurrence_count", 0),
            "first_seen": primary.get("first_seen", ""),
            "last_seen": primary.get("last_seen", ""),
            "median_interval_minutes": primary.get("median_interval_minutes"),
            "success_ratio": primary.get("success_ratio"),
            "window_start": primary_schedule.get("window_start", ""),
            "window_end": primary_schedule.get("window_end", ""),
        },
        "supporting_patterns": [
            {
                "sequence_key": p.get("sequence_key", ""),
                "occurrence_count": p.get("occurrence_count", 0),
                "confidence": p.get("confidence", 0.0),
            }
            for p in selected_patterns[1:]
        ],
        "sample_events": primary.get("sample_events", []),
    }

    skeleton = {
        "id": f"proposal-{_sequence_slug(primary_sequence)}",
        "name": proposal_name,
        "enabled": enabled,
        "version": 1,
        "trigger": {
            "type": "schedule",
            "cron": str(primary_schedule.get("cron", "0 9 * * 1-5")),
            "timezone": config.TIMEZONE,
        },
        "min_interval": f"{max(5, int(min_interval_s / 60))}m",
        "conditions": conditions,
        "pipeline": [
            {
                "step": "execute_pattern",
                "agent": "analyst",
                "prompt": pipeline_prompt,
            },
            {
                "step": "deliver",
                "channel": "whatsapp",
                "to": "owner",
                "message": deliver_message,
            },
        ],
        "meta": {
            "source_mode": source_mode,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_from_log_count": len(operational_logs),
            "parsed_event_count": len(detected_events),
            "schema_fields_detected": detected_schema_fields,
            "operational_log_schema_reference": _OPERATIONAL_LOG_SCHEMA_DOC,
            "fields_used_for_pattern_detection": [
                "tool_name",
                "created_at",
                "success",
                "latency_ms",
                "error_message",
                "parameters",
            ],
            "proposal_confidence": confidence,
            "auto_enable_threshold": auto_enable_threshold,
            "why_proposed": rationale,
            "dedupe_notes": dedupe_notes,
            "evidence": evidence,
        },
    }
    return yaml.safe_dump(skeleton, sort_keys=False)
