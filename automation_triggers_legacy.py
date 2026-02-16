import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import config

try:
    from croniter import croniter
except Exception:  # pragma: no cover - handled at runtime if dependency missing
    croniter = None

log = logging.getLogger(__name__)


def _parse_iso(dt_str: str) -> datetime | None:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_interval_seconds(value, default_seconds: int) -> int:
    if value is None:
        return default_seconds
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, str):
        match = re.fullmatch(r"(\d+)\s*([smhd]?)", value.strip().lower())
        if not match:
            return default_seconds
        amount = int(match.group(1))
        unit = match.group(2) or "s"
        mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        return amount * mult
    return default_seconds


def _normalize_now(now: datetime | None) -> datetime:
    ts = now or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


class BaseTrigger:
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.last_payload: dict = {}

    async def should_fire(self, context: dict) -> bool:
        return False

    async def next_fire_time(self, context: dict) -> datetime | None:
        return None


class ScheduleTrigger(BaseTrigger):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._last_match_key = ""
        self._warned_missing_croniter = False

    def _is_due_minute(self, now_local: datetime, cron_expr: str) -> bool:
        if croniter is None:
            if not self._warned_missing_croniter:
                log.warning("croniter is not installed; schedule triggers are disabled")
                self._warned_missing_croniter = True
            return False
        if croniter.match(cron_expr, now_local):
            return True
        it = croniter(cron_expr, now_local + timedelta(minutes=1))
        return it.get_prev(datetime).replace(second=0, microsecond=0) == now_local

    async def should_fire(self, context: dict) -> bool:
        cron_expr = self.cfg.get("cron", "").strip()
        if not cron_expr:
            return False

        tz_name = self.cfg.get("timezone", config.TIMEZONE)
        tz = ZoneInfo(tz_name)
        now = _normalize_now(context.get("now")).astimezone(tz).replace(second=0, microsecond=0)

        try:
            if not self._is_due_minute(now, cron_expr):
                return False
        except Exception:
            log.warning("Invalid cron expression: %s", cron_expr, exc_info=True)
            return False

        key = f"{cron_expr}:{now.isoformat()}"
        if key == self._last_match_key:
            return False

        self._last_match_key = key
        self.last_payload = {"scheduled_for": now.isoformat(), "cron": cron_expr}
        return True

    async def next_fire_time(self, context: dict) -> datetime | None:
        if croniter is None:
            return None
        cron_expr = self.cfg.get("cron", "").strip()
        if not cron_expr:
            return None
        tz_name = self.cfg.get("timezone", config.TIMEZONE)
        tz = ZoneInfo(tz_name)
        now_local = _normalize_now(context.get("now")).astimezone(tz)
        try:
            return croniter(cron_expr, now_local).get_next(datetime)
        except Exception:
            return None


class EventTrigger(BaseTrigger):
    async def _fetch_events(self, time_min: datetime, time_max: datetime) -> list[dict]:
        from tools.google_auth import get_calendar_service

        def _run() -> list[dict]:
            service = get_calendar_service()
            result = (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=time_min.isoformat(),
                    timeMax=time_max.isoformat(),
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=20,
                )
                .execute()
            )
            return result.get("items", [])

        return await asyncio.to_thread(_run)

    async def should_fire(self, context: dict) -> bool:
        now = _normalize_now(context.get("now"))
        minutes_before = int(self.cfg.get("minutes_before", 30))
        window_minutes = int(self.cfg.get("window_minutes", 2))
        has_attendees = bool(self.cfg.get("has_attendees", False))
        min_attendees = int(self.cfg.get("min_attendees", 2 if has_attendees else 0))
        summary_contains = str(self.cfg.get("summary_contains", "")).strip().lower()

        window_start = now + timedelta(minutes=max(0, minutes_before - window_minutes))
        window_end = now + timedelta(minutes=minutes_before + window_minutes)

        try:
            events = await self._fetch_events(window_start, window_end)
        except Exception:
            log.debug("EventTrigger calendar check failed", exc_info=True)
            return False

        matches: list[dict] = []
        for event in events:
            attendees = event.get("attendees", []) or []
            attendee_count = len([a for a in attendees if not a.get("self", False)])
            if attendee_count < min_attendees:
                continue
            if summary_contains and summary_contains not in event.get("summary", "").lower():
                continue
            matches.append(
                {
                    "event_id": event.get("id", ""),
                    "title": event.get("summary", "(untitled)"),
                    "start": (event.get("start", {}) or {}).get("dateTime", ""),
                    "attendee_count": attendee_count,
                    "attendees": [
                        a.get("email", "")
                        for a in attendees
                        if a.get("email") and not a.get("self", False)
                    ],
                }
            )

        if not matches:
            return False

        self.last_payload = {"events": matches, "event": matches[0]}
        return True

    async def next_fire_time(self, context: dict) -> datetime | None:
        # Event-driven triggers depend on external calendar state and are not
        # deterministic without querying the API. Keep status calls lightweight.
        return None


class EmailTrigger(BaseTrigger):
    _MAX_SEEN_IDS = 500
    _MAX_HIGH_WATER_IDS = 50

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._seen_ids: set[str] = set()
        self._automation_id = str(self.cfg.get("_automation_id", "")).strip()
        state_path = str(self.cfg.get("_state_path", "")).strip()
        self._state_path = Path(state_path) if state_path else config.AUTOMATIONS_STATE_FILE

    def _load_high_water(self) -> tuple[int, set[str]]:
        """Load persisted high-water state for this email automation."""
        if not self._automation_id or not self._state_path.exists():
            return 0, set()

        try:
            raw = json.loads(self._state_path.read_text())
            automations = raw.get("automations", {})
            automation_state = automations.get(self._automation_id, {})
            trigger_state = automation_state.get("trigger_state", {})
            email_state = trigger_state.get("email", {})
        except Exception:
            log.debug(
                "EmailTrigger could not load state from %s",
                self._state_path,
                exc_info=True,
            )
            return 0, set()

        try:
            high_water_ts_ms = int(email_state.get("high_water_internal_ts_ms", 0))
        except Exception:
            high_water_ts_ms = 0
        if high_water_ts_ms < 0:
            high_water_ts_ms = 0

        high_water_ids: set[str] = set()
        raw_ids = email_state.get("high_water_ids", [])
        if isinstance(raw_ids, list):
            for value in raw_ids:
                msg_id = str(value).strip()
                if msg_id:
                    high_water_ids.add(msg_id)
        legacy_id = str(email_state.get("high_water_message_id", "")).strip()
        if legacy_id:
            high_water_ids.add(legacy_id)

        return high_water_ts_ms, high_water_ids

    def _build_query(self, base_query: str, high_water_ts_ms: int) -> str:
        query = (base_query or "").strip()
        if high_water_ts_ms <= 0:
            return query

        # Replace relative-time filters with a strict high-water boundary.
        query = re.sub(r"(?i)\b(?:after|newer_than|older_than):\S+\b", "", query)
        query = re.sub(r"\s+", " ", query).strip()

        # Gmail "after:" is second-granular; include one-second overlap and
        # filter exact duplicates client-side via (timestamp,id).
        after_s = max(0, (high_water_ts_ms // 1000) - 1)
        if query:
            return f"{query} after:{after_s}"
        return f"after:{after_s}"

    @staticmethod
    def _is_newer_than_high_water(message: dict, high_water_ts_ms: int, high_water_ids: set[str]) -> bool:
        msg_id = str(message.get("id", "")).strip()
        if not msg_id:
            return False

        try:
            internal_ts_ms = int(message.get("internal_ts_ms", 0))
        except Exception:
            internal_ts_ms = 0

        if high_water_ts_ms <= 0:
            return msg_id not in high_water_ids
        if internal_ts_ms <= 0:
            return msg_id not in high_water_ids
        if internal_ts_ms > high_water_ts_ms:
            return True
        if internal_ts_ms == high_water_ts_ms and msg_id not in high_water_ids:
            return True
        return False

    async def _search_emails(self, query: str, max_results: int) -> list[dict]:
        from tools.google_auth import get_gmail_service

        def _run() -> list[dict]:
            service = get_gmail_service()
            result = (
                service.users()
                .messages()
                .list(userId="me", q=query, maxResults=max_results)
                .execute()
            )
            refs = result.get("messages", [])
            details = []
            for ref in refs[: max_results]:
                msg = (
                    service.users()
                    .messages()
                    .get(
                        userId="me",
                        id=ref["id"],
                        format="metadata",
                        metadataHeaders=["From", "Subject", "Date"],
                    )
                    .execute()
                )
                headers = {
                    h["name"].lower(): h["value"]
                    for h in msg.get("payload", {}).get("headers", [])
                }
                try:
                    internal_ts_ms = int(msg.get("internalDate", 0))
                except Exception:
                    internal_ts_ms = 0
                details.append(
                    {
                        "id": msg.get("id", ""),
                        "from": headers.get("from", ""),
                        "subject": headers.get("subject", ""),
                        "date": headers.get("date", ""),
                        "snippet": msg.get("snippet", ""),
                        "internal_ts_ms": internal_ts_ms,
                    }
                )
            return details

        return await asyncio.to_thread(_run)

    async def should_fire(self, context: dict) -> bool:
        query = str(self.cfg.get("query", "")).strip()
        max_results = int(self.cfg.get("max_results", 10))
        from_domain = str(self.cfg.get("from_domain", "")).strip().lower()
        unread = self.cfg.get("is_unread", True)
        if not query:
            query = "newer_than:1h"
            if unread:
                query = f"is:unread {query}"
            if from_domain:
                query = f"from:{from_domain} {query}"

        high_water_ts_ms, high_water_ids = self._load_high_water()
        query = self._build_query(query, high_water_ts_ms)

        try:
            messages = await self._search_emails(query, max_results=max_results)
        except Exception:
            log.debug("EmailTrigger query failed", exc_info=True)
            return False

        new_messages = []
        for msg in messages:
            msg_id = msg.get("id", "")
            if not msg_id or msg_id in self._seen_ids:
                continue
            if not self._is_newer_than_high_water(msg, high_water_ts_ms, high_water_ids):
                continue
            new_messages.append(msg)
            self._seen_ids.add(msg_id)

        if len(self._seen_ids) > self._MAX_SEEN_IDS and new_messages:
            self._seen_ids = {
                str(msg.get("id", "")).strip()
                for msg in new_messages[-self._MAX_SEEN_IDS:]
                if str(msg.get("id", "")).strip()
            }

        if not new_messages:
            return False

        latest_ts_ms = 0
        latest_ids: list[str] = []
        for msg in new_messages:
            msg_id = str(msg.get("id", "")).strip()
            if not msg_id:
                continue
            try:
                ts_ms = int(msg.get("internal_ts_ms", 0))
            except Exception:
                ts_ms = 0
            if ts_ms > latest_ts_ms:
                latest_ts_ms = ts_ms
                latest_ids = [msg_id]
            elif ts_ms == latest_ts_ms:
                latest_ids.append(msg_id)

        latest_ids = list(dict.fromkeys([msg_id for msg_id in latest_ids if msg_id]))
        latest_ids = latest_ids[-self._MAX_HIGH_WATER_IDS:]
        fallback_id = str(new_messages[0].get("id", "")).strip() if new_messages else ""

        self.last_payload = {
            "messages": new_messages,
            "message": new_messages[0],
            "query": query,
            "_email_trigger_state": {
                "high_water_internal_ts_ms": latest_ts_ms,
                "high_water_ids": latest_ids,
                "high_water_message_id": latest_ids[-1] if latest_ids else fallback_id,
            },
        }
        return True

    async def next_fire_time(self, context: dict) -> datetime | None:
        now = _normalize_now(context.get("now"))
        interval_minutes = int(self.cfg.get("interval_minutes", 10))
        return now + timedelta(minutes=interval_minutes)


class MessageTrigger(BaseTrigger):
    async def should_fire(self, context: dict) -> bool:
        msg = context.get("message") or {}
        if not msg:
            return False

        text = str(msg.get("content", ""))
        sender = str(msg.get("sender_jid", ""))
        is_owner = bool(context.get("is_owner", False))

        source_filter = str(self.cfg.get("from", "")).strip().lower()
        if source_filter == "owner" and not is_owner:
            return False
        if source_filter == "non_owner" and is_owner:
            return False
        if source_filter and source_filter not in {"owner", "non_owner"}:
            if source_filter not in sender.lower():
                return False

        contains = self.cfg.get("contains")
        if isinstance(contains, str):
            contains = [contains]
        if contains:
            lowered = text.lower()
            if not any(token.lower() in lowered for token in contains):
                return False

        pattern = str(self.cfg.get("pattern", "")).strip()
        if pattern:
            try:
                if re.search(pattern, text, flags=re.IGNORECASE) is None:
                    return False
            except re.error:
                log.warning("Invalid message trigger regex: %s", pattern)
                return False

        self.last_payload = {
            "message_text": text,
            "sender_jid": sender,
            "sender_name": msg.get("sender_name", ""),
            "chat_jid": msg.get("chat_jid", ""),
            "timestamp": msg.get("timestamp", ""),
        }
        return True


class CommitmentTrigger(MessageTrigger):
    async def should_fire(self, context: dict) -> bool:
        if "pattern" not in self.cfg:
            self.cfg["pattern"] = r"\b(i (will|can)|i'?ll|remind me|follow up|by (monday|tomorrow|\d{4}-\d{2}-\d{2}))\b"
        return await super().should_fire(context)


class ConditionTrigger(BaseTrigger):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._last_eval_at: datetime | None = None
        self._last_value = False

    async def should_fire(self, context: dict) -> bool:
        interval_s = _parse_interval_seconds(self.cfg.get("interval"), default_seconds=300)
        now = _normalize_now(context.get("now"))
        if self._last_eval_at and (now - self._last_eval_at).total_seconds() < interval_s:
            return False

        expression = str(self.cfg.get("check", "")).strip()
        if not expression:
            return False

        safe_context = {
            "message_count_today": context.get("message_count_today", 0),
            "owner_online": context.get("owner_online", False),
            "now_hour": now.hour,
        }
        try:
            value = bool(eval(expression, {"__builtins__": {}}, safe_context))
        except Exception:
            log.warning("Condition trigger eval failed: %s", expression, exc_info=True)
            self._last_eval_at = now
            return False

        self._last_eval_at = now
        should_fire = value and not self._last_value
        self._last_value = value
        if should_fire:
            self.last_payload = {"check": expression, "value": value}
        return should_fire


class WebhookTrigger(BaseTrigger):
    async def should_fire(self, context: dict) -> bool:
        event = context.get("webhook_event")
        if not event:
            return False
        path = str(self.cfg.get("path", "")).strip()
        if path and str(event.get("path", "")).strip() != path:
            return False
        self.last_payload = dict(event)
        return True


def create_trigger(cfg: dict) -> BaseTrigger:
    trigger_type = str((cfg or {}).get("type", "")).strip().lower()
    mapping = {
        "schedule": ScheduleTrigger,
        "event": EventTrigger,
        "email": EmailTrigger,
        "message": MessageTrigger,
        "commitment": CommitmentTrigger,
        "condition": ConditionTrigger,
        "webhook": WebhookTrigger,
    }
    trigger_cls = mapping.get(trigger_type)
    if not trigger_cls:
        raise ValueError(f"Unsupported trigger type: {trigger_type}")
    return trigger_cls(cfg)
