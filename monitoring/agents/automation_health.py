"""Automation Health — 7 automation engine checks."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import config
from monitoring._base import HealthCheck, _now_local, _parse_iso, _short_ts

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_automation_health(molly=None) -> list[HealthCheck]:
    """Run all automation health checks and return results."""
    checks: list[HealthCheck] = []

    state_ok, state_data, state_detail = _load_automation_state()
    checks.append(
        HealthCheck(
            check_id="automation.state_integrity",
            layer="Automation Health",
            label="state.json integrity",
            status="green" if state_ok else "red",
            detail=state_detail,
            action_required=not state_ok,
        )
    )
    if not state_ok:
        return checks

    automations = state_data.get("automations", {})
    if not automations:
        checks.append(
            HealthCheck(
                check_id="automation.loaded",
                layer="Automation Health",
                label="Automation status",
                status="yellow",
                detail="No automation state entries found",
                watch_item=True,
            )
        )
        return checks

    # Per-automation failures
    failed = [
        aid for aid, row in automations.items()
        if str(row.get("last_status", "")).lower() == "failed"
    ]
    status = "green"
    if failed:
        status = "yellow" if len(failed) <= 2 else "red"
    checks.append(
        HealthCheck(
            check_id="automation.failures",
            layer="Automation Health",
            label="Per-automation last_result",
            status=status,
            detail=f"failed={len(failed)} / total={len(automations)}",
            watch_item=(status == "yellow"),
            action_required=(status == "red"),
        )
    )

    # Morning briefing (weekdays only)
    now_local = _now_local()
    weekday = now_local.weekday() < 5
    morning_row = _find_automation_row(automations, ["morning", "digest", "brief"])
    if weekday:
        morning_ok = bool(
            morning_row and _is_same_day(morning_row.get("last_run", ""), now_local)
        )
        checks.append(
            HealthCheck(
                check_id="automation.morning_briefing",
                layer="Automation Health",
                label="Morning briefing",
                status="green" if morning_ok else "red",
                detail="fired today" if morning_ok else "missed weekday run",
                action_required=not morning_ok,
            )
        )

    # Email triage staleness
    email_row = _find_automation_row(automations, ["email", "triage"])
    checks.append(
        HealthCheck(
            check_id="automation.email_triage",
            layer="Automation Health",
            label="Email triage staleness",
            status=_staleness_status(email_row, max_minutes=30),
            detail=_staleness_detail(email_row),
            watch_item=True,
        )
    )

    # Meeting prep coverage
    prep_row = _find_automation_row(automations, ["meeting", "prep"])
    checks.append(
        HealthCheck(
            check_id="automation.meeting_prep",
            layer="Automation Health",
            label="Meeting prep coverage",
            status=_staleness_status(prep_row, max_minutes=24 * 60),
            detail=_staleness_detail(prep_row),
            watch_item=True,
        )
    )

    # Commitment tracker freshness
    followups = config.WORKSPACE / "memory" / "followups.md"
    if not followups.exists():
        fu_status = "yellow"
        fu_detail = "followups.md missing"
    else:
        age_days = (datetime.now(timezone.utc) - datetime.fromtimestamp(followups.stat().st_mtime, tz=timezone.utc)).days
        fu_status = "green" if age_days <= 3 else ("yellow" if age_days <= 7 else "red")
        fu_detail = f"updated {age_days} day(s) ago"
    checks.append(
        HealthCheck(
            check_id="automation.commitment_tracker",
            layer="Automation Health",
            label="Commitment tracker freshness",
            status=fu_status,
            detail=fu_detail,
            watch_item=(fu_status == "yellow"),
            action_required=(fu_status == "red"),
        )
    )

    return checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_automation_state() -> tuple[bool, dict[str, Any], str]:
    path = config.AUTOMATIONS_STATE_FILE
    if not path.exists():
        return False, {}, f"missing: {path}"
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return False, {}, "invalid top-level JSON type"
        if "automations" not in data or not isinstance(data["automations"], dict):
            return False, {}, "missing 'automations' object"
        return True, data, f"loaded {len(data['automations'])} automation entries"
    except Exception as exc:
        return False, {}, f"parse error ({exc})"


def _find_automation_row(automations: dict[str, dict], tokens: list[str]) -> dict | None:
    lowered = [t.lower() for t in tokens]
    for aid, row in automations.items():
        name = f"{aid} {row.get('name', '')}".lower()
        if all(token in name for token in lowered):
            return row
    return None


def _is_same_day(timestamp: str, local_now: datetime) -> bool:
    dt = _parse_iso(timestamp)
    if not dt:
        return False
    return dt.astimezone(local_now.tzinfo).date() == local_now.date()


def _staleness_status(row: dict | None, max_minutes: int) -> str:
    if not row:
        return "yellow"
    dt = _parse_iso(str(row.get("last_run", "")))
    if not dt:
        return "yellow"
    age_min = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 60
    if age_min <= max_minutes:
        return "green"
    if age_min <= max_minutes * 3:
        return "yellow"
    return "red"


def _staleness_detail(row: dict | None) -> str:
    if not row:
        return "automation not found"
    dt_str = str(row.get("last_run", ""))
    if not dt_str:
        return "never run"
    return f"last_run={_short_ts(dt_str)} status={row.get('last_status', '-')}"
