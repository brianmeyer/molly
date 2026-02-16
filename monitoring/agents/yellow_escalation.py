"""Yellow Escalation — streak tracking and auto-escalation to red."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import config
from monitoring._base import (
    HEALTH_YELLOW_ESCALATION_DAYS,
    HealthCheck,
    _load_embedded_report_data,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def apply_yellow_escalation(
    checks: list[HealthCheck],
    report_dir: Path | None = None,
    threshold_days: int | None = None,
) -> list[HealthCheck]:
    """Escalate yellow checks that have persisted for *threshold_days* to red."""
    threshold = threshold_days or max(2, HEALTH_YELLOW_ESCALATION_DAYS)
    rdir = report_dir or config.HEALTH_REPORT_DIR
    updated: list[HealthCheck] = []
    for check in checks:
        if check.status != "yellow":
            updated.append(check)
            continue
        prev_streak = previous_yellow_streak(check.check_id, rdir)
        if prev_streak + 1 >= threshold:
            updated.append(
                HealthCheck(
                    check_id=check.check_id,
                    layer=check.layer,
                    label=check.label,
                    status="red",
                    detail=f"{check.detail} (yellow persisted {prev_streak + 1} days)",
                    action_required=True,
                    watch_item=check.watch_item,
                )
            )
        else:
            updated.append(check)
    return updated


def previous_yellow_streak(check_id: str, report_dir: Path | None = None) -> int:
    """Count consecutive previous days where *check_id* was yellow."""
    rdir = report_dir or config.HEALTH_REPORT_DIR
    paths = sorted(rdir.glob("????-??-??.md"))
    if not paths:
        return 0
    today = date.today().isoformat()
    streak = 0
    for path in reversed(paths):
        if path.stem == today:
            continue
        payload = _load_embedded_report_data(path)
        rows = payload.get("checks", [])
        if not isinstance(rows, list):
            continue
        status = ""
        for row in rows:
            if str(row.get("id", "")) == check_id:
                status = str(row.get("status", ""))
                break
        if status == "yellow":
            streak += 1
            continue
        break
    return streak
