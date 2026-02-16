"""Issue Registry Sync — upsert health checks into issue registry + notifications."""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
import db_pool
from monitoring._base import HEALTH_NOTIFY_COOLDOWN_HOURS, HealthCheck, _issue_last_notified_at
from monitoring.agents.yellow_escalation import previous_yellow_streak

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def sync_issue_registry(
    checks: list[HealthCheck],
    report_dir: Path | None = None,
) -> dict[str, Any]:
    """Upsert health checks into the issue registry, route via remediation, notify with cooldown."""
    from memory.issue_registry import (
        append_issue_event,
        ensure_issue_registry_tables,
        should_notify,
        upsert_issue,
    )
    from monitoring.remediation import route_health_signal

    rdir = report_dir or config.HEALTH_REPORT_DIR
    synced = 0
    notified = 0
    remediation_rows: list[dict[str, Any]] = []

    conn: sqlite3.Connection | None = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        ensure_issue_registry_tables(conn)
        observed_at = datetime.now(timezone.utc).isoformat()

        for check in checks:
            issue = upsert_issue(
                conn,
                check_id=check.check_id,
                severity=check.status,
                detail=check.detail,
                source="health",
                observed_at=observed_at,
            )
            synced += 1

            yellow_streak_days = 1
            if check.status == "yellow":
                yellow_streak_days = previous_yellow_streak(check.check_id, rdir) + 1

            plan = route_health_signal(
                check.check_id,
                check.status,
                yellow_streak_days=yellow_streak_days,
            )
            remediation_row = {
                "check_id": plan.check_id,
                "severity": plan.severity,
                "action": plan.action,
                "suggested_action": plan.suggested_action,
                "rationale": plan.rationale,
                "escalate_owner_now": plan.escalation.escalate_owner_now,
                "immediate_investigation_candidate": (
                    plan.escalation.immediate_investigation_candidate
                ),
                "yellow_streak_days": plan.escalation.yellow_streak_days,
                "yellow_days_until_escalation": (
                    plan.escalation.yellow_days_until_escalation
                ),
            }
            remediation_rows.append(remediation_row)

            if check.status not in {"yellow", "red"}:
                continue
            last_notified = _issue_last_notified_at(conn, issue["fingerprint"])
            if should_notify(
                fingerprint=issue["fingerprint"],
                cooldown_hours=HEALTH_NOTIFY_COOLDOWN_HOURS,
                last_notified_at=last_notified,
                severity_changed=bool(issue.get("severity_changed", False)),
            ):
                append_issue_event(
                    conn,
                    issue_fingerprint=issue["fingerprint"],
                    event_type="notified",
                    created_at=observed_at,
                    payload={
                        "check_id": check.check_id,
                        "status": check.status,
                        "detail": check.detail,
                        "action": plan.action,
                        "suggested_action": plan.suggested_action,
                        "escalate_owner_now": plan.escalation.escalate_owner_now,
                    },
                )
                notified += 1

        conn.commit()
        return {
            "synced": synced,
            "notified": notified,
            "remediation": remediation_rows,
        }
    except Exception:
        log.debug("Health issue registry sync failed", exc_info=True)
        return {
            "synced": synced,
            "notified": notified,
            "remediation": remediation_rows,
        }
    finally:
        if conn is not None:
            conn.close()


# _issue_last_notified_at imported from monitoring._base
