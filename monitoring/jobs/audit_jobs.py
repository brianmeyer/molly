"""Audit Jobs — maintenance issue tracking via issue registry."""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

import config
import db_pool
from monitoring._base import _issue_last_notified_at

log = logging.getLogger(__name__)

MAINTENANCE_STEP_PREFIX = "maint."
MAINTENANCE_NOTIFY_COOLDOWN_HOURS = 12


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def record_maintenance_issues(
    results: dict[str, str],
    run_status: str,
) -> tuple[int, int]:
    """Sync step results into issue registry, route notifications via route_health_signal()."""
    from memory.issue_registry import (
        append_issue_event,
        ensure_issue_registry_tables,
        should_notify,
        upsert_issue,
    )
    from monitoring.remediation import route_health_signal

    synced = 0
    notified = 0
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text or ""))
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return collapsed or "unknown"


def _severity_from_result(result: str) -> str:
    lower = str(result or "").lower()
    if "failed" in lower or "error" in lower:
        return "red"
    if "skipped" in lower or "timeout" in lower:
        return "yellow"
    return "green"


# _issue_last_notified_at imported from monitoring._base
