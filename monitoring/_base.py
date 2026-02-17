"""Shared types, constants, and helpers for monitoring agents and jobs."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import config
import db_pool
from utils import normalize_timestamp

log = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    check_id: str
    layer: str
    label: str
    status: str  # green | yellow | red
    detail: str
    action_required: bool = False
    watch_item: bool = False


_STATUS_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
_STATUS_SEVERITY = {"green": 0, "yellow": 1, "red": 2}

HEALTH_NOTIFY_COOLDOWN_HOURS = max(
    1, int(os.getenv("MOLLY_HEALTH_NOTIFY_COOLDOWN_HOURS", "24"))
)
HEALTH_PIPELINE_WINDOW_HOURS = max(
    1, int(os.getenv("MOLLY_HEALTH_PIPELINE_WINDOW_HOURS", "24"))
)
HEALTH_ENTITY_SAMPLE_SIZE = max(
    5, int(os.getenv("MOLLY_HEALTH_ENTITY_SAMPLE_SIZE", "20"))
)
HEALTH_YELLOW_ESCALATION_DAYS = max(
    2, int(os.getenv("MOLLY_HEALTH_YELLOW_ESCALATION_DAYS", "3"))
)
HEALTH_SKILL_WINDOW_DAYS = max(
    1, int(os.getenv("MOLLY_HEALTH_SKILL_WINDOW_DAYS", "7"))
)
HEALTH_SKILL_LOW_WATERMARK = max(
    1, int(os.getenv("MOLLY_HEALTH_SKILL_LOW_WATERMARK", "3"))
)
HEALTH_SKILL_BASH_RATIO_RED = max(
    0.0, float(os.getenv("MOLLY_HEALTH_SKILL_BASH_RATIO_RED", "0.30"))
)
HEALTH_SKILL_BASH_RATIO_YELLOW = max(
    HEALTH_SKILL_BASH_RATIO_RED,
    float(os.getenv("MOLLY_HEALTH_SKILL_BASH_RATIO_YELLOW", "0.75")),
)


def _status_emoji(status: str) -> str:
    return _STATUS_EMOJI.get(status, "⚪")


def _now_local() -> datetime:
    return datetime.now(ZoneInfo(config.TIMEZONE))


def _parse_iso(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        normalized = normalize_timestamp(raw)
        return datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None


def _short_ts(value: str) -> str:
    dt = _parse_iso(value)
    if not dt:
        return value or "-"
    return dt.astimezone(ZoneInfo(config.TIMEZONE)).strftime("%Y-%m-%d %H:%M")


def _load_embedded_report_data(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {}
    text = report_path.read_text()
    marker = "<!-- HEALTH_DATA:"
    start = text.rfind(marker)
    if start < 0:
        return {}
    end = text.find("-->", start)
    if end < 0:
        return {}
    raw = text[start + len(marker) : end].strip()
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _count_rows(db_path: Path, sql: str, params: tuple[Any, ...] = ()) -> int:
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(db_path))
        row = conn.execute(sql, params).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0
    finally:
        if conn is not None:
            conn.close()


def _issue_last_notified_at(
    conn: sqlite3.Connection,
    fingerprint: str,
) -> str | None:
    """Return the most recent 'notified' event timestamp for an issue, or None."""
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


def _sqlite_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None
