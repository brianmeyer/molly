import hashlib
import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

log = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")
_NUMERIC_TS_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_ISO_TS_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:\d{2})?\b",
    re.IGNORECASE,
)
_HEX_RE = re.compile(r"\b0x[0-9a-f]+\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+\b")
_IDENTIFIER_CLEAN_RE = re.compile(r"[^a-z0-9._:/-]+")
_DETAIL_KEEP_RE = re.compile(r"[^a-z0-9_=:/.-]+")

_FAIL_SEVERITIES = {"yellow", "red"}

_SEVERITY_ALIASES = {
    "green": "green",
    "ok": "green",
    "healthy": "green",
    "success": "green",
    "pass": "green",
    "yellow": "yellow",
    "warn": "yellow",
    "warning": "yellow",
    "degraded": "yellow",
    "red": "red",
    "critical": "red",
    "crit": "red",
    "error": "red",
    "fail": "red",
    "failed": "red",
}

_STATUS_ALIASES = {
    "open": "open",
    "new": "open",
    "active": "open",
    "investigating": "open",
    "resolved": "resolved",
    "closed": "resolved",
    "fixed": "resolved",
    "suppressed": "suppressed",
    "muted": "suppressed",
    "ignored": "suppressed",
}

_ISSUE_COLUMNS = {
    "fingerprint": "TEXT PRIMARY KEY",
    "check_id": "TEXT NOT NULL DEFAULT ''",
    "severity": "TEXT NOT NULL DEFAULT 'yellow'",
    "status": "TEXT NOT NULL DEFAULT 'open'",
    "first_seen": "TEXT NOT NULL DEFAULT ''",
    "last_seen": "TEXT NOT NULL DEFAULT ''",
    "consecutive_failures": "INTEGER NOT NULL DEFAULT 0",
    "last_detail": "TEXT DEFAULT ''",
    "source": "TEXT DEFAULT ''",
}

_ISSUE_EVENT_COLUMNS = {
    "issue_fingerprint": "TEXT NOT NULL DEFAULT ''",
    "event_type": "TEXT NOT NULL DEFAULT 'observed'",
    "created_at": "TEXT NOT NULL DEFAULT ''",
    "payload": "TEXT DEFAULT '{}'",
}


def _normalize_identifier(value: str | None, default: str) -> str:
    text = _WS_RE.sub(" ", str(value or "").strip().lower())
    text = text.replace(" ", "_")
    text = _IDENTIFIER_CLEAN_RE.sub("_", text).strip("_")
    return text or default


def normalize_severity(severity: str | None) -> str:
    return _SEVERITY_ALIASES.get(str(severity or "").strip().lower(), "yellow")


def normalize_status(status: str | None) -> str:
    return _STATUS_ALIASES.get(str(status or "").strip().lower(), "open")


def normalize_issue_detail(detail: str | None) -> str:
    text = _WS_RE.sub(" ", str(detail or "").strip().lower())
    text = _UUID_RE.sub("<uuid>", text)
    text = _ISO_TS_RE.sub("<ts>", text)
    text = _HEX_RE.sub("<hex>", text)
    text = _NUMBER_RE.sub("<num>", text)
    return text[:800]


def _detail_signature(detail: str | None) -> str:
    normalized = normalize_issue_detail(detail)
    if not normalized:
        return ""
    scrubbed = (
        normalized
        .replace("<uuid>", " ")
        .replace("<ts>", " ")
        .replace("<hex>", " ")
        .replace("<num>", " ")
    )
    scrubbed = _DETAIL_KEEP_RE.sub(" ", scrubbed)
    tokens = [
        token
        for token in scrubbed.split()
        if token not in {"a", "an", "the", "at", "on", "for", "in", "to", "from"}
    ]
    return " ".join(tokens[:40])


def build_issue_fingerprint(
    check_id: str,
    detail: str = "",
    source: str = "health",
) -> str:
    canonical_check_id = _normalize_identifier(check_id, "unknown.check")
    canonical_source = _normalize_identifier(source, "health")
    canonical_detail = _detail_signature(detail)
    raw = f"{canonical_source}|{canonical_check_id}|{canonical_detail}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _coerce_unix_epoch(value: float) -> float:
    abs_value = abs(value)
    if abs_value >= 1e17:
        return value / 1_000_000_000
    if abs_value >= 1e14:
        return value / 1_000_000
    if abs_value >= 1e11:
        return value / 1_000
    return value


def _parse_timestamp(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        raw = str(value).strip()
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            if not _NUMERIC_TS_RE.fullmatch(raw):
                return None
            try:
                dt = datetime.fromtimestamp(
                    _coerce_unix_epoch(float(raw)),
                    tz=timezone.utc,
                )
            except (ValueError, OverflowError, OSError):
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_iso_utc(value: str | datetime | None) -> str:
    dt = _parse_timestamp(value)
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.isoformat()


def _ensure_columns(
    conn: sqlite3.Connection,
    table_name: str,
    required_columns: dict[str, str],
):
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {str(row[1]) for row in rows}
    for column, definition in required_columns.items():
        if column in existing:
            continue
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {definition}")


def ensure_issue_registry_tables(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS maintenance_issues (
            fingerprint TEXT PRIMARY KEY,
            check_id TEXT NOT NULL,
            severity TEXT NOT NULL,
            status TEXT NOT NULL,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            last_detail TEXT,
            source TEXT
        );

        CREATE TABLE IF NOT EXISTS maintenance_issue_events (
            issue_fingerprint TEXT NOT NULL,
            event_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT
        );
        """
    )
    _ensure_columns(conn, "maintenance_issues", _ISSUE_COLUMNS)
    _ensure_columns(conn, "maintenance_issue_events", _ISSUE_EVENT_COLUMNS)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_maintenance_issues_status "
        "ON maintenance_issues(status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_maintenance_issues_last_seen "
        "ON maintenance_issues(last_seen)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_maintenance_issue_events_fingerprint "
        "ON maintenance_issue_events(issue_fingerprint)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_maintenance_issue_events_created_at "
        "ON maintenance_issue_events(created_at)"
    )
    conn.commit()


def append_issue_event(
    conn: sqlite3.Connection,
    issue_fingerprint: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    created_at: str | datetime | None = None,
):
    ensure_issue_registry_tables(conn)
    normalized_fingerprint = str(issue_fingerprint or "").strip().lower()
    if not normalized_fingerprint:
        raise ValueError("issue_fingerprint is required")
    normalized_event = _normalize_identifier(event_type, "observed")
    created = _to_iso_utc(created_at)
    payload_json = json.dumps(payload or {}, ensure_ascii=True, sort_keys=True)
    conn.execute(
        """
        INSERT INTO maintenance_issue_events
        (issue_fingerprint, event_type, created_at, payload)
        VALUES (?, ?, ?, ?)
        """,
        (normalized_fingerprint, normalized_event, created, payload_json),
    )


def upsert_issue(
    conn: sqlite3.Connection,
    *,
    check_id: str,
    severity: str,
    detail: str = "",
    source: str = "health",
    observed_at: str | datetime | None = None,
    status: str | None = None,
    fingerprint: str | None = None,
) -> dict[str, Any]:
    ensure_issue_registry_tables(conn)
    canonical_check_id = _normalize_identifier(check_id, "unknown.check")
    canonical_source = _normalize_identifier(source, "health")
    canonical_severity = normalize_severity(severity)
    canonical_status = (
        normalize_status(status)
        if status is not None
        else ("open" if canonical_severity in _FAIL_SEVERITIES else "resolved")
    )
    observed_iso = _to_iso_utc(observed_at)
    issue_fingerprint = (
        str(fingerprint or "").strip().lower()
        or build_issue_fingerprint(
            check_id=canonical_check_id,
            detail=detail,
            source=canonical_source,
        )
    )
    trimmed_detail = str(detail or "").strip()[:2000]

    existing = conn.execute(
        """
        SELECT severity, status, first_seen, consecutive_failures
        FROM maintenance_issues
        WHERE fingerprint = ?
        """,
        (issue_fingerprint,),
    ).fetchone()

    if existing is None:
        first_seen = observed_iso
        previous_severity = None
        previous_status = None
        consecutive_failures = 1 if canonical_severity in _FAIL_SEVERITIES else 0
        conn.execute(
            """
            INSERT INTO maintenance_issues
            (fingerprint, check_id, severity, status, first_seen, last_seen, consecutive_failures, last_detail, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                issue_fingerprint,
                canonical_check_id,
                canonical_severity,
                canonical_status,
                first_seen,
                observed_iso,
                consecutive_failures,
                trimmed_detail,
                canonical_source,
            ),
        )
        event_type = "created"
    else:
        previous_severity = str(existing[0] or "")
        previous_status = str(existing[1] or "")
        first_seen = str(existing[2] or observed_iso)
        prev_failures = int(existing[3] or 0)
        consecutive_failures = (
            prev_failures + 1 if canonical_severity in _FAIL_SEVERITIES else 0
        )
        conn.execute(
            """
            UPDATE maintenance_issues
            SET check_id = ?,
                severity = ?,
                status = ?,
                first_seen = ?,
                last_seen = ?,
                consecutive_failures = ?,
                last_detail = ?,
                source = ?
            WHERE fingerprint = ?
            """,
            (
                canonical_check_id,
                canonical_severity,
                canonical_status,
                first_seen,
                observed_iso,
                consecutive_failures,
                trimmed_detail,
                canonical_source,
                issue_fingerprint,
            ),
        )
        if canonical_status != previous_status:
            event_type = "status_changed"
        elif canonical_severity != previous_severity:
            event_type = "severity_changed"
        else:
            event_type = "observed"

    append_issue_event(
        conn,
        issue_fingerprint=issue_fingerprint,
        event_type=event_type,
        created_at=observed_iso,
        payload={
            "check_id": canonical_check_id,
            "severity": canonical_severity,
            "status": canonical_status,
            "source": canonical_source,
            "detail": trimmed_detail,
        },
    )
    # NOTE: Caller is responsible for conn.commit().  Both sync_issue_registry
    # and record_maintenance_issues batch multiple upserts into a single
    # transaction so the registry stays atomically consistent.
    return {
        "fingerprint": issue_fingerprint,
        "check_id": canonical_check_id,
        "severity": canonical_severity,
        "status": canonical_status,
        "first_seen": first_seen,
        "last_seen": observed_iso,
        "consecutive_failures": consecutive_failures,
        "source": canonical_source,
        "event_type": event_type,
        "severity_changed": previous_severity not in {None, canonical_severity},
    }


def should_notify(
    fingerprint: str,
    cooldown_hours: float,
    last_notified_at: str | datetime | None,
    severity_changed: bool,
) -> bool:
    if not str(fingerprint or "").strip():
        return False
    if severity_changed:
        return True
    if float(cooldown_hours or 0) <= 0:
        return True
    if not last_notified_at:
        return True

    last_dt = _parse_timestamp(last_notified_at)
    if last_dt is None:
        return True
    now = datetime.now(timezone.utc)
    return now >= last_dt + timedelta(hours=float(cooldown_hours))


__all__ = [
    "append_issue_event",
    "build_issue_fingerprint",
    "ensure_issue_registry_tables",
    "normalize_issue_detail",
    "normalize_severity",
    "normalize_status",
    "should_notify",
    "upsert_issue",
]
