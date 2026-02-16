"""Track F Pre-Prod — 4 readiness checks + audit report generation."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config
import db_pool
from monitoring._base import (
    HealthCheck,
    _now_local,
    _parse_iso,
    _sqlite_table_exists,
    _status_emoji,
)
from utils import normalize_timestamp

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_track_f_checks() -> list[HealthCheck]:
    """Run all 4 Track F pre-production readiness checks."""
    return [
        _parser_compatibility_check(),
        _skill_telemetry_presence_check(),
        _foundry_ingestion_health_check(),
        _promotion_drift_status_check(),
    ]


def run_track_f_audit(output_dir: Path | None = None) -> Path:
    """Generate a Track F pre-prod readiness audit report and return its path."""
    checks = run_track_f_checks()
    summary = {
        "green": sum(1 for c in checks if c.status == "green"),
        "yellow": sum(1 for c in checks if c.status == "yellow"),
        "red": sum(1 for c in checks if c.status == "red"),
    }
    verdict = "NO-GO" if summary["red"] else "GO"

    lines = [f"# Track F Pre-Prod Readiness Audit — {_now_local().strftime('%Y-%m-%d %H:%M %Z')}", ""]
    lines.append("## Readiness Checks")
    for check in checks:
        lines.append(f"- {_status_emoji(check.status)} `{check.check_id}`: {check.detail}")
    lines += ["", f"## Verdict: **{verdict}**", ""]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "verdict": verdict,
        "checks": [
            {
                "id": c.check_id,
                "label": c.label,
                "status": c.status,
                "detail": c.detail,
                "action_required": bool(c.action_required),
            }
            for c in checks
        ],
    }
    lines.append(f"<!-- TRACK_F_AUDIT_DATA: {json.dumps(payload, ensure_ascii=True)} -->")

    audit_dir = output_dir or config.TRACK_F_AUDIT_DIR
    audit_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = audit_dir / f"track-f-preprod-{stamp}.md"
    report_path.write_text("\n".join(lines).rstrip() + "\n")
    return report_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_result(
    *,
    check_id: str,
    label: str,
    ok: bool,
    success_detail: str,
    failure_detail: str,
    enforce_flag: bool,
) -> HealthCheck:
    if ok:
        return HealthCheck(check_id, "Track F Pre-Prod", label, "green", success_detail)
    hard_enforcement = bool(enforce_flag) and not bool(getattr(config, "TRACK_F_REPORT_ONLY", True))
    mode = "enforced" if hard_enforcement else "report-only"
    return HealthCheck(
        check_id,
        "Track F Pre-Prod",
        label,
        "red" if hard_enforcement else "yellow",
        f"{failure_detail} [{mode}]",
        action_required=hard_enforcement,
        watch_item=not hard_enforcement,
    )


def _parser_compatibility_check() -> HealthCheck:
    base = 1770598413
    variants = {
        "seconds": str(base),
        "milliseconds": str(base * 1_000),
        "microseconds": str(base * 1_000_000),
        "nanoseconds": str(base * 1_000_000_000),
    }
    failures: list[str] = []
    for label, raw in variants.items():
        parsed = _parse_iso(raw)
        if parsed is None or int(parsed.timestamp()) != base:
            failures.append(f"{label}:parse")
        normalized = normalize_timestamp(raw)
        reparsed = _parse_iso(normalized)
        if reparsed is None or int(reparsed.timestamp()) != base:
            failures.append(f"{label}:normalize")

    if _parse_iso("2026-02-09T00:53:33+00:00") is None:
        failures.append("iso:parse")

    return _check_result(
        check_id="trackf.parser_compatibility",
        label="Parser compatibility",
        ok=not failures,
        success_detail="seconds/ms/us/ns epoch + ISO parse paths are compatible",
        failure_detail=f"parser mismatch in {', '.join(failures)}",
        enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_PARSER_COMPAT", False)),
    )


def _skill_telemetry_presence_check() -> HealthCheck:
    required = {"id", "skill_name", "trigger", "outcome", "user_approval", "edits_made", "created_at"}
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        try:
            if not _sqlite_table_exists(conn, "skill_executions"):
                return _check_result(
                    check_id="trackf.skill_telemetry_presence",
                    label="Skill telemetry presence",
                    ok=False,
                    success_detail="",
                    failure_detail="table missing: skill_executions",
                    enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_SKILL_TELEMETRY", False)),
                )
            rows = conn.execute("PRAGMA table_info(skill_executions)").fetchall()
            cols = {str(row[1]) for row in rows}
            missing = sorted(required - cols)
        finally:
            conn.close()
    except Exception as exc:
        return _check_result(
            check_id="trackf.skill_telemetry_presence",
            label="Skill telemetry presence",
            ok=False,
            success_detail="",
            failure_detail=f"telemetry probe failed ({exc})",
            enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_SKILL_TELEMETRY", False)),
        )

    return _check_result(
        check_id="trackf.skill_telemetry_presence",
        label="Skill telemetry presence",
        ok=not missing,
        success_detail="schema present",
        failure_detail=("missing columns: " + ", ".join(missing)) if missing else "schema unavailable",
        enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_SKILL_TELEMETRY", False)),
    )


def _foundry_ingestion_health_check() -> HealthCheck:
    categories = ("skill", "tool", "core")
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        try:
            if not _sqlite_table_exists(conn, "self_improvement_events"):
                return _check_result(
                    check_id="trackf.foundry_ingestion_health",
                    label="Foundry ingestion health",
                    ok=False,
                    success_detail="",
                    failure_detail="table missing: self_improvement_events",
                    enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_FOUNDRY_INGESTION", False)),
                )
            placeholders = ",".join("?" for _ in categories)
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM self_improvement_events WHERE category IN ({placeholders})",
                    categories,
                ).fetchone()[0]
                or 0
            )
        finally:
            conn.close()
    except Exception as exc:
        return _check_result(
            check_id="trackf.foundry_ingestion_health",
            label="Foundry ingestion health",
            ok=False,
            success_detail="",
            failure_detail=f"ingestion probe failed ({exc})",
            enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_FOUNDRY_INGESTION", False)),
        )

    return _check_result(
        check_id="trackf.foundry_ingestion_health",
        label="Foundry ingestion health",
        ok=total > 0,
        success_detail=f"events={total}",
        failure_detail="no ingestion events",
        enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_FOUNDRY_INGESTION", False)),
    )


def _promotion_drift_status_check() -> HealthCheck:
    categories = ("skill", "tool", "core")
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        try:
            if not _sqlite_table_exists(conn, "self_improvement_events"):
                return _check_result(
                    check_id="trackf.promotion_drift_status",
                    label="Promotion drift status",
                    ok=False,
                    success_detail="",
                    failure_detail="table missing: self_improvement_events",
                    enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_PROMOTION_DRIFT", False)),
                )
            placeholders = ",".join("?" for _ in categories)
            rows = conn.execute(
                f"""
                SELECT lower(status) AS status, COUNT(*) AS c
                FROM self_improvement_events
                WHERE category IN ({placeholders})
                GROUP BY lower(status)
                """,
                categories,
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        return _check_result(
            check_id="trackf.promotion_drift_status",
            label="Promotion drift status",
            ok=False,
            success_detail="",
            failure_detail=f"promotion drift probe failed ({exc})",
            enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_PROMOTION_DRIFT", False)),
        )

    counts: dict[str, int] = {str(status or "").lower(): int(count or 0) for status, count in rows}
    proposed = counts.get("proposed", 0)
    promoted = counts.get("approved", 0) + counts.get("deployed", 0)
    pending = max(0, proposed - promoted - counts.get("rejected", 0))
    ok = pending <= int(getattr(config, "TRACK_F_PROMOTION_DRIFT_MAX_PENDING", 20))

    return _check_result(
        check_id="trackf.promotion_drift_status",
        label="Promotion drift status",
        ok=ok,
        success_detail=f"proposed={proposed}, promoted={promoted}, pending={pending}",
        failure_detail=f"pending={pending}",
        enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_PROMOTION_DRIFT", False)),
    )
