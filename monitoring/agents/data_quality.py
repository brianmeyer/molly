"""Data Quality — 7+ data integrity and freshness checks."""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import config
import db_pool
from monitoring._base import (
    HEALTH_ENTITY_SAMPLE_SIZE,
    HealthCheck,
    _count_rows,
    _load_embedded_report_data,
    _parse_iso,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_data_quality(molly=None, latest_report_path: Path | None = None) -> list[HealthCheck]:
    """Run all data quality checks and return results."""
    checks: list[HealthCheck] = []

    # 1. Entity strength decay
    strength_status, strength_detail = _entity_strength_decay_check()
    checks.append(
        HealthCheck(
            check_id="quality.entity_strength_decay",
            layer="Data Quality",
            label="Entity strength decay",
            status=strength_status,
            detail=strength_detail,
            watch_item=(strength_status == "yellow"),
            action_required=(strength_status == "red"),
        )
    )

    # 2-3. MEMORY.md checks
    memory_path = config.WORKSPACE / "MEMORY.md"
    if not memory_path.exists():
        checks.append(
            HealthCheck(
                check_id="quality.memory_file",
                layer="Data Quality",
                label="MEMORY.md",
                status="red",
                detail="file missing",
                action_required=True,
            )
        )
    else:
        age_days = (datetime.now() - datetime.fromtimestamp(memory_path.stat().st_mtime)).days
        freshness = "green" if age_days <= 7 else ("yellow" if age_days <= 14 else "red")
        checks.append(
            HealthCheck(
                check_id="quality.memory_freshness",
                layer="Data Quality",
                label="MEMORY.md freshness",
                status=freshness,
                detail=f"Last modified {age_days} day(s) ago",
                watch_item=(freshness == "yellow"),
                action_required=(freshness == "red"),
            )
        )
        size = memory_path.stat().st_size
        content_status = "green" if size >= 100 else "red"
        checks.append(
            HealthCheck(
                check_id="quality.memory_content",
                layer="Data Quality",
                label="MEMORY.md content",
                status=content_status,
                detail=f"{size} bytes",
                action_required=(content_status == "red"),
            )
        )

    # 4. Orphan trend
    orphan_status, orphan_detail = _orphan_trend_check(latest_report_path)
    checks.append(
        HealthCheck(
            check_id="quality.orphan_trend",
            layer="Data Quality",
            label="Orphaned entities",
            status=orphan_status,
            detail=orphan_detail,
            watch_item=(orphan_status == "yellow"),
        )
    )

    # 5. Duplicate entities
    dup_status, dup_detail = _duplicate_entity_check()
    checks.append(
        HealthCheck(
            check_id="quality.duplicates",
            layer="Data Quality",
            label="Duplicate entity fuzzy match",
            status=dup_status,
            detail=dup_detail,
            watch_item=(dup_status == "yellow"),
            action_required=(dup_status == "red"),
        )
    )

    # 6. Chunk retention
    chunk_status, chunk_detail = _chunk_retention_check()
    checks.append(
        HealthCheck(
            check_id="quality.chunk_retention",
            layer="Data Quality",
            label="Conversation chunk retention",
            status=chunk_status,
            detail=chunk_detail,
            watch_item=(chunk_status == "yellow"),
            action_required=(chunk_status == "red"),
        )
    )

    # 7. Maintenance log
    maintenance_status, maintenance_detail = _maintenance_log_check()
    checks.append(
        HealthCheck(
            check_id="quality.maintenance_log",
            layer="Data Quality",
            label="Maintenance log",
            status=maintenance_status,
            detail=maintenance_detail,
            action_required=(maintenance_status == "red"),
        )
    )

    # 8. Operational tables
    op_status, op_detail = _operational_tables_check()
    checks.append(
        HealthCheck(
            check_id="quality.operational_tables",
            layer="Data Quality",
            label="Operational tables",
            status=op_status,
            detail=op_detail,
            watch_item=(op_status == "yellow"),
        )
    )

    return checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity_strength_decay_check() -> tuple[str, str]:
    try:
        from memory.graph import get_driver

        sample_size = max(1, int(HEALTH_ENTITY_SAMPLE_SIZE))
        driver = get_driver()
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (e:Entity)
                RETURN e.strength AS strength
                ORDER BY e.last_mentioned DESC
                LIMIT $n
                """,
                n=sample_size,
            )
            values = [float(r["strength"]) for r in rows if r["strength"] is not None]
        if not values:
            return "yellow", "No entities sampled"
        stuck = sum(1 for v in values if abs(v - 1.0) < 1e-6)
        if stuck == len(values):
            return "red", f"{stuck}/{len(values)} stuck at 1.0"
        if stuck > len(values) * 0.4:
            return "yellow", f"{stuck}/{len(values)} still at 1.0"
        return "green", f"{stuck}/{len(values)} at 1.0"
    except Exception as exc:
        return "red", f"sample failed ({exc})"


def _orphan_trend_check(latest_report_path: Path | None = None) -> tuple[str, str]:
    try:
        from memory.graph import get_driver

        driver = get_driver()
        with driver.session() as session:
            total = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            orphans = session.run(
                "MATCH (e:Entity) WHERE NOT (e)--() RETURN count(e) AS c"
            ).single()["c"]
        ratio = (orphans / total) if total else 0.0

        prev_ratio = None
        if latest_report_path:
            payload = _load_embedded_report_data(latest_report_path)
            for check in payload.get("checks", []):
                if check.get("id") == "quality.orphan_trend":
                    m = re.search(r"ratio=([0-9.]+)", str(check.get("detail", "")))
                    if m:
                        prev_ratio = float(m.group(1))
                        break

        detail = f"{orphans}/{total} (ratio={ratio:.3f})"
        if prev_ratio is None:
            return "green", detail
        if prev_ratio > 0 and ratio > prev_ratio * 1.2 and orphans > 10:
            return "yellow", f"{detail}, up from {prev_ratio:.3f}"
        return "green", detail
    except Exception as exc:
        return "red", f"trend check failed ({exc})"


def _duplicate_entity_check() -> tuple[str, str]:
    try:
        from memory.dedup import find_near_duplicates
        from memory.graph import get_driver

        driver = get_driver()
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (e:Entity)
                RETURN e.name AS name
                ORDER BY e.mention_count DESC
                LIMIT 160
                """
            )
            names = [str(r["name"]) for r in rows if r["name"]]

        near_matches = find_near_duplicates(names)
        near = len(near_matches)
        sample = ", ".join(
            f"{pair.left} ~ {pair.right}" for pair in near_matches[:3]
        )
        detail = f"{near} near-duplicates detected"
        if sample:
            detail = f"{detail} ({sample})"

        if near > 10:
            return "red", detail
        if near > 3:
            return "yellow", detail
        return "green", detail
    except Exception as exc:
        return "red", f"duplicate check failed ({exc})"


def _chunk_retention_check() -> tuple[str, str]:
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        row = conn.execute(
            "SELECT MIN(created_at), MAX(created_at), COUNT(*) FROM conversation_chunks"
        ).fetchone()
        conn.close()
        oldest, newest, count = row
        if int(count or 0) == 0:
            return "red", "0 chunks"
        oldest_dt = _parse_iso(str(oldest))
        newest_dt = _parse_iso(str(newest))
        if not oldest_dt or not newest_dt:
            return "yellow", f"oldest={oldest}, newest={newest}"
        age_days = (datetime.now(timezone.utc) - oldest_dt.astimezone(timezone.utc)).days
        if age_days > 90:
            return "yellow", f"oldest={oldest_dt.date()}, newest={newest_dt.date()}"
        return "green", f"oldest={oldest_dt.date()}, newest={newest_dt.date()}"
    except Exception as exc:
        return "red", f"retention check failed ({exc})"


def _maintenance_log_check() -> tuple[str, str]:
    maint_dir = config.WORKSPACE / "memory" / "maintenance"
    if not maint_dir.exists():
        return "red", "maintenance directory missing"
    today = date.today()
    candidates = [
        maint_dir / f"{today.isoformat()}.md",
        maint_dir / f"{(today - timedelta(days=1)).isoformat()}.md",
    ]
    for path in candidates:
        if path.exists():
            return "green", f"last report {path.stem}"
    return "red", "no report for today/yesterday"


def _operational_tables_check() -> tuple[str, str]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    skills = _count_rows(
        config.MOLLYGRAPH_PATH,
        "SELECT COUNT(*) FROM skill_executions WHERE created_at > ?",
        (cutoff,),
    )
    corrections = _count_rows(
        config.MOLLYGRAPH_PATH,
        "SELECT COUNT(*) FROM corrections WHERE created_at > ?",
        (cutoff,),
    )
    if skills == 0 and corrections == 0:
        return "yellow", "skill_executions=0, corrections=0 (7d)"
    return "green", f"skill_executions={skills}, corrections={corrections} (7d)"
