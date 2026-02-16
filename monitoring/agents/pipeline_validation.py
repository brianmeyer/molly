"""Pipeline Validation — 8 data-flow pipeline checks."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import config
import db_pool
from monitoring._base import (
    HEALTH_PIPELINE_WINDOW_HOURS,
    HealthCheck,
    _count_rows,
    _parse_iso,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline_validation(molly=None) -> list[HealthCheck]:
    """Run all 8 pipeline validation checks and return results."""
    checks: list[HealthCheck] = []
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=HEALTH_PIPELINE_WINDOW_HOURS)
    ).isoformat()

    # 1. Message → Embedding ratio
    msg_count, emb_count = _message_embedding_counts(cutoff)
    ratio = (emb_count / msg_count) if msg_count > 0 else 1.0
    if msg_count == 0:
        status = "yellow"
        detail = "No new messages in window"
    elif 0.95 <= ratio <= 1.05:
        status = "green"
        detail = f"Messages {msg_count} → embeddings {emb_count} ({ratio:.0%})"
    elif 0.85 <= ratio <= 1.15:
        status = "yellow"
        detail = f"Messages {msg_count} → embeddings {emb_count} ({ratio:.0%})"
    else:
        status = "red"
        detail = f"Messages {msg_count} → embeddings {emb_count} ({ratio:.0%})"
    checks.append(
        HealthCheck(
            check_id="pipeline.message_to_embedding",
            layer="Pipeline Validation",
            label="Message → Embedding",
            status=status,
            detail=detail,
            action_required=(status == "red"),
        )
    )

    # 2. Embedding → Entity extraction
    entity_count = _recent_entity_count(cutoff)
    if msg_count > 5 and entity_count == 0:
        status = "red"
    elif entity_count == 0:
        status = "yellow"
    else:
        status = "green"
    checks.append(
        HealthCheck(
            check_id="pipeline.embedding_to_entity",
            layer="Pipeline Validation",
            label="Embedding → Entity extraction",
            status=status,
            detail=f"Recent entities: {entity_count} (messages: {msg_count})",
            action_required=(status == "red"),
        )
    )

    # 3. Entity → Relationship sampling
    rel_status, rel_detail = _entity_relationship_sampling()
    checks.append(
        HealthCheck(
            check_id="pipeline.entity_to_relationship",
            layer="Pipeline Validation",
            label="Entity → Relationship sampling",
            status=rel_status,
            detail=rel_detail,
            watch_item=(rel_status == "yellow"),
        )
    )

    # 4. Source distribution
    source_status, source_detail = _source_distribution(cutoff)
    checks.append(
        HealthCheck(
            check_id="pipeline.source_distribution",
            layer="Pipeline Validation",
            label="Source distribution",
            status=source_status,
            detail=source_detail,
            action_required=(source_status == "red"),
        )
    )

    # 5. Timestamp format
    ts_status, ts_detail = _timestamp_format_check()
    checks.append(
        HealthCheck(
            check_id="pipeline.timestamp_format",
            layer="Pipeline Validation",
            label="Timestamp format",
            status=ts_status,
            detail=ts_detail,
            action_required=(ts_status == "red"),
        )
    )

    # 6. Approval system dry-run
    from approval import get_action_tier

    tier = get_action_tier("Bash")
    tier_status = "green" if tier == "CONFIRM" else "red"
    checks.append(
        HealthCheck(
            check_id="pipeline.approval_dry_run",
            layer="Pipeline Validation",
            label="Approval system dry-run",
            status=tier_status,
            detail=f"Bash tier = {tier}",
            action_required=(tier_status == "red"),
        )
    )

    # 7. Sub-agent Task calls
    task_calls = _tool_call_count("routing:subagent_start:%", cutoff)
    task_status = "green"
    if msg_count > 0 and task_calls == 0:
        task_status = "red"
    checks.append(
        HealthCheck(
            check_id="pipeline.subagent_task_calls",
            layer="Pipeline Validation",
            label="Sub-agent Task calls",
            status=task_status,
            detail=f"Logged Task calls: {task_calls}",
            action_required=(task_status == "red"),
        )
    )

    # 8. Preference signals table writable
    pref_status, pref_detail = _preference_table_writable()
    checks.append(
        HealthCheck(
            check_id="pipeline.preference_signals_writable",
            layer="Pipeline Validation",
            label="Preference signals table",
            status=pref_status,
            detail=pref_detail,
            action_required=(pref_status == "red"),
        )
    )

    return checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _message_embedding_counts(cutoff: str) -> tuple[int, int]:
    msg_count = _count_rows(
        config.DATABASE_PATH,
        "SELECT COUNT(*) FROM messages WHERE timestamp > ?",
        (cutoff,),
    )
    emb_count = _count_rows(
        config.MOLLYGRAPH_PATH,
        "SELECT COUNT(*) FROM conversation_chunks WHERE created_at > ?",
        (cutoff,),
    )
    return msg_count, emb_count


def _recent_entity_count(cutoff: str) -> int:
    try:
        from memory.graph import get_driver

        driver = get_driver()
        with driver.session() as session:
            record = session.run(
                "MATCH (e:Entity) WHERE e.last_mentioned > $cutoff RETURN count(e) AS c",
                cutoff=cutoff,
            ).single()
        return int(record["c"]) if record else 0
    except Exception:
        return 0


def _entity_relationship_sampling() -> tuple[str, str]:
    try:
        from memory.graph import get_driver

        driver = get_driver()
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (e:Entity)
                WHERE e.last_mentioned IS NOT NULL
                RETURN e.name AS name
                ORDER BY e.last_mentioned DESC
                LIMIT 5
                """
            )
            entities = [r["name"] for r in rows]
            if not entities:
                return "yellow", "No recent entities sampled"
            has_rel = 0
            for name in entities:
                rec = session.run(
                    "MATCH (e:Entity {name: $name}) OPTIONAL MATCH (e)-[r]-() RETURN count(r) AS c",
                    name=name,
                ).single()
                if rec and int(rec["c"]) > 0:
                    has_rel += 1
            ratio = has_rel / len(entities)
            if ratio >= 0.5:
                status = "green"
            elif ratio >= 0.3:
                status = "yellow"
            else:
                status = "red"
            return status, f"{has_rel}/{len(entities)} sampled entities have relationships"
    except Exception as exc:
        return "red", f"sampling failed ({exc})"


def _source_distribution(cutoff: str) -> tuple[str, str]:
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        cur = conn.execute(
            """
            SELECT source, COUNT(*) AS c
            FROM conversation_chunks
            WHERE created_at > ?
            GROUP BY source
            ORDER BY c DESC
            """,
            (cutoff,),
        )
        rows = cur.fetchall()
        if not rows:
            return "yellow", "No recent chunks"
        dist = {str(row[0] or "unknown"): int(row[1]) for row in rows}
        if "unknown" in dist:
            return "red", ", ".join(f"{k}={v}" for k, v in dist.items())
        if "whatsapp" not in dist:
            return "yellow", ", ".join(f"{k}={v}" for k, v in dist.items())
        return "green", ", ".join(f"{k}={v}" for k, v in dist.items())
    except Exception as exc:
        return "red", f"distribution query failed ({exc})"
    finally:
        if conn is not None:
            conn.close()


def _timestamp_format_check() -> tuple[str, str]:
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.DATABASE_PATH))
        cur = conn.execute(
            "SELECT timestamp FROM messages ORDER BY timestamp DESC LIMIT 5"
        )
        rows = [str(r[0]) for r in cur.fetchall()]
        if not rows:
            return "yellow", "No messages sampled"
        bad = 0
        for ts in rows:
            if _parse_iso(ts) is None:
                bad += 1
        if bad == 0:
            return "green", "All sampled timestamps ISO-compatible"
        return "red", f"{bad}/{len(rows)} sampled timestamps invalid"
    except Exception as exc:
        return "red", f"timestamp check failed ({exc})"
    finally:
        if conn is not None:
            conn.close()


def _tool_call_count(pattern: str, cutoff: str) -> int:
    return _count_rows(
        config.MOLLYGRAPH_PATH,
        "SELECT COUNT(*) FROM tool_calls WHERE tool_name LIKE ? AND created_at > ?",
        (pattern, cutoff),
    )


def _preference_table_writable() -> tuple[str, str]:
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        conn.execute("BEGIN")
        signal_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO preference_signals
            (id, signal_type, source, surfaced_summary, sender_pattern, owner_feedback, context, timestamp, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal_id, "health_check", "health", "health probe",
                "health:probe", "health check", "{}", now, now,
            ),
        )
        conn.execute("DELETE FROM preference_signals WHERE id = ?", (signal_id,))
        conn.rollback()
        return "green", "Writable"
    except Exception as exc:
        try:
            if conn is not None:
                conn.rollback()
        except Exception:
            pass
        return "red", f"Write probe failed ({exc})"
    finally:
        if conn is not None:
            conn.close()
