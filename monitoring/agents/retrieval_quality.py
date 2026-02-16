"""Retrieval Quality — hit rate, relevance scoring, latency, and coverage checks."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import config
import db_pool
from monitoring._base import HealthCheck, _count_rows, _sqlite_table_exists

log = logging.getLogger(__name__)

# Table name for retrieval stats logged by memory/retriever.py
_STATS_TABLE = "retrieval_stats"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_retrieval_quality(molly=None) -> list[HealthCheck]:
    """Run all 5 retrieval quality checks and return results."""
    checks: list[HealthCheck] = []

    # Bail early if the instrumentation table doesn't exist yet
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        table_exists = _sqlite_table_exists(conn, _STATS_TABLE)
    except Exception as exc:
        checks.append(
            HealthCheck(
                check_id="retrieval.stats_table",
                layer="Retrieval Quality",
                label="Retrieval stats table",
                status="red",
                detail=f"DB probe failed ({exc})",
                action_required=True,
            )
        )
        return checks
    finally:
        if conn is not None:
            conn.close()

    if not table_exists:
        checks.append(
            HealthCheck(
                check_id="retrieval.stats_table",
                layer="Retrieval Quality",
                label="Retrieval stats table",
                status="yellow",
                detail=f"{_STATS_TABLE} table not yet created — instrument retriever first",
                watch_item=True,
            )
        )
        return checks

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

    # 1. Hit rate
    hit_status, hit_detail = _retrieval_hit_rate(cutoff)
    checks.append(
        HealthCheck(
            check_id="retrieval.hit_rate",
            layer="Retrieval Quality",
            label="Retrieval hit rate",
            status=hit_status,
            detail=hit_detail,
            watch_item=(hit_status == "yellow"),
            action_required=(hit_status == "red"),
        )
    )

    # 2. Relevance score
    rel_status, rel_detail = _retrieval_relevance_score(cutoff)
    checks.append(
        HealthCheck(
            check_id="retrieval.relevance_score",
            layer="Retrieval Quality",
            label="Avg relevance score",
            status=rel_status,
            detail=rel_detail,
            watch_item=(rel_status == "yellow"),
        )
    )

    # 3. Latency
    lat_status, lat_detail = _retrieval_latency(cutoff)
    checks.append(
        HealthCheck(
            check_id="retrieval.latency",
            layer="Retrieval Quality",
            label="Retrieval latency",
            status=lat_status,
            detail=lat_detail,
            watch_item=(lat_status == "yellow"),
            action_required=(lat_status == "red"),
        )
    )

    # 4. Source coverage
    cov_status, cov_detail = _retrieval_source_coverage(cutoff)
    checks.append(
        HealthCheck(
            check_id="retrieval.source_coverage",
            layer="Retrieval Quality",
            label="Source coverage in results",
            status=cov_status,
            detail=cov_detail,
            watch_item=(cov_status == "yellow"),
        )
    )

    # 5. Embedding staleness
    stale_status, stale_detail = _embedding_staleness()
    checks.append(
        HealthCheck(
            check_id="retrieval.embedding_staleness",
            layer="Retrieval Quality",
            label="Embedding coverage",
            status=stale_status,
            detail=stale_detail,
            watch_item=(stale_status == "yellow"),
            action_required=(stale_status == "red"),
        )
    )

    return checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retrieval_hit_rate(cutoff: str) -> tuple[str, str]:
    """Percentage of queries that returned >= 1 result."""
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        total = conn.execute(
            f"SELECT COUNT(*) FROM {_STATS_TABLE} WHERE created_at > ?", (cutoff,)
        ).fetchone()[0] or 0
        hits = conn.execute(
            f"SELECT COUNT(*) FROM {_STATS_TABLE} WHERE created_at > ? AND result_count > 0",
            (cutoff,),
        ).fetchone()[0] or 0
        if total == 0:
            return "yellow", "No retrieval queries in 24h"
        rate = hits / total
        detail = f"{rate:.0%} ({hits}/{total} queries returned results)"
        if rate >= 0.80:
            return "green", detail
        if rate >= 0.60:
            return "yellow", detail
        return "red", detail
    except Exception as exc:
        return "red", f"hit rate query failed ({exc})"
    finally:
        if conn is not None:
            conn.close()


def _retrieval_relevance_score(cutoff: str) -> tuple[str, str]:
    """Average cosine similarity of top-K results across recent queries."""
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        row = conn.execute(
            f"SELECT AVG(avg_similarity), COUNT(*) FROM {_STATS_TABLE} "
            f"WHERE created_at > ? AND avg_similarity IS NOT NULL",
            (cutoff,),
        ).fetchone()
        avg_sim = float(row[0]) if row and row[0] is not None else 0.0
        count = int(row[1]) if row else 0
        if count == 0:
            return "yellow", "No similarity data in 24h"
        detail = f"avg_similarity={avg_sim:.3f} across {count} queries"
        if avg_sim >= 0.7:
            return "green", detail
        if avg_sim >= 0.5:
            return "yellow", detail
        return "red", detail
    except Exception as exc:
        return "red", f"relevance score query failed ({exc})"
    finally:
        if conn is not None:
            conn.close()


def _retrieval_latency(cutoff: str) -> tuple[str, str]:
    """P50 and P95 query latency from retrieval_stats."""
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        rows = conn.execute(
            f"SELECT latency_ms FROM {_STATS_TABLE} "
            f"WHERE created_at > ? AND latency_ms IS NOT NULL "
            f"ORDER BY latency_ms",
            (cutoff,),
        ).fetchall()
        if not rows:
            return "yellow", "No latency data in 24h"
        latencies = [int(r[0]) for r in rows]
        n = len(latencies)
        p50 = latencies[n // 2]
        p95 = latencies[int(n * 0.95)]
        detail = f"P50={p50}ms, P95={p95}ms (n={n})"
        if p95 < 100:
            return "green", detail
        if p95 < 500:
            return "yellow", detail
        return "red", detail
    except Exception as exc:
        return "red", f"latency query failed ({exc})"
    finally:
        if conn is not None:
            conn.close()


def _retrieval_source_coverage(cutoff: str) -> tuple[str, str]:
    """Check if all expected sources (whatsapp, email, imessage) appear in recent results."""
    expected_sources = {"whatsapp", "email", "imessage"}
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        rows = conn.execute(
            f"SELECT sources FROM {_STATS_TABLE} WHERE created_at > ? AND sources IS NOT NULL",
            (cutoff,),
        ).fetchall()
        if not rows:
            return "yellow", "No source data in 24h"
        import json

        seen: set[str] = set()
        for (sources_json,) in rows:
            try:
                sources = json.loads(sources_json)
                if isinstance(sources, list):
                    seen.update(str(s).lower() for s in sources)
            except Exception:
                continue
        missing = expected_sources - seen
        if not missing:
            return "green", f"All sources represented: {', '.join(sorted(seen))}"
        return "yellow", f"Missing sources: {', '.join(sorted(missing))} (seen: {', '.join(sorted(seen))})"
    except Exception as exc:
        return "red", f"source coverage query failed ({exc})"
    finally:
        if conn is not None:
            conn.close()


def _embedding_staleness() -> tuple[str, str]:
    """Percentage of conversation chunks that have embeddings vs total."""
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
        total_row = conn.execute("SELECT COUNT(*) FROM conversation_chunks").fetchone()
        total = int(total_row[0]) if total_row else 0
        if total == 0:
            return "yellow", "No conversation chunks"
        embedded_row = conn.execute(
            "SELECT COUNT(*) FROM conversation_chunks WHERE embedding IS NOT NULL"
        ).fetchone()
        embedded = int(embedded_row[0]) if embedded_row else 0
        pct = embedded / total
        detail = f"{pct:.0%} ({embedded}/{total} chunks have embeddings)"
        if pct >= 0.95:
            return "green", detail
        if pct >= 0.85:
            return "yellow", detail
        return "red", detail
    except Exception as exc:
        return "red", f"embedding staleness check failed ({exc})"
    finally:
        if conn is not None:
            conn.close()
