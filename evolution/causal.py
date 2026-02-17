"""Causal self-tracking: change → consequence mapping.

When Molly modifies herself, this module records what happened.
Over time it builds a dependency graph: "touching graph.py breaks
dedup.py 40% of the time."  The DGM uses ``estimate_risk()`` before
proposing changes to high-risk files.
"""
from __future__ import annotations

import logging
import time

from evolution.db import get_connection

log = logging.getLogger(__name__)


def snapshot_baselines(affected_files: list[str]) -> dict[str, dict]:
    """Capture current test results for files that will be affected.

    Returns ``{file: {test: result, ...}}`` for later comparison.
    """
    # In production this runs pytest on specific test files and captures
    # per-test pass/fail.  For now returns empty baselines.
    baselines: dict[str, dict] = {}
    for f in affected_files:
        baselines[f] = {"_snapshot_time": time.time()}
    return baselines


def record_consequences(
    commit_hash: str,
    file_changed: str,
    baseline: dict,
    post: dict,
) -> int:
    """Store causal edges showing what broke (or didn't) after a change.

    Returns number of edges recorded.
    """
    conn = get_connection()
    edges = 0
    try:
        # Compare baseline vs post for each test
        all_tests = set(list(baseline.keys()) + list(post.keys()))
        for test_key in all_tests:
            if test_key.startswith("_"):
                continue
            base_result = str(baseline.get(test_key, "unknown"))
            post_result = str(post.get(test_key, "unknown"))
            delta = 0.0
            if base_result == "pass" and post_result != "pass":
                delta = -1.0
            elif base_result != "pass" and post_result == "pass":
                delta = 1.0

            conn.execute(
                """INSERT INTO causal_edges
                   (commit_hash, file_changed, test_file, test_name,
                    baseline_result, post_result, outcome_delta)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (commit_hash, file_changed, "", test_key,
                 base_result, post_result, delta),
            )
            edges += 1

        conn.commit()
        if edges:
            log.info("Recorded %d causal edges for %s", edges, file_changed)
    finally:
        conn.close()
    return edges


def estimate_risk(target_file: str) -> float:
    """Estimate probability of breaking something if *target_file* is changed.

    Based on historical causal edges: ratio of negative outcomes to total
    changes affecting this file.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome_delta < 0 THEN 1 ELSE 0 END) as breaks
               FROM causal_edges
               WHERE file_changed = ?""",
            (target_file,),
        ).fetchone()

        if not row or row["total"] == 0:
            return 0.0  # no history → no estimated risk

        risk = row["breaks"] / row["total"]
        log.debug("Risk for %s: %.2f (%d breaks / %d total)", target_file, risk, row["breaks"], row["total"])
        return risk
    finally:
        conn.close()


def get_dependency_map() -> dict[str, list[str]]:
    """Build file → [files it tends to break] map from causal edges."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT file_changed, test_file
               FROM causal_edges
               WHERE outcome_delta < 0 AND test_file != ''
               GROUP BY file_changed, test_file"""
        ).fetchall()

        dep_map: dict[str, list[str]] = {}
        for r in rows:
            dep_map.setdefault(r["file_changed"], []).append(r["test_file"])
        return dep_map
    finally:
        conn.close()


def prune_stale(max_age_days: int = 90) -> int:
    """Remove causal edges older than *max_age_days*. Returns count removed."""
    cutoff = time.time() - max_age_days * 86400
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM causal_edges WHERE timestamp < ?", (cutoff,))
        conn.commit()
        count = cur.rowcount
        if count:
            log.info("Pruned %d stale causal edges", count)
        return count
    finally:
        conn.close()
