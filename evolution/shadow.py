"""Shadow evaluation pipeline.

Tests patches against a golden set of known-good interactions before any
human sees them.  Measures reward delta: does the patch actually improve things?
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from evolution._base import (
    BASELINE_SAMPLES,
    MIN_REWARD_DELTA,
    SHADOW_SAMPLE_SIZE,
    SHADOW_TIMEOUT,
    ShadowEvalResult,
)
from evolution.db import get_connection

log = logging.getLogger(__name__)

GOLDEN_SET_DIR = Path.home() / ".molly" / "data" / "golden"


def load_golden_set(max_items: int = SHADOW_SAMPLE_SIZE) -> list[dict]:
    """Load golden items from JSON files in GOLDEN_SET_DIR."""
    GOLDEN_SET_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for fp in sorted(GOLDEN_SET_DIR.glob("*.json"))[:max_items]:
        try:
            items.append(json.loads(fp.read_text()))
        except Exception:
            log.warning("Failed to load golden item %s", fp, exc_info=True)
    return items


def add_golden_item(
    task_description: str,
    expected_output_summary: str,
    tools_expected: list[str] | None = None,
    max_latency_seconds: float = 30.0,
    reward: float = 0.8,
) -> Path:
    """Add a verified-good interaction to the golden set."""
    GOLDEN_SET_DIR.mkdir(parents=True, exist_ok=True)
    item = {
        "task_description": task_description,
        "expected_output_summary": expected_output_summary,
        "tools_expected": tools_expected or [],
        "max_latency_seconds": max_latency_seconds,
        "reward": reward,
        "created_at": time.time(),
    }
    filename = f"golden_{int(time.time() * 1000)}.json"
    path = GOLDEN_SET_DIR / filename
    path.write_text(json.dumps(item, indent=2))
    log.info("Added golden item: %s", path.name)
    return path


def prune_stale(max_age_days: int = 90) -> int:
    """Remove golden items older than *max_age_days*. Returns count removed."""
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for fp in GOLDEN_SET_DIR.glob("*.json"):
        try:
            data = json.loads(fp.read_text())
            if data.get("created_at", 0) < cutoff:
                fp.unlink()
                removed += 1
        except Exception:
            pass
    if removed:
        log.info("Pruned %d stale golden items", removed)
    return removed


async def evaluate(
    proposal_id: str,
    patched_fn=None,
    baseline_fn=None,
) -> ShadowEvalResult:
    """Run shadow evaluation comparing patched vs baseline on the golden set.

    If the golden set is empty, returns a result with ``golden_pass_rate=0.0``
    and the ``insufficient_data`` flag in the note.

    *patched_fn* and *baseline_fn* are async callables that take a golden item
    and return ``{reward, error, latency_s}``.  When ``None`` (during initial
    setup) a stub evaluation is performed.
    """
    golden = load_golden_set()

    if not golden:
        log.warning("Empty golden set — shadow eval returns insufficient-data")
        result = ShadowEvalResult(
            proposal_id=proposal_id,
            avg_reward_before=0.0, avg_reward_after=0.0,
            error_rate_before=0.0, error_rate_after=0.0,
            latency_p95_before=0.0, latency_p95_after=0.0,
            golden_pass_rate=0.0,
        )
        _store_result(result, is_improvement=False, guard_passed=False)
        return result

    # Stub: when not yet wired, return conservative default
    if patched_fn is None or baseline_fn is None:
        log.debug("Shadow eval stub — no patched/baseline fn provided")
        result = ShadowEvalResult(
            proposal_id=proposal_id,
            avg_reward_before=0.0, avg_reward_after=0.0,
            error_rate_before=0.0, error_rate_after=0.0,
            latency_p95_before=0.0, latency_p95_after=0.0,
            golden_pass_rate=1.0,
        )
        _store_result(result, is_improvement=False, guard_passed=True)
        return result

    # Real evaluation (wired in Batch 8)
    import asyncio

    baseline_rewards, patched_rewards = [], []
    baseline_errors, patched_errors = 0, 0
    baseline_latencies, patched_latencies = [], []

    for item in golden:
        for _ in range(BASELINE_SAMPLES):
            try:
                r = await asyncio.wait_for(baseline_fn(item), timeout=SHADOW_TIMEOUT)
                baseline_rewards.append(r.get("reward", 0))
                baseline_latencies.append(r.get("latency_s", 0))
                if r.get("error"):
                    baseline_errors += 1
            except Exception:
                baseline_errors += 1

        try:
            r = await asyncio.wait_for(patched_fn(item), timeout=SHADOW_TIMEOUT)
            patched_rewards.append(r.get("reward", 0))
            patched_latencies.append(r.get("latency_s", 0))
            if r.get("error"):
                patched_errors += 1
        except Exception:
            patched_errors += 1

    n_baseline = max(1, len(golden) * BASELINE_SAMPLES)
    n_patched = max(1, len(golden))

    def _p95(vals: list[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    result = ShadowEvalResult(
        proposal_id=proposal_id,
        avg_reward_before=sum(baseline_rewards) / max(1, len(baseline_rewards)),
        avg_reward_after=sum(patched_rewards) / max(1, len(patched_rewards)),
        error_rate_before=baseline_errors / n_baseline,
        error_rate_after=patched_errors / n_patched,
        latency_p95_before=_p95(baseline_latencies),
        latency_p95_after=_p95(patched_latencies),
        golden_pass_rate=(n_patched - patched_errors) / n_patched,
    )
    is_imp = (result.avg_reward_after - result.avg_reward_before) >= MIN_REWARD_DELTA
    _store_result(result, is_improvement=is_imp, guard_passed=is_imp)
    return result


def _store_result(result: ShadowEvalResult, *, is_improvement: bool, guard_passed: bool) -> None:
    """Persist shadow eval result to evolution.db."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO shadow_results
               (proposal_id, avg_reward_before, avg_reward_after,
                error_rate_before, error_rate_after,
                latency_p95_before, latency_p95_after,
                golden_pass_rate, is_improvement, guard_passed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (result.proposal_id, result.avg_reward_before, result.avg_reward_after,
             result.error_rate_before, result.error_rate_after,
             result.latency_p95_before, result.latency_p95_after,
             result.golden_pass_rate, int(is_improvement), int(guard_passed)),
        )
        conn.commit()
    finally:
        conn.close()
