"""Three Laws guard framework — hard safety thresholds for evolution patches.

Law 1 — ENDURE (Stability):  pytest must pass, error rate <15%, clean imports.
Law 2 — EXCEL  (Performance): latency regression <500ms, reward regression <5%.
Law 3 — EVOLVE (Adaptation):  meaningful delta >0.02, LOC delta <50.

Auto-reject on critical ENDURE / EXCEL violations.
EVOLVE violations are warnings (logged, not blocking).
"""
from __future__ import annotations

import logging

from evolution._base import GuardResult, GuardViolation, ShadowEvalResult
from evolution.db import get_connection

log = logging.getLogger(__name__)

# ENDURE thresholds
_ERROR_RATE_MAX = 0.15

# EXCEL thresholds
_LATENCY_REGRESSION_MAX_MS = 500.0
_REWARD_REGRESSION_MAX = 0.05
_COST_RATIO_MAX = 0.9

# EVOLVE thresholds
_MIN_MEANINGFUL_DELTA = 0.02
_MAX_LOC_DELTA = 50


def validate(
    shadow: ShadowEvalResult | None = None,
    test_passed: bool = True,
    patch_loc_delta: int = 0,
) -> GuardResult:
    """Run all Three Laws guards and return result.

    Auto-rejects on critical ENDURE / EXCEL failures.
    EVOLVE violations are warnings only.
    """
    violations: list[GuardViolation] = []

    # --- Law 1: ENDURE (Stability) ---
    if not test_passed:
        violations.append(GuardViolation(
            guard_name="pytest_pass", law="endure",
            threshold=1.0, actual=0.0, severity="critical",
        ))

    if shadow and shadow.error_rate_after > _ERROR_RATE_MAX:
        violations.append(GuardViolation(
            guard_name="error_rate", law="endure",
            threshold=_ERROR_RATE_MAX, actual=shadow.error_rate_after,
            severity="critical",
        ))

    # --- Law 2: EXCEL (Performance) ---
    if shadow:
        latency_delta_ms = (
            (shadow.latency_p95_after - shadow.latency_p95_before) * 1000
        )
        if latency_delta_ms > _LATENCY_REGRESSION_MAX_MS:
            violations.append(GuardViolation(
                guard_name="latency_regression", law="excel",
                threshold=_LATENCY_REGRESSION_MAX_MS, actual=latency_delta_ms,
                severity="critical",
            ))

        reward_delta = shadow.avg_reward_after - shadow.avg_reward_before
        if reward_delta < -_REWARD_REGRESSION_MAX:
            violations.append(GuardViolation(
                guard_name="reward_regression", law="excel",
                threshold=-_REWARD_REGRESSION_MAX, actual=reward_delta,
                severity="critical",
            ))

    # --- Law 3: EVOLVE (Adaptation) — warnings only ---
    if shadow:
        delta = shadow.avg_reward_after - shadow.avg_reward_before
        if delta < _MIN_MEANINGFUL_DELTA:
            violations.append(GuardViolation(
                guard_name="min_delta", law="evolve",
                threshold=_MIN_MEANINGFUL_DELTA, actual=delta,
                severity="warning",
            ))

    if abs(patch_loc_delta) > _MAX_LOC_DELTA:
        violations.append(GuardViolation(
            guard_name="max_loc_delta", law="evolve",
            threshold=_MAX_LOC_DELTA, actual=float(abs(patch_loc_delta)),
            severity="warning",
        ))

    critical = [v for v in violations if v.severity == "critical"]
    passed = len(critical) == 0

    result = GuardResult(passed=passed, violations=violations)
    log.info(
        "Guard check: passed=%s, violations=%d (critical=%d)",
        passed, len(violations), len(critical),
    )
    return result


def log_violations(proposal_id: str, result: GuardResult) -> None:
    """Persist guard violations to evolution.db for audit trail."""
    if not result.violations:
        return
    conn = get_connection()
    try:
        for v in result.violations:
            conn.execute(
                """INSERT INTO guard_violations
                   (proposal_id, guard_name, threshold, actual_value, severity)
                   VALUES (?, ?, ?, ?, ?)""",
                (proposal_id, v.guard_name, v.threshold, v.actual, v.severity),
            )
        conn.commit()
    finally:
        conn.close()
