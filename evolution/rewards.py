"""Five-component reward function with EWMA normalization.

Components:
- Outcome (0.4)  — task completion quality
- Process (0.3)  — efficiency (latency, tokens, tool count)
- Safety  (0.1)  — no policy violations, no errors
- Cost    (-0.1) — API token cost penalty
- Diversity (0.1) — exploration bonus for underused arms
"""
from __future__ import annotations

import logging
import math

log = logging.getLogger(__name__)

# Weights
W_OUTCOME = 0.4
W_PROCESS = 0.3
W_SAFETY = 0.1
W_COST = -0.1
W_DIVERSITY = 0.1

# Baselines for process normalization
_BASELINE_LATENCY = 10.0   # seconds
_BASELINE_TOKENS = 2000
_BASELINE_TOOLS = 3

# EWMA smoothing
_EWMA_ALPHA = 0.01
_ewma_state: dict[str, float] = {}


def _normalize(value: float, baseline: float) -> float:
    """Normalize relative to baseline: 1.0 = at baseline, >1.0 = worse."""
    if baseline <= 0:
        return 1.0
    return max(0.0, min(2.0, value / baseline))


def _process_score(latency_s: float, tokens: int, tools: int) -> float:
    """Efficiency: lower latency/tokens/tools → higher score."""
    lat = 1.0 - _normalize(latency_s, _BASELINE_LATENCY) * 0.5
    tok = 1.0 - _normalize(tokens, _BASELINE_TOKENS) * 0.3
    tl = 1.0 - _normalize(tools, _BASELINE_TOOLS) * 0.2
    return max(0.0, min(1.0, lat + tok + tl))


def _diversity_score(arm_id: str, total_pulls: int, arm_pulls: int) -> float:
    """Exploration bonus for underused arms (inverse frequency)."""
    if total_pulls <= 0 or arm_pulls <= 0:
        return 1.0
    expected_freq = 1.0 / max(1, total_pulls)
    actual_freq = arm_pulls / total_pulls
    if actual_freq < expected_freq:
        return 1.0  # underused → full bonus
    return max(0.0, 1.0 - (actual_freq - expected_freq))


def _ewma_normalize(key: str, raw: float) -> float:
    """Exponentially weighted moving average normalization."""
    prev = _ewma_state.get(key, raw)
    smoothed = _EWMA_ALPHA * raw + (1.0 - _EWMA_ALPHA) * prev
    _ewma_state[key] = smoothed
    return smoothed


def compute_reward(
    *,
    outcome: float,
    latency_s: float = 0.0,
    tokens: int = 0,
    tools: int = 0,
    safety: float = 1.0,
    cost_usd: float = 0.0,
    arm_id: str = "",
    total_pulls: int = 0,
    arm_pulls: int = 0,
) -> dict:
    """Compute five-component reward.

    Returns dict with component scores and final ``reward`` in [0, 1].
    """
    proc = _process_score(latency_s, tokens, tools)
    div = _diversity_score(arm_id, total_pulls, arm_pulls)
    cost = min(1.0, cost_usd * 100)  # $0.01 → 1.0

    raw = (
        W_OUTCOME * outcome
        + W_PROCESS * proc
        + W_SAFETY * safety
        + W_COST * cost
        + W_DIVERSITY * div
    )
    reward = max(0.0, min(1.0, _ewma_normalize(arm_id or "_global", raw)))

    return {
        "reward": reward,
        "outcome_score": outcome,
        "process_score": proc,
        "safety_score": safety,
        "cost_penalty": cost,
        "diversity_bonus": div,
    }
