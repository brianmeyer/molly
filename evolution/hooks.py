"""Pre/post execution hooks for the evolution engine.

Called by ``agent.py`` around every message handling cycle:

- ``pre_execution_hook()`` — select bandit arm, retrieve memories, inject guidelines
- ``post_execution_hook()`` — compute reward, update bandit, log trajectory, store experience

**SAFETY:** Both hooks are wrapped in try/except → no-op on any
evolution.db failure.  They NEVER block message handling.
"""
from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)


def pre_execution_hook(
    task_hash: str = "",
    task_class: str = "",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select bandit arm and inject relevant context before execution.

    Returns a context dict with keys:
    - ``arm_id``: selected bandit arm (empty string if unavailable)
    - ``memories``: list of relevant past experiences
    - ``guidelines``: list of applicable IF-THEN rules
    - ``start_time``: high-resolution timer for latency tracking

    NEVER raises — returns empty context on any error.
    """
    result: dict[str, Any] = {
        "arm_id": "",
        "memories": [],
        "guidelines": [],
        "start_time": time.time(),
    }

    try:
        from evolution.bandit import ThompsonBandit
        from evolution.db import ensure_schema

        ensure_schema()
        bandit = ThompsonBandit()
        arm = bandit.select_arm()
        result["arm_id"] = arm
    except Exception:
        log.debug("pre_execution_hook: bandit selection failed", exc_info=True)

    try:
        from evolution.memory import retrieve_similar

        memories = retrieve_similar(task_class=task_class)
        result["memories"] = memories
    except Exception:
        log.debug("pre_execution_hook: memory retrieval failed", exc_info=True)

    try:
        from evolution.memory import get_relevant_guidelines

        if task_class:
            guidelines = get_relevant_guidelines(task_class)
            result["guidelines"] = guidelines
    except Exception:
        log.debug("pre_execution_hook: guideline retrieval failed", exc_info=True)

    return result


async def post_execution_hook(
    task_hash: str = "",
    task_class: str = "",
    arm_id: str = "",
    start_time: float = 0.0,
    outcome: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute reward, update bandit, log trajectory, store experience.

    *outcome* should contain:
    - ``success``: bool
    - ``tool_calls``: int
    - ``error``: optional str
    - ``safety_flags``: optional list[str]
    - ``tokens_used``: optional int

    NEVER raises — returns empty result on any error.
    """
    result: dict[str, Any] = {
        "reward": 0.0,
        "bandit_updated": False,
        "trajectory_logged": False,
        "experience_stored": False,
    }

    if outcome is None:
        outcome = {}

    elapsed = time.time() - start_time if start_time else 0.0

    # Compute reward
    try:
        from evolution.rewards import compute_reward

        reward = compute_reward(
            outcome_success=outcome.get("success", False),
            tool_calls=outcome.get("tool_calls", 0),
            error=outcome.get("error"),
            safety_flags=outcome.get("safety_flags", []),
            tokens_used=outcome.get("tokens_used", 0),
            latency_seconds=elapsed,
        )
        result["reward"] = reward
    except Exception:
        log.debug("post_execution_hook: reward computation failed", exc_info=True)

    # Update bandit
    try:
        if arm_id and result["reward"] > 0:
            from evolution.bandit import ThompsonBandit

            bandit = ThompsonBandit()
            bandit.update_arm(arm_id, success=result["reward"] >= 0.5)
            result["bandit_updated"] = True
    except Exception:
        log.debug("post_execution_hook: bandit update failed", exc_info=True)

    # Log trajectory
    try:
        from evolution.trajectory import log_trajectory

        log_trajectory(
            arm_id=arm_id,
            task_hash=task_hash,
            task_class=task_class,
            reward=result["reward"],
            latency_seconds=elapsed,
            metadata=outcome,
        )
        result["trajectory_logged"] = True
    except Exception:
        log.debug("post_execution_hook: trajectory logging failed", exc_info=True)

    # Store experience
    try:
        from evolution.memory import store_experience

        exp_id = store_experience(
            task_hash=task_hash,
            task_class=task_class,
            reward=result["reward"],
            confidence=1.0 if outcome.get("success") else 0.3,
            content={
                "arm_id": arm_id,
                "outcome": outcome,
                "latency_seconds": elapsed,
            },
        )
        result["experience_stored"] = bool(exp_id)
    except Exception:
        log.debug("post_execution_hook: experience storage failed", exc_info=True)

    return result
