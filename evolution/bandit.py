"""Thompson Sampling bandit for arm selection.

Arms represent sub-agents + skills.  Beta(1+successes, 1+failures) posterior
per arm, SQLite-backed via ``evolution.db``.
"""
from __future__ import annotations

import logging
import random
import time

from evolution.db import get_connection

log = logging.getLogger(__name__)

_WARM_START_PULLS = 3  # round-robin first N pulls per arm

BANDIT_ARMS = ["baseline", "concise_prompt", "thorough_prompt", "memory_heavy", "graph_context"]


def register_default_arms() -> None:
    """Ensure all standard arms exist in the database."""
    bandit = ThompsonBandit()
    for arm_id in BANDIT_ARMS:
        bandit.ensure_arm(arm_id)


class ThompsonBandit:
    """Thompson Sampling with Beta posteriors, backed by evolution.db."""

    # ------------------------------------------------------------------
    def select_arm(self, arm_ids: list[str] | None = None) -> str:
        """Pick the best arm via Thompson Sampling (or warm-start round-robin)."""
        conn = get_connection()
        try:
            rows = conn.execute("SELECT * FROM bandit_arms").fetchall()
            arms = {r["arm_id"]: dict(r) for r in rows}
        finally:
            conn.close()

        # If specific arm_ids requested, filter
        if arm_ids:
            arms = {k: v for k, v in arms.items() if k in arm_ids}

        if not arms:
            return arm_ids[0] if arm_ids else "default"

        # Warm-start: round-robin until every arm has _WARM_START_PULLS
        under_explored = [
            aid for aid, a in arms.items() if a["pulls"] < _WARM_START_PULLS
        ]
        if under_explored:
            chosen = min(under_explored, key=lambda aid: arms[aid]["pulls"])
            log.debug("Warm-start pull: %s (pulls=%d)", chosen, arms[chosen]["pulls"])
            return chosen

        # Thompson sampling: draw from Beta(1+s, 1+f)
        samples = {
            aid: random.betavariate(1 + a["successes"], 1 + a["failures"])
            for aid, a in arms.items()
        }
        chosen = max(samples, key=samples.get)  # type: ignore[arg-type]
        log.debug("Thompson selected %s (sample=%.3f)", chosen, samples[chosen])
        return chosen

    # ------------------------------------------------------------------
    def update_arm(self, arm_id: str, reward: float) -> None:
        """Update arm stats after observing a reward in [0, 1]."""
        success = int(reward >= 0.5)
        failure = 1 - success
        conn = get_connection()
        try:
            conn.execute(
                """INSERT INTO bandit_arms (arm_id, successes, failures, pulls,
                        total_reward, last_pulled)
                   VALUES (?, ?, ?, 1, ?, ?)
                   ON CONFLICT(arm_id) DO UPDATE SET
                        successes = successes + excluded.successes,
                        failures  = failures  + excluded.failures,
                        pulls     = pulls + 1,
                        total_reward = total_reward + excluded.total_reward,
                        last_pulled  = excluded.last_pulled
                """,
                (arm_id, success, failure, reward, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def get_stats(self) -> dict[str, dict]:
        """Return per-arm statistics."""
        conn = get_connection()
        try:
            rows = conn.execute("SELECT * FROM bandit_arms").fetchall()
            return {r["arm_id"]: dict(r) for r in rows}
        finally:
            conn.close()

    def ensure_arm(self, arm_id: str) -> None:
        """Register an arm if it doesn't exist yet."""
        conn = get_connection()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO bandit_arms (arm_id) VALUES (?)",
                (arm_id,),
            )
            conn.commit()
        finally:
            conn.close()
