"""Trajectory logging — dual JSONL + SQLite storage.

Every tool execution is logged with full context (arm, task, reward breakdown,
latency, tokens, tools, git commit).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from evolution.db import get_connection

log = logging.getLogger(__name__)

_JSONL_PATH = Path.home() / ".molly" / "data" / "trajectories.jsonl"


def log_trajectory(
    *,
    arm_id: str,
    task_hash: str,
    action: str,
    reward: float,
    outcome_score: float = 0.0,
    process_score: float = 0.0,
    safety_score: float = 0.0,
    cost_penalty: float = 0.0,
    diversity_bonus: float = 0.0,
    latency_seconds: float = 0.0,
    tokens_used: int = 0,
    tools_used: int = 0,
    git_commit: str = "",
) -> int:
    """Log a trajectory to both JSONL and SQLite. Returns the SQLite row id."""
    ts = time.time()

    # JSONL append
    _JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": ts, "arm_id": arm_id, "task_hash": task_hash,
        "action": action, "reward": reward,
        "outcome_score": outcome_score, "process_score": process_score,
        "safety_score": safety_score, "cost_penalty": cost_penalty,
        "diversity_bonus": diversity_bonus,
        "latency_seconds": latency_seconds, "tokens_used": tokens_used,
        "tools_used": tools_used, "git_commit": git_commit,
    }
    try:
        with open(_JSONL_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        log.warning("Failed to write trajectory JSONL", exc_info=True)

    # SQLite insert
    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO trajectories
               (arm_id, task_hash, action, reward,
                outcome_score, process_score, safety_score,
                cost_penalty, diversity_bonus,
                latency_seconds, tokens_used, tools_used, git_commit)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (arm_id, task_hash, action, reward,
             outcome_score, process_score, safety_score,
             cost_penalty, diversity_bonus,
             latency_seconds, tokens_used, tools_used, git_commit),
        )
        conn.commit()
        return cur.lastrowid or 0
    finally:
        conn.close()


def get_recent(days: int = 7, limit: int = 100) -> list[dict]:
    """Get recent trajectories from the last *days* days."""
    cutoff = time.time() - days * 86400
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM trajectories WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
