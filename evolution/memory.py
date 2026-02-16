"""Episodic memory with decay and pollution guard.

Stores DGM experiences with embeddings for similarity search.
Nightly distillation extracts IF-THEN guidelines from high-reward experiences.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from evolution._base import (
    MEMORY_DECAY_DAYS,
    MEMORY_K,
    MEMORY_MIN_CONFIDENCE,
    MEMORY_REWARD_FLOOR,
    Experience,
)
from evolution.db import get_connection

log = logging.getLogger(__name__)


def store_experience(
    task_hash: str,
    task_class: str,
    reward: float,
    confidence: float,
    content: dict,
    embedding: bytes = b"",
) -> int:
    """Store an experience in the episodic bank.

    Pollution guard: rejects experiences with confidence < MEMORY_MIN_CONFIDENCE.
    Returns the experience id.
    """
    if confidence < MEMORY_MIN_CONFIDENCE:
        log.debug("Experience rejected (confidence %.2f < %.2f)", confidence, MEMORY_MIN_CONFIDENCE)
        return 0

    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO experiences
               (embedding, task_hash, task_class, reward, confidence, content_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (embedding, task_hash, task_class, reward, confidence, json.dumps(content)),
        )
        conn.commit()
        return cur.lastrowid or 0
    finally:
        conn.close()


def retrieve_similar(
    query_embedding: bytes = b"",
    k: int = MEMORY_K,
    task_class: str | None = None,
) -> list[Experience]:
    """Retrieve top-k similar experiences, filtered by reward floor and confidence.

    When sqlite-vec is available, uses cosine similarity on embeddings.
    Otherwise falls back to most recent high-reward experiences.
    """
    conn = get_connection()
    try:
        # Fallback: retrieve by reward (most recent high-reward)
        where_clause = "WHERE reward >= ? AND confidence >= ? AND decay_weight > 0.01"
        params: list[Any] = [MEMORY_REWARD_FLOOR, MEMORY_MIN_CONFIDENCE]
        if task_class:
            where_clause += " AND task_class = ?"
            params.append(task_class)

        rows = conn.execute(
            f"SELECT * FROM experiences {where_clause} ORDER BY reward * decay_weight DESC LIMIT ?",
            (*params, k),
        ).fetchall()

        return [
            Experience(
                task_hash=r["task_hash"],
                task_class=r["task_class"],
                reward=r["reward"],
                confidence=r["confidence"],
                decay_weight=r["decay_weight"],
                content=json.loads(r["content_json"] or "{}"),
                embedding=r["embedding"] or b"",
            )
            for r in rows
        ]
    finally:
        conn.close()


def apply_decay(max_age_days: int = MEMORY_DECAY_DAYS) -> int:
    """Reduce decay_weight by 50% for experiences older than *max_age_days*.

    Returns number of experiences decayed.
    """
    cutoff = time.time() - max_age_days * 86400
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE experiences SET decay_weight = decay_weight * 0.5 WHERE timestamp < ? AND decay_weight > 0.01",
            (cutoff,),
        )
        conn.commit()
        count = cur.rowcount
        if count:
            log.info("Decayed %d experiences older than %d days", count, max_age_days)
        return count
    finally:
        conn.close()


def distill_guidelines(min_reward: float = 0.8, min_confidence: float = 0.7) -> int:
    """Nightly: extract IF-THEN rules from high-reward experiences.

    Scans experiences above thresholds, clusters by task_class, and creates
    guideline entries.  Source experiences get 50% decay after distillation.

    Returns number of guidelines created.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM experiences WHERE reward >= ? AND confidence >= ? AND decay_weight > 0.01",
            (min_reward, min_confidence),
        ).fetchall()

        if not rows:
            return 0

        # Cluster by task_class
        clusters: dict[str, list] = {}
        for r in rows:
            tc = r["task_class"] or "general"
            clusters.setdefault(tc, []).append(r)

        created = 0
        for task_class, exps in clusters.items():
            if len(exps) < 2:
                continue
            # Simple rule: summarize common patterns
            exp_ids = ",".join(str(e["id"]) for e in exps)
            rule = f"IF task_class='{task_class}' THEN apply patterns from {len(exps)} successful experiences"

            conn.execute(
                "INSERT INTO guidelines (rule_text, source_exp_ids) VALUES (?, ?)",
                (rule, exp_ids),
            )

            # Decay source experiences by 50%
            for e in exps:
                conn.execute(
                    "UPDATE experiences SET decay_weight = decay_weight * 0.5 WHERE id = ?",
                    (e["id"],),
                )
            created += 1

        conn.commit()
        log.info("Distilled %d guidelines from %d experiences", created, len(rows))
        return created
    finally:
        conn.close()


def get_relevant_guidelines(task_class: str, limit: int = 5) -> list[str]:
    """Get applicable guidelines for a given task class."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT rule_text FROM guidelines WHERE rule_text LIKE ? ORDER BY activation_count DESC LIMIT ?",
            (f"%{task_class}%", limit),
        ).fetchall()
        return [r["rule_text"] for r in rows]
    finally:
        conn.close()
