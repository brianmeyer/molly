"""Analytics helpers for skill performance and unresolved skill gaps."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
import db_pool

_SUCCESS_MARKERS = (
    "success",
    "succeeded",
    "complete",
    "completed",
    "approved",
    "resolved",
    "pass",
)
_FAILURE_MARKERS = (
    "fail",
    "failed",
    "error",
    "denied",
    "reject",
    "timeout",
    "cancel",
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "you",
    "your",
}


def _db_path(path: Path | None = None) -> Path:
    return path or config.MOLLYGRAPH_PATH


def _query_rows(
    query: str,
    params: tuple | None = None,
    path: Path | None = None,
) -> list[sqlite3.Row]:
    conn = db_pool.sqlite_connect(str(_db_path(path)))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(query, params or ())
        return cursor.fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _is_success_outcome(outcome: str) -> bool:
    normalized = str(outcome or "").strip().lower()
    if not normalized:
        return False
    if any(marker in normalized for marker in _FAILURE_MARKERS):
        return False
    return any(marker in normalized for marker in _SUCCESS_MARKERS)


def _build_skill_stats(rows: list[sqlite3.Row]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for row in rows:
        skill_name = str(row["skill_name"] or "").strip()
        if not skill_name:
            continue
        entry = stats.setdefault(
            skill_name,
            {
                "skill_name": skill_name,
                "invocations": 0,
                "successes": 0,
                "failures": 0,
                "success_rate": 0.0,
                "last_invoked_at": None,
            },
        )
        entry["invocations"] += 1
        if _is_success_outcome(str(row["outcome"] or "")):
            entry["successes"] += 1
        created_at = str(row["created_at"] or "")
        if created_at and (entry["last_invoked_at"] is None or created_at > entry["last_invoked_at"]):
            entry["last_invoked_at"] = created_at

    for entry in stats.values():
        entry["failures"] = entry["invocations"] - entry["successes"]
        if entry["invocations"] > 0:
            entry["success_rate"] = entry["successes"] / entry["invocations"]
    return stats


def get_skill_stats(skill_name: str) -> dict:
    """Return invocation/success stats for one skill."""
    target = str(skill_name or "").strip()
    rows = _query_rows(
        """
        SELECT skill_name, outcome, created_at
        FROM skill_executions
        WHERE skill_name = ?
        ORDER BY created_at ASC
        """,
        (target,),
    )
    stats = _build_skill_stats(rows).get(
        target,
        {
            "skill_name": target,
            "invocations": 0,
            "successes": 0,
            "failures": 0,
            "success_rate": 0.0,
            "last_invoked_at": None,
        },
    )
    return stats


def get_underperforming_skills(
    min_invocations: int = 5,
    max_success_rate: float = 0.6,
) -> list[dict]:
    """Return skills with low success rate and enough invocation volume."""
    min_count = max(1, int(min_invocations))
    max_rate = float(max_success_rate)
    rows = _query_rows(
        """
        SELECT skill_name, outcome, created_at
        FROM skill_executions
        ORDER BY skill_name ASC, created_at ASC
        """
    )
    stats = _build_skill_stats(rows)
    underperforming = [
        entry
        for entry in stats.values()
        if entry["invocations"] >= min_count and entry["success_rate"] <= max_rate
    ]
    return sorted(
        underperforming,
        key=lambda item: (item["success_rate"], -item["invocations"], item["skill_name"]),
    )


def _parse_tools(tools_used: str) -> list[str]:
    try:
        parsed = json.loads(tools_used or "[]")
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    tools: list[str] = []
    for item in parsed:
        name = str(item or "").strip()
        if name:
            tools.append(name)
    return tools


def _extract_keywords(user_message: str, tools: list[str]) -> set[str]:
    text_parts = [str(user_message or "")]
    for tool in tools:
        # Break mcp__foo__bar and routing:foo into searchable tokens.
        normalized = tool.replace("__", " ").replace(":", " ").replace(".", " ").replace("-", " ")
        text_parts.append(normalized)
    tokens = re.findall(r"[a-z0-9_]+", " ".join(text_parts).lower())
    keywords = {
        token
        for token in tokens
        if len(token) >= 3 and token not in _STOPWORDS and not token.isdigit()
    }
    return keywords


def get_skill_gap_clusters(days: int = 30) -> list[dict]:
    """Cluster unresolved skill gaps by keyword overlap (deterministic baseline)."""
    lookback_days = max(1, int(days))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    rows = _query_rows(
        """
        SELECT id, user_message, tools_used, session_id, created_at
        FROM skill_gaps
        WHERE created_at > ?
          AND addressed = 0
        ORDER BY created_at ASC, id ASC
        """,
        (cutoff,),
    )
    if not rows:
        return []

    clusters: list[dict] = []
    for row in rows:
        tools = _parse_tools(str(row["tools_used"] or "[]"))
        keywords = _extract_keywords(str(row["user_message"] or ""), tools)
        gap = {
            "id": int(row["id"]),
            "user_message": str(row["user_message"] or ""),
            "tools_used": tools,
            "session_id": str(row["session_id"] or ""),
            "created_at": str(row["created_at"] or ""),
            "keywords": keywords,
        }

        best_cluster = None
        best_overlap = 0
        for cluster in clusters:
            overlap = len(keywords & cluster["keyword_set"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_cluster = cluster

        if best_cluster is None or best_overlap < 2:
            best_cluster = {
                "cluster_id": len(clusters) + 1,
                "gaps": [],
                "keyword_set": set(),
                "keyword_counter": Counter(),
                "tool_counter": Counter(),
            }
            clusters.append(best_cluster)

        best_cluster["gaps"].append(gap)
        best_cluster["keyword_set"].update(keywords)
        best_cluster["keyword_counter"].update(keywords)
        best_cluster["tool_counter"].update(tools)

    result: list[dict] = []
    for cluster in clusters:
        gaps = cluster["gaps"]
        top_keywords = [
            key
            for key, _count in sorted(
                cluster["keyword_counter"].items(),
                key=lambda item: (-item[1], item[0]),
            )[:8]
        ]
        top_tools = [
            name
            for name, _count in sorted(
                cluster["tool_counter"].items(),
                key=lambda item: (-item[1], item[0]),
            )[:8]
        ]
        result.append(
            {
                "cluster_id": cluster["cluster_id"],
                "gap_count": len(gaps),
                "gap_ids": [gap["id"] for gap in gaps],
                "top_keywords": top_keywords,
                "top_tools": top_tools,
                "sessions": sorted({gap["session_id"] for gap in gaps if gap["session_id"]}),
                "examples": [gap["user_message"] for gap in gaps[:3]],
                "latest_created_at": max((gap["created_at"] for gap in gaps), default=""),
            }
        )

    return sorted(result, key=lambda item: (-item["gap_count"], item["cluster_id"]))
