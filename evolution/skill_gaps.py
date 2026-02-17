"""Skill-gap detection and proposal service.

Extracted from ``evolution/skills.py`` (SelfImprovementEngine).  Owns pattern
step helpers, Foundry signal loading, skill-gap cluster queries, the
track-a-skill hook integration, and row-level addressed/cooldown bookkeeping.
"""
from __future__ import annotations

import inspect
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import config
import db_pool
from evolution.infra import (
    _parse_datetime,
    _LOW_VALUE_WORKFLOW_TOOL_NAMES,
    _SKILL_GAP_MIN_CLUSTER_SIZE,
    _SKILL_GAP_RECENT_PROPOSAL_DAYS,
    _SKILL_GAP_PROPOSAL_COOLDOWN_DAYS,
)
from foundry_adapter import FoundrySequenceSignal, load_foundry_sequence_signals

log = logging.getLogger(__name__)


def _normalize_track_a_skill_hook_result(result: Any) -> dict[str, Any]:
    """Normalize return value of a track-a-skill hook to a standard status dict."""
    if isinstance(result, dict):
        normalized = dict(result)
    elif isinstance(result, str):
        normalized = {"proposal_id": result, "status": "pending"}
    elif isinstance(result, bool):
        normalized = {"status": "pending" if result else "rejected"}
    elif result is None:
        normalized = {"status": "pending"}
    else:
        normalized = {"status": "pending", "detail": str(result)}

    proposal_id = (
        str(
            normalized.get("proposal_id")
            or normalized.get("id")
            or normalized.get("pending_id")
            or ""
        )
        .strip()
    )
    status = str(normalized.get("status", "")).strip().lower()
    if not status:
        status = "pending" if proposal_id else "unavailable"
    status_map = {
        "created": "pending",
        "queued": "pending",
        "proposed": "pending",
        "drafted": "pending",
        "pending": "pending",
        "approved": "activated",
        "active": "activated",
        "activated": "activated",
        "deployed": "activated",
    }
    resolved_status = status_map.get(status, status)
    return {
        "status": resolved_status,
        "proposal_id": proposal_id,
        "reason": str(normalized.get("reason", "")),
    }


class SkillGapsService:
    """Skill-gap detection and proposal service.

    Receives explicit ``InfraService``, ``OwnerCommsService``, and
    ``SkillLifecycleService`` dependencies.
    """

    def __init__(self, ctx, infra, comms, lifecycle):
        from evolution.context import EngineContext
        from evolution.infra import InfraService
        from evolution.owner_comms import OwnerCommsService
        from evolution.skill_lifecycle import SkillLifecycleService
        self.ctx: EngineContext = ctx
        self.infra: InfraService = infra
        self.comms: OwnerCommsService = comms
        self.lifecycle: SkillLifecycleService = lifecycle

    # -- pure helpers (delegate to pattern_helpers) -------------------------

    @staticmethod
    def pattern_steps(pattern: dict[str, Any]) -> list[str]:
        from evolution.pattern_helpers import pattern_steps as _pattern_steps
        return _pattern_steps(pattern)

    @staticmethod
    def is_low_value_workflow_tool(tool_name: str) -> bool:
        from evolution.pattern_helpers import is_low_value_workflow_tool as _is_low_value
        return _is_low_value(tool_name)

    def load_foundry_sequence_signals(self, days: int = 30) -> dict[str, FoundrySequenceSignal]:
        from evolution.pattern_helpers import is_low_value_workflow_tool as _is_low_value
        try:
            return load_foundry_sequence_signals(
                days=days,
                is_low_value_tool=_is_low_value,
            )
        except Exception:
            log.debug("Failed to load Foundry observation signals", exc_info=True)
            return {}

    # -- static row helpers ----------------------------------------------------

    @staticmethod
    def _is_truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        text = str(value or "").strip().lower()
        return text in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _skill_gap_cluster_column(columns: list[str]) -> str:
        for candidate in (
            "cluster_key",
            "gap_cluster",
            "cluster",
            "skill_gap_cluster",
            "skill_name",
            "skill",
            "name",
            "title",
        ):
            if candidate in columns:
                return candidate
        return ""

    def _skill_gap_row_is_open(self, row: dict[str, Any]) -> bool:
        status = str(row.get("status", "") or "").strip().lower()
        if status in {"addressed", "resolved", "closed", "activated", "active"}:
            return False

        proposal_status = str(row.get("proposal_status", "") or "").strip().lower()
        if proposal_status in {"pending", "proposed", "activated", "active", "approved"}:
            return False

        for flag_field in ("addressed", "resolved"):
            if flag_field in row and self._is_truthy(row.get(flag_field)):
                return False

        for timestamp_field in ("addressed_at", "resolved_at"):
            if str(row.get(timestamp_field, "") or "").strip():
                return False

        return True

    @staticmethod
    def _skill_gap_row_cooldown_until(row: dict[str, Any]) -> datetime | None:
        latest: datetime | None = None
        for field_name in ("cooldown_until", "proposal_cooldown_until", "next_proposal_after"):
            dt = _parse_datetime(str(row.get(field_name, "") or ""))
            if not dt:
                continue
            if latest is None or dt > latest:
                latest = dt
        return latest

    # -- event bookkeeping --------------------------------------------------

    def has_recent_event(
        self,
        category: str,
        title: str,
        days: int = 30,
    ) -> bool:
        if not title:
            return False
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self.infra.rows(
            config.MOLLYGRAPH_PATH,
            """
            SELECT COUNT(*) AS c
            FROM self_improvement_events
            WHERE category = ?
              AND lower(title) = ?
              AND created_at > ?
            """,
            (category, title.strip().lower(), cutoff),
        )
        if not rows:
            return False
        try:
            return int(rows[0].get("c", 0) or 0) > 0
        except Exception:
            return False

    # -- pattern-based proposals --------------------------------------------

    async def propose_skill_updates_from_patterns(self, patterns: list[dict[str, Any]]) -> dict[str, Any]:
        if not patterns:
            return {"status": "skipped", "reason": "no patterns"}

        for pattern in patterns:
            steps = self.pattern_steps(pattern)
            if len(steps) < 3:
                continue
            result = await self.lifecycle.propose_skill_from_patterns([pattern], source="pattern")
            if str(result.get("status", "")).lower() == "skipped":
                continue
            return result

        return {"status": "skipped", "reason": "no new skill candidates"}

    # -- cluster-based proposals --------------------------------------------

    async def propose_skill_updates_from_gap_clusters(
        self,
        min_cluster_size: int = _SKILL_GAP_MIN_CLUSTER_SIZE,
    ) -> dict[str, Any]:
        clusters = self.query_skill_gap_clusters(min_cluster_size=min_cluster_size)
        if not clusters:
            return {"status": "skipped", "reason": "no skill gap clusters"}

        drafted = 0
        skipped = 0
        failures = 0
        now = datetime.now(timezone.utc)

        for cluster in clusters:
            cluster_key = str(cluster.get("cluster_key", "")).strip()
            cluster_size = int(cluster.get("count", 0) or 0)
            if not cluster_key or cluster_size < min_cluster_size:
                skipped += 1
                continue

            cooldown_until = _parse_datetime(str(cluster.get("cooldown_until", "")))
            if cooldown_until and cooldown_until > now:
                skipped += 1
                continue

            if self.has_recent_skill_gap_proposal(
                cluster_key,
                days=_SKILL_GAP_RECENT_PROPOSAL_DAYS,
            ):
                skipped += 1
                continue

            proposal_title = f"Skill gap cluster: {cluster_key}"
            hook_result = await self.invoke_track_a_skill_hook(cluster)
            hook_status = str(hook_result.get("status", "")).strip().lower()
            proposal_id = str(hook_result.get("proposal_id", "")).strip()

            if hook_status in {"unavailable", "rejected"}:
                skipped += 1
                self.comms.log_improvement_event(
                    event_type="proposal",
                    category="skill-gap",
                    title=proposal_title,
                    payload=json.dumps(
                        {
                            "cluster_key": cluster_key,
                            "cluster_size": cluster_size,
                            "reason": str(hook_result.get("reason", hook_status)),
                        },
                        ensure_ascii=True,
                    ),
                    status="skipped",
                )
                continue

            if hook_status in {"error", "failed"}:
                failures += 1
                self.comms.log_improvement_event(
                    event_type="proposal",
                    category="skill-gap",
                    title=proposal_title,
                    payload=json.dumps(
                        {
                            "cluster_key": cluster_key,
                            "cluster_size": cluster_size,
                            "reason": str(hook_result.get("reason", hook_status)),
                        },
                        ensure_ascii=True,
                    ),
                    status="failed",
                )
                continue

            lifecycle_status = "activated" if hook_status in {"activated", "approved", "active"} else "proposed"
            payload = json.dumps(
                {
                    "cluster_key": cluster_key,
                    "cluster_size": cluster_size,
                    "proposal_id": proposal_id,
                    "hook_status": hook_status or "pending",
                },
                ensure_ascii=True,
            )
            self.comms.log_improvement_event(
                event_type="proposal",
                category="skill-gap",
                title=proposal_title,
                payload=payload,
                status="pending",
            )
            if lifecycle_status == "activated":
                self.comms.log_improvement_event(
                    event_type="proposal",
                    category="skill-gap",
                    title=proposal_title,
                    payload=payload,
                    status="activated",
                )
            self.mark_skill_gap_rows_addressed(
                cluster_key=cluster_key,
                proposal_id=proposal_id,
                lifecycle_status=lifecycle_status,
            )
            drafted += 1

        if drafted == 0 and failures == 0:
            return {"status": "skipped", "reason": "no eligible skill gap clusters"}
        return {
            "status": "completed" if drafted > 0 else "failed",
            "drafted": drafted,
            "skipped": skipped,
            "failures": failures,
        }

    def query_skill_gap_clusters(
        self,
        min_cluster_size: int = _SKILL_GAP_MIN_CLUSTER_SIZE,
    ) -> list[dict[str, Any]]:
        db_path = config.MOLLYGRAPH_PATH
        cluster_totals: dict[str, dict[str, Any]] = {}

        for table_name in ("skill_gap_clusters", "skill_gaps"):
            columns = self.infra.table_columns(db_path, table_name)
            if not columns:
                continue
            cluster_column = self._skill_gap_cluster_column(columns)
            if not cluster_column:
                continue
            count_column = next(
                (
                    name
                    for name in (
                        "cluster_size",
                        "occurrences",
                        "count",
                        "frequency",
                        "invocations",
                        "hits",
                    )
                    if name in columns
                ),
                "",
            )
            rows = self.infra.rows(db_path, f"SELECT * FROM {table_name}", ())
            for row in rows:
                if not self._skill_gap_row_is_open(row):
                    continue
                cluster_key = str(row.get(cluster_column, "")).strip()
                if not cluster_key:
                    continue
                raw_count = row.get(count_column, 1) if count_column else 1
                try:
                    count = int(raw_count or 0)
                except (TypeError, ValueError):
                    count = 0
                if count <= 0:
                    count = 1
                entry = cluster_totals.setdefault(
                    cluster_key,
                    {
                        "cluster_key": cluster_key,
                        "count": 0,
                        "cooldown_until": "",
                        "source_tables": set(),
                    },
                )
                entry["count"] += count
                entry["source_tables"].add(table_name)

                cooldown_until = self._skill_gap_row_cooldown_until(row)
                if cooldown_until:
                    current = _parse_datetime(str(entry.get("cooldown_until", "")))
                    if not current or cooldown_until > current:
                        entry["cooldown_until"] = cooldown_until.isoformat()

        clusters = []
        for entry in cluster_totals.values():
            if int(entry.get("count", 0) or 0) < min_cluster_size:
                continue
            source_tables = entry.get("source_tables", set())
            entry["source_tables"] = sorted(source_tables) if isinstance(source_tables, set) else []
            clusters.append(entry)

        return sorted(clusters, key=lambda item: int(item.get("count", 0) or 0), reverse=True)

    def has_recent_skill_gap_proposal(self, cluster_key: str, days: int = _SKILL_GAP_RECENT_PROPOSAL_DAYS) -> bool:
        title = f"Skill gap cluster: {cluster_key}".strip().lower()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self.infra.rows(
            config.MOLLYGRAPH_PATH,
            """
            SELECT COUNT(*) AS c
            FROM self_improvement_events
            WHERE category = 'skill-gap'
              AND lower(title) = ?
              AND created_at > ?
              AND lower(coalesce(status, '')) IN ('pending', 'proposed', 'activated', 'active', 'approved')
            """,
            (title, cutoff),
        )
        if rows:
            try:
                if int(rows[0].get("c", 0) or 0) > 0:
                    return True
            except Exception:
                pass
        return self.has_recent_event(category="skill-gap", title=f"Skill gap cluster: {cluster_key}", days=days)

    # -- track-a-skill hook -------------------------------------------------

    def resolve_track_a_skill_hook(self):
        molly = self.ctx.molly
        candidates = [
            molly,
            getattr(molly, "track_a", None) if molly else None,
            getattr(molly, "track_a_hooks", None) if molly else None,
            getattr(molly, "health_remediation", None) if molly else None,
        ]
        method_names = (
            "draft_pending_skill_proposal",
            "create_pending_skill_proposal",
            "auto_draft_skill_proposal",
            "draft_skill_proposal",
            "propose_skill",
        )
        for candidate in candidates:
            if candidate is None:
                continue
            for method_name in method_names:
                hook = getattr(candidate, method_name, None)
                if callable(hook):
                    return hook
        return None

    async def invoke_track_a_skill_hook(self, cluster: dict[str, Any]) -> dict[str, Any]:
        hook = self.resolve_track_a_skill_hook()
        if hook is None:
            return {"status": "unavailable", "reason": "track_a_hook_missing"}

        cluster_key = str(cluster.get("cluster_key", "")).strip()
        cluster_size = int(cluster.get("count", 0) or 0)
        payload = {
            "cluster_key": cluster_key,
            "cluster_size": cluster_size,
            "source": "weekly_self_improve",
            "summary": f"Recurring skill gap '{cluster_key}' detected ({cluster_size} observations).",
        }

        invocations = [
            lambda: hook(cluster=payload),
            lambda: hook(**payload),
            lambda: hook(cluster_key=cluster_key, cluster_size=cluster_size, context=payload),
            lambda: hook(cluster_key, cluster_size, payload),
            lambda: hook(cluster_key, cluster_size),
            lambda: hook(payload),
        ]

        last_type_error: Exception | None = None
        for invoke in invocations:
            try:
                result = invoke()
            except TypeError as exc:
                last_type_error = exc
                continue
            except Exception as exc:
                return {"status": "error", "reason": str(exc)}

            try:
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:
                return {"status": "error", "reason": str(exc)}
            return _normalize_track_a_skill_hook_result(result)

        reason = "unsupported_track_a_hook_signature"
        if last_type_error is not None:
            reason = f"{reason}: {last_type_error}"
        return {"status": "error", "reason": reason}

    # -- row-level bookkeeping ----------------------------------------------

    def mark_skill_gap_rows_addressed(
        self,
        *,
        cluster_key: str,
        proposal_id: str,
        lifecycle_status: str,
    ) -> None:
        if not cluster_key:
            return

        db_path = config.MOLLYGRAPH_PATH
        now = datetime.now(timezone.utc).isoformat()
        cooldown_until = (datetime.now(timezone.utc) + timedelta(days=_SKILL_GAP_PROPOSAL_COOLDOWN_DAYS)).isoformat()
        status_value = "activated" if lifecycle_status == "activated" else "proposed"

        for table_name in ("skill_gaps", "skill_gap_clusters"):
            columns = self.infra.table_columns(db_path, table_name)
            if not columns:
                continue
            cluster_column = self._skill_gap_cluster_column(columns)
            if not cluster_column:
                continue

            assignments: list[tuple[str, Any]] = []
            if "status" in columns:
                assignments.append(("status", status_value))
            if "proposal_status" in columns:
                assignments.append(("proposal_status", status_value))
            for field in ("proposal_id", "active_proposal_id", "latest_proposal_id", "addressed_by_proposal"):
                if field in columns:
                    assignments.append((field, proposal_id))
            if "addressed" in columns:
                assignments.append(("addressed", 1))
            if lifecycle_status == "activated" and "resolved" in columns:
                assignments.append(("resolved", 1))
            for timestamp_field in ("updated_at", "last_proposed_at", "addressed_at"):
                if timestamp_field in columns:
                    assignments.append((timestamp_field, now))
            if lifecycle_status == "activated":
                for timestamp_field in ("activated_at", "resolved_at"):
                    if timestamp_field in columns:
                        assignments.append((timestamp_field, now))
            for cooldown_field in ("cooldown_until", "proposal_cooldown_until", "next_proposal_after"):
                if cooldown_field in columns:
                    assignments.append((cooldown_field, cooldown_until))

            if not assignments:
                continue

            try:
                conn = db_pool.sqlite_connect(str(db_path))
                try:
                    values = [value for _, value in assignments]
                    set_clause = ", ".join(f"{field} = ?" for field, _ in assignments)
                    conn.execute(
                        f"UPDATE {table_name} SET {set_clause} WHERE {cluster_column} = ?",
                        (*values, cluster_key),
                    )
                    conn.commit()
                finally:
                    conn.close()
            except Exception:
                log.debug("Failed marking skill gap rows addressed for cluster=%s", cluster_key, exc_info=True)
