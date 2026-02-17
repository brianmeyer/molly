"""Metrics and statistics service for SelfImprovementEngine.

Provides system performance, memory health, automation status,
health summaries, and self-improvement event statistics.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import config

log = logging.getLogger(__name__)


class StatsService:
    """Metrics and statistics service.

    Receives explicit dependencies on ``InfraService`` and ``GraphOpsService``.
    """

    def __init__(self, ctx, infra, graph):
        from evolution.context import EngineContext
        from evolution.infra import InfraService
        from evolution.graph_ops import GraphOpsService
        self.ctx: EngineContext = ctx
        self.infra: InfraService = infra
        self.graph: GraphOpsService = graph

    def performance_metrics(self, days: int = 7) -> dict[str, Any]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        messages = self.infra.count_rows(
            config.DATABASE_PATH,
            "SELECT COUNT(*) FROM messages WHERE timestamp > ?",
            (cutoff,),
        )
        tool_calls = self.infra.count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM tool_calls WHERE created_at > ?",
            (cutoff,),
        )
        avg_latency = self.infra.scalar(
            config.MOLLYGRAPH_PATH,
            "SELECT AVG(latency_ms) FROM tool_calls WHERE created_at > ?",
            (cutoff,),
            default=0.0,
        )
        routing_rows = self.infra.rows(
            config.MOLLYGRAPH_PATH,
            """
            SELECT tool_name, COUNT(*) AS c
            FROM tool_calls
            WHERE created_at > ? AND tool_name LIKE 'routing:subagent_start:%'
            GROUP BY tool_name
            ORDER BY c DESC
            """,
            (cutoff,),
        )
        routing_total = sum(int(r["c"]) for r in routing_rows) or 1
        distribution_parts = []
        for row in routing_rows:
            agent = str(row["tool_name"]).split(":")[-1]
            pct = int((int(row["c"]) / routing_total) * 100)
            distribution_parts.append(f"{agent} {pct}%")
        routing_distribution = ", ".join(distribution_parts) if distribution_parts else "none"

        extraction_count = self.infra.count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM tool_calls WHERE created_at > ? AND tool_name = 'extraction'",
            (cutoff,),
        )
        correction_count = self.infra.count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM corrections WHERE created_at > ?",
            (cutoff,),
        )
        extraction_quality = (
            f"extractions={extraction_count}, corrections={correction_count}, "
            f"delta={max(0, extraction_count - correction_count)}"
        )
        return {
            "messages": messages,
            "tool_calls": tool_calls,
            "avg_latency_ms": int(avg_latency or 0),
            "routing_distribution": routing_distribution,
            "extraction_quality": extraction_quality,
        }

    def memory_stats(self) -> dict[str, Any]:
        from memory.graph import entity_count, relationship_count, get_driver

        entities = entity_count()
        relationships = relationship_count()
        stale = len(self.graph.stale_entities(days=60))
        orphans = 0
        try:
            driver = get_driver()
            with driver.session() as session:
                rec = session.run("MATCH (e:Entity) WHERE NOT (e)--() RETURN count(e) AS c").single()
                orphans = int(rec["c"]) if rec else 0
        except Exception:
            pass
        return {
            "entities": entities,
            "relationships": relationships,
            "stale_entities": stale,
            "orphans": orphans,
        }

    def automation_stats(self) -> dict[str, Any]:
        path = config.AUTOMATIONS_STATE_FILE
        if not path.exists():
            return {"loaded": 0, "last_run": "-", "failed": 0}
        try:
            data = json.loads(path.read_text())
            automations = data.get("automations", {})
            last_runs = [str(row.get("last_run", "")) for row in automations.values() if row.get("last_run")]
            failed = sum(1 for row in automations.values() if str(row.get("last_status", "")).lower() == "failed")
            return {
                "loaded": len(automations),
                "last_run": max(last_runs) if last_runs else "-",
                "failed": failed,
            }
        except Exception:
            return {"loaded": 0, "last_run": "-", "failed": 0}

    def latest_health_summary(self) -> str:
        from monitoring import get_health_doctor

        doctor = get_health_doctor(self.ctx.molly)
        path = doctor.latest_report_path()
        if not path:
            return "no recent health report"
        payload = {}
        try:
            payload = json.loads(
                doctor.latest_report_text().split("<!-- HEALTH_DATA:", 1)[1].split("-->", 1)[0].strip()
            )
        except Exception:
            payload = {}
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        if summary:
            return f"{summary.get('green', 0)} green / {summary.get('yellow', 0)} yellow / {summary.get('red', 0)} red"
        return f"latest={path.stem}"

    def self_improvement_stats(self, days: int = 30) -> dict[str, int]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self.infra.rows(
            config.MOLLYGRAPH_PATH,
            """
            SELECT status, COUNT(*) AS c
            FROM self_improvement_events
            WHERE created_at > ?
            GROUP BY status
            """,
            (cutoff,),
        )
        stats = {"proposed": 0, "approved": 0, "rejected": 0}
        for row in rows:
            status = str(row["status"]).lower()
            count = int(row["c"])
            if status in stats:
                stats[status] += count
            if status == "deployed":
                stats["approved"] += count
        return stats
