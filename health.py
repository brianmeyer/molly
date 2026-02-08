import json
import logging
import os
import re
import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import config

log = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    check_id: str
    layer: str
    label: str
    status: str  # green | yellow | red
    detail: str
    action_required: bool = False
    watch_item: bool = False


_STATUS_EMOJI = {
    "green": "🟢",
    "yellow": "🟡",
    "red": "🔴",
}
_STATUS_SEVERITY = {
    "green": 0,
    "yellow": 1,
    "red": 2,
}


def _status_emoji(status: str) -> str:
    return _STATUS_EMOJI.get(status, "⚪")


def _now_local() -> datetime:
    return datetime.now(ZoneInfo(config.TIMEZONE))


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _short_ts(value: str) -> str:
    dt = _parse_iso(value)
    if not dt:
        return value or "-"
    return dt.astimezone(ZoneInfo(config.TIMEZONE)).strftime("%Y-%m-%d %H:%M")


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        curr = [i]
        for j, ch_b in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ch_a != ch_b)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def _load_embedded_report_data(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {}
    text = report_path.read_text()
    marker = "<!-- HEALTH_DATA:"
    start = text.rfind(marker)
    if start < 0:
        return {}
    end = text.find("-->", start)
    if end < 0:
        return {}
    raw = text[start + len(marker):end].strip()
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


class HealthDoctor:
    def __init__(self, molly=None):
        self.molly = molly
        self.report_dir = config.HEALTH_REPORT_DIR
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def latest_report_path(self) -> Path | None:
        reports = sorted(self.report_dir.glob("????-??-??.md"))
        return reports[-1] if reports else None

    def latest_report_text(self) -> str | None:
        path = self.latest_report_path()
        if not path:
            return None
        return path.read_text()

    def get_or_generate_latest_report(self, fresh: bool = False) -> str:
        today_path = self.report_dir / f"{date.today().isoformat()}.md"
        if fresh or not today_path.exists():
            return self.generate_report(abbreviated=False, trigger="manual")
        payload = _load_embedded_report_data(today_path)
        if (
            self.molly is not None
            and bool(payload.get("abbreviated", False))
            and str(payload.get("trigger", "")).strip().lower() == "startup"
        ):
            # Replace startup preflight snapshot with a full runtime probe.
            return self.generate_report(abbreviated=False, trigger="manual")
        return today_path.read_text()

    def history_markdown(self, days: int = 7) -> str:
        paths = sorted(self.report_dir.glob("????-??-??.md"))[-days:]
        if not paths:
            return "No health reports found."

        lines = [f"Health History ({len(paths)} days)", ""]
        for p in paths:
            payload = _load_embedded_report_data(p)
            summary = payload.get("summary", {})
            green = int(summary.get("green", 0))
            yellow = int(summary.get("yellow", 0))
            red = int(summary.get("red", 0))
            lines.append(
                f"- {p.stem}: {_status_emoji('green')} {green} / "
                f"{_status_emoji('yellow')} {yellow} / {_status_emoji('red')} {red}"
            )
        return "\n".join(lines)

    def run_daily(self) -> str:
        return self.generate_report(abbreviated=False, trigger="daily")

    def run_abbreviated_preflight(self) -> str:
        return self.generate_report(abbreviated=True, trigger="startup")

    def generate_report(self, abbreviated: bool = False, trigger: str = "manual") -> str:
        checks = self._run_checks(abbreviated=abbreviated)
        checks = self._apply_yellow_escalation(checks)

        summary = {
            "green": sum(1 for c in checks if c.status == "green"),
            "yellow": sum(1 for c in checks if c.status == "yellow"),
            "red": sum(1 for c in checks if c.status == "red"),
        }

        action_required = [c for c in checks if c.status == "red" or c.action_required]
        watch_items = [c for c in checks if c.status == "yellow" or c.watch_item]
        generated_at = _now_local().strftime("%Y-%m-%d %H:%M %Z")

        sections: dict[str, list[HealthCheck]] = {}
        for check in checks:
            sections.setdefault(check.layer, []).append(check)

        lines = [f"# 🩺 Molly Health Report — {generated_at}", ""]

        ordered_layers = [
            "Component Heartbeats",
            "Pipeline Validation",
            "Data Quality",
            "Automation Health",
            "Learning Loop",
        ]
        for layer_name in ordered_layers:
            if layer_name not in sections:
                continue
            suffix = " (abbreviated)" if abbreviated and layer_name != "Component Heartbeats" else ""
            lines.append(f"## {layer_name}{suffix}")
            for check in sections[layer_name]:
                lines.append(f"{_status_emoji(check.status)} {check.label}: {check.detail}")
            lines.append("")

        lines.append(
            f"## Summary: {summary['green']} {_status_emoji('green')} / "
            f"{summary['yellow']} {_status_emoji('yellow')} / {summary['red']} {_status_emoji('red')}"
        )
        lines.append("")

        lines.append("### Action Required")
        if action_required:
            for check in action_required:
                lines.append(f"- {_status_emoji('red')} {check.label}: {check.detail}")
        else:
            lines.append("- None")
        lines.append("")

        lines.append("### Watch Items")
        if watch_items:
            for check in watch_items:
                lines.append(f"- {_status_emoji('yellow')} {check.label}: {check.detail}")
        else:
            lines.append("- None")
        lines.append("")

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "trigger": trigger,
            "abbreviated": abbreviated,
            "summary": summary,
            "checks": [
                {
                    "id": c.check_id,
                    "layer": c.layer,
                    "status": c.status,
                    "label": c.label,
                    "detail": c.detail,
                }
                for c in checks
            ],
        }
        lines.append(f"<!-- HEALTH_DATA: {json.dumps(payload, ensure_ascii=True)} -->")

        markdown = "\n".join(lines).rstrip() + "\n"
        report_path = self.report_dir / f"{date.today().isoformat()}.md"
        report_path.write_text(markdown)
        self._prune_old_reports()
        return markdown

    def compare_green_to_red_regressions(
        self,
        baseline_report: str,
        candidate_report: str,
    ) -> list[str]:
        regressions = self.compare_worsened_components(
            baseline_report=baseline_report,
            candidate_report=candidate_report,
        )
        return [row["id"] for row in regressions if row["before"] == "green" and row["after"] == "red"]

    def extract_status_map(self, report_text: str) -> dict[str, str]:
        return self._extract_status_map_from_text(report_text)

    def compare_worsened_components(
        self,
        baseline_report: str,
        candidate_report: str,
    ) -> list[dict[str, str]]:
        baseline_checks = self.extract_status_map(baseline_report)
        candidate_checks = self.extract_status_map(candidate_report)
        regressions: list[dict[str, str]] = []
        for check_id, before in baseline_checks.items():
            after = candidate_checks.get(check_id)
            if before not in _STATUS_SEVERITY or after not in _STATUS_SEVERITY:
                continue
            if _STATUS_SEVERITY[after] > _STATUS_SEVERITY[before]:
                regressions.append({"id": check_id, "before": before, "after": after})
        return regressions

    def _extract_status_map_from_text(self, report_text: str) -> dict[str, str]:
        marker = "<!-- HEALTH_DATA:"
        idx = report_text.rfind(marker)
        if idx < 0:
            return {}
        end = report_text.find("-->", idx)
        if end < 0:
            return {}
        raw = report_text[idx + len(marker):end].strip()
        try:
            payload = json.loads(raw)
            rows = payload.get("checks", [])
            return {
                str(row.get("id", "")): str(row.get("status", ""))
                for row in rows
                if row.get("id")
            }
        except Exception:
            return {}

    def _prune_old_reports(self):
        cutoff = date.today() - timedelta(days=max(1, config.HEALTH_REPORT_RETENTION_DAYS))
        for path in self.report_dir.glob("????-??-??.md"):
            try:
                d = date.fromisoformat(path.stem)
            except ValueError:
                continue
            if d < cutoff:
                path.unlink(missing_ok=True)

    def _apply_yellow_escalation(self, checks: list[HealthCheck]) -> list[HealthCheck]:
        threshold = max(2, config.HEALTH_YELLOW_ESCALATION_DAYS)
        updated: list[HealthCheck] = []
        for check in checks:
            if check.status != "yellow":
                updated.append(check)
                continue
            prev_streak = self._previous_yellow_streak(check.check_id)
            if prev_streak + 1 >= threshold:
                updated.append(
                    HealthCheck(
                        check_id=check.check_id,
                        layer=check.layer,
                        label=check.label,
                        status="red",
                        detail=f"{check.detail} (yellow persisted {prev_streak + 1} days)",
                        action_required=True,
                        watch_item=check.watch_item,
                    )
                )
            else:
                updated.append(check)
        return updated

    def _previous_yellow_streak(self, check_id: str) -> int:
        paths = sorted(self.report_dir.glob("????-??-??.md"))
        if not paths:
            return 0
        today = date.today().isoformat()
        streak = 0
        for path in reversed(paths):
            if path.stem == today:
                continue
            payload = _load_embedded_report_data(path)
            rows = payload.get("checks", [])
            if not isinstance(rows, list):
                continue
            status = ""
            for row in rows:
                if str(row.get("id", "")) == check_id:
                    status = str(row.get("status", ""))
                    break
            if status == "yellow":
                streak += 1
                continue
            break
        return streak

    def _run_checks(self, abbreviated: bool) -> list[HealthCheck]:
        checks: list[HealthCheck] = []
        checks.extend(self._layer_component_heartbeats())
        if abbreviated:
            return checks
        checks.extend(self._layer_pipeline_validation())
        checks.extend(self._layer_data_quality())
        checks.extend(self._layer_automation_health())
        checks.extend(self._layer_learning_loop())
        return checks

    def _layer_component_heartbeats(self) -> list[HealthCheck]:
        checks: list[HealthCheck] = []

        wa_status, wa_detail = self._whatsapp_status()
        checks.append(
            HealthCheck(
                check_id="component.whatsapp",
                layer="Component Heartbeats",
                label="WhatsApp",
                status=wa_status,
                detail=wa_detail,
                action_required=(wa_status == "red"),
            )
        )

        neo_status, neo_detail = self._neo4j_heartbeat()
        checks.append(
            HealthCheck(
                check_id="component.neo4j",
                layer="Component Heartbeats",
                label="Neo4j",
                status=neo_status,
                detail=neo_detail,
                action_required=(neo_status == "red"),
            )
        )

        triage_loaded = self._module_attr_loaded("memory.triage", "_TRIAGE_MODEL")
        checks.append(
            HealthCheck(
                check_id="component.triage_model",
                layer="Component Heartbeats",
                label="Triage model",
                status="green" if triage_loaded else "red",
                detail="loaded" if triage_loaded else "not loaded",
                action_required=not triage_loaded,
            )
        )

        embedding_status, embedding_detail = self._embedding_model_status()
        checks.append(
            HealthCheck(
                check_id="component.embedding_model",
                layer="Component Heartbeats",
                label="EmbeddingGemma",
                status=embedding_status,
                detail=embedding_detail,
                action_required=(embedding_status == "red"),
            )
        )

        extractor_status, extractor_detail = self._gliner_model_status()
        checks.append(
            HealthCheck(
                check_id="component.gliner_model",
                layer="Component Heartbeats",
                label="GLiNER2",
                status=extractor_status,
                detail=extractor_detail,
                action_required=(extractor_status == "red"),
            )
        )

        oauth_status, oauth_detail = self._google_oauth_status()
        checks.append(
            HealthCheck(
                check_id="component.google_oauth",
                layer="Component Heartbeats",
                label="Google OAuth",
                status=oauth_status,
                detail=oauth_detail,
                action_required=(oauth_status == "red"),
            )
        )

        mcp_status, mcp_detail = self._mcp_servers_status()
        checks.append(
            HealthCheck(
                check_id="component.mcp_servers",
                layer="Component Heartbeats",
                label="MCP tools",
                status=mcp_status,
                detail=mcp_detail,
                action_required=(mcp_status == "red"),
            )
        )

        auto_status, auto_detail = self._automation_engine_status()
        checks.append(
            HealthCheck(
                check_id="component.automation_engine",
                layer="Component Heartbeats",
                label="Automation engine",
                status=auto_status,
                detail=auto_detail,
                action_required=(auto_status == "red"),
            )
        )

        disk_status, disk_detail = self._disk_status()
        checks.append(
            HealthCheck(
                check_id="component.disk_space",
                layer="Component Heartbeats",
                label="Disk",
                status=disk_status,
                detail=disk_detail,
                action_required=(disk_status == "red"),
            )
        )

        ram_status, ram_detail = self._ram_status()
        checks.append(
            HealthCheck(
                check_id="component.ram_usage",
                layer="Component Heartbeats",
                label="RAM",
                status=ram_status,
                detail=ram_detail,
                action_required=(ram_status == "red"),
            )
        )

        return checks

    def _layer_pipeline_validation(self) -> list[HealthCheck]:
        checks: list[HealthCheck] = []
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=config.HEALTH_PIPELINE_WINDOW_HOURS)).isoformat()

        msg_count, emb_count = self._message_embedding_counts(cutoff)
        ratio = (emb_count / msg_count) if msg_count > 0 else 1.0
        if msg_count == 0:
            status = "yellow"
            detail = "No new messages in window"
        elif 0.95 <= ratio <= 1.05:
            status = "green"
            detail = f"Messages {msg_count} → embeddings {emb_count} ({ratio:.0%})"
        elif 0.85 <= ratio <= 1.15:
            status = "yellow"
            detail = f"Messages {msg_count} → embeddings {emb_count} ({ratio:.0%})"
        else:
            status = "red"
            detail = f"Messages {msg_count} → embeddings {emb_count} ({ratio:.0%})"
        checks.append(
            HealthCheck(
                check_id="pipeline.message_to_embedding",
                layer="Pipeline Validation",
                label="Message → Embedding",
                status=status,
                detail=detail,
                action_required=(status == "red"),
            )
        )

        entity_count = self._recent_entity_count(cutoff)
        if msg_count > 5 and entity_count == 0:
            status = "red"
        elif entity_count == 0:
            status = "yellow"
        else:
            status = "green"
        checks.append(
            HealthCheck(
                check_id="pipeline.embedding_to_entity",
                layer="Pipeline Validation",
                label="Embedding → Entity extraction",
                status=status,
                detail=f"Recent entities: {entity_count} (messages: {msg_count})",
                action_required=(status == "red"),
            )
        )

        rel_status, rel_detail = self._entity_relationship_sampling()
        checks.append(
            HealthCheck(
                check_id="pipeline.entity_to_relationship",
                layer="Pipeline Validation",
                label="Entity → Relationship sampling",
                status=rel_status,
                detail=rel_detail,
                watch_item=(rel_status == "yellow"),
            )
        )

        source_status, source_detail = self._source_distribution(cutoff)
        checks.append(
            HealthCheck(
                check_id="pipeline.source_distribution",
                layer="Pipeline Validation",
                label="Source distribution",
                status=source_status,
                detail=source_detail,
                action_required=(source_status == "red"),
            )
        )

        ts_status, ts_detail = self._timestamp_format_check()
        checks.append(
            HealthCheck(
                check_id="pipeline.timestamp_format",
                layer="Pipeline Validation",
                label="Timestamp format",
                status=ts_status,
                detail=ts_detail,
                action_required=(ts_status == "red"),
            )
        )

        from approval import get_action_tier

        tier = get_action_tier("Bash")
        tier_status = "green" if tier == "CONFIRM" else "red"
        checks.append(
            HealthCheck(
                check_id="pipeline.approval_dry_run",
                layer="Pipeline Validation",
                label="Approval system dry-run",
                status=tier_status,
                detail=f"Bash tier = {tier}",
                action_required=(tier_status == "red"),
            )
        )

        task_calls = self._tool_call_count("routing:subagent_start:%", cutoff)
        task_status = "green"
        if msg_count > 0 and task_calls == 0:
            task_status = "red"
        checks.append(
            HealthCheck(
                check_id="pipeline.subagent_task_calls",
                layer="Pipeline Validation",
                label="Sub-agent Task calls",
                status=task_status,
                detail=f"Logged Task calls: {task_calls}",
                action_required=(task_status == "red"),
            )
        )

        pref_status, pref_detail = self._preference_table_writable()
        checks.append(
            HealthCheck(
                check_id="pipeline.preference_signals_writable",
                layer="Pipeline Validation",
                label="Preference signals table",
                status=pref_status,
                detail=pref_detail,
                action_required=(pref_status == "red"),
            )
        )

        return checks

    def _layer_data_quality(self) -> list[HealthCheck]:
        checks: list[HealthCheck] = []

        strength_status, strength_detail = self._entity_strength_decay_check()
        checks.append(
            HealthCheck(
                check_id="quality.entity_strength_decay",
                layer="Data Quality",
                label="Entity strength decay",
                status=strength_status,
                detail=strength_detail,
                watch_item=(strength_status == "yellow"),
                action_required=(strength_status == "red"),
            )
        )

        memory_path = config.WORKSPACE / "MEMORY.md"
        if not memory_path.exists():
            checks.append(
                HealthCheck(
                    check_id="quality.memory_file",
                    layer="Data Quality",
                    label="MEMORY.md",
                    status="red",
                    detail="file missing",
                    action_required=True,
                )
            )
        else:
            age_days = (datetime.now() - datetime.fromtimestamp(memory_path.stat().st_mtime)).days
            freshness = "green" if age_days <= 7 else ("yellow" if age_days <= 14 else "red")
            checks.append(
                HealthCheck(
                    check_id="quality.memory_freshness",
                    layer="Data Quality",
                    label="MEMORY.md freshness",
                    status=freshness,
                    detail=f"Last modified {age_days} day(s) ago",
                    watch_item=(freshness == "yellow"),
                    action_required=(freshness == "red"),
                )
            )
            size = memory_path.stat().st_size
            content_status = "green" if size >= 100 else "red"
            checks.append(
                HealthCheck(
                    check_id="quality.memory_content",
                    layer="Data Quality",
                    label="MEMORY.md content",
                    status=content_status,
                    detail=f"{size} bytes",
                    action_required=(content_status == "red"),
                )
            )

        orphan_status, orphan_detail = self._orphan_trend_check()
        checks.append(
            HealthCheck(
                check_id="quality.orphan_trend",
                layer="Data Quality",
                label="Orphaned entities",
                status=orphan_status,
                detail=orphan_detail,
                watch_item=(orphan_status == "yellow"),
            )
        )

        dup_status, dup_detail = self._duplicate_entity_check()
        checks.append(
            HealthCheck(
                check_id="quality.duplicates",
                layer="Data Quality",
                label="Duplicate entity fuzzy match",
                status=dup_status,
                detail=dup_detail,
                watch_item=(dup_status == "yellow"),
                action_required=(dup_status == "red"),
            )
        )

        chunk_status, chunk_detail = self._chunk_retention_check()
        checks.append(
            HealthCheck(
                check_id="quality.chunk_retention",
                layer="Data Quality",
                label="Conversation chunk retention",
                status=chunk_status,
                detail=chunk_detail,
                watch_item=(chunk_status == "yellow"),
                action_required=(chunk_status == "red"),
            )
        )

        maintenance_status, maintenance_detail = self._maintenance_log_check()
        checks.append(
            HealthCheck(
                check_id="quality.maintenance_log",
                layer="Data Quality",
                label="Maintenance log",
                status=maintenance_status,
                detail=maintenance_detail,
                action_required=(maintenance_status == "red"),
            )
        )

        op_status, op_detail = self._operational_tables_check()
        checks.append(
            HealthCheck(
                check_id="quality.operational_tables",
                layer="Data Quality",
                label="Operational tables",
                status=op_status,
                detail=op_detail,
                watch_item=(op_status == "yellow"),
            )
        )

        return checks

    def _layer_automation_health(self) -> list[HealthCheck]:
        checks: list[HealthCheck] = []
        state_ok, state_data, state_detail = self._load_automation_state()
        checks.append(
            HealthCheck(
                check_id="automation.state_integrity",
                layer="Automation Health",
                label="state.json integrity",
                status="green" if state_ok else "red",
                detail=state_detail,
                action_required=not state_ok,
            )
        )
        if not state_ok:
            return checks

        automations = state_data.get("automations", {})
        if not automations:
            checks.append(
                HealthCheck(
                    check_id="automation.loaded",
                    layer="Automation Health",
                    label="Automation status",
                    status="yellow",
                    detail="No automation state entries found",
                    watch_item=True,
                )
            )
            return checks

        failed = [
            aid for aid, row in automations.items()
            if str(row.get("last_status", "")).lower() == "failed"
        ]
        status = "green"
        if failed:
            status = "yellow" if len(failed) <= 2 else "red"
        checks.append(
            HealthCheck(
                check_id="automation.failures",
                layer="Automation Health",
                label="Per-automation last_result",
                status=status,
                detail=f"failed={len(failed)} / total={len(automations)}",
                watch_item=(status == "yellow"),
                action_required=(status == "red"),
            )
        )

        now_local = _now_local()
        weekday = now_local.weekday() < 5
        morning_row = self._find_automation_row(automations, ["morning", "digest", "brief"])
        if weekday:
            morning_ok = bool(morning_row and self._is_same_day(morning_row.get("last_run", ""), now_local))
            checks.append(
                HealthCheck(
                    check_id="automation.morning_briefing",
                    layer="Automation Health",
                    label="Morning briefing",
                    status="green" if morning_ok else "red",
                    detail="fired today" if morning_ok else "missed weekday run",
                    action_required=not morning_ok,
                )
            )

        email_row = self._find_automation_row(automations, ["email", "triage"])
        checks.append(
            HealthCheck(
                check_id="automation.email_triage",
                layer="Automation Health",
                label="Email triage staleness",
                status=self._staleness_status(email_row, max_minutes=30),
                detail=self._staleness_detail(email_row),
                watch_item=True,
            )
        )

        prep_row = self._find_automation_row(automations, ["meeting", "prep"])
        checks.append(
            HealthCheck(
                check_id="automation.meeting_prep",
                layer="Automation Health",
                label="Meeting prep coverage",
                status=self._staleness_status(prep_row, max_minutes=24 * 60),
                detail=self._staleness_detail(prep_row),
                watch_item=True,
            )
        )

        followups = config.WORKSPACE / "memory" / "followups.md"
        if not followups.exists():
            status = "yellow"
            detail = "followups.md missing"
        else:
            age_days = (datetime.now() - datetime.fromtimestamp(followups.stat().st_mtime)).days
            status = "green" if age_days <= 3 else ("yellow" if age_days <= 7 else "red")
            detail = f"updated {age_days} day(s) ago"
        checks.append(
            HealthCheck(
                check_id="automation.commitment_tracker",
                layer="Automation Health",
                label="Commitment tracker freshness",
                status=status,
                detail=detail,
                watch_item=(status == "yellow"),
                action_required=(status == "red"),
            )
        )
        return checks

    def _layer_learning_loop(self) -> list[HealthCheck]:
        checks: list[HealthCheck] = []
        cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        pref_count = self._count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM preference_signals WHERE created_at > ?",
            (cutoff_7d,),
        )
        pref_status = "green" if pref_count > 0 else "yellow"
        checks.append(
            HealthCheck(
                check_id="learning.preference_signals",
                layer="Learning Loop",
                label="Preference signals accumulating",
                status=pref_status,
                detail=f"{pref_count} in last 7 days",
                watch_item=(pref_status == "yellow"),
            )
        )

        maintenance_status, maintenance_detail = self._maintenance_log_check()
        checks.append(
            HealthCheck(
                check_id="learning.maintenance_completion",
                layer="Learning Loop",
                label="Nightly maintenance completion",
                status=maintenance_status,
                detail=maintenance_detail,
                action_required=(maintenance_status == "red"),
            )
        )

        action_status, action_detail = self._maintenance_action_check()
        checks.append(
            HealthCheck(
                check_id="learning.maintenance_actions",
                layer="Learning Loop",
                label="Maintenance actions taken",
                status=action_status,
                detail=action_detail,
                watch_item=(action_status == "yellow"),
                action_required=(action_status == "red"),
            )
        )

        cutoff_30d = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        proposals = self._count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM self_improvement_events WHERE created_at > ?",
            (cutoff_30d,),
        )
        status = "green" if proposals > 0 else "yellow"
        checks.append(
            HealthCheck(
                check_id="learning.self_improvement_proposals",
                layer="Learning Loop",
                label="Self-improvement proposals",
                status=status,
                detail=f"{proposals} in last 30 days",
                watch_item=(status == "yellow"),
            )
        )

        weekly_status, weekly_detail = self._weekly_assessment_check()
        checks.append(
            HealthCheck(
                check_id="learning.weekly_assessment",
                layer="Learning Loop",
                label="Weekly assessment",
                status=weekly_status,
                detail=weekly_detail,
                action_required=(weekly_status == "red"),
            )
        )

        reject_status, reject_detail = self._rejected_resubmission_check()
        checks.append(
            HealthCheck(
                check_id="learning.rejected_resubmission",
                layer="Learning Loop",
                label="Rejected proposal resubmission",
                status=reject_status,
                detail=reject_detail,
                action_required=(reject_status == "red"),
            )
        )
        return checks

    def _module_attr_loaded(self, module_name: str, attr_name: str) -> bool:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            return getattr(module, attr_name, None) is not None
        except Exception:
            return False

    def _whatsapp_status(self) -> tuple[str, str]:
        wa = getattr(self.molly, "wa", None) if self.molly else None
        if wa is None:
            return "red", "client not initialized"

        connected = bool(getattr(wa, "connected", False))
        bridge_client = getattr(wa, "client", None)
        bridge_identity = getattr(bridge_client, "me", None) if bridge_client else None
        if connected:
            return "green", "connected"
        if bridge_identity:
            return "green", "authenticated (bridge)"
        return "red", "disconnected"

    def _embedding_model_status(self) -> tuple[str, str]:
        try:
            module = __import__("memory.embeddings", fromlist=["_model"])
            model = getattr(module, "_model", None)
            if model is None:
                return "red", "not loaded"

            try:
                vec = model.encode("health", normalize_embeddings=True)
            except TypeError:
                vec = model.encode("health")

            dim = 0
            shape = getattr(vec, "shape", None)
            if shape and len(shape) > 0:
                dim = int(shape[-1])
            elif isinstance(vec, (list, tuple)):
                if vec and isinstance(vec[0], (list, tuple)):
                    dim = len(vec[0])
                else:
                    dim = len(vec)

            if dim <= 0:
                return "red", "loaded, encode sanity failed"
            return "green", f"loaded ({dim}d sanity ok)"
        except Exception as exc:
            return "red", f"probe failed ({exc})"

    def _gliner_model_status(self) -> tuple[str, str]:
        try:
            module = __import__("memory.extractor", fromlist=["_model"])
            model = getattr(module, "_model", None)
            if model is None:
                return "red", "not loaded"
            if not callable(getattr(model, "extract", None)):
                return "red", "loaded, extract unavailable"
            return "green", "loaded"
        except Exception as exc:
            return "red", f"probe failed ({exc})"

    def _neo4j_heartbeat(self) -> tuple[str, str]:
        try:
            from memory.graph import get_driver

            t0 = time.monotonic()
            driver = get_driver()
            with driver.session() as session:
                session.run("RETURN 1").single()
            ms = int((time.monotonic() - t0) * 1000)
            if ms < 100:
                return "green", f"responding ({ms}ms)"
            if ms <= 500:
                return "yellow", f"slow ({ms}ms)"
            return "red", f"very slow ({ms}ms)"
        except Exception as exc:
            return "red", f"unreachable ({exc})"

    def _google_oauth_status(self) -> tuple[str, str]:
        try:
            from tools.google_auth import get_credentials

            creds = get_credentials()
            if not creds or not creds.valid:
                return "red", "invalid or expired"
            expiry = getattr(creds, "expiry", None)
            if expiry is None:
                return "green", "valid"
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            mins = int((expiry - datetime.now(timezone.utc)).total_seconds() // 60)
            if mins < 0:
                return "red", "expired"
            if mins < 60:
                return "yellow", f"valid (<1h remaining, {mins}m)"
            return "green", f"valid ({mins}m remaining)"
        except Exception as exc:
            return "red", f"refresh failed ({exc})"

    def _mcp_servers_status(self) -> tuple[str, str]:
        try:
            from agent import _MCP_SERVER_SPECS, _MCP_SERVER_TOOL_NAMES

            failures: list[str] = []
            total_servers = 0
            total_tools = 0
            healthy_tools = 0
            for server_name, spec in _MCP_SERVER_SPECS.items():
                total_servers += 1
                tool_count = max(1, len(_MCP_SERVER_TOOL_NAMES.get(server_name, set())))
                total_tools += tool_count
                if server_name in getattr(config, "DISABLED_MCP_SERVERS", set()):
                    failures.append(server_name)
                    continue
                try:
                    if isinstance(spec, tuple):
                        module_name, attr_name = spec
                        module = __import__(module_name, fromlist=[attr_name])
                        getattr(module, attr_name)
                    elif isinstance(spec, dict):
                        command = str(spec.get("command", "")).strip()
                        if not command:
                            raise RuntimeError("missing command")
                        if shutil.which(command) is None:
                            raise RuntimeError(f"command not found: {command}")
                    else:
                        raise RuntimeError(f"unsupported MCP spec: {type(spec)!r}")
                    healthy_tools += tool_count
                except Exception:
                    failures.append(server_name)
            if not failures:
                return (
                    "green",
                    f"{healthy_tools}/{total_tools} tools responding "
                    f"({total_servers}/{total_servers} servers)",
                )
            if len(failures) <= 2:
                return (
                    "yellow",
                    f"{healthy_tools}/{total_tools} tools responding; "
                    f"down={', '.join(failures)}",
                )
            return (
                "red",
                f"{healthy_tools}/{total_tools} tools responding; "
                f"down={', '.join(failures)}",
            )
        except Exception as exc:
            return "red", f"probe failed ({exc})"

    def _automation_engine_status(self) -> tuple[str, str]:
        engine = getattr(self.molly, "automations", None) if self.molly else None
        if not engine:
            return "red", "not initialized"
        loaded = len(getattr(engine, "_automations", {}) or {})
        initialized = bool(getattr(engine, "_initialized", False))
        if initialized and loaded > 0:
            return "green", f"initialized ({loaded} loaded)"
        if initialized:
            return "yellow", "initialized (0 loaded)"
        return "red", "not initialized"

    def _disk_status(self) -> tuple[str, str]:
        usage = shutil.disk_usage(str(config.WORKSPACE))
        free_gb = usage.free / (1024**3)
        if free_gb > 5:
            status = "green"
        elif free_gb > 1:
            status = "yellow"
        else:
            status = "red"
        return status, f"{free_gb:.1f}GB free"

    def _ram_status(self) -> tuple[str, str]:
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Darwin reports bytes, Linux KB
            rss_bytes = rss if os.uname().sysname == "Darwin" else rss * 1024
            rss_gb = rss_bytes / (1024**3)
            if rss_gb < 10:
                return "green", f"{rss_gb:.1f}GB used"
            if rss_gb < 14:
                return "yellow", f"{rss_gb:.1f}GB used"
            return "red", f"{rss_gb:.1f}GB used"
        except Exception as exc:
            return "yellow", f"unavailable ({exc})"

    def _message_embedding_counts(self, cutoff: str) -> tuple[int, int]:
        msg_count = self._count_rows(
            config.DATABASE_PATH,
            "SELECT COUNT(*) FROM messages WHERE timestamp > ?",
            (cutoff,),
        )
        emb_count = self._count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM conversation_chunks WHERE created_at > ?",
            (cutoff,),
        )
        return msg_count, emb_count

    def _recent_entity_count(self, cutoff: str) -> int:
        try:
            from memory.graph import get_driver

            driver = get_driver()
            with driver.session() as session:
                record = session.run(
                    "MATCH (e:Entity) WHERE e.last_mentioned > $cutoff RETURN count(e) AS c",
                    cutoff=cutoff,
                ).single()
            return int(record["c"]) if record else 0
        except Exception:
            return 0

    def _entity_relationship_sampling(self) -> tuple[str, str]:
        try:
            from memory.graph import get_driver

            driver = get_driver()
            with driver.session() as session:
                rows = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.last_mentioned IS NOT NULL
                    RETURN e.name AS name
                    ORDER BY e.last_mentioned DESC
                    LIMIT 5
                    """
                )
                entities = [r["name"] for r in rows]
                if not entities:
                    return "yellow", "No recent entities sampled"
                has_rel = 0
                for name in entities:
                    rec = session.run(
                        "MATCH (e:Entity {name: $name}) OPTIONAL MATCH (e)-[r]-() RETURN count(r) AS c",
                        name=name,
                    ).single()
                    if rec and int(rec["c"]) > 0:
                        has_rel += 1
                ratio = has_rel / len(entities)
                if ratio >= 0.5:
                    status = "green"
                elif ratio >= 0.3:
                    status = "yellow"
                else:
                    status = "red"
                return status, f"{has_rel}/{len(entities)} sampled entities have relationships"
        except Exception as exc:
            return "red", f"sampling failed ({exc})"

    def _source_distribution(self, cutoff: str) -> tuple[str, str]:
        try:
            conn = sqlite3.connect(str(config.MOLLYGRAPH_PATH))
            cur = conn.execute(
                """
                SELECT source, COUNT(*) AS c
                FROM conversation_chunks
                WHERE created_at > ?
                GROUP BY source
                ORDER BY c DESC
                """,
                (cutoff,),
            )
            rows = cur.fetchall()
            conn.close()
            if not rows:
                return "yellow", "No recent chunks"
            dist = {str(row[0] or "unknown"): int(row[1]) for row in rows}
            if "unknown" in dist:
                return "red", ", ".join(f"{k}={v}" for k, v in dist.items())
            if "whatsapp" not in dist:
                return "yellow", ", ".join(f"{k}={v}" for k, v in dist.items())
            return "green", ", ".join(f"{k}={v}" for k, v in dist.items())
        except Exception as exc:
            return "red", f"distribution query failed ({exc})"

    def _timestamp_format_check(self) -> tuple[str, str]:
        try:
            conn = sqlite3.connect(str(config.DATABASE_PATH))
            cur = conn.execute(
                "SELECT timestamp FROM messages ORDER BY timestamp DESC LIMIT 5"
            )
            rows = [str(r[0]) for r in cur.fetchall()]
            conn.close()
            if not rows:
                return "yellow", "No messages sampled"
            bad = 0
            for ts in rows:
                if _parse_iso(ts) is None:
                    bad += 1
            if bad == 0:
                return "green", "All sampled timestamps ISO-compatible"
            return "red", f"{bad}/{len(rows)} sampled timestamps invalid"
        except Exception as exc:
            return "red", f"timestamp check failed ({exc})"

    def _tool_call_count(self, pattern: str, cutoff: str) -> int:
        return self._count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM tool_calls WHERE tool_name LIKE ? AND created_at > ?",
            (pattern, cutoff),
        )

    def _preference_table_writable(self) -> tuple[str, str]:
        conn = sqlite3.connect(str(config.MOLLYGRAPH_PATH))
        try:
            conn.execute("BEGIN")
            signal_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO preference_signals
                (id, signal_type, source, surfaced_summary, sender_pattern, owner_feedback, context, timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal_id, "health_check", "health", "health probe",
                    "health:probe", "health check", "{}", now, now,
                ),
            )
            conn.execute("DELETE FROM preference_signals WHERE id = ?", (signal_id,))
            conn.rollback()
            return "green", "Writable"
        except Exception as exc:
            return "red", f"Write probe failed ({exc})"
        finally:
            conn.close()

    def _entity_strength_decay_check(self) -> tuple[str, str]:
        try:
            from memory.graph import get_driver

            sample_size = max(1, int(config.HEALTH_ENTITY_SAMPLE_SIZE))
            driver = get_driver()
            with driver.session() as session:
                rows = session.run(
                    """
                    MATCH (e:Entity)
                    RETURN e.strength AS strength
                    ORDER BY e.last_mentioned DESC
                    LIMIT $n
                    """,
                    n=sample_size,
                )
                values = [float(r["strength"]) for r in rows if r["strength"] is not None]
            if not values:
                return "yellow", "No entities sampled"
            stuck = sum(1 for v in values if abs(v - 1.0) < 1e-6)
            if stuck == len(values):
                return "red", f"{stuck}/{len(values)} stuck at 1.0"
            if stuck > len(values) * 0.4:
                return "yellow", f"{stuck}/{len(values)} still at 1.0"
            return "green", f"{stuck}/{len(values)} at 1.0"
        except Exception as exc:
            return "red", f"sample failed ({exc})"

    def _orphan_trend_check(self) -> tuple[str, str]:
        try:
            from memory.graph import get_driver

            driver = get_driver()
            with driver.session() as session:
                total = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
                orphans = session.run(
                    "MATCH (e:Entity) WHERE NOT (e)--() RETURN count(e) AS c"
                ).single()["c"]
            ratio = (orphans / total) if total else 0.0

            prev_ratio = None
            latest = self.latest_report_path()
            if latest:
                payload = _load_embedded_report_data(latest)
                for check in payload.get("checks", []):
                    if check.get("id") == "quality.orphan_trend":
                        m = re.search(r"ratio=([0-9.]+)", str(check.get("detail", "")))
                        if m:
                            prev_ratio = float(m.group(1))
                            break

            detail = f"{orphans}/{total} (ratio={ratio:.3f})"
            if prev_ratio is None:
                return "green", detail
            if prev_ratio > 0 and ratio > prev_ratio * 1.2 and orphans > 10:
                return "yellow", f"{detail}, up from {prev_ratio:.3f}"
            return "green", detail
        except Exception as exc:
            return "red", f"trend check failed ({exc})"

    def _duplicate_entity_check(self) -> tuple[str, str]:
        try:
            from memory.graph import get_driver

            driver = get_driver()
            with driver.session() as session:
                rows = session.run(
                    """
                    MATCH (e:Entity)
                    RETURN e.name AS name
                    ORDER BY e.mention_count DESC
                    LIMIT 160
                    """
                )
                names = [str(r["name"]) for r in rows if r["name"]]
            near = 0
            threshold = max(1, int(config.HEALTH_DUPLICATE_THRESHOLD))
            for i, left in enumerate(names):
                for right in names[i + 1:]:
                    if abs(len(left) - len(right)) > threshold:
                        continue
                    if _levenshtein(left.lower(), right.lower()) < threshold:
                        near += 1
                        if near > 20:
                            break
                if near > 20:
                    break
            if near > 10:
                return "red", f"{near} near-duplicates detected"
            if near > 3:
                return "yellow", f"{near} near-duplicates detected"
            return "green", f"{near} near-duplicates detected"
        except Exception as exc:
            return "red", f"duplicate check failed ({exc})"

    def _chunk_retention_check(self) -> tuple[str, str]:
        try:
            conn = sqlite3.connect(str(config.MOLLYGRAPH_PATH))
            row = conn.execute(
                "SELECT MIN(created_at), MAX(created_at), COUNT(*) FROM conversation_chunks"
            ).fetchone()
            conn.close()
            oldest, newest, count = row
            if int(count or 0) == 0:
                return "red", "0 chunks"
            oldest_dt = _parse_iso(str(oldest))
            newest_dt = _parse_iso(str(newest))
            if not oldest_dt or not newest_dt:
                return "yellow", f"oldest={oldest}, newest={newest}"
            age_days = (datetime.now(timezone.utc) - oldest_dt.astimezone(timezone.utc)).days
            if age_days > 90:
                return "yellow", f"oldest={oldest_dt.date()}, newest={newest_dt.date()}"
            return "green", f"oldest={oldest_dt.date()}, newest={newest_dt.date()}"
        except Exception as exc:
            return "red", f"retention check failed ({exc})"

    def _maintenance_log_check(self) -> tuple[str, str]:
        maint_dir = config.WORKSPACE / "memory" / "maintenance"
        if not maint_dir.exists():
            return "red", "maintenance directory missing"
        today = date.today()
        candidates = [maint_dir / f"{today.isoformat()}.md", maint_dir / f"{(today - timedelta(days=1)).isoformat()}.md"]
        for path in candidates:
            if path.exists():
                return "green", f"last report {path.stem}"
        return "red", "no report for today/yesterday"

    def _operational_tables_check(self) -> tuple[str, str]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        skills = self._count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM skill_executions WHERE created_at > ?",
            (cutoff,),
        )
        corrections = self._count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM corrections WHERE created_at > ?",
            (cutoff,),
        )
        if skills == 0 and corrections == 0:
            return "yellow", "skill_executions=0, corrections=0 (7d)"
        return "green", f"skill_executions={skills}, corrections={corrections} (7d)"

    def _load_automation_state(self) -> tuple[bool, dict[str, Any], str]:
        path = config.AUTOMATIONS_STATE_FILE
        if not path.exists():
            return False, {}, f"missing: {path}"
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                return False, {}, "invalid top-level JSON type"
            if "automations" not in data or not isinstance(data["automations"], dict):
                return False, {}, "missing 'automations' object"
            return True, data, f"loaded {len(data['automations'])} automation entries"
        except Exception as exc:
            return False, {}, f"parse error ({exc})"

    def _find_automation_row(self, automations: dict[str, dict], tokens: list[str]) -> dict | None:
        lowered = [t.lower() for t in tokens]
        for aid, row in automations.items():
            name = f"{aid} {row.get('name', '')}".lower()
            if all(token in name for token in lowered):
                return row
        return None

    def _is_same_day(self, timestamp: str, local_now: datetime) -> bool:
        dt = _parse_iso(timestamp)
        if not dt:
            return False
        return dt.astimezone(local_now.tzinfo).date() == local_now.date()

    def _staleness_status(self, row: dict | None, max_minutes: int) -> str:
        if not row:
            return "yellow"
        dt = _parse_iso(str(row.get("last_run", "")))
        if not dt:
            return "yellow"
        age_min = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 60
        if age_min <= max_minutes:
            return "green"
        if age_min <= max_minutes * 3:
            return "yellow"
        return "red"

    def _staleness_detail(self, row: dict | None) -> str:
        if not row:
            return "automation not found"
        dt_str = str(row.get("last_run", ""))
        if not dt_str:
            return "never run"
        return f"last_run={_short_ts(dt_str)} status={row.get('last_status', '-')}"

    def _maintenance_action_check(self) -> tuple[str, str]:
        maint_dir = config.WORKSPACE / "memory" / "maintenance"
        if not maint_dir.exists():
            return "red", "maintenance directory missing"
        paths = sorted(maint_dir.glob("????-??-??.md"))
        if not paths:
            return "red", "no maintenance reports"
        latest = paths[-1]
        text = latest.read_text()
        table_lines = [line for line in text.splitlines() if line.startswith("| ")]
        if not table_lines:
            return "yellow", f"{latest.stem}: no task table"

        action_like = 0
        for line in table_lines:
            if any(token in line.lower() for token in ("updated", "merged", "deleted", "archived", "found")):
                action_like += 1
        if action_like == 0:
            return "yellow", f"{latest.stem}: no explicit actions"
        return "green", f"{latest.stem}: {action_like} action-like entries"

    def _weekly_assessment_check(self) -> tuple[str, str]:
        weekly_dir = config.WEEKLY_ASSESSMENT_DIR
        weekly_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(weekly_dir.glob("????-??-??.md"))
        if not files:
            return "red", "no weekly assessments"
        latest = files[-1]
        d = date.fromisoformat(latest.stem)
        today = date.today()
        days_since = (today - d).days
        if days_since <= 7:
            return "green", f"latest {latest.stem}"
        if days_since <= 14:
            return "yellow", f"stale {days_since} days (latest {latest.stem})"
        return "red", f"stale {days_since} days (latest {latest.stem})"

    def _rejected_resubmission_check(self) -> tuple[str, str]:
        try:
            conn = sqlite3.connect(str(config.MOLLYGRAPH_PATH))
            rows = conn.execute(
                """
                SELECT title, status, created_at
                FROM self_improvement_events
                WHERE category IN ('skill', 'tool', 'core')
                ORDER BY title, created_at
                """
            ).fetchall()
            conn.close()
        except Exception as exc:
            return "yellow", f"unavailable ({exc})"

        seen_rejected: set[str] = set()
        repeats: set[str] = set()
        for title, status, _created_at in rows:
            title_norm = str(title or "").strip().lower()
            status_norm = str(status or "").strip().lower()
            if not title_norm:
                continue
            if status_norm in {"rejected", "denied"}:
                seen_rejected.add(title_norm)
            elif status_norm in {"proposed", "approved", "deployed"} and title_norm in seen_rejected:
                repeats.add(title_norm)

        if repeats:
            return "red", f"re-submitted after rejection: {', '.join(sorted(repeats)[:3])}"
        return "green", "no rejected proposal re-submissions detected"

    def _count_rows(self, db_path: Path, sql: str, params: tuple[Any, ...] = ()) -> int:
        try:
            conn = sqlite3.connect(str(db_path))
            value = conn.execute(sql, params).fetchone()[0]
            conn.close()
            return int(value or 0)
        except Exception:
            return 0


def get_health_doctor(molly=None) -> HealthDoctor:
    return HealthDoctor(molly=molly)
