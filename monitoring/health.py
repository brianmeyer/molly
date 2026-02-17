"""Health Doctor — thin orchestrator that delegates to agent modules.

All check implementations live in monitoring/agents/*.py. This module
coordinates them, applies yellow escalation, syncs the issue registry,
and generates the health report.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import config
from monitoring._base import (
    HealthCheck,
    _load_embedded_report_data,
    _now_local,
    _status_emoji,
)
from monitoring.remediation import route_health_signal  # noqa: F401 — re-exported

log = logging.getLogger(__name__)

# Layer display order for the report
_LAYER_ORDER = (
    "Component Heartbeats",
    "Pipeline Validation",
    "Data Quality",
    "Automation Health",
    "Learning Loop",
    "Retrieval Quality",
    "Track F Pre-Prod",
)


class HealthDoctor:
    def __init__(self, molly=None):
        self.molly = molly
        self.report_dir = config.HEALTH_REPORT_DIR
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Report access
    # ------------------------------------------------------------------

    def latest_report_path(self) -> Path | None:
        reports = sorted(self.report_dir.glob("????-??-??.md"))
        return reports[-1] if reports else None

    def latest_report_text(self) -> str | None:
        path = self.latest_report_path()
        return path.read_text() if path else None

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_daily(self) -> str:
        return self.generate_report(abbreviated=False, trigger="daily")

    def run_abbreviated_preflight(self) -> str:
        return self.generate_report(abbreviated=True, trigger="startup")

    def generate_report(self, abbreviated: bool = False, trigger: str = "manual") -> str:
        checks = self._run_checks(abbreviated=abbreviated)

        # Apply yellow escalation
        if not abbreviated:
            from monitoring.agents.yellow_escalation import apply_yellow_escalation
            checks = apply_yellow_escalation(checks, self.report_dir)

        summary = {
            "green": sum(1 for c in checks if c.status == "green"),
            "yellow": sum(1 for c in checks if c.status == "yellow"),
            "red": sum(1 for c in checks if c.status == "red"),
        }

        # Build report markdown
        lines = [f"# 🩺 Molly Health Report — {_now_local().strftime('%Y-%m-%d %H:%M %Z')}", ""]

        sections: dict[str, list[HealthCheck]] = {}
        for check in checks:
            sections.setdefault(check.layer, []).append(check)

        for layer_name in _LAYER_ORDER:
            if layer_name not in sections:
                continue
            lines.append(f"## {layer_name}")
            for check in sections[layer_name]:
                lines.append(f"{_status_emoji(check.status)} {check.label}: {check.detail}")
            lines.append("")

        # Action Required section
        action_items = [c for c in checks if c.action_required]
        if action_items:
            lines.append("## Action Required")
            for c in action_items:
                lines.append(f"- {_status_emoji(c.status)} **{c.label}** (`{c.check_id}`): {c.detail}")
            lines.append("")

        # Watch Items section
        watch_items = [c for c in checks if c.watch_item and not c.action_required]
        if watch_items:
            lines.append("## Watch Items")
            for c in watch_items:
                lines.append(f"- {_status_emoji(c.status)} {c.label}: {c.detail}")
            lines.append("")

        lines.append(
            f"## Summary: {summary['green']} {_status_emoji('green')} / "
            f"{summary['yellow']} {_status_emoji('yellow')} / {summary['red']} {_status_emoji('red')}"
        )
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
        from utils import atomic_write
        atomic_write(report_path, markdown)
        self._prune_old_reports()

        # Sync issue registry (non-blocking — errors are swallowed)
        if not abbreviated:
            try:
                from monitoring.agents.issue_sync import sync_issue_registry
                sync_issue_registry(checks, self.report_dir)
            except Exception:
                log.warning("Issue registry sync failed", exc_info=True)

        return markdown

    # ------------------------------------------------------------------
    # Track F (delegate to agent module, keep methods for backward compat)
    # ------------------------------------------------------------------

    def run_track_f_preprod_audit(self, output_dir: Path | None = None) -> Path:
        from monitoring.agents.track_f_preprod import run_track_f_audit
        return run_track_f_audit(output_dir)

    def track_f_preprod_checks(self) -> list[HealthCheck]:
        from monitoring.agents.track_f_preprod import run_track_f_checks
        return run_track_f_checks()

    # ------------------------------------------------------------------
    # Comparison helpers (restored from legacy)
    # ------------------------------------------------------------------

    def extract_status_map(self, report_text: str) -> dict[str, str]:
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
            return {str(row.get("id", "")): str(row.get("status", "")) for row in rows if row.get("id")}
        except Exception:
            return {}

    def get_or_generate_latest_report(self) -> str:
        """Return today's report if it exists, otherwise generate one."""
        today_path = self.report_dir / f"{date.today().isoformat()}.md"
        if today_path.exists():
            return today_path.read_text()
        return self.generate_report(abbreviated=False, trigger="on-demand")

    def history_markdown(self, days: int = 7) -> str:
        """Return markdown showing check status over the last N days."""
        paths = sorted(self.report_dir.glob("????-??-??.md"))[-days:]
        if not paths:
            return "No health history available."
        lines = ["| Date | Green | Yellow | Red |", "|------|-------|--------|-----|"]
        for path in paths:
            payload = _load_embedded_report_data(path)
            s = payload.get("summary", {})
            lines.append(
                f"| {path.stem} | {s.get('green', '?')} | {s.get('yellow', '?')} | {s.get('red', '?')} |"
            )
        return "\n".join(lines)

    def compare_green_to_red_regressions(self, previous_text: str, current_text: str) -> list[str]:
        """Return check_ids that regressed from green to red between two reports."""
        prev = self.extract_status_map(previous_text)
        curr = self.extract_status_map(current_text)
        return [
            cid for cid in curr
            if curr.get(cid) == "red" and prev.get(cid) == "green"
        ]

    def compare_worsened_components(
        self,
        previous_text: str = "",
        current_text: str = "",
        *,
        baseline_report: str = "",
        candidate_report: str = "",
    ) -> list[dict[str, str]]:
        """Return dicts with ``id``, ``before``, ``after`` for worsened checks.

        Accepts both positional (previous_text, current_text) and keyword
        (baseline_report, candidate_report) call styles for backward compat.
        """
        prev_text = baseline_report or previous_text
        curr_text = candidate_report or current_text
        severity = {"green": 0, "yellow": 1, "red": 2}
        prev = self.extract_status_map(prev_text)
        curr = self.extract_status_map(curr_text)
        return [
            {"id": cid, "before": prev.get(cid, ""), "after": curr.get(cid, "")}
            for cid in curr
            if severity.get(curr.get(cid, ""), 0) > severity.get(prev.get(cid, ""), 0)
        ]

    # ------------------------------------------------------------------
    # Check orchestration
    # ------------------------------------------------------------------

    def _run_checks(self, abbreviated: bool) -> list[HealthCheck]:
        checks: list[HealthCheck] = []

        def _crash_check(layer: str, label: str) -> HealthCheck:
            """Return a red HealthCheck when an agent layer crashes."""
            return HealthCheck(
                check_id=f"meta.{label.replace(' ', '_').lower()}_crash",
                layer=layer,
                label=f"{label} agent",
                status="red",
                detail="Agent crashed during check execution",
                action_required=True,
            )

        # Layer 1: Component Heartbeats (always run)
        try:
            from monitoring.agents.component_heartbeats import run_component_heartbeats
            checks.extend(run_component_heartbeats(self.molly))
        except Exception:
            log.error("Component heartbeats failed", exc_info=True)
            checks.append(_crash_check("Component Heartbeats", "Component Heartbeats"))

        if abbreviated:
            return checks

        # Layer 2: Pipeline Validation
        try:
            from monitoring.agents.pipeline_validation import run_pipeline_validation
            checks.extend(run_pipeline_validation(self.molly))
        except Exception:
            log.error("Pipeline validation failed", exc_info=True)
            checks.append(_crash_check("Pipeline Validation", "Pipeline Validation"))

        # Layer 3: Data Quality
        try:
            from monitoring.agents.data_quality import run_data_quality
            checks.extend(run_data_quality(self.molly, self.latest_report_path()))
        except Exception:
            log.error("Data quality checks failed", exc_info=True)
            checks.append(_crash_check("Data Quality", "Data Quality"))

        # Layer 4: Automation Health
        try:
            from monitoring.agents.automation_health import run_automation_health
            checks.extend(run_automation_health(self.molly))
        except Exception:
            log.error("Automation health checks failed", exc_info=True)
            checks.append(_crash_check("Automation Health", "Automation Health"))

        # Layer 5: Learning Loop
        try:
            from monitoring.agents.learning_loop import run_learning_loop
            checks.extend(run_learning_loop(self.molly))
        except Exception:
            log.error("Learning loop checks failed", exc_info=True)
            checks.append(_crash_check("Learning & Self-Improvement", "Learning Loop"))

        # Layer 6: Retrieval Quality
        try:
            from monitoring.agents.retrieval_quality import run_retrieval_quality
            checks.extend(run_retrieval_quality(self.molly))
        except Exception:
            log.error("Retrieval quality checks failed", exc_info=True)
            checks.append(_crash_check("Retrieval Quality", "Retrieval Quality"))

        # Layer 7: Track F Pre-Prod
        try:
            from monitoring.agents.track_f_preprod import run_track_f_checks
            checks.extend(run_track_f_checks())
        except Exception:
            log.error("Track F checks failed", exc_info=True)
            checks.append(_crash_check("Track F Pre-Prod", "Track F"))

        return checks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prune_old_reports(self) -> None:
        cutoff = date.today() - timedelta(days=max(1, config.HEALTH_REPORT_RETENTION_DAYS))
        for path in self.report_dir.glob("????-??-??.md"):
            try:
                d = date.fromisoformat(path.stem)
            except ValueError:
                continue
            if d < cutoff:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_doctor: HealthDoctor | None = None
_doctor_lock = threading.Lock()


def get_health_doctor(molly=None) -> HealthDoctor:
    global _default_doctor
    with _doctor_lock:
        if molly is not None:
            # Always update the singleton with the molly reference so that
            # subsequent calls without molly still get a doctor with a valid
            # molly reference (avoids stale molly=None in cached singleton).
            _default_doctor = HealthDoctor(molly=molly)
            return _default_doctor
        if _default_doctor is None:
            _default_doctor = HealthDoctor(molly=None)
        return _default_doctor
