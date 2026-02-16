from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, cast
from zoneinfo import ZoneInfo

import config
import db_pool
from contract_audit_legacy import run_contract_audits
from utils import atomic_write, atomic_write_json, load_json, normalize_timestamp

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Remediation router
# ---------------------------------------------------------------------------

Severity = Literal["green", "yellow", "red"]
RemediationAction = Literal[
    "auto_fix",
    "propose_skill",
    "propose_tool",
    "propose_core_patch",
    "escalate_owner",
    "observe_only",
]

ACTION_AUTO_FIX: RemediationAction = "auto_fix"
ACTION_PROPOSE_SKILL: RemediationAction = "propose_skill"
ACTION_PROPOSE_TOOL: RemediationAction = "propose_tool"
ACTION_PROPOSE_CORE_PATCH: RemediationAction = "propose_core_patch"
ACTION_ESCALATE_OWNER: RemediationAction = "escalate_owner"
ACTION_OBSERVE_ONLY: RemediationAction = "observe_only"

YELLOW_ESCALATION_DAYS = 3

_DEFAULT_ACTION_BY_SEVERITY: dict[Severity, RemediationAction] = {
    "green": ACTION_OBSERVE_ONLY,
    "yellow": ACTION_OBSERVE_ONLY,
    "red": ACTION_ESCALATE_OWNER,
}

_EXACT_ACTIONS: dict[str, dict[Severity, RemediationAction]] = {
    "component.whatsapp": {"yellow": ACTION_PROPOSE_TOOL},
    "component.neo4j": {"yellow": ACTION_ESCALATE_OWNER},
    "component.google_oauth": {"yellow": ACTION_PROPOSE_TOOL},
    "component.mcp_servers": {"yellow": ACTION_PROPOSE_TOOL},
    "component.automation_engine": {"yellow": ACTION_AUTO_FIX},
    "component.disk_space": {"yellow": ACTION_AUTO_FIX},
    "component.ram_usage": {"yellow": ACTION_AUTO_FIX},
    "pipeline.entity_to_relationship": {"yellow": ACTION_OBSERVE_ONLY},
    "automation.state_integrity": {"yellow": ACTION_ESCALATE_OWNER},
    "automation.failures": {"yellow": ACTION_PROPOSE_TOOL},
    "automation.loaded": {"yellow": ACTION_AUTO_FIX},
    "learning.preference_signals": {"yellow": ACTION_PROPOSE_SKILL},
    "learning.maintenance_actions": {"yellow": ACTION_PROPOSE_SKILL},
    "learning.self_improvement_proposals": {"yellow": ACTION_PROPOSE_SKILL},
    "learning.rejected_resubmission": {"yellow": ACTION_PROPOSE_SKILL},
}

_PREFIX_ACTIONS: list[tuple[str, dict[Severity, RemediationAction]]] = [
    ("pipeline.", {"yellow": ACTION_PROPOSE_CORE_PATCH}),
    ("quality.", {"yellow": ACTION_PROPOSE_CORE_PATCH}),
    ("automation.", {"yellow": ACTION_AUTO_FIX}),
    ("learning.", {"yellow": ACTION_PROPOSE_SKILL}),
    ("component.", {"yellow": ACTION_AUTO_FIX}),
]

_ACTION_RATIONALE: dict[RemediationAction, str] = {
    ACTION_AUTO_FIX: "Operational issue with bounded repair path; queue automated remediation.",
    ACTION_PROPOSE_SKILL: "Pattern indicates repeated workflow gap; draft or update a skill.",
    ACTION_PROPOSE_TOOL: "Capability/tooling gap detected; propose tool-level improvement.",
    ACTION_PROPOSE_CORE_PATCH: "Core behavior needs code-level correction; propose guarded patch.",
    ACTION_ESCALATE_OWNER: "Requires owner-level investigation and explicit decision.",
    ACTION_OBSERVE_ONLY: "Signal is informational; monitor without active intervention.",
}


@dataclass(frozen=True)
class HealthSignal:
    check_id: str
    severity: str
    yellow_streak_days: int = 1


@dataclass(frozen=True)
class EscalationPlan:
    immediate_investigation_candidate: bool
    escalate_owner_now: bool
    yellow_streak_days: int
    yellow_escalation_threshold_days: int
    yellow_days_until_escalation: int


@dataclass(frozen=True)
class RemediationPlan:
    check_id: str
    severity: Severity
    action: RemediationAction
    suggested_action: RemediationAction | None
    rationale: str
    escalation: EscalationPlan


def route_health_signal(
    check_id: str,
    severity: str,
    *,
    yellow_streak_days: int = 1,
) -> RemediationPlan:
    severity_norm = _normalize_severity(severity)
    streak_days = max(0, int(yellow_streak_days))

    base_action = _lookup_action(check_id=check_id, severity=severity_norm)
    if severity_norm == "red" and base_action == ACTION_ESCALATE_OWNER:
        base_action = _lookup_action(check_id=check_id, severity="yellow")

    immediate_candidate = severity_norm == "red"
    yellow_contract_triggered = severity_norm == "yellow" and streak_days >= YELLOW_ESCALATION_DAYS
    escalate_owner_now = immediate_candidate or yellow_contract_triggered

    action = ACTION_ESCALATE_OWNER if escalate_owner_now else base_action
    suggested_action = base_action if action != base_action else None
    days_until = max(0, YELLOW_ESCALATION_DAYS - streak_days) if severity_norm == "yellow" else 0

    return RemediationPlan(
        check_id=check_id,
        severity=severity_norm,
        action=action,
        suggested_action=suggested_action,
        rationale=_build_rationale(
            severity=severity_norm,
            action=action,
            suggested_action=suggested_action,
            yellow_streak_days=streak_days,
        ),
        escalation=EscalationPlan(
            immediate_investigation_candidate=immediate_candidate,
            escalate_owner_now=escalate_owner_now,
            yellow_streak_days=streak_days,
            yellow_escalation_threshold_days=YELLOW_ESCALATION_DAYS,
            yellow_days_until_escalation=days_until,
        ),
    )


def build_remediation_plan(
    signals: Iterable[HealthSignal | Mapping[str, Any] | Any],
    *,
    yellow_streak_by_check: Mapping[str, int] | None = None,
) -> list[RemediationPlan]:
    plans: list[RemediationPlan] = []
    streak_map = yellow_streak_by_check or {}

    for signal in signals:
        check_id, severity, signal_streak = _coerce_signal(signal)
        plans.append(
            route_health_signal(
                check_id=check_id,
                severity=severity,
                yellow_streak_days=int(streak_map.get(check_id, signal_streak)),
            )
        )
    return plans


def resolve_router_row(check_id: str) -> dict[str, RemediationAction]:
    return {
        "green": ACTION_OBSERVE_ONLY,
        "yellow": _lookup_action(check_id=check_id, severity="yellow"),
        "red": ACTION_ESCALATE_OWNER,
    }


def router_table_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for check_id in sorted(_EXACT_ACTIONS):
        row = resolve_router_row(check_id)
        rows.append({"scope": check_id, "green": row["green"], "yellow": row["yellow"], "red": row["red"]})

    for prefix, actions in _PREFIX_ACTIONS:
        rows.append(
            {
                "scope": f"{prefix}*",
                "green": ACTION_OBSERVE_ONLY,
                "yellow": actions.get("yellow", ACTION_OBSERVE_ONLY),
                "red": ACTION_ESCALATE_OWNER,
            }
        )

    rows.append({"scope": "*", "green": ACTION_OBSERVE_ONLY, "yellow": ACTION_OBSERVE_ONLY, "red": ACTION_ESCALATE_OWNER})
    return rows


def _coerce_signal(signal: HealthSignal | Mapping[str, Any] | Any) -> tuple[str, str, int]:
    if isinstance(signal, HealthSignal):
        return signal.check_id, signal.severity, signal.yellow_streak_days

    if isinstance(signal, Mapping):
        check_id = str(signal.get("check_id", signal.get("id", ""))).strip()
        severity = str(signal.get("severity", signal.get("status", ""))).strip()
        return check_id, severity, _safe_int(signal.get("yellow_streak_days", 1), default=1)

    check_id = str(getattr(signal, "check_id", getattr(signal, "id", "")) or "").strip()
    severity = str(getattr(signal, "severity", getattr(signal, "status", "")) or "").strip()
    return check_id, severity, _safe_int(getattr(signal, "yellow_streak_days", 1), default=1)


def _lookup_action(check_id: str, severity: Severity) -> RemediationAction:
    exact = _EXACT_ACTIONS.get(check_id, {})
    if severity in exact:
        return exact[severity]

    for prefix, actions in _PREFIX_ACTIONS:
        if check_id.startswith(prefix) and severity in actions:
            return actions[severity]

    return _DEFAULT_ACTION_BY_SEVERITY[severity]


def _normalize_severity(raw_severity: str) -> Severity:
    severity = str(raw_severity or "").strip().lower()
    if severity in {"green", "yellow", "red"}:
        return cast(Severity, severity)
    raise ValueError(f"unsupported severity: {raw_severity!r}")


def _build_rationale(
    *,
    severity: Severity,
    action: RemediationAction,
    suggested_action: RemediationAction | None,
    yellow_streak_days: int,
) -> str:
    if severity == "red":
        suffix = f" Suggested remediation after triage: {suggested_action}." if suggested_action else ""
        return "Red severity is an immediate investigation candidate." + suffix

    if severity == "yellow" and action == ACTION_ESCALATE_OWNER:
        return (
            f"Yellow persisted {yellow_streak_days} days and crossed the "
            f"{YELLOW_ESCALATION_DAYS}-day escalation contract."
        )

    return _ACTION_RATIONALE[action]


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# HealthDoctor
# ---------------------------------------------------------------------------


@dataclass
class HealthCheck:
    check_id: str
    layer: str
    label: str
    status: str
    detail: str
    action_required: bool = False
    watch_item: bool = False


_STATUS_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
HEALTH_SKILL_WINDOW_DAYS = max(1, int(os.getenv("MOLLY_HEALTH_SKILL_WINDOW_DAYS", "7")))
HEALTH_SKILL_LOW_WATERMARK = max(1, int(os.getenv("MOLLY_HEALTH_SKILL_LOW_WATERMARK", "3")))
HEALTH_SKILL_BASH_RATIO_RED = max(0.0, float(os.getenv("MOLLY_HEALTH_SKILL_BASH_RATIO_RED", "0.30")))
HEALTH_SKILL_BASH_RATIO_YELLOW = max(
    HEALTH_SKILL_BASH_RATIO_RED,
    float(os.getenv("MOLLY_HEALTH_SKILL_BASH_RATIO_YELLOW", "0.75")),
)


def _status_emoji(status: str) -> str:
    return _STATUS_EMOJI.get(status, "⚪")


def _now_local() -> datetime:
    return datetime.now(ZoneInfo(config.TIMEZONE))


def _parse_iso(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        normalized = normalize_timestamp(raw)
        return datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None


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
        return payload if isinstance(payload, dict) else {}
    except Exception:
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
        return path.read_text() if path else None

    def run_daily(self) -> str:
        return self.generate_report(abbreviated=False, trigger="daily")

    def run_abbreviated_preflight(self) -> str:
        return self.generate_report(abbreviated=True, trigger="startup")

    def generate_report(self, abbreviated: bool = False, trigger: str = "manual") -> str:
        checks = self._run_checks(abbreviated=abbreviated)
        summary = {
            "green": sum(1 for c in checks if c.status == "green"),
            "yellow": sum(1 for c in checks if c.status == "yellow"),
            "red": sum(1 for c in checks if c.status == "red"),
        }

        lines = [f"# 🩺 Molly Health Report — {_now_local().strftime('%Y-%m-%d %H:%M %Z')}", ""]
        sections: dict[str, list[HealthCheck]] = {}
        for check in checks:
            sections.setdefault(check.layer, []).append(check)
        for layer_name in ("Component Heartbeats", "Data Quality", "Learning Loop", "Track F Pre-Prod"):
            if layer_name not in sections:
                continue
            lines.append(f"## {layer_name}")
            for check in sections[layer_name]:
                lines.append(f"{_status_emoji(check.status)} {check.label}: {check.detail}")
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
        report_path.write_text(markdown)
        self._prune_old_reports()
        return markdown

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

    def _run_checks(self, abbreviated: bool) -> list[HealthCheck]:
        checks: list[HealthCheck] = []

        wa_status, wa_detail = self._whatsapp_status()
        checks.append(HealthCheck("component.whatsapp", "Component Heartbeats", "WhatsApp", wa_status, wa_detail, wa_status == "red"))

        maintenance_status, maintenance_detail = self._maintenance_log_check()
        checks.append(
            HealthCheck(
                "quality.maintenance_log",
                "Data Quality",
                "Maintenance log",
                maintenance_status,
                maintenance_detail,
                maintenance_status == "red",
            )
        )

        op_status, op_detail = self._operational_tables_check()
        checks.append(HealthCheck("quality.operational_tables", "Data Quality", "Operational tables", op_status, op_detail, watch_item=op_status == "yellow"))

        if abbreviated:
            return checks

        cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        pref_count = self._count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM preference_signals WHERE created_at > ?",
            (cutoff_7d,),
        )
        pref_status = "green" if pref_count > 0 else "yellow"
        checks.append(
            HealthCheck(
                "learning.preference_signals",
                "Learning Loop",
                "Preference signals accumulating",
                pref_status,
                f"{pref_count} in last 7 days",
                watch_item=pref_status == "yellow",
            )
        )

        skill_status, skill_detail = self._skill_execution_volume_check()
        checks.append(
            HealthCheck(
                "learning.skill_execution_volume",
                "Learning Loop",
                "Skill execution volume",
                skill_status,
                skill_detail,
                action_required=skill_status == "red",
                watch_item=skill_status == "yellow",
            )
        )

        ratio_status, ratio_detail = self._skill_vs_direct_bash_ratio_check()
        checks.append(
            HealthCheck(
                "learning.skill_vs_direct_bash_ratio",
                "Learning Loop",
                "Skill vs direct Bash ratio",
                ratio_status,
                ratio_detail,
                action_required=ratio_status == "red",
                watch_item=ratio_status == "yellow",
            )
        )

        checks.extend(self.track_f_preprod_checks())
        return checks

    def _whatsapp_status(self) -> tuple[str, str]:
        wa = getattr(self.molly, "wa", None) if self.molly else None
        if wa is None:
            return "red", "client not initialized"
        connected = bool(getattr(wa, "connected", False))
        if connected:
            return "green", "connected"
        bridge_client = getattr(wa, "client", None)
        bridge_identity = getattr(bridge_client, "me", None) if bridge_client else None
        if bridge_identity:
            return "green", "authenticated (bridge)"
        return "red", "disconnected"

    def _maintenance_log_check(self) -> tuple[str, str]:
        maint_dir = config.WORKSPACE / "memory" / "maintenance"
        if not maint_dir.exists():
            return "red", "maintenance directory missing"
        today = date.today()
        for path in (maint_dir / f"{today.isoformat()}.md", maint_dir / f"{(today - timedelta(days=1)).isoformat()}.md"):
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

    def _skill_execution_volume_check(self) -> tuple[str, str]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=HEALTH_SKILL_WINDOW_DAYS)).isoformat()
        total = self._count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM skill_executions WHERE created_at > ?",
            (cutoff,),
        )
        success = self._count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM skill_executions WHERE created_at > ? AND lower(outcome) = 'success'",
            (cutoff,),
        )
        failure = self._count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM skill_executions WHERE created_at > ? AND lower(outcome) = 'failure'",
            (cutoff,),
        )
        unknown = max(0, total - success - failure)
        detail = (
            f"executions={total} (success={success}, failure={failure}, unknown={unknown}) "
            f"in last {HEALTH_SKILL_WINDOW_DAYS}d"
        )
        if total == 0:
            return "red", detail
        if total < HEALTH_SKILL_LOW_WATERMARK:
            return "yellow", detail
        if failure > success:
            return "yellow", detail
        return "green", detail

    def _skill_vs_direct_bash_ratio_check(self) -> tuple[str, str]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=HEALTH_SKILL_WINDOW_DAYS)).isoformat()
        skills = self._count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM skill_executions WHERE created_at > ?",
            (cutoff,),
        )
        bash_calls = self._count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM tool_calls WHERE created_at > ? AND lower(tool_name) = 'bash'",
            (cutoff,),
        )

        if bash_calls == 0:
            if skills == 0:
                return "yellow", f"skill_executions=0, direct_bash=0 (last {HEALTH_SKILL_WINDOW_DAYS}d)"
            return "green", f"skill_executions={skills}, direct_bash=0 (last {HEALTH_SKILL_WINDOW_DAYS}d)"

        ratio = skills / bash_calls
        detail = (
            f"skill_executions={skills}, direct_bash={bash_calls}, "
            f"ratio={ratio:.2f} (last {HEALTH_SKILL_WINDOW_DAYS}d)"
        )
        if ratio < HEALTH_SKILL_BASH_RATIO_RED:
            return "red", detail
        if ratio < HEALTH_SKILL_BASH_RATIO_YELLOW:
            return "yellow", detail
        return "green", detail

    def run_track_f_preprod_audit(self, output_dir: Path | None = None) -> Path:
        checks = self.track_f_preprod_checks()
        summary = {
            "green": sum(1 for c in checks if c.status == "green"),
            "yellow": sum(1 for c in checks if c.status == "yellow"),
            "red": sum(1 for c in checks if c.status == "red"),
        }
        verdict = "NO-GO" if summary["red"] else "GO"

        lines = [f"# Track F Pre-Prod Readiness Audit — {_now_local().strftime('%Y-%m-%d %H:%M %Z')}", ""]
        lines.append("## Readiness Checks")
        for check in checks:
            lines.append(f"- {_status_emoji(check.status)} `{check.check_id}`: {check.detail}")
        lines += ["", f"## Verdict: **{verdict}**", ""]

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "verdict": verdict,
            "checks": [
                {
                    "id": c.check_id,
                    "label": c.label,
                    "status": c.status,
                    "detail": c.detail,
                    "action_required": bool(c.action_required),
                }
                for c in checks
            ],
        }
        lines.append(f"<!-- TRACK_F_AUDIT_DATA: {json.dumps(payload, ensure_ascii=True)} -->")

        audit_dir = output_dir or config.TRACK_F_AUDIT_DIR
        audit_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        report_path = audit_dir / f"track-f-preprod-{stamp}.md"
        report_path.write_text("\n".join(lines).rstrip() + "\n")
        return report_path

    def track_f_preprod_checks(self) -> list[HealthCheck]:
        return [
            self._track_f_parser_compatibility_check(),
            self._track_f_skill_telemetry_presence_check(),
            self._track_f_foundry_ingestion_health_check(),
            self._track_f_promotion_drift_status_check(),
        ]

    def _track_f_check_result(
        self,
        *,
        check_id: str,
        label: str,
        ok: bool,
        success_detail: str,
        failure_detail: str,
        enforce_flag: bool,
    ) -> HealthCheck:
        if ok:
            return HealthCheck(check_id, "Track F Pre-Prod", label, "green", success_detail)
        hard_enforcement = bool(enforce_flag) and not bool(getattr(config, "TRACK_F_REPORT_ONLY", True))
        mode = "enforced" if hard_enforcement else "report-only"
        return HealthCheck(
            check_id,
            "Track F Pre-Prod",
            label,
            "red" if hard_enforcement else "yellow",
            f"{failure_detail} [{mode}]",
            action_required=hard_enforcement,
            watch_item=not hard_enforcement,
        )

    def _track_f_parser_compatibility_check(self) -> HealthCheck:
        base = 1770598413
        variants = {
            "seconds": str(base),
            "milliseconds": str(base * 1_000),
            "microseconds": str(base * 1_000_000),
            "nanoseconds": str(base * 1_000_000_000),
        }
        failures: list[str] = []
        for label, raw in variants.items():
            parsed = _parse_iso(raw)
            if parsed is None or int(parsed.timestamp()) != base:
                failures.append(f"{label}:parse")
            normalized = normalize_timestamp(raw)
            reparsed = _parse_iso(normalized)
            if reparsed is None or int(reparsed.timestamp()) != base:
                failures.append(f"{label}:normalize")

        if _parse_iso("2026-02-09T00:53:33+00:00") is None:
            failures.append("iso:parse")

        return self._track_f_check_result(
            check_id="trackf.parser_compatibility",
            label="Parser compatibility",
            ok=not failures,
            success_detail="seconds/ms/us/ns epoch + ISO parse paths are compatible",
            failure_detail=f"parser mismatch in {', '.join(failures)}",
            enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_PARSER_COMPAT", False)),
        )

    def _track_f_skill_telemetry_presence_check(self) -> HealthCheck:
        required = {"id", "skill_name", "trigger", "outcome", "user_approval", "edits_made", "created_at"}
        try:
            conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
            try:
                if not self._sqlite_table_exists(conn, "skill_executions"):
                    return self._track_f_check_result(
                        check_id="trackf.skill_telemetry_presence",
                        label="Skill telemetry presence",
                        ok=False,
                        success_detail="",
                        failure_detail="table missing: skill_executions",
                        enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_SKILL_TELEMETRY", False)),
                    )
                rows = conn.execute("PRAGMA table_info(skill_executions)").fetchall()
                cols = {str(row[1]) for row in rows}
                missing = sorted(required - cols)
            finally:
                conn.close()
        except Exception as exc:
            return self._track_f_check_result(
                check_id="trackf.skill_telemetry_presence",
                label="Skill telemetry presence",
                ok=False,
                success_detail="",
                failure_detail=f"telemetry probe failed ({exc})",
                enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_SKILL_TELEMETRY", False)),
            )

        return self._track_f_check_result(
            check_id="trackf.skill_telemetry_presence",
            label="Skill telemetry presence",
            ok=not missing,
            success_detail="schema present",
            failure_detail=("missing columns: " + ", ".join(missing)) if missing else "schema unavailable",
            enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_SKILL_TELEMETRY", False)),
        )

    def _track_f_foundry_ingestion_health_check(self) -> HealthCheck:
        categories = ("skill", "tool", "core")
        try:
            conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
            try:
                if not self._sqlite_table_exists(conn, "self_improvement_events"):
                    return self._track_f_check_result(
                        check_id="trackf.foundry_ingestion_health",
                        label="Foundry ingestion health",
                        ok=False,
                        success_detail="",
                        failure_detail="table missing: self_improvement_events",
                        enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_FOUNDRY_INGESTION", False)),
                    )
                placeholders = ",".join("?" for _ in categories)
                total = int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM self_improvement_events WHERE category IN ({placeholders})",
                        categories,
                    ).fetchone()[0]
                    or 0
                )
            finally:
                conn.close()
        except Exception as exc:
            return self._track_f_check_result(
                check_id="trackf.foundry_ingestion_health",
                label="Foundry ingestion health",
                ok=False,
                success_detail="",
                failure_detail=f"ingestion probe failed ({exc})",
                enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_FOUNDRY_INGESTION", False)),
            )

        return self._track_f_check_result(
            check_id="trackf.foundry_ingestion_health",
            label="Foundry ingestion health",
            ok=total > 0,
            success_detail=f"events={total}",
            failure_detail="no ingestion events",
            enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_FOUNDRY_INGESTION", False)),
        )

    def _track_f_promotion_drift_status_check(self) -> HealthCheck:
        categories = ("skill", "tool", "core")
        try:
            conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
            try:
                if not self._sqlite_table_exists(conn, "self_improvement_events"):
                    return self._track_f_check_result(
                        check_id="trackf.promotion_drift_status",
                        label="Promotion drift status",
                        ok=False,
                        success_detail="",
                        failure_detail="table missing: self_improvement_events",
                        enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_PROMOTION_DRIFT", False)),
                    )
                placeholders = ",".join("?" for _ in categories)
                rows = conn.execute(
                    f"""
                    SELECT lower(status) AS status, COUNT(*) AS c
                    FROM self_improvement_events
                    WHERE category IN ({placeholders})
                    GROUP BY lower(status)
                    """,
                    categories,
                ).fetchall()
            finally:
                conn.close()
        except Exception as exc:
            return self._track_f_check_result(
                check_id="trackf.promotion_drift_status",
                label="Promotion drift status",
                ok=False,
                success_detail="",
                failure_detail=f"promotion drift probe failed ({exc})",
                enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_PROMOTION_DRIFT", False)),
            )

        counts: dict[str, int] = {str(status or "").lower(): int(count or 0) for status, count in rows}
        proposed = counts.get("proposed", 0)
        promoted = counts.get("approved", 0) + counts.get("deployed", 0)
        pending = max(0, proposed - promoted - counts.get("rejected", 0))
        ok = pending <= int(getattr(config, "TRACK_F_PROMOTION_DRIFT_MAX_PENDING", 20))

        return self._track_f_check_result(
            check_id="trackf.promotion_drift_status",
            label="Promotion drift status",
            ok=ok,
            success_detail=f"proposed={proposed}, promoted={promoted}, pending={pending}",
            failure_detail=f"pending={pending}",
            enforce_flag=bool(getattr(config, "TRACK_F_ENFORCE_PROMOTION_DRIFT", False)),
        )

    @staticmethod
    def _sqlite_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            (table_name,),
        ).fetchone()
        return row is not None

    def _count_rows(self, db_path: Path, sql: str, params: tuple[Any, ...] = ()) -> int:
        try:
            conn = db_pool.sqlite_connect(str(db_path))
            value = conn.execute(sql, params).fetchone()[0]
            conn.close()
            return int(value or 0)
        except Exception:
            return 0

    def _prune_old_reports(self) -> None:
        cutoff = date.today() - timedelta(days=max(1, config.HEALTH_REPORT_RETENTION_DAYS))
        for path in self.report_dir.glob("????-??-??.md"):
            try:
                d = date.fromisoformat(path.stem)
            except ValueError:
                continue
            if d < cutoff:
                path.unlink(missing_ok=True)


_default_doctor: HealthDoctor | None = None


def get_health_doctor(molly=None) -> HealthDoctor:
    global _default_doctor
    if molly is not None:
        return HealthDoctor(molly=molly)
    if _default_doctor is None:
        _default_doctor = HealthDoctor(molly=None)
    return _default_doctor


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------


MAINTENANCE_DIR = config.WORKSPACE / "memory" / "maintenance"
HEALTH_LOG_PATH = MAINTENANCE_DIR / "health-log.md"
MAINTENANCE_HOUR = 23


@dataclass
class MaintenanceRunState:
    status: str = "idle"
    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    last_error: str = ""
    queued_requests: int = 0
    failed_steps: list[str] = field(default_factory=list)
    results: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_error": self.last_error,
            "queued_requests": int(self.queued_requests),
            "failed_steps": list(self.failed_steps),
            "results": dict(self.results),
        }


_MAINTENANCE_LOCK: asyncio.Lock | None = None
_MAINTENANCE_LOCK_LOOP_ID: int | None = None
_RUN_STATE = MaintenanceRunState()


def _maintenance_checkpoint_scope() -> str:
    try:
        return str(MAINTENANCE_DIR.resolve())
    except Exception:
        return str(MAINTENANCE_DIR)


def _load_checkpoint(run_date: str) -> set[int]:
    payload = load_json(config.MAINTENANCE_STATE_FILE, {})
    if not isinstance(payload, dict):
        return set()
    if str(payload.get("run_date", "")) != run_date:
        return set()
    if str(payload.get("maintenance_dir", "")) != _maintenance_checkpoint_scope():
        return set()
    completed: set[int] = set()
    for value in payload.get("completed_steps", []):
        try:
            completed.add(int(value))
        except (TypeError, ValueError):
            continue
    return completed


def _save_checkpoint(run_date: str, completed_steps: set[int]) -> None:
    atomic_write_json(
        config.MAINTENANCE_STATE_FILE,
        {
            "run_date": run_date,
            "completed_steps": sorted(completed_steps),
            "maintenance_dir": _maintenance_checkpoint_scope(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _finalize_step(run_date: str, completed_steps: set[int], step_no: int) -> None:
    completed_steps.add(int(step_no))
    _save_checkpoint(run_date, completed_steps)


def _clear_checkpoint() -> None:
    atomic_write_json(
        config.MAINTENANCE_STATE_FILE,
        {
            "run_date": "",
            "completed_steps": [],
            "maintenance_dir": _maintenance_checkpoint_scope(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _get_maintenance_lock() -> asyncio.Lock:
    global _MAINTENANCE_LOCK, _MAINTENANCE_LOCK_LOOP_ID
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    if _MAINTENANCE_LOCK is None or _MAINTENANCE_LOCK_LOOP_ID != loop_id:
        _MAINTENANCE_LOCK = asyncio.Lock()
        _MAINTENANCE_LOCK_LOOP_ID = loop_id
    return _MAINTENANCE_LOCK


def should_run_maintenance(last_run: datetime | None) -> bool:
    now = datetime.now()
    if last_run is None:
        return now.hour >= MAINTENANCE_HOUR
    if last_run.date() >= now.date():
        return False
    if now.hour >= MAINTENANCE_HOUR:
        return True
    return (now.date() - last_run.date()).days > 1


def run_health_check() -> str:
    return (
        f"## Health Check: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        "| Status | ok |\n\n"
    )


def write_health_check() -> None:
    MAINTENANCE_DIR.mkdir(parents=True, exist_ok=True)
    existing = ""
    if HEALTH_LOG_PATH.exists():
        try:
            existing = HEALTH_LOG_PATH.read_text()
        except Exception:
            existing = ""
    atomic_write(HEALTH_LOG_PATH, existing + run_health_check())


async def _run_strength_decay() -> int:
    from memory.graph import run_strength_decay

    return await run_strength_decay()


def _run_dedup_sweep() -> int:
    from memory.dedup import run_dedup

    return int(run_dedup())


def _run_orphan_cleanup() -> int:
    from memory.graph import get_driver

    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """MATCH (e:Entity)
               WHERE NOT (e)--()
                 AND e.mention_count <= 1
                 AND (e.strength IS NULL OR e.strength < 0.3)
               DELETE e
               RETURN count(e) AS deleted"""
        )
        return int(result.single()["deleted"])


def _run_self_ref_cleanup() -> int:
    from memory.graph import delete_self_referencing_rels

    return int(delete_self_referencing_rels())


def _run_blocklist_cleanup() -> int:
    from memory.graph import delete_blocklisted_entities
    from memory.processor import _ENTITY_BLOCKLIST

    return int(delete_blocklisted_entities(_ENTITY_BLOCKLIST))


def _prune_daily_logs() -> int:
    memory_dir = config.WORKSPACE / "memory"
    archive_dir = memory_dir / "archive"
    cutoff = (date.today() - timedelta(days=30)).isoformat()
    archived = 0
    for path in memory_dir.glob("????-??-??.md"):
        if path.stem < cutoff:
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(archive_dir / path.name))
            archived += 1

    deleted = 0
    gs_dir = config.WORKSPACE / "memory" / "graph_suggestions"
    if gs_dir.is_dir():
        for path in gs_dir.glob("????-??-??.jsonl"):
            if path.stem < cutoff:
                path.unlink(missing_ok=True)
                deleted += 1

    from memory.email_digest import cleanup_old_files

    deleted += cleanup_old_files(keep_days=3)
    return archived + deleted


def _latest_scheduled_weekly_date(now_local: datetime) -> date:
    weekday_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    target_idx = weekday_map.get(str(config.WEEKLY_ASSESSMENT_DAY).strip().lower(), 6)
    return (now_local - timedelta(days=(now_local.weekday() - target_idx) % 7)).date()


def _parse_iso_date(value: str) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(normalize_timestamp(text)[:10])
    except ValueError:
        return None


def _weekly_assessment_due_or_overdue(improver: Any, now_local: datetime) -> bool:
    state = getattr(improver, "_state", {}) or {}
    last_date = _parse_iso_date(str(state.get("last_weekly_assessment", "")))
    target_date = _latest_scheduled_weekly_date(now_local)
    if last_date and last_date >= target_date:
        return False
    if now_local.date() > target_date:
        return True
    return now_local.hour >= int(config.WEEKLY_ASSESSMENT_HOUR)


def _build_maintenance_report(
    results: dict[str, str],
    *,
    run_status: str = "",
    failed_steps: list[str] | None = None,
) -> str:
    today = date.today().isoformat()
    lines = [f"# Maintenance Report — {today}\n"]
    if run_status:
        lines += ["## Run Status\n", f"- Status: {run_status}"]
        if failed_steps:
            lines.append(f"- Failed steps: {', '.join(failed_steps)}")
        lines.append("")

    lines += ["## Task Results\n", "| Task | Result |", "|------|--------|"]
    for task, result in results.items():
        lines.append(f"| {task} | {result} |")
    lines.append("")
    return "\n".join(lines)


def _send_summary_to_owner(molly, summary_text: str) -> bool:
    if not molly:
        return False
    wa = getattr(molly, "wa", None)
    owner_getter = getattr(molly, "_get_owner_dm_jid", None)
    if wa is None or not callable(owner_getter):
        return False
    owner_jid = owner_getter()
    if not owner_jid:
        return False
    try:
        send_result = wa.send_message(owner_jid, summary_text)
        tracker = getattr(molly, "_track_send", None)
        if callable(tracker):
            tracker(send_result)
        return bool(send_result)
    except Exception:
        return False


ANALYSIS_SYSTEM_PROMPT = """\
You are Molly's maintenance analyst.
Produce two sections separated by ---WHATSAPP---.
"""


async def _run_opus_analysis(report: str, graph_summary: str, today: str) -> str:
    return ""


async def run_maintenance(molly=None) -> dict[str, Any]:
    lock = _get_maintenance_lock()
    if lock.locked():
        _RUN_STATE.queued_requests += 1
        return {
            "status": "queued",
            "run_id": _RUN_STATE.run_id,
            "queued_requests": _RUN_STATE.queued_requests,
        }

    async with lock:
        today = date.today().isoformat()
        completed_steps = _load_checkpoint(today)
        run_id = uuid.uuid4().hex
        report_path = MAINTENANCE_DIR / f"{today}.md"
        pending_summary_path = MAINTENANCE_DIR / "pending_summary.txt"

        results: dict[str, str] = {}
        failed_steps: list[str] = []
        analysis_text = ""
        whatsapp_summary = ""

        _RUN_STATE.status = "running"
        _RUN_STATE.run_id = run_id
        _RUN_STATE.started_at = datetime.now(timezone.utc).isoformat()
        _RUN_STATE.finished_at = ""
        _RUN_STATE.last_error = ""
        _RUN_STATE.failed_steps = []
        _RUN_STATE.results = {}
        _RUN_STATE.queued_requests = 0

        MAINTENANCE_DIR.mkdir(parents=True, exist_ok=True)

        improver = None
        weekly_due = False
        weekly_result = "not evaluated"

        async def _ensure_improver():
            nonlocal improver
            if improver is not None:
                return improver
            improver = getattr(molly, "self_improvement", None) if molly else None
            if improver is None:
                from self_improve import SelfImprovementEngine

                improver = SelfImprovementEngine(molly=molly)
                await improver.initialize()
            return improver

        def _record(name: str, result: str, *, failed: bool = False) -> None:
            results[name] = str(result)
            _RUN_STATE.results[name] = str(result)
            if failed and name not in failed_steps:
                failed_steps.append(name)
                _RUN_STATE.failed_steps = list(failed_steps)

        def _is_done(step_no: int, name: str) -> bool:
            if step_no in completed_steps:
                _record(name, "skipped (checkpoint resume)")
                return True
            return False

        def _final(step_no: int) -> None:
            _finalize_step(today, completed_steps, step_no)

        def _status() -> str:
            if not results:
                return "failed"
            if not failed_steps:
                return "success"
            return "failed" if len(failed_steps) >= len(results) else "partial"

        async def _run_step(step_no: int, name: str, fn) -> None:
            if _is_done(step_no, name):
                return
            try:
                value = fn()
                if asyncio.iscoroutine(value):
                    value = await value
                if isinstance(value, tuple) and len(value) == 2:
                    text, failed = str(value[0]), bool(value[1])
                else:
                    text, failed = str(value), False
            except Exception:
                text, failed = "failed", True
            _record(name, text, failed=failed)
            _final(step_no)

        async def _step_relationship_audit() -> tuple[str, bool]:
            from memory.relationship_audit import run_relationship_audit

            rel_audit = await run_relationship_audit(
                model_enabled=config.REL_AUDIT_MODEL_ENABLED,
                molly=molly,
            )
            ra_auto = rel_audit.get("auto_fixes_applied", 0)
            ra_quar = rel_audit.get("quarantined_count", 0)
            ra_status = rel_audit.get("deterministic_result", {}).get("status", "pass")
            return f"{ra_auto} auto-fixed, {ra_quar} quarantined ({ra_status})", ra_status == "fail"

        async def _step_strength_decay() -> str:
            return f"{await _run_strength_decay()} entities updated"

        async def _step_memory_optimization() -> str:
            mem_opt = await (await _ensure_improver()).run_memory_optimization()
            return (
                f"consolidated={mem_opt.get('entity_consolidations', 0)}, "
                f"stale={mem_opt.get('stale_entities', 0)}, "
                f"contradictions={mem_opt.get('contradictions', 0)}"
            )

        async def _step_gliner_loop() -> str:
            gliner_cycle = await (await _ensure_improver()).run_gliner_nightly_cycle()
            return str(gliner_cycle.get("message") or gliner_cycle.get("status", "unknown"))

        async def _step_weekly_assessment() -> str:
            nonlocal weekly_due, weekly_result
            now_local = datetime.now(ZoneInfo(config.TIMEZONE))
            impr = await _ensure_improver()
            weekly_due = _weekly_assessment_due_or_overdue(impr, now_local)
            if not weekly_due:
                weekly_result = "not due"
                return "not due"
            weekly_name = Path(str(await impr.run_weekly_assessment())).name
            weekly_result = f"generated {weekly_name}"
            return weekly_result

        async def _step_contract_audits() -> None:
            if _is_done(17, "Contract audit nightly (deterministic)"):
                for skipped_name in (
                    "Contract audit weekly (deterministic)",
                    "Contract audit nightly (model)",
                    "Contract audit weekly (model)",
                    "Contract audit artifacts",
                ):
                    _record(skipped_name, "skipped (checkpoint resume)")
                return

            try:
                audit_bundle = await run_contract_audits(
                    today=today,
                    task_results=results,
                    weekly_due=weekly_due,
                    weekly_result=weekly_result,
                    maintenance_dir=MAINTENANCE_DIR,
                    health_dir=config.HEALTH_REPORT_DIR,
                )

                nightly_det = dict(audit_bundle.get("nightly_deterministic", {}))
                weekly_det = dict(audit_bundle.get("weekly_deterministic", {}))
                nightly_model = dict(audit_bundle.get("nightly_model", {}))
                weekly_model = dict(audit_bundle.get("weekly_model", {}))
                artifacts = dict(audit_bundle.get("artifacts", {}))

                _record(
                    "Contract audit nightly (deterministic)",
                    str(nightly_det.get("summary", "pass")),
                    failed=str(nightly_det.get("status", "pass")) == "fail",
                )
                _record(
                    "Contract audit weekly (deterministic)",
                    str(weekly_det.get("summary", "pass")),
                    failed=str(weekly_det.get("status", "pass")) == "fail",
                )

                model_blocking = bool(config.CONTRACT_AUDIT_LLM_BLOCKING)
                nightly_model_status = str(nightly_model.get("status", "disabled")).strip().lower()
                weekly_model_status = str(weekly_model.get("status", "disabled")).strip().lower()
                _record(
                    "Contract audit nightly (model)",
                    str(nightly_model.get("summary", "disabled by config")),
                    failed=model_blocking and nightly_model_status in {"error", "unavailable"},
                )
                _record(
                    "Contract audit weekly (model)",
                    str(weekly_model.get("summary", "disabled by config")),
                    failed=model_blocking and weekly_model_status in {"error", "unavailable"},
                )

                artifact_error = str(artifacts.get("error", "")).strip()
                if artifact_error:
                    _record("Contract audit artifacts", f"write error: {artifact_error}")
                else:
                    _record(
                        "Contract audit artifacts",
                        (
                            f"maintenance={Path(str(artifacts.get('maintenance', '-'))).name}, "
                            f"health={Path(str(artifacts.get('health', '-'))).name}"
                        ),
                    )
            except Exception:
                _record("Contract audit nightly (deterministic)", "failed", failed=True)
                _record("Contract audit weekly (deterministic)", "failed", failed=True)
                if config.CONTRACT_AUDIT_LLM_BLOCKING:
                    _record("Contract audit nightly (model)", "failed", failed=True)
                    _record("Contract audit weekly (model)", "failed", failed=True)
                else:
                    _record("Contract audit nightly (model)", "error (report-only)")
                    _record("Contract audit weekly (model)", "error (report-only)")
                _record("Contract audit artifacts", "unavailable")
            finally:
                _final(17)

        try:
            await _run_step(1, "Health check", lambda: (write_health_check(), "completed")[1])
            await _run_step(2, "Strength decay", _step_strength_decay)
            await _run_step(3, "Deduplication", lambda: f"{_run_dedup_sweep()} entities merged")
            await _run_step(
                4,
                "Orphan cleanup",
                lambda: (
                    f"{_run_orphan_cleanup()} orphans, {_run_self_ref_cleanup()} self-refs, "
                    f"{_run_blocklist_cleanup()} blocklisted"
                ),
            )
            await _run_step(5, "Relationship audit", _step_relationship_audit)
            await _run_step(7, "Memory optimization", _step_memory_optimization)
            await _run_step(8, "Daily log pruning", lambda: f"{_prune_daily_logs()} logs archived")
            await _run_step(9, "GLiNER loop", _step_gliner_loop)
            await _run_step(14, "Weekly assessment", _step_weekly_assessment)
            await _run_step(15, "Health Doctor", lambda: (get_health_doctor(molly=molly).run_daily(), "completed")[1])
            await _step_contract_audits()

            if not _is_done(18, "Analysis"):
                try:
                    from memory.graph import get_graph_summary

                    summary = get_graph_summary()
                    graph_text = (
                        f"Entities: {summary['entity_count']}, Relationships: {summary['relationship_count']}\n"
                        f"Top connected: {summary['top_connected']}\nRecent: {summary['recent']}"
                    )
                    analysis = await _run_opus_analysis(
                        _build_maintenance_report(results, run_status=_status(), failed_steps=failed_steps),
                        graph_text,
                        today,
                    )
                    if "---WHATSAPP---" in analysis:
                        analysis_text, whatsapp_summary = [
                            part.strip() for part in analysis.split("---WHATSAPP---", 1)
                        ]
                    else:
                        analysis_text = analysis.strip()
                        whatsapp_summary = analysis.strip()
                    if analysis_text:
                        memory_path = config.WORKSPACE / "MEMORY.md"
                        existing = memory_path.read_text() if memory_path.exists() else ""
                        atomic_write(memory_path, (existing.rstrip() + "\n\n" + analysis_text + "\n").lstrip())
                        _record("Analysis", "MEMORY.md updated")
                    else:
                        _record("Analysis", "empty response")
                except Exception:
                    _record("Analysis", "failed", failed=True)
                finally:
                    _final(18)

            if not _is_done(19, "Report"):
                report = _build_maintenance_report(results, run_status=_status(), failed_steps=failed_steps)
                if analysis_text.strip():
                    report = report.rstrip() + f"\n\n## Analysis\n\n{analysis_text.strip()}\n"
                atomic_write(report_path, report)
                _record("Report", "written")
                _final(19)

            if not _is_done(20, "Summary"):
                try:
                    from memory.graph import entity_count, relationship_count

                    if whatsapp_summary:
                        message = f"*🧠 Nightly Maintenance — {today}*\n\n{whatsapp_summary}"
                    else:
                        message = (
                            f"*⚙️ Maintenance complete — {today}*\n"
                            f"Graph: {entity_count()} entities, {relationship_count()} relationships."
                        )
                        if failed_steps:
                            message += f"\n⚠️ {len(failed_steps)} step(s) failed: " + ", ".join(failed_steps)
                    atomic_write(pending_summary_path, message[:2000])
                    _record("Summary", "saved for morning delivery")
                except Exception:
                    _record("Summary", "failed", failed=True)
                finally:
                    _final(20)

            _RUN_STATE.status = _status()
            _RUN_STATE.last_error = ""
            _clear_checkpoint()
            return _RUN_STATE.as_dict()

        except Exception as exc:
            _RUN_STATE.status = "failed"
            _RUN_STATE.last_error = str(exc)
            return _RUN_STATE.as_dict()
        finally:
            _RUN_STATE.finished_at = datetime.now(timezone.utc).isoformat()
            _RUN_STATE.failed_steps = list(failed_steps)
            _RUN_STATE.results = dict(results)


__all__ = [
    "ACTION_AUTO_FIX",
    "ACTION_ESCALATE_OWNER",
    "ACTION_OBSERVE_ONLY",
    "ACTION_PROPOSE_CORE_PATCH",
    "ACTION_PROPOSE_SKILL",
    "ACTION_PROPOSE_TOOL",
    "ANALYSIS_SYSTEM_PROMPT",
    "EscalationPlan",
    "HealthDoctor",
    "HealthSignal",
    "MaintenanceRunState",
    "RemediationPlan",
    "YELLOW_ESCALATION_DAYS",
    "build_remediation_plan",
    "get_health_doctor",
    "resolve_router_row",
    "route_health_signal",
    "router_table_rows",
    "run_health_check",
    "run_maintenance",
    "should_run_maintenance",
    "write_health_check",
]
