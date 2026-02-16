from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, cast

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
    "component.triage_model": {"yellow": ACTION_PROPOSE_CORE_PATCH},
    "component.embedding_model": {"yellow": ACTION_PROPOSE_CORE_PATCH},
    "component.gliner_model": {"yellow": ACTION_PROPOSE_CORE_PATCH},
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
        # Reuse yellow mapping as the recommended remediation once triaged.
        base_action = _lookup_action(check_id=check_id, severity="yellow")

    immediate_candidate = severity_norm == "red"
    yellow_contract_triggered = severity_norm == "yellow" and streak_days >= YELLOW_ESCALATION_DAYS
    escalate_owner_now = immediate_candidate or yellow_contract_triggered

    action = ACTION_ESCALATE_OWNER if escalate_owner_now else base_action
    suggested_action = base_action if action != base_action else None

    if severity_norm == "yellow":
        days_until = max(0, YELLOW_ESCALATION_DAYS - streak_days)
    else:
        days_until = 0

    rationale = _build_rationale(
        severity=severity_norm,
        action=action,
        suggested_action=suggested_action,
        yellow_streak_days=streak_days,
    )

    return RemediationPlan(
        check_id=check_id,
        severity=severity_norm,
        action=action,
        suggested_action=suggested_action,
        rationale=rationale,
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
        streak_days = int(streak_map.get(check_id, signal_streak))
        plans.append(
            route_health_signal(
                check_id=check_id,
                severity=severity,
                yellow_streak_days=streak_days,
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
        rows.append(
            {
                "scope": check_id,
                "green": row["green"],
                "yellow": row["yellow"],
                "red": row["red"],
            }
        )

    for prefix, actions in _PREFIX_ACTIONS:
        rows.append(
            {
                "scope": f"{prefix}*",
                "green": ACTION_OBSERVE_ONLY,
                "yellow": actions.get("yellow", ACTION_OBSERVE_ONLY),
                "red": ACTION_ESCALATE_OWNER,
            }
        )

    rows.append(
        {
            "scope": "*",
            "green": ACTION_OBSERVE_ONLY,
            "yellow": ACTION_OBSERVE_ONLY,
            "red": ACTION_ESCALATE_OWNER,
        }
    )
    return rows


def _coerce_signal(signal: HealthSignal | Mapping[str, Any] | Any) -> tuple[str, str, int]:
    if isinstance(signal, HealthSignal):
        return signal.check_id, signal.severity, signal.yellow_streak_days

    if isinstance(signal, Mapping):
        check_id = str(signal.get("check_id", signal.get("id", ""))).strip()
        severity = str(signal.get("severity", signal.get("status", ""))).strip()
        streak_days = _safe_int(signal.get("yellow_streak_days", 1), default=1)
        return check_id, severity, streak_days

    check_id = str(
        getattr(signal, "check_id", getattr(signal, "id", "")) or ""
    ).strip()
    severity = str(
        getattr(signal, "severity", getattr(signal, "status", "")) or ""
    ).strip()
    streak_days = _safe_int(getattr(signal, "yellow_streak_days", 1), default=1)
    return check_id, severity, streak_days


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
        suffix = (
            f" Suggested remediation after triage: {suggested_action}."
            if suggested_action
            else ""
        )
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
