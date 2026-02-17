"""Learning Loop — 8 learning and self-improvement checks."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import config
import db_pool
from monitoring._base import (
    HEALTH_SKILL_BASH_RATIO_RED,
    HEALTH_SKILL_BASH_RATIO_YELLOW,
    HEALTH_SKILL_LOW_WATERMARK,
    HEALTH_SKILL_WINDOW_DAYS,
    HealthCheck,
    _count_rows,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_learning_loop(molly=None) -> list[HealthCheck]:
    """Run all 8 learning loop checks and return results."""
    checks: list[HealthCheck] = []
    cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    # 1. Preference signals accumulating
    pref_count = _count_rows(
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

    # 2. Skill execution volume
    skill_status, skill_detail = _skill_execution_volume_check()
    checks.append(
        HealthCheck(
            check_id="learning.skill_execution_volume",
            layer="Learning Loop",
            label="Skill execution volume",
            status=skill_status,
            detail=skill_detail,
            watch_item=(skill_status == "yellow"),
            action_required=(skill_status == "red"),
        )
    )

    # 3. Skill vs direct Bash ratio
    ratio_status, ratio_detail = _skill_vs_direct_bash_ratio_check()
    checks.append(
        HealthCheck(
            check_id="learning.skill_vs_direct_bash_ratio",
            layer="Learning Loop",
            label="Skill vs direct Bash ratio",
            status=ratio_status,
            detail=ratio_detail,
            watch_item=(ratio_status == "yellow"),
            action_required=(ratio_status == "red"),
        )
    )

    # 4. Nightly maintenance completion
    maintenance_status, maintenance_detail = _maintenance_log_check()
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

    # 5. Maintenance actions taken
    action_status, action_detail = _maintenance_action_check()
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

    # 6. Self-improvement proposals
    cutoff_30d = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    proposals = _count_rows(
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

    # 7. Weekly assessment
    weekly_status, weekly_detail = _weekly_assessment_check()
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

    # 8. Rejected proposal resubmission
    reject_status, reject_detail = _rejected_resubmission_check()
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

    # 9–11. Evolution engine health (evolution.db)
    checks.extend(_evolution_health_checks())

    return checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skill_execution_volume_check() -> tuple[str, str]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=HEALTH_SKILL_WINDOW_DAYS)).isoformat()
    total = _count_rows(
        config.MOLLYGRAPH_PATH,
        "SELECT COUNT(*) FROM skill_executions WHERE created_at > ?",
        (cutoff,),
    )
    success = _count_rows(
        config.MOLLYGRAPH_PATH,
        "SELECT COUNT(*) FROM skill_executions WHERE created_at > ? AND lower(outcome) = 'success'",
        (cutoff,),
    )
    failure = _count_rows(
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


def _skill_vs_direct_bash_ratio_check() -> tuple[str, str]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=HEALTH_SKILL_WINDOW_DAYS)).isoformat()
    skills = _count_rows(
        config.MOLLYGRAPH_PATH,
        "SELECT COUNT(*) FROM skill_executions WHERE created_at > ?",
        (cutoff,),
    )
    bash_calls = _count_rows(
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


def _maintenance_log_check() -> tuple[str, str]:
    maint_dir = config.WORKSPACE / "memory" / "maintenance"
    if not maint_dir.exists():
        return "red", "maintenance directory missing"
    today = date.today()
    candidates = [
        maint_dir / f"{today.isoformat()}.md",
        maint_dir / f"{(today - timedelta(days=1)).isoformat()}.md",
    ]
    for path in candidates:
        if path.exists():
            return "green", f"last report {path.stem}"
    return "red", "no report for today/yesterday"


def _maintenance_action_check() -> tuple[str, str]:
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


def _weekly_assessment_check() -> tuple[str, str]:
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


def _evolution_health_checks() -> list[HealthCheck]:
    """Query evolution.db for bandit, shadow eval, and guard violation health."""
    checks: list[HealthCheck] = []

    try:
        from evolution.db import get_connection
        conn = get_connection()
    except Exception as exc:
        checks.append(HealthCheck(
            check_id="learning.evolution_db",
            layer="Learning Loop",
            label="Evolution DB accessible",
            status="red",
            detail=f"cannot open evolution.db: {exc}",
            action_required=True,
        ))
        return checks

    try:
        cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).timestamp()

        # 9. Bandit arm convergence — are arms being pulled and rewarded?
        arm_row = conn.execute(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(pulls), 0) as total_pulls, "
            "COALESCE(AVG(CASE WHEN pulls > 0 THEN 1.0 * successes / pulls END), 0) as avg_rate "
            "FROM bandit_arms"
        ).fetchone()
        arm_count = arm_row["cnt"] if arm_row else 0
        total_pulls = arm_row["total_pulls"] if arm_row else 0
        avg_rate = arm_row["avg_rate"] if arm_row else 0

        if arm_count == 0:
            bandit_status, bandit_detail = "yellow", "no bandit arms registered"
        elif total_pulls == 0:
            bandit_status, bandit_detail = "yellow", f"{arm_count} arms, 0 pulls"
        else:
            bandit_status = "green"
            bandit_detail = f"{arm_count} arms, {total_pulls} pulls, avg success rate {avg_rate:.1%}"
        checks.append(HealthCheck(
            check_id="learning.bandit_convergence",
            layer="Learning Loop",
            label="Bandit arm convergence",
            status=bandit_status,
            detail=bandit_detail,
            watch_item=(bandit_status == "yellow"),
        ))

        # 10. Shadow evaluation pass rate — are proposed changes improving?
        shadow_row = conn.execute(
            "SELECT COUNT(*) as cnt, "
            "COALESCE(SUM(is_improvement), 0) as improved, "
            "COALESCE(SUM(guard_passed), 0) as guard_ok "
            "FROM shadow_results WHERE timestamp > ?",
            (cutoff_7d,),
        ).fetchone()
        shadow_cnt = shadow_row["cnt"] if shadow_row else 0

        if shadow_cnt == 0:
            shadow_status, shadow_detail = "yellow", "no shadow evaluations in last 7 days"
        else:
            improved = shadow_row["improved"]
            guard_ok = shadow_row["guard_ok"]
            pass_rate = improved / shadow_cnt
            shadow_status = "green" if pass_rate >= 0.3 else "yellow" if pass_rate > 0 else "red"
            shadow_detail = (
                f"{shadow_cnt} evals, {improved} improved ({pass_rate:.0%}), "
                f"{guard_ok} guard-passed (last 7d)"
            )
        checks.append(HealthCheck(
            check_id="learning.shadow_eval_pass_rate",
            layer="Learning Loop",
            label="Shadow eval pass rate",
            status=shadow_status,
            detail=shadow_detail,
            watch_item=(shadow_status == "yellow"),
            action_required=(shadow_status == "red"),
        ))

        # 11. Guard violations — are we hitting safety limits?
        violation_row = conn.execute(
            "SELECT COUNT(*) as cnt, "
            "COALESCE(SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END), 0) as critical "
            "FROM guard_violations WHERE timestamp > ?",
            (cutoff_7d,),
        ).fetchone()
        v_cnt = violation_row["cnt"] if violation_row else 0
        v_critical = violation_row["critical"] if violation_row else 0

        if v_critical > 0:
            guard_status = "red"
        elif v_cnt > 5:
            guard_status = "yellow"
        else:
            guard_status = "green"
        guard_detail = f"{v_cnt} violations ({v_critical} critical) in last 7 days"
        checks.append(HealthCheck(
            check_id="learning.guard_violations",
            layer="Learning Loop",
            label="Guard violations",
            status=guard_status,
            detail=guard_detail,
            watch_item=(guard_status == "yellow"),
            action_required=(guard_status == "red"),
        ))

    except Exception as exc:
        log.debug("Evolution health checks failed: %s", exc, exc_info=True)
        checks.append(HealthCheck(
            check_id="learning.evolution_health",
            layer="Learning Loop",
            label="Evolution engine health",
            status="yellow",
            detail=f"checks failed: {exc}",
            watch_item=True,
        ))
    finally:
        conn.close()

    return checks


def _rejected_resubmission_check() -> tuple[str, str]:
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
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
