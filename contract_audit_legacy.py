import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Mapping

import config
import db_pool

log = logging.getLogger(__name__)

SUPPORTED_MODEL_ROUTES = {"opus", "kimi", "gemini"}
DISABLED_MODEL_ROUTES = {"", "off", "none", "disabled"}

REQUIRED_NIGHTLY_STEPS = (
    "Health check",
    "Strength decay",
    "Deduplication",
    "Orphan cleanup",
    "Relationship audit",
    "Memory optimization",
    "Daily log pruning",
    "GLiNER loop",
    "Weekly assessment",
    "Health Doctor",
)

_WEEKDAY_MAP = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_MODEL_AUDIT_SYSTEM_PROMPT = """\
You audit maintenance contract integrity.

Use deterministic findings as the source of truth, then provide:
1) Verdict: pass | warn | fail
2) Evidence: short bullets tied to deterministic checks
3) Recommendations: concrete next actions
"""


class AuditModelUnavailable(RuntimeError):
    """Raised when a selected model route cannot be used in this environment."""


def _overall_status(checks: list[dict[str, str]]) -> str:
    statuses = {row.get("status", "pass") for row in checks}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _summarize_checks(checks: list[dict[str, str]]) -> str:
    failed = sum(1 for row in checks if row.get("status") == "fail")
    warned = sum(1 for row in checks if row.get("status") == "warn")
    total = len(checks)
    return f"{_overall_status(checks)} ({failed} failed, {warned} warned, {total} checks)"


def query_underperforming_skills(
    *,
    db_path: Path | None = None,
    min_invocations: int = 5,
    success_rate_threshold: float = 0.6,
) -> list[dict[str, Any]]:
    """Query low-performing skills from analytics table with execution fallback."""
    path = Path(db_path or config.MOLLYGRAPH_PATH)
    if not path.exists():
        return []

    conn = db_pool.sqlite_connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            if row and row[0]
        }

        rows: list[sqlite3.Row] = []
        if "skill_analytics" in tables:
            rows = conn.execute(
                """
                SELECT skill_name, invocations, success_rate
                FROM skill_analytics
                WHERE invocations >= ?
                  AND success_rate < ?
                ORDER BY success_rate ASC, invocations DESC, skill_name ASC
                """,
                (int(min_invocations), float(success_rate_threshold)),
            ).fetchall()
        elif "skill_executions" in tables:
            rows = conn.execute(
                """
                SELECT skill_name,
                       COUNT(*) AS invocations,
                       AVG(
                           CASE
                               WHEN lower(coalesce(outcome, '')) IN (
                                   'success', 'ok', 'completed', 'succeeded', 'approved', 'active', 'activated'
                               )
                                   THEN 1.0
                               ELSE 0.0
                           END
                       ) AS success_rate
                FROM skill_executions
                WHERE trim(coalesce(skill_name, '')) <> ''
                GROUP BY skill_name
                HAVING COUNT(*) >= ?
                   AND AVG(
                       CASE
                           WHEN lower(coalesce(outcome, '')) IN (
                               'success', 'ok', 'completed', 'succeeded', 'approved', 'active', 'activated'
                           )
                               THEN 1.0
                           ELSE 0.0
                       END
                   ) < ?
                ORDER BY success_rate ASC, invocations DESC, skill_name ASC
                """,
                (int(min_invocations), float(success_rate_threshold)),
            ).fetchall()

        underperforming: list[dict[str, Any]] = []
        for row in rows:
            skill_name = str(row["skill_name"] or "").strip()
            if not skill_name:
                continue
            try:
                invocations = int(row["invocations"] or 0)
            except (TypeError, ValueError):
                invocations = 0
            try:
                success_rate = float(row["success_rate"] or 0.0)
            except (TypeError, ValueError):
                success_rate = 0.0
            underperforming.append(
                {
                    "skill_name": skill_name,
                    "invocations": max(0, invocations),
                    "success_rate": round(success_rate, 4),
                }
            )
        return underperforming
    except Exception:
        log.debug("Failed to load underperforming skills", exc_info=True)
        return []
    finally:
        conn.close()


def run_nightly_deterministic_checks(
    task_results: Mapping[str, str],
    *,
    underperforming_skills: list[dict[str, Any]] | None = None,
    enforce_underperforming: bool | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    missing_steps = [step for step in REQUIRED_NIGHTLY_STEPS if step not in task_results]
    if missing_steps:
        checks.append(
            {
                "check_id": "nightly.required_steps",
                "status": "fail",
                "detail": f"missing required steps: {', '.join(missing_steps)}",
            }
        )
    else:
        checks.append(
            {
                "check_id": "nightly.required_steps",
                "status": "pass",
                "detail": "all required deterministic steps recorded",
            }
        )

    failed_steps = [
        name
        for name, result in task_results.items()
        if str(result).strip().lower() == "failed"
    ]
    if failed_steps:
        checks.append(
            {
                "check_id": "nightly.step_failures",
                "status": "fail",
                "detail": f"{len(failed_steps)} failed steps: {', '.join(failed_steps)}",
            }
        )
    else:
        checks.append(
            {
                "check_id": "nightly.step_failures",
                "status": "pass",
                "detail": "no failed deterministic steps",
            }
        )

    checks.append(
        {
            "check_id": "nightly.step_count",
            "status": "pass" if len(task_results) >= len(REQUIRED_NIGHTLY_STEPS) else "warn",
            "detail": f"{len(task_results)} step results captured",
        }
    )

    candidates = underperforming_skills or []
    enforce = bool(
        bool(getattr(config, "CONTRACT_AUDIT_ENFORCE_UNDERPERFORMING_SKILLS", False))
        if enforce_underperforming is None
        else enforce_underperforming
    )
    if candidates:
        checks.append(
            {
                "check_id": "nightly.underperforming_skills",
                "status": "fail" if enforce else "warn",
                "detail": (
                    f"{len(candidates)} underperforming skills "
                    "(invocations>=5, success_rate<0.60)"
                ),
            }
        )
    else:
        checks.append(
            {
                "check_id": "nightly.underperforming_skills",
                "status": "pass",
                "detail": "no underperforming skills detected",
            }
        )

    return {
        "status": _overall_status(checks),
        "summary": _summarize_checks(checks),
        "checks": checks,
        "underperforming_skills": candidates,
    }


def run_weekly_deterministic_checks(
    *,
    weekly_due: bool,
    weekly_result: str,
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []

    day_value = str(config.WEEKLY_ASSESSMENT_DAY).strip().lower()
    if day_value in _WEEKDAY_MAP:
        checks.append(
            {
                "check_id": "weekly.schedule_day",
                "status": "pass",
                "detail": f"weekly day configured as '{day_value}'",
            }
        )
    else:
        checks.append(
            {
                "check_id": "weekly.schedule_day",
                "status": "fail",
                "detail": f"invalid WEEKLY_ASSESSMENT_DAY='{day_value}'",
            }
        )

    hour_value = int(config.WEEKLY_ASSESSMENT_HOUR)
    if 0 <= hour_value <= 23:
        checks.append(
            {
                "check_id": "weekly.schedule_hour",
                "status": "pass",
                "detail": f"weekly hour configured as {hour_value:02d}:00",
            }
        )
    else:
        checks.append(
            {
                "check_id": "weekly.schedule_hour",
                "status": "fail",
                "detail": f"invalid WEEKLY_ASSESSMENT_HOUR={hour_value}",
            }
        )

    normalized_result = str(weekly_result or "").strip()
    normalized_lower = normalized_result.lower()
    if not weekly_due:
        checks.append(
            {
                "check_id": "weekly.execution",
                "status": "pass",
                "detail": "weekly assessment not due in this run window",
            }
        )
    elif normalized_lower.startswith("generated "):
        checks.append(
            {
                "check_id": "weekly.execution",
                "status": "pass",
                "detail": normalized_result,
            }
        )
    elif normalized_lower == "failed":
        checks.append(
            {
                "check_id": "weekly.execution",
                "status": "fail",
                "detail": "weekly assessment was due but failed",
            }
        )
    else:
        checks.append(
            {
                "check_id": "weekly.execution",
                "status": "warn",
                "detail": f"weekly due with non-standard result: {normalized_result or '-'}",
            }
        )

    return {
        "status": _overall_status(checks),
        "summary": _summarize_checks(checks),
        "checks": checks,
    }


def _route_for_pass(audit_pass: str) -> str:
    if audit_pass == "weekly":
        return str(config.CONTRACT_AUDIT_WEEKLY_MODEL).strip().lower()
    return str(config.CONTRACT_AUDIT_NIGHTLY_MODEL).strip().lower()


def _routes_for_pass(audit_pass: str) -> list[str]:
    raw = _route_for_pass(audit_pass)
    parts = [part.strip().lower() for part in raw.split(",")]
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        deduped.append(part)
    return deduped or [""]


def _model_prompt(
    *,
    audit_pass: str,
    deterministic_result: Mapping[str, Any],
    context: Mapping[str, Any],
) -> str:
    checks = deterministic_result.get("checks", [])
    check_lines = []
    for row in checks:
        check_lines.append(
            f"- {row.get('check_id', 'unknown')}: "
            f"{row.get('status', 'pass')} ({row.get('detail', '-')})"
        )
    context_json = json.dumps(context, ensure_ascii=True, indent=2)
    return (
        f"Audit pass: {audit_pass}\n"
        f"Deterministic status: {deterministic_result.get('status', 'pass')}\n"
        f"Deterministic summary: {deterministic_result.get('summary', '-')}\n"
        "Deterministic checks:\n"
        f"{chr(10).join(check_lines) if check_lines else '- none'}\n\n"
        "Runtime context (JSON):\n"
        f"{context_json[:8000]}"
    )


async def _invoke_opus(prompt: str) -> str:
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )
    except Exception as exc:
        raise AuditModelUnavailable(f"Opus SDK unavailable: {exc}") from exc

    options = ClaudeAgentOptions(
        system_prompt=_MODEL_AUDIT_SYSTEM_PROMPT,
        model="opus",
        allowed_tools=[],
        cwd=str(config.WORKSPACE),
    )

    async def _prompt():
        yield {
            "type": "user",
            "session_id": "",
            "message": {"role": "user", "content": prompt},
            "parent_tool_use_id": None,
        }

    response_text = ""
    async for message in query(prompt=_prompt(), options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_text += block.text
        elif isinstance(message, ResultMessage) and message.is_error:
            raise RuntimeError(str(message.result))

    if not response_text.strip():
        raise RuntimeError("Opus returned an empty response")
    return response_text.strip()


async def _invoke_kimi(prompt: str) -> str:
    if not config.MOONSHOT_API_KEY:
        raise AuditModelUnavailable("MOONSHOT_API_KEY not configured")

    try:
        import httpx
    except Exception as exc:
        raise AuditModelUnavailable(f"httpx unavailable for Kimi: {exc}") from exc

    body = {
        "model": config.CONTRACT_AUDIT_KIMI_MODEL,
        "messages": [
            {"role": "system", "content": _MODEL_AUDIT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {config.MOONSHOT_API_KEY}",
        "Content-Type": "application/json",
    }

    timeout = float(config.CONTRACT_AUDIT_MODEL_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{config.MOONSHOT_BASE_URL}/chat/completions",
            headers=headers,
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "\n".join(str(chunk) for chunk in content)
    text = str(content).strip()
    if not text:
        raise RuntimeError("Kimi returned an empty response")
    return text


async def _invoke_gemini(prompt: str) -> str:
    if not config.GEMINI_API_KEY:
        raise AuditModelUnavailable("GEMINI_API_KEY not configured")

    try:
        import httpx
    except Exception as exc:
        raise AuditModelUnavailable(f"httpx unavailable for Gemini: {exc}") from exc

    endpoint = (
        f"{config.GEMINI_BASE_URL}/models/"
        f"{config.CONTRACT_AUDIT_GEMINI_MODEL}:generateContent"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": _MODEL_AUDIT_SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1},
    }
    timeout = float(config.CONTRACT_AUDIT_MODEL_TIMEOUT_SECONDS)

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            endpoint,
            params={"key": config.GEMINI_API_KEY},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    candidates = data.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = str(part.get("text", "")).strip()
            if text:
                return text
    raise RuntimeError("Gemini returned an empty response")


async def _invoke_model_route(route: str, prompt: str) -> str:
    if route == "opus":
        return await _invoke_opus(prompt)
    if route == "kimi":
        return await _invoke_kimi(prompt)
    if route == "gemini":
        return await _invoke_gemini(prompt)
    raise RuntimeError(f"Unsupported model route: {route}")


async def run_model_audit(
    *,
    audit_pass: str,
    deterministic_result: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, str]:
    routes = _routes_for_pass(audit_pass)
    primary_route = routes[0]

    if not config.CONTRACT_AUDIT_LLM_ENABLED:
        return {
            "status": "disabled",
            "route": primary_route or "disabled",
            "summary": "disabled by config",
            "output": "",
            "attempts": json.dumps([], ensure_ascii=True),
        }

    enabled_routes = [route for route in routes if route not in DISABLED_MODEL_ROUTES]
    if not enabled_routes:
        return {
            "status": "disabled",
            "route": primary_route or "disabled",
            "summary": "disabled by model route",
            "output": "",
            "attempts": json.dumps([], ensure_ascii=True),
        }

    supported_routes = [route for route in enabled_routes if route in SUPPORTED_MODEL_ROUTES]
    if not supported_routes:
        return {
            "status": "error",
            "route": primary_route or "unknown",
            "summary": f"unsupported routes: {', '.join(enabled_routes)}",
            "output": "",
            "attempts": json.dumps([], ensure_ascii=True),
        }

    prompt = _model_prompt(
        audit_pass=audit_pass,
        deterministic_result=deterministic_result,
        context=context,
    )
    attempts: list[dict[str, str]] = []
    for route in supported_routes:
        try:
            output = await _invoke_model_route(route, prompt)
            attempts.append(
                {
                    "route": route,
                    "status": "completed",
                    "detail": "ok",
                }
            )
            return {
                "status": "completed",
                "route": route,
                "summary": f"completed via {route}",
                "output": output.strip(),
                "attempts": json.dumps(attempts, ensure_ascii=True),
            }
        except AuditModelUnavailable as exc:
            attempts.append(
                {
                    "route": route,
                    "status": "unavailable",
                    "detail": str(exc),
                }
            )
            continue
        except Exception as exc:
            log.error("Contract model audit failed for route=%s", route, exc_info=True)
            attempts.append(
                {
                    "route": route,
                    "status": "error",
                    "detail": str(exc),
                }
            )
            continue

    if attempts and all(item.get("status") == "unavailable" for item in attempts):
        detail = "; ".join(
            f"{item.get('route')}={item.get('detail')}" for item in attempts
        )
        return {
            "status": "unavailable",
            "route": attempts[0].get("route", primary_route),
            "summary": f"unavailable ({detail})",
            "output": "",
            "attempts": json.dumps(attempts, ensure_ascii=True),
        }

    if attempts:
        detail = "; ".join(
            f"{item.get('route')}={item.get('status')}" for item in attempts
        )
        return {
            "status": "error",
            "route": attempts[0].get("route", primary_route),
            "summary": f"error (all routes failed: {detail})",
            "output": "",
            "attempts": json.dumps(attempts, ensure_ascii=True),
        }

    return {
        "status": "error",
        "route": primary_route or "unknown",
        "summary": "error (no routable model configured)",
        "output": "",
        "attempts": json.dumps([], ensure_ascii=True),
    }


def _render_check_lines(checks: list[dict[str, str]]) -> str:
    lines = ["| Check | Status | Detail |", "|---|---|---|"]
    for row in checks:
        lines.append(
            f"| {row.get('check_id', 'unknown')} | "
            f"{row.get('status', 'pass')} | "
            f"{row.get('detail', '-')} |"
        )
    return "\n".join(lines)


def _render_underperforming_skills_output(rows: list[dict[str, Any]]) -> str:
    lines = [
        "### Deterministic Check Output",
        "- Criteria: invocations >= 5 and success_rate < 0.60",
    ]
    if not rows:
        lines.append("- Underperforming skills: none")
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "| Skill | Invocations | Success Rate |",
            "|---|---:|---:|",
        ]
    )
    for row in rows:
        skill = str(row.get("skill_name", "-") or "-")
        invocations = int(row.get("invocations", 0) or 0)
        success_rate = float(row.get("success_rate", 0.0) or 0.0)
        lines.append(f"| {skill} | {invocations} | {success_rate:.2f} |")
    return "\n".join(lines)


def _render_model_section(title: str, result: Mapping[str, str]) -> str:
    lines = [
        f"## {title}",
        f"- Status: {result.get('status', '-')}",
        f"- Route: {result.get('route', '-')}",
        f"- Summary: {result.get('summary', '-')}",
    ]
    output = str(result.get("output", "")).strip()
    attempts_raw = str(result.get("attempts", "")).strip()
    if attempts_raw:
        try:
            attempts = json.loads(attempts_raw)
        except Exception:
            attempts = []
        if isinstance(attempts, list) and attempts:
            lines.append("- Attempts:")
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                route = str(attempt.get("route", "-"))
                status = str(attempt.get("status", "-"))
                detail = str(attempt.get("detail", "-"))
                lines.append(f"  - {route}: {status} ({detail})")
    if output:
        lines.extend(
            [
                "",
                "### Output",
                output[:5000],
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _render_audit_report(
    *,
    today: str,
    nightly_deterministic: Mapping[str, Any],
    weekly_deterministic: Mapping[str, Any],
    nightly_model: Mapping[str, str],
    weekly_model: Mapping[str, str],
    underperforming_skills: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Contract Audit — {today}",
        "",
        "## Nightly Deterministic",
        f"- Status: {nightly_deterministic.get('status', '-')}",
        f"- Summary: {nightly_deterministic.get('summary', '-')}",
        "",
        _render_underperforming_skills_output(underperforming_skills),
        "",
        _render_check_lines(list(nightly_deterministic.get("checks", []))),
        "",
        "## Weekly Deterministic",
        f"- Status: {weekly_deterministic.get('status', '-')}",
        f"- Summary: {weekly_deterministic.get('summary', '-')}",
        "",
        _render_check_lines(list(weekly_deterministic.get("checks", []))),
        "",
        _render_model_section("Nightly Model Audit", nightly_model),
        _render_model_section("Weekly Model Audit", weekly_model),
    ]
    return "\n".join(lines).rstrip() + "\n"


async def run_contract_audits(
    *,
    today: str,
    task_results: Mapping[str, str],
    weekly_due: bool,
    weekly_result: str,
    maintenance_dir: Path,
    health_dir: Path,
) -> dict[str, Any]:
    underperforming_skills = query_underperforming_skills()
    nightly_deterministic = run_nightly_deterministic_checks(
        task_results,
        underperforming_skills=underperforming_skills,
    )
    weekly_deterministic = run_weekly_deterministic_checks(
        weekly_due=weekly_due,
        weekly_result=weekly_result,
    )

    context = {
        "today": today,
        "weekly_due": bool(weekly_due),
        "weekly_result": str(weekly_result),
        "task_results": dict(task_results),
        "underperforming_skills": underperforming_skills,
    }
    nightly_model = await run_model_audit(
        audit_pass="nightly",
        deterministic_result=nightly_deterministic,
        context=context,
    )
    weekly_context = {
        "today": today,
        "weekly_due": bool(weekly_due),
        "weekly_result": str(weekly_result),
    }
    weekly_model = await run_model_audit(
        audit_pass="weekly",
        deterministic_result=weekly_deterministic,
        context=weekly_context,
    )

    report_text = _render_audit_report(
        today=today,
        nightly_deterministic=nightly_deterministic,
        weekly_deterministic=weekly_deterministic,
        nightly_model=nightly_model,
        weekly_model=weekly_model,
        underperforming_skills=underperforming_skills,
    )

    maintenance_path = Path(maintenance_dir) / f"{today}-contract-audit.md"
    health_path = Path(health_dir) / f"{today}-contract-audit.md"
    artifact_error = ""
    try:
        maintenance_path.parent.mkdir(parents=True, exist_ok=True)
        health_path.parent.mkdir(parents=True, exist_ok=True)
        maintenance_path.write_text(report_text)
        health_path.write_text(report_text)
    except Exception as exc:
        artifact_error = str(exc)
        log.error("Failed to persist contract audit artifacts", exc_info=True)

    return {
        "nightly_deterministic": nightly_deterministic,
        "weekly_deterministic": weekly_deterministic,
        "nightly_model": nightly_model,
        "weekly_model": weekly_model,
        "underperforming_skills": underperforming_skills,
        "artifacts": {
            "maintenance": str(maintenance_path),
            "health": str(health_path),
            "error": artifact_error,
        },
        "markdown": report_text,
    }
