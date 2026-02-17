"""Multi-model judge pool for pairwise evaluation.

Uses Gemini + Groq as primary judges (+ Kimi tie-breaker) to prevent
single-provider blind spots.  Gracefully degrades when API keys are
missing — falls back to available models.  If ALL fail, returns
``JudgeResult(passed=None)``.

Note: Main Molly brain uses Claude Agent SDK via Max subscription.
The judge pool intentionally uses *different* providers to avoid
self-evaluation bias.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field

from evolution.db import get_connection

log = logging.getLogger(__name__)


@dataclass
class JudgeScore:
    model: str
    functional: float
    efficiency: float
    final: float


@dataclass
class JudgeResult:
    passed: bool | None  # None = judges unavailable
    scores: list[JudgeScore] = field(default_factory=list)
    tie_broken: bool = False
    note: str = ""


def _available_judges() -> list[dict]:
    """Return judge configs for models whose API keys are present.

    Priority order: Gemini (primary), Groq (primary), Kimi (tie-breaker).
    """
    import config
    judges = []
    # Gemini Flash-Lite — primary judge #1
    key = getattr(config, "GEMINI_API_KEY", "")
    if key:
        judges.append({"model": "gemini-2.5-flash-lite", "provider": "gemini", "api_key": key})
    # Groq (GPT-OSS 120B) — primary judge #2
    key = getattr(config, "GROQ_API_KEY", "")
    if key:
        judges.append({"model": "openai/gpt-oss-120b", "provider": "groq", "api_key": key})
    # Kimi K2.5 (tie-breaker)
    key = getattr(config, "MOONSHOT_API_KEY", "")
    if key:
        judges.append({"model": "kimi-k2.5", "provider": "moonshot", "api_key": key})
    return judges


async def _call_judge(judge: dict, prompt: str) -> JudgeScore:
    """Call a single judge model and parse functional + efficiency scores.

    Gracefully degrades: returns 0-scores if the API call or parsing fails.
    """
    provider = judge.get("provider", "")
    model = judge.get("model", "")
    api_key = judge.get("api_key", "")
    if not api_key:
        log.debug("Judge %s skipped — no API key", model)
        return JudgeScore(model=model, functional=0.0, efficiency=0.0, final=0.0)

    system_msg = (
        "You are a code review judge. Evaluate the proposed change.\n"
        "Respond ONLY with a JSON object: {\"functional\": <0.0-1.0>, \"efficiency\": <0.0-1.0>}\n"
        "functional = correctness and reliability of the change.\n"
        "efficiency = performance, safety, and code quality.\n"
    )

    try:
        response_text = await _call_provider(provider, model, api_key, system_msg, prompt)
        return _parse_judge_response(model, response_text)
    except Exception:
        log.debug("Judge %s evaluation failed", model, exc_info=True)
        return JudgeScore(model=model, functional=0.0, efficiency=0.0, final=0.0)


async def _call_provider(
    provider: str, model: str, api_key: str, system_msg: str, user_msg: str,
) -> str:
    """Route to the correct provider API. Returns response text."""
    if provider == "gemini":
        import httpx

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {"contents": [{"parts": [{"text": f"{system_msg}\n\n{user_msg}"}]}]}
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return parts[0].get("text", "") if parts else ""

    if provider == "groq":
        import httpx

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model, "max_tokens": 256,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        }
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    if provider == "moonshot":
        import httpx

        import config as _cfg
        url = f"{_cfg.MOONSHOT_BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model, "max_tokens": 256,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        }
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    raise ValueError(f"Unknown judge provider: {provider}")


def _parse_judge_response(model: str, text: str) -> JudgeScore:
    """Extract functional/efficiency scores from judge response text."""
    import json as _json
    import re as _re

    # Try to find JSON in the response
    match = _re.search(r"\{[^}]*\"functional\"[^}]*\}", text, _re.DOTALL)
    if match:
        try:
            data = _json.loads(match.group())
            func = max(0.0, min(1.0, float(data.get("functional", 0))))
            eff = max(0.0, min(1.0, float(data.get("efficiency", 0))))
            final = func * 0.6 + eff * 0.4
            return JudgeScore(model=model, functional=func, efficiency=eff, final=final)
        except (ValueError, KeyError, TypeError):
            pass

    log.debug("Judge %s returned unparseable response: %s", model, text[:200])
    return JudgeScore(model=model, functional=0.0, efficiency=0.0, final=0.0)


async def evaluate(
    proposal_description: str,
    diff_text: str,
    test_results: str = "",
) -> JudgeResult:
    """Run multi-model judge evaluation.

    Protocol: 2 judges + optional tie-breaker if they disagree by >20%.
    Conservative: reject unless at least 2/3 approve.
    """
    judges = _available_judges()
    if not judges:
        log.warning("No judge models available — all API keys missing")
        return JudgeResult(passed=None, note="judges unavailable")

    # Select 2 different judges (+ optional tie-breaker)
    random.shuffle(judges)
    primary = judges[:min(2, len(judges))]

    prompt = (
        f"Evaluate this code change proposal:\n\n"
        f"Description: {proposal_description}\n\n"
        f"Diff:\n{diff_text[:3000]}\n\n"
        f"Test results: {test_results[:1000]}\n\n"
        f"Score functional correctness (0-1) and efficiency/safety (0-1)."
    )

    try:
        tasks = [_call_judge(j, prompt) for j in primary]
        scores = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception:
        log.error("Judge evaluation failed", exc_info=True)
        return JudgeResult(passed=None, note="judge evaluation error")

    # Filter out exceptions
    valid: list[JudgeScore] = [s for s in scores if isinstance(s, JudgeScore)]
    if not valid:
        return JudgeResult(passed=None, note="all judges failed")

    # Check for tie-breaker need
    tie_broken = False
    if len(valid) >= 2:
        diff = abs(valid[0].final - valid[1].final)
        if diff > 0.2 and len(judges) > 2:
            # Invoke tie-breaker
            tiebreaker = judges[2]
            try:
                tb_score = await _call_judge(tiebreaker, prompt)
                if isinstance(tb_score, JudgeScore):
                    valid.append(tb_score)
                    tie_broken = True
            except Exception:
                log.warning("Tie-breaker judge failed", exc_info=True)

    # Conservative: approve only if majority approve (final >= 0.5)
    approvals = sum(1 for s in valid if s.final >= 0.5)
    passed = approvals >= max(2, len(valid) // 2 + 1) if valid else None

    result = JudgeResult(passed=passed, scores=valid, tie_broken=tie_broken)

    # Persist to evolution.db
    _log_evaluations(valid, tie_broken)

    return result


def _log_evaluations(scores: list[JudgeScore], tie_broken: bool) -> None:
    """Store judge evaluations in evolution.db."""
    conn = get_connection()
    try:
        for s in scores:
            conn.execute(
                """INSERT INTO judge_evaluations
                   (judge_model, functional_score, efficiency_score, final_score, tie_broken)
                   VALUES (?, ?, ?, ?, ?)""",
                (s.model, s.functional, s.efficiency, s.final, int(tie_broken)),
            )
        conn.commit()
    finally:
        conn.close()
