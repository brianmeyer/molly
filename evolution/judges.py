"""Multi-model judge pool for pairwise evaluation.

Uses up to 4 different models to prevent single-provider blind spots.
Gracefully degrades when API keys are missing — falls back to available
models.  If ALL fail, returns ``JudgeResult(passed=None)``.
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
    """Return judge configs for models whose API keys are present."""
    import config
    judges = []
    # Claude Haiku — always available (Anthropic key assumed present)
    judges.append({
        "model": "claude-haiku-4-20250514",
        "provider": "anthropic",
        "api_key": getattr(config, "ANTHROPIC_API_KEY", ""),
    })
    # Gemini Flash-Lite
    key = getattr(config, "GEMINI_API_KEY", "")
    if key:
        judges.append({"model": "gemini-2.5-flash-lite", "provider": "gemini", "api_key": key})
    # GPT-OSS 120B
    key = getattr(config, "OPENAI_API_KEY", "")
    if key:
        judges.append({"model": "openai/gpt-oss-120b", "provider": "openai", "api_key": key})
    # Kimi K2.5 (tie-breaker)
    key = getattr(config, "MOONSHOT_API_KEY", "")
    if key:
        judges.append({"model": "kimi-k2.5", "provider": "moonshot", "api_key": key})
    return judges


async def _call_judge(judge: dict, prompt: str) -> JudgeScore:
    """Call a single judge model and parse scores.

    This is a stub — actual API calls will be wired during Batch 8.
    Returns a placeholder score based on the model name for now.
    """
    # Placeholder: in production this makes an API call to the judge
    # and parses functional + efficiency scores from the response.
    log.debug("Judge %s evaluating (stub)", judge["model"])
    await asyncio.sleep(0.01)  # simulate async
    return JudgeScore(
        model=judge["model"],
        functional=0.0,
        efficiency=0.0,
        final=0.0,
    )


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
