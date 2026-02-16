"""Shared dataclasses and constants for the evolution package.

Follows the monitoring/_base.py pattern — cross-cutting types live here,
not in the orchestrator (dgm.py), to avoid circular imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Proposal:
    target_file: str
    search_anchor: str
    replace_block: str
    validation_tests: list[str]
    description: str
    rationale: str
    mutation_operator: str


@dataclass
class Mutation:
    operator: str
    area: str        # "prompts", "inference", "context", "code", "reasoning"
    description: str
    payload: dict


@dataclass
class PatchValidation:
    """Result of validating a proposed patch (tests, ruff, git apply)."""
    ok: bool
    reason: str = ""
    touched_files: list[str] | None = None
    changed_lines: int = 0


@dataclass
class EnforcedPatch:
    target_file: str
    diff_text: str
    strategy_used: str
    original_output: str


@dataclass
class ShadowEvalResult:
    proposal_id: str
    avg_reward_before: float
    avg_reward_after: float
    error_rate_before: float
    error_rate_after: float
    latency_p95_before: float
    latency_p95_after: float
    golden_pass_rate: float


@dataclass
class GuardViolation:
    guard_name: str
    law: str
    threshold: float
    actual: float
    severity: str


@dataclass
class GuardResult:
    passed: bool
    violations: list[GuardViolation] = field(default_factory=list)


@dataclass
class Experience:
    task_hash: str
    task_class: str
    reward: float
    confidence: float
    decay_weight: float
    content: dict
    embedding: bytes = b""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def is_code_loop_enabled() -> bool:
    """Read the single source of truth from config (lazy to avoid import cycles).

    Callers should use ``is_code_loop_enabled()`` rather than checking a bare
    constant, so that the runtime env-var toggle in ``config.CODE_LOOP_ENABLED``
    is always respected.
    """
    try:
        import config
        return bool(getattr(config, "CODE_LOOP_ENABLED", False))
    except Exception:
        return False

CODE_LOOP_MAX_PER_HOUR = 3
CODE_LOOP_TIMEOUT = 600
MEMORY_K = 3
MEMORY_REWARD_FLOOR = 0.6
MEMORY_DECAY_DAYS = 30
MEMORY_MIN_CONFIDENCE = 0.5
MIN_REWARD_DELTA = 0.02
SHADOW_SAMPLE_SIZE = 15
BASELINE_SAMPLES = 3
SHADOW_TIMEOUT = 180
