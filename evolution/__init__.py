"""Evolution Engine — self-improvement infrastructure for Molly.

Public API facade.  Consumers should ``from evolution import X`` rather
than reaching into deep sub-modules.

Lazy imports are used so that this package can be imported without
pulling in every dependency at once (some modules depend on optional
packages like ``psutil`` or ``sqlite-vec``).
"""
from __future__ import annotations

from evolution._base import (           # always available — pure dataclasses
    EnforcedPatch,
    Experience,
    GuardResult,
    GuardViolation,
    Mutation,
    PatchValidation,
    Proposal,
    ShadowEvalResult,
    is_code_loop_enabled,
)

__all__ = [
    # dataclasses
    "EnforcedPatch",
    "Experience",
    "GuardResult",
    "GuardViolation",
    "Mutation",
    "PatchValidation",
    "Proposal",
    "ShadowEvalResult",
    # helpers
    "is_code_loop_enabled",
    # lazy-loaded — listed for documentation
    "DGM",
    "ThompsonBandit",
    "SelfImprovementEngine",
    "run_code_loop",
    "get_resource_tier",
    "pre_execution_hook",
    "post_execution_hook",
    "compute_reward",
    "estimate_risk",
    "generate_code",
    # composed services
    "EngineContext",
    "InfraService",
    "OwnerCommsService",
    "GraphOpsService",
    "StatsService",
    "AutomationPatternsService",
    "SkillLifecycleService",
    "SkillGapsService",
    "ToolGapsService",
    "GLiNERTrainingService",
    "QwenTrainingService",
]


def __getattr__(name: str):  # noqa: C901
    """Lazy-load heavy symbols on first access.

    This avoids importing the entire evolution stack when only _base
    dataclasses are needed.
    """
    if name == "DGM":
        from evolution.dgm import DGM
        return DGM

    if name == "ThompsonBandit":
        from evolution.bandit import ThompsonBandit
        return ThompsonBandit

    if name == "SelfImprovementEngine":
        from evolution.skills import SelfImprovementEngine
        return SelfImprovementEngine

    if name == "run_code_loop":
        from evolution.code_loop import run_code_loop
        return run_code_loop

    if name == "get_resource_tier":
        from evolution.resources import get_resource_tier
        return get_resource_tier

    if name == "pre_execution_hook":
        from evolution.hooks import pre_execution_hook
        return pre_execution_hook

    if name == "post_execution_hook":
        from evolution.hooks import post_execution_hook
        return post_execution_hook

    if name == "compute_reward":
        from evolution.rewards import compute_reward
        return compute_reward

    if name == "estimate_risk":
        from evolution.causal import estimate_risk
        return estimate_risk

    if name == "generate_code":
        from evolution.codegen import generate_code
        return generate_code

    # Composed services
    if name == "EngineContext":
        from evolution.context import EngineContext
        return EngineContext
    if name == "InfraService":
        from evolution.infra import InfraService
        return InfraService
    if name == "OwnerCommsService":
        from evolution.owner_comms import OwnerCommsService
        return OwnerCommsService
    if name == "GraphOpsService":
        from evolution.graph_ops import GraphOpsService
        return GraphOpsService
    if name == "StatsService":
        from evolution.stats import StatsService
        return StatsService
    if name == "AutomationPatternsService":
        from evolution.automation_patterns import AutomationPatternsService
        return AutomationPatternsService
    if name == "SkillLifecycleService":
        from evolution.skill_lifecycle import SkillLifecycleService
        return SkillLifecycleService
    if name == "SkillGapsService":
        from evolution.skill_gaps import SkillGapsService
        return SkillGapsService
    if name == "ToolGapsService":
        from evolution.tool_gaps import ToolGapsService
        return ToolGapsService
    if name == "GLiNERTrainingService":
        from evolution.gliner_training import GLiNERTrainingService
        return GLiNERTrainingService
    if name == "QwenTrainingService":
        from evolution.qwen_training import QwenTrainingService
        return QwenTrainingService

    raise AttributeError(f"module 'evolution' has no attribute {name!r}")
