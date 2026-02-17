"""Pure helper functions for workflow-pattern analysis.

Extracted to break the circular dependency chain:
    SkillLifecycle -> AutomationPatterns -> SkillGaps -> SkillLifecycle

These functions have zero state dependencies — they operate on plain dicts
and strings, making them safe to import from any module.
"""
from __future__ import annotations

import logging
from typing import Any

from evolution.infra import _LOW_VALUE_WORKFLOW_TOOL_NAMES

log = logging.getLogger(__name__)


def pattern_steps(pattern: dict[str, Any]) -> list[str]:
    """Extract step names from a workflow pattern dict."""
    raw_steps = pattern.get("steps", [])
    if isinstance(raw_steps, str):
        return [step.strip() for step in raw_steps.split("->") if step.strip()]
    if isinstance(raw_steps, list):
        return [str(step).strip() for step in raw_steps if str(step).strip()]
    return []


def is_low_value_workflow_tool(tool_name: str) -> bool:
    """Return True if *tool_name* is too generic for workflow pattern detection."""
    normalized = str(tool_name or "").strip()
    if not normalized:
        return True
    lowered = normalized.lower()
    return lowered in _LOW_VALUE_WORKFLOW_TOOL_NAMES or lowered.startswith("approval:")


def load_foundry_signals(days: int = 30) -> dict:
    """Load Foundry sequence signals, filtering low-value tools."""
    from foundry_adapter import load_foundry_sequence_signals

    try:
        return load_foundry_sequence_signals(
            days=days,
            is_low_value_tool=is_low_value_workflow_tool,
        )
    except Exception:
        log.debug("Failed to load Foundry observation signals", exc_info=True)
        return {}
