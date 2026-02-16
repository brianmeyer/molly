"""Backward-compatible re-exports from commitments and proposals modules.

All real code lives in commitments.py and proposals.py.
This module provides ``from automations import ...`` compatibility
for tests and any remaining runtime callers.
"""

from commitments import (  # noqa: F401
    Automation,
    AutomationEngine,
    _COMMITMENT_TRAILING_DUE_RE,
    _extract_commitment_title,
    _extract_due_datetime,
    _extract_due_datetime_with_llm,
    _titles_similar,
)

from proposals import propose_automation  # noqa: F401
