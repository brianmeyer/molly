"""Backward-compatible re-exports from automation_triggers_legacy.

All trigger implementations live in automation_triggers_legacy.py.
This module provides ``from automation_triggers import ...`` compatibility.
"""

from automation_triggers_legacy import *  # noqa: F401,F403
from automation_triggers_legacy import (  # noqa: F401
    BaseTrigger,
    EmailTrigger,
    create_trigger,
)
