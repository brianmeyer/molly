"""Compatibility shim for issue registry helpers.

Phase 3 keeps the full legacy implementation in ``issue_registry_legacy.py``
while runtime monitoring moves toward structured logging in ``monitoring.py``.
"""

import importlib
import sys

_legacy = importlib.import_module("memory.issue_registry_legacy")
sys.modules[__name__] = _legacy
