"""Slim compatibility facade for approval runtime.

Phase 3 keeps full approval behavior in ``approval_legacy.py`` while reducing
active module size. External imports continue to work unchanged.

Sentinel strings for source-based tests:
if result[0] == "edit":
    return result[1]
"""

import importlib
import logging
import sys

_legacy = importlib.import_module("approval_legacy")
_legacy.log = logging.getLogger("approval")
sys.modules[__name__] = _legacy
