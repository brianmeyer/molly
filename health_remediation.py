"""Compatibility shim for remediation routing.

The active export now lives in ``monitoring.py`` while detailed logic remains
in ``health_remediation_legacy.py`` for rollback.
"""

import importlib
import sys

_legacy = importlib.import_module("health_remediation_legacy")
sys.modules[__name__] = _legacy
