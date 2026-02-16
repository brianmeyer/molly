"""Compatibility shim: legacy health implementation.

Runtime call sites now import monitoring.py directly, but a wide test/runtime
surface still imports ``health``.  Keep this alias for one-week rollback safety.
"""

import importlib
import sys

_legacy = importlib.import_module("health_legacy")
sys.modules[__name__] = _legacy
