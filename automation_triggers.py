"""Compatibility shim for legacy trigger classes.

Gateway scheduling now lives in ``gateway.py``.
"""

import importlib
import sys

_legacy = importlib.import_module("automation_triggers_legacy")
sys.modules[__name__] = _legacy
