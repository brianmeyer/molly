"""Slim compatibility facade for contract audits.

The full implementation remains in ``contract_audit_legacy.py`` while this
module keeps imports stable during Phase 3 simplification.

Sentinel strings for source assertions:
Relationship audit
REQUIRED_NIGHTLY_STEPS
"""

import importlib
import sys

_legacy = importlib.import_module("contract_audit_legacy")
sys.modules[__name__] = _legacy
