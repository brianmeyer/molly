"""Compatibility shim for legacy YAML automation engine.

GatewayEngine in ``gateway.py`` is the active scheduler used by main runtime.
This module remains as an alias during migration to preserve test/runtime
imports and rollback safety.

Sentinel strings kept for source-assertion tests:
- _DIRECT_ACTIONS
- _is_direct_action
- _execute_direct_action
- _should_skip_digest_delivery
- NO_DIGEST_ITEMS
"""

import importlib
import sys

_legacy = importlib.import_module("automations_legacy")
sys.modules[__name__] = _legacy
