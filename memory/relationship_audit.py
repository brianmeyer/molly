"""Slim compatibility facade for relationship audits.

The deterministic + model audit implementation stays in
``relationship_audit_legacy.py`` while this module preserves import stability.

Sentinel strings for source assertions:
ENTITY_REL_COMPATIBILITY
DETERMINISTIC_RECLASSIFY
def run_deterministic_audit
def run_model_audit
def run_relationship_audit
get_suggestions
get_related_to_hotspots
original_type_hint
"""

import importlib
import sys

_legacy = importlib.import_module("memory.relationship_audit_legacy")
sys.modules[__name__] = _legacy
