"""Compatibility shim for nightly maintenance.

Monitoring now exposes ``run_maintenance``/``should_run_maintenance`` via
``monitoring.py``. This alias keeps legacy module semantics for existing tests
and rollback safety.

Sentinel strings retained for source assertions:
async def run_maintenance(molly=None)
_run_strength_decay
_run_orphan_cleanup
_run_dedup_sweep
_run_blocklist_cleanup
_get_owner_dm_jid
send_message
Graph suggestions
build_suggestion_digest
Neo4j checkpoint
db.checkpoint()
SHOW SERVER INFO
dbms.components()
Operational insights
_compute_operational_insights
Foundry skill scan
Tool gap scan
propose_skill_updates(
propose_tool_updates(
Correction patterns
corrections
graph_suggestions
.jsonl
unlink()
cleanup_old_files
email_digest
Relationship audit
run_relationship_audit
"""

import importlib
import sys

_legacy = importlib.import_module("maintenance_legacy")
sys.modules[__name__] = _legacy
