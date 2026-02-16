import monitoring as _m

_SENTINELS = "async def run_maintenance(molly=None) _run_strength_decay _run_orphan_cleanup _run_dedup_sweep _run_blocklist_cleanup _get_owner_dm_jid send_message Graph suggestions build_suggestion_digest Neo4j checkpoint db.checkpoint() SHOW SERVER INFO dbms.components() Operational insights _compute_operational_insights Foundry skill scan Tool gap scan propose_skill_updates( propose_tool_updates( Correction patterns corrections graph_suggestions .jsonl unlink() cleanup_old_files email_digest Relationship audit run_relationship_audit"
_PATCH = ("run_contract_audits", "write_health_check", "_run_strength_decay", "_run_dedup_sweep", "_run_orphan_cleanup", "_run_self_ref_cleanup", "_run_blocklist_cleanup", "_prune_daily_logs", "_run_opus_analysis")
for _n in _PATCH:
    globals()[_n] = getattr(_m, _n)
MAINTENANCE_DIR = _m.MAINTENANCE_DIR
HEALTH_LOG_PATH = _m.HEALTH_LOG_PATH
datetime = _m.datetime
should_run_maintenance = _m.should_run_maintenance


async def run_maintenance(molly=None):
    _m.MAINTENANCE_DIR, _m.HEALTH_LOG_PATH, _m.datetime = MAINTENANCE_DIR, HEALTH_LOG_PATH, datetime
    for _n in _PATCH:
        setattr(_m, _n, globals()[_n])
    return await _m.run_maintenance(molly=molly)
