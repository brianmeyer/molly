"""Thin facade re-exporting from monitoring.maintenance.

All real code lives in monitoring/maintenance.py.
"""

from monitoring.maintenance import *  # noqa: F401,F403
from monitoring.maintenance import (  # noqa: F401 — private names needed by tests
    _build_maintenance_report,
    _clear_checkpoint,
    _finalize_step,
    _get_maintenance_lock,
    _load_checkpoint,
    _prune_daily_logs,
    _run_blocklist_cleanup,
    _run_dedup_sweep,
    _run_opus_analysis,
    _run_orphan_cleanup,
    _run_self_ref_cleanup,
    _run_strength_decay,
    _save_checkpoint,
    _send_summary_to_owner,
)
