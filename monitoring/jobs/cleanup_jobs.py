"""Cleanup Jobs — daily log pruning including foundry observations."""
from __future__ import annotations

import logging
import shutil
from datetime import date, timedelta

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def prune_daily_logs() -> str:
    """Archive daily logs > 30 days, clean up JSONL files in graph_suggestions, email_digest, and foundry/observations."""
    memory_dir = config.WORKSPACE / "memory"
    archive_dir = memory_dir / "archive"
    cutoff = (date.today() - timedelta(days=30)).isoformat()
    archived = 0
    deleted = 0

    # Archive daily markdown logs
    for path in memory_dir.glob("????-??-??.md"):
        if path.stem < cutoff:
            try:
                archive_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(archive_dir / path.name))
                archived += 1
            except OSError:
                log.debug("Failed to archive %s", path, exc_info=True)

    # Cleanup JSONL files in graph_suggestions/ older than 30 days
    gs_dir = config.WORKSPACE / "memory" / "graph_suggestions"
    if gs_dir.is_dir():
        for path in gs_dir.glob("????-??-??.jsonl"):
            if path.stem < cutoff:
                try:
                    path.unlink()
                    deleted += 1
                except OSError:
                    log.debug("Failed to delete %s", path, exc_info=True)

    # Cleanup email digest queue files older than 3 days
    from memory.email_digest import cleanup_old_files as _cleanup_digest

    deleted += _cleanup_digest(keep_days=3)

    # Cleanup JSONL files in foundry/observations/ older than 30 days
    fo_dir = config.WORKSPACE / "foundry" / "observations"
    if fo_dir.is_dir():
        for path in fo_dir.glob("????-??-??.jsonl"):
            if path.stem < cutoff:
                try:
                    path.unlink()
                    deleted += 1
                except OSError:
                    log.debug("Failed to delete %s", path, exc_info=True)

    parts = []
    if archived:
        parts.append(f"archived {archived} daily log(s)")
    if deleted:
        parts.append(f"deleted {deleted} JSONL file(s)")
    return ", ".join(parts) if parts else "nothing to prune"
