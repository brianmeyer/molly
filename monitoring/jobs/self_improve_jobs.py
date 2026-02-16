"""Self-Improvement Jobs — memory optimization, GLiNER loop, weekly assessment."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def run_memory_optimization(improver) -> str:
    """Run Phase 7 memory optimization (entity consolidation, stale cleanup, contradiction detection)."""
    mem_opt = await improver.run_memory_optimization()
    return (
        f"consolidated={mem_opt.get('entity_consolidations', 0)}, "
        f"stale={mem_opt.get('stale_entities', 0)}, "
        f"contradictions={mem_opt.get('contradictions', 0)}"
    )


async def run_gliner_loop(improver) -> str:
    """Run the nightly GLiNER training/evaluation cycle."""
    gliner_cycle = await improver.run_gliner_nightly_cycle()
    return str(gliner_cycle.get("message") or gliner_cycle.get("status", "unknown"))


async def run_weekly_assessment(improver, now_local: datetime) -> tuple[bool, str]:
    """Run the weekly assessment if due. Returns (ran, result_description)."""
    import config

    weekly_due = _weekly_assessment_due_or_overdue(improver, now_local)
    if not weekly_due:
        return False, "not due"
    weekly_name = Path(str(await improver.run_weekly_assessment())).name
    return True, f"generated {weekly_name}"


def _weekly_assessment_due_or_overdue(improver, now_local: datetime) -> bool:
    """Check if a weekly assessment is due based on scheduled day/time."""
    import config

    target_date = _latest_scheduled_weekly_date(now_local)
    assessment_dir = config.WEEKLY_ASSESSMENT_DIR
    assessment_dir.mkdir(parents=True, exist_ok=True)
    expected = assessment_dir / f"{target_date.isoformat()}.md"
    return not expected.exists()


def _latest_scheduled_weekly_date(now_local: datetime) -> date:
    """Find the most recent scheduled weekly assessment date relative to *now_local*."""
    import config

    weekday_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    target_weekday = weekday_map.get(
        str(getattr(config, "WEEKLY_ASSESSMENT_DAY", "sunday")).lower(), 6
    )
    d = now_local.date()
    while d.weekday() != target_weekday:
        d -= timedelta(days=1)
    return d
