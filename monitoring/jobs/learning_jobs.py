"""Learning Jobs — foundry skill scan, tool gap scan, correction patterns."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import config
import db_pool

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def run_foundry_skill_scan(improver) -> str:
    """Load foundry sequence signals and propose skill updates from qualifying patterns."""
    from foundry_adapter import load_foundry_sequence_signals

    signals = load_foundry_sequence_signals(days=7)
    patterns = [
        {
            "steps": list(sig.steps),
            "count": sig.count,
            "confidence": sig.success_rate,
            "name": key,
            "steps_text": key,
        }
        for key, sig in signals.items()
        if sig.count >= 3
    ]
    if patterns:
        skill_result = await improver.propose_skill_updates(patterns)
        return str(skill_result.get("status", "no candidates"))
    return "no qualifying patterns"


async def run_tool_gap_scan(improver) -> str:
    """Analyze 7-day failure data and propose tool updates for failing tools."""
    gap_result = await improver.propose_tool_updates(
        days=7, min_failures=5,
    )
    return str(gap_result.get("status", "no gaps"))


def run_correction_patterns() -> str:
    """Analyze 24h correction data — pattern counts + examples."""
    cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))

        cursor = conn.execute(
            "SELECT pattern, COUNT(*) as cnt FROM corrections "
            "WHERE created_at > ? GROUP BY pattern ORDER BY cnt DESC LIMIT 10",
            (cutoff_24h,),
        )
        pattern_rows = cursor.fetchall()

        cursor = conn.execute(
            "SELECT molly_output, user_correction, pattern FROM corrections "
            "WHERE created_at > ? ORDER BY created_at DESC LIMIT 5",
            (cutoff_24h,),
        )
        example_rows = cursor.fetchall()

        total_corrections = sum(row[1] for row in pattern_rows)
        if total_corrections == 0:
            return "0 corrections in last 24h"

        parts = [f"{total_corrections} correction(s) in last 24h"]
        for row in pattern_rows[:5]:
            parts.append(f"  '{row[0]}': {row[1]}x")
        if example_rows:
            parts.append("Recent examples:")
            for ex in example_rows[:3]:
                molly_out = (ex[0] or "")[:80]
                user_corr = (ex[1] or "")[:80]
                parts.append(f"  Molly: {molly_out}... -> User: {user_corr}...")
        return "\n".join(parts)
    except Exception as exc:
        return f"failed ({exc})"
    finally:
        if conn is not None:
            conn.close()
