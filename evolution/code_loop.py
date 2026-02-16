"""Automated evolution loop — human-gated.

Runs during nightly maintenance (step 21).  Generates proposals
automatically, but every proposal goes through the full DGM pipeline
including human approval.

Key constraint: this loop PROPOSES only — never auto-commits.
"""
from __future__ import annotations

import logging
import time

from evolution._base import CODE_LOOP_MAX_PER_HOUR, is_code_loop_enabled
from evolution.db import get_connection

log = logging.getLogger(__name__)


def should_run() -> bool:
    """Check all preconditions for running the code loop.

    Returns False if:
    - Feature flag is OFF
    - Resource tier is RED
    - Rate limit exceeded (3+ proposals in last hour)
    - DGM is not idle (proposal already pending)
    """
    if not is_code_loop_enabled():
        log.debug("Code loop: FF is OFF — skipping")
        return False

    # Check resource tier
    try:
        from evolution.resources import get_resource_tier
        if get_resource_tier() == "RED":
            log.info("Code loop: resource tier RED — deferring")
            return False
    except Exception:
        pass

    # Check rate limit
    conn = get_connection()
    try:
        cutoff = time.time() - 3600
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM shadow_results WHERE timestamp > ?",
            (cutoff,),
        ).fetchone()
        if row and row["cnt"] >= CODE_LOOP_MAX_PER_HOUR:
            log.info("Code loop: rate limit reached (%d/%d) — skipping",
                     row["cnt"], CODE_LOOP_MAX_PER_HOUR)
            return False
    finally:
        conn.close()

    # Check DGM idle (max 1 pending proposal)
    try:
        from evolution.dgm import DGM
        dgm = DGM()
        current = dgm.get_current()
        if current.get("state") != "idle":
            log.info("Code loop: DGM not idle (state=%s) — skipping", current.get("state"))
            return False
    except Exception:
        log.warning("Code loop: failed to check DGM state", exc_info=True)
        return False

    return True


async def run_code_loop(improver=None) -> str:
    """Main code loop entry point.  Called by maintenance step 21.

    Returns a status string for the maintenance summary.
    """
    if not should_run():
        return "skipped (preconditions not met)"

    try:
        # Find weakest capability area
        target = select_improvement_target()
        if not target:
            return "skipped (no improvement target found)"

        # Select mutation
        from evolution.mutations import select_mutation
        mutation = select_mutation(context=target)

        # Retrieve relevant memories
        memories = []
        try:
            from evolution.memory import retrieve_similar
            memories = retrieve_similar(task_class=target.get("area", ""))
        except Exception:
            pass

        # Estimate risk
        risk = 0.0
        try:
            from evolution.causal import estimate_risk
            risk = estimate_risk(target.get("file", ""))
        except Exception:
            pass

        # Generate proposal and feed into DGM
        from evolution._base import Proposal
        from evolution.dgm import DGM

        proposal = Proposal(
            target_file=target.get("file", ""),
            search_anchor="",
            replace_block="",
            validation_tests=target.get("tests", []),
            description=f"Improve {target.get('area', 'unknown')}: {mutation.description}",
            rationale=f"Weakest area (avg reward {target.get('avg_reward', 0):.2f}, "
                      f"risk {risk:.2f}). {len(memories)} relevant memories.",
            mutation_operator=mutation.operator,
        )

        dgm = DGM()
        proposal_id = dgm.propose(proposal)
        return f"proposed {proposal_id} ({mutation.operator} on {target.get('area', '?')})"

    except Exception as exc:
        log.error("Code loop failed: %s", exc, exc_info=True)
        return f"error: {exc}"


def select_improvement_target() -> dict | None:
    """Analyze trajectories to find the weakest capability area.

    Returns ``{area, file, avg_reward, tests}`` or None if no data.
    """
    from evolution.trajectory import get_recent

    recent = get_recent(days=7)
    if not recent:
        return None

    # Group by arm/area and find lowest average reward
    areas: dict[str, list[float]] = {}
    for t in recent:
        area = t.get("arm_id") or "default"
        areas.setdefault(area, []).append(t.get("reward", 0))

    if not areas:
        return None

    weakest = min(areas, key=lambda a: sum(areas[a]) / len(areas[a]))
    avg = sum(areas[weakest]) / len(areas[weakest])

    return {"area": weakest, "file": "", "avg_reward": avg, "tests": []}


def get_loop_stats() -> dict:
    """Return code loop statistics for status reporting."""
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) as cnt FROM shadow_results").fetchone()
        cutoff = time.time() - 3600
        recent = conn.execute(
            "SELECT COUNT(*) as cnt FROM shadow_results WHERE timestamp > ?",
            (cutoff,),
        ).fetchone()
        return {
            "total_proposals": total["cnt"] if total else 0,
            "proposals_last_hour": recent["cnt"] if recent else 0,
            "rate_limit": CODE_LOOP_MAX_PER_HOUR,
            "enabled": is_code_loop_enabled(),
        }
    finally:
        conn.close()
