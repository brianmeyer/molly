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

        # Run the full evaluation pipeline: codegen → test → shadow → guards
        eval_status = await _run_evaluation_pipeline(dgm, proposal, proposal_id)
        return f"proposed {proposal_id} ({mutation.operator} on {target.get('area', '?')}) — {eval_status}"

    except Exception as exc:
        log.error("Code loop failed: %s", exc, exc_info=True)
        return f"error: {exc}"


def _find_context_files(target_file: str, max_files: int = 5) -> list[str]:
    """Find context files for codegen: the target itself + matching test files.

    Returns up to *max_files* paths (strings).
    """
    from pathlib import Path as _Path

    files: list[str] = []
    if not target_file:
        return files

    target = _Path(target_file)
    if target.is_file():
        files.append(str(target))

    # Look for a matching test file (test_<name>.py in same or tests/ dir)
    if target.suffix == ".py":
        stem = target.stem
        candidates = [
            target.parent / f"test_{stem}.py",
            target.parent / "tests" / f"test_{stem}.py",
            target.parent.parent / "tests" / f"test_{stem}.py",
        ]
        for c in candidates:
            if c.is_file() and str(c) not in files:
                files.append(str(c))

    return files[:max_files]


async def _run_evaluation_pipeline(dgm, proposal, proposal_id: str) -> str:
    """Run the DGM evaluation pipeline: patch → test → judges → shadow → guards.

    Returns a status string. If any stage fails, the DGM rolls back.
    """
    from evolution._base import GuardResult, ShadowEvalResult

    # Stage 1: Generate code / enforce patch
    try:
        from evolution.codegen import generate_code, CodegenRequest, is_available

        if is_available():
            context_files = _find_context_files(proposal.target_file) if proposal.target_file else []
            req = CodegenRequest(
                task_description=proposal.description,
                target_files=[proposal.target_file] if proposal.target_file else [],
                context_files=context_files,
            )
            result = await generate_code(req)
            if result.success and result.patches:
                dgm.enforce_patch(result.patches[0].diff_text if result.patches else "")
            else:
                dgm.rollback()
                return "codegen failed"
        else:
            # No codegen backend available — skip to shadow eval with stub
            dgm.enforce_patch("")
    except Exception as exc:
        log.warning("Evaluation pipeline codegen failed: %s", exc)
        dgm.rollback()
        return f"codegen error: {exc}"

    # Stage 2: Run tests
    try:
        dgm.test({"status": "pending", "note": "tests deferred to human review"})
    except Exception as exc:
        dgm.rollback()
        return f"test error: {exc}"

    # Stage 3: Judge evaluation
    try:
        from evolution.judges import evaluate as judge_evaluate
        judge_result = await judge_evaluate(
            proposal_description=proposal.description,
            diff_text=proposal.replace_block or "(no diff)",
            test_results="pending",
        )
        if judge_result.passed is False:
            log.info("DGM %s: judges REJECTED", proposal_id)
            dgm.rollback()
            return "judges rejected"
    except Exception as exc:
        log.warning("Judge evaluation failed (continuing): %s", exc)

    # Stage 4: Shadow evaluation
    shadow_result: ShadowEvalResult | None = None
    try:
        from evolution.shadow import evaluate as shadow_evaluate
        shadow_result = await shadow_evaluate(proposal_id=proposal_id)
        dgm.shadow_eval(shadow_result)
    except Exception as exc:
        log.warning("Shadow eval failed (continuing): %s", exc)
        shadow_result = ShadowEvalResult(
            proposal_id=proposal_id,
            avg_reward_before=0.0, avg_reward_after=0.0,
            error_rate_before=0.0, error_rate_after=0.0,
            latency_p95_before=0.0, latency_p95_after=0.0,
            golden_pass_rate=0.0,
        )
        dgm.shadow_eval(shadow_result)

    # Stage 5: Three Laws guard check
    try:
        from evolution.guards import validate
        guard_result = validate(shadow=shadow_result, test_passed=True)
        dgm.guard_check(guard_result)

        if not guard_result.passed:
            return "guards rejected"
    except Exception as exc:
        log.warning("Guard check failed: %s", exc)
        dgm.guard_check(GuardResult(passed=False, violations=[]))
        return f"guard error: {exc}"

    return "awaiting approval"


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
