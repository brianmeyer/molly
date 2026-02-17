"""Darwin Gödel Machine — self-modification state machine.

States: idle → proposing → patching → testing → shadow_eval →
        guard_check → awaiting_approval → committing / rolling_back

SQLite-persisted (survives restarts).  Single row in ``dgm_state``.
Every state transition wrapped in try/except — never leaves the machine
in a broken intermediate state.

Proposals include a sequential ID for the approval protocol
(e.g. ``DGM-047``).  Proposals in ``awaiting_approval`` auto-expire
after 72 hours.
"""
from __future__ import annotations

import json
import logging
import time

from evolution._base import GuardResult, Proposal, ShadowEvalResult
from evolution.db import ensure_schema, get_connection

log = logging.getLogger(__name__)

_VALID_STATES = frozenset({
    "idle", "proposing", "patching", "testing",
    "shadow_eval", "guard_check", "awaiting_approval",
    "committing", "rolling_back",
})

_TTL_SECONDS = 72 * 3600  # 72 hours


class DGM:
    """Darwin Gödel Machine — the evolution orchestrator."""

    def __init__(self) -> None:
        ensure_schema()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def get_current(self) -> dict:
        """Return the current DGM state row as a dict."""
        conn = get_connection()
        try:
            row = conn.execute("SELECT * FROM dgm_state WHERE id = 1").fetchone()
            if row is None:
                conn.execute("INSERT OR IGNORE INTO dgm_state (id, state) VALUES (1, 'idle')")
                conn.commit()
                return {"state": "idle"}
            state = dict(row)
            # Auto-expire stale proposals
            if (
                state.get("state") == "awaiting_approval"
                and state.get("updated_at")
                and time.time() - state["updated_at"] > _TTL_SECONDS
            ):
                self._transition("idle", note="auto-expired (72hr TTL)")
                state["state"] = "idle"
            return state
        finally:
            conn.close()

    def _transition(self, new_state: str, **updates: str) -> None:
        """Transition to a new state with optional column updates.

        Every transition is recorded in ``proposal_history`` for full audit
        trail — the single-row ``dgm_state`` only holds *current* state.
        """
        if new_state not in _VALID_STATES:
            raise ValueError(f"Invalid state: {new_state}")

        set_parts = ["state = ?", "updated_at = ?"]
        params: list = [new_state, time.time()]

        for col, val in updates.items():
            set_parts.append(f"{col} = ?")
            params.append(val)

        params.append(1)  # WHERE id = 1
        sql = f"UPDATE dgm_state SET {', '.join(set_parts)} WHERE id = ?"

        conn = get_connection()
        try:
            # Read old state for audit log
            old_row = conn.execute("SELECT state, proposal_json FROM dgm_state WHERE id = 1").fetchone()
            old_state = old_row["state"] if old_row else "unknown"
            proposal_json = old_row["proposal_json"] if old_row else None

            # Extract proposal_id from proposal_json if available
            proposal_id = None
            if proposal_json:
                try:
                    proposal_id = json.loads(proposal_json).get("id")
                except (json.JSONDecodeError, TypeError):
                    pass
            # Check if new proposal_json is being set in this transition
            if "proposal_json" in updates and updates["proposal_json"] != "null":
                try:
                    proposal_id = json.loads(updates["proposal_json"]).get("id") or proposal_id
                except (json.JSONDecodeError, TypeError):
                    pass

            conn.execute(sql, params)

            # Write audit log entry
            conn.execute(
                """INSERT INTO proposal_history
                   (proposal_id, old_state, new_state, note, proposal_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    proposal_id,
                    old_state,
                    new_state,
                    updates.get("note", ""),
                    updates.get("proposal_json", proposal_json),
                ),
            )
            conn.commit()
            log.info("DGM transition → %s%s", new_state,
                     f" ({updates.get('note', '')})" if 'note' in updates else "")
        finally:
            conn.close()

    def _next_proposal_id(self) -> str:
        """Generate sequential proposal ID like DGM-047."""
        conn = get_connection()
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM shadow_results").fetchone()
            n = (row["cnt"] if row else 0) + 1
            return f"DGM-{n:03d}"
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def propose(self, proposal: Proposal) -> str:
        """Start proposing stage.  Returns proposal_id."""
        current = self.get_current()
        if current.get("state") != "idle":
            raise RuntimeError(f"Cannot propose: DGM is in '{current.get('state')}' state")

        proposal_id = self._next_proposal_id()
        proposal_data = {
            "id": proposal_id,
            "target_file": proposal.target_file,
            "description": proposal.description,
            "rationale": proposal.rationale,
            "mutation_operator": proposal.mutation_operator,
            "search_anchor": proposal.search_anchor,
            "replace_block": proposal.replace_block,
            "validation_tests": proposal.validation_tests,
        }
        try:
            self._transition(
                "proposing",
                proposal_json=json.dumps(proposal_data),
                started_at=str(time.time()),
            )
        except Exception as exc:
            self._transition("rolling_back", note=f"propose error: {exc}")
            raise
        return proposal_id

    def enforce_patch(self, enforced_diff: str) -> None:
        """Move to patching stage with enforced diff."""
        try:
            self._transition("patching")
        except Exception as exc:
            self._transition("rolling_back", note=f"patch error: {exc}")
            raise

    def test(self, test_results: dict) -> None:
        """Record test results and advance to testing stage."""
        try:
            self._transition("testing", test_results_json=json.dumps(test_results))
        except Exception as exc:
            self._transition("rolling_back", note=f"test error: {exc}")
            raise

    def shadow_eval(self, result: ShadowEvalResult) -> None:
        """Record shadow evaluation results."""
        data = {
            "proposal_id": result.proposal_id,
            "avg_reward_before": result.avg_reward_before,
            "avg_reward_after": result.avg_reward_after,
            "golden_pass_rate": result.golden_pass_rate,
        }
        try:
            self._transition("shadow_eval", shadow_results_json=json.dumps(data))
        except Exception as exc:
            self._transition("rolling_back", note=f"shadow_eval error: {exc}")
            raise

    def guard_check(self, result: GuardResult) -> None:
        """Run guard check.  Auto-rejects on critical failures."""
        data = {
            "passed": result.passed,
            "violations": [
                {"name": v.guard_name, "law": v.law, "severity": v.severity}
                for v in result.violations
            ],
        }
        try:
            if not result.passed:
                log.warning("Guard check FAILED — auto-rejecting")
                self._transition("rolling_back", guard_results_json=json.dumps(data))
                return
            self._transition("awaiting_approval", guard_results_json=json.dumps(data))
        except Exception as exc:
            self._transition("rolling_back", note=f"guard_check error: {exc}")
            raise

    def approve(self, proposal_id: str | None = None) -> None:
        """Approve the current proposal.  Transitions to committing."""
        current = self.get_current()
        if current.get("state") != "awaiting_approval":
            raise RuntimeError(f"Cannot approve: DGM is in '{current.get('state')}' state")

        # Verify proposal_id if provided
        if proposal_id:
            pdata = json.loads(current.get("proposal_json") or "{}")
            if pdata.get("id") != proposal_id:
                raise ValueError(f"Proposal ID mismatch: expected {pdata.get('id')}, got {proposal_id}")

        try:
            self._transition("committing")
        except Exception as exc:
            self._transition("rolling_back", note=f"approve error: {exc}")
            raise

    def reject(self, proposal_id: str | None = None) -> None:
        """Reject the current proposal.  Returns to idle."""
        current = self.get_current()
        if current.get("state") != "awaiting_approval":
            raise RuntimeError(f"Cannot reject: DGM is in '{current.get('state')}' state")

        try:
            self._transition("idle", proposal_json="null", note="rejected")
        except Exception as exc:
            self._transition("rolling_back", note=f"reject error: {exc}")
            raise

    def commit(self, git_commit: str = "") -> None:
        """Finalize commit and return to idle."""
        try:
            self._transition("idle", git_branch="", proposal_json="null", note=f"committed {git_commit}")
        except Exception as exc:
            self._transition("rolling_back", note=f"commit error: {exc}")
            raise

    def rollback(self) -> None:
        """Roll back current operation and return to idle."""
        self._transition("idle", proposal_json="null", note="rolled back")

    def reset_state(self) -> None:
        """Emergency reset: wipe all DGM state and return to idle.

        Use only when the state machine is corrupted or stuck.
        """
        conn = get_connection()
        try:
            conn.execute(
                """UPDATE dgm_state SET
                   state='idle', proposal_json=NULL, test_results_json=NULL,
                   shadow_results_json=NULL, guard_results_json=NULL,
                   git_branch=NULL, started_at=NULL, updated_at=?
                   WHERE id=1""",
                (time.time(),),
            )
            conn.commit()
            log.warning("DGM state RESET to idle (emergency)")
        finally:
            conn.close()
