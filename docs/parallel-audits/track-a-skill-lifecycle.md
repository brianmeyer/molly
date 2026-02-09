# Track A Skill Lifecycle Audit

## Behavior Contract
- New skill proposals now flow through a unified lifecycle API: `SelfImprovementEngine.propose_skill_lifecycle(...)`.
- Trigger paths:
  - Explicit owner phrases: `make a skill for that`, `save that as a skill`.
  - Pattern-driven triggers: `propose_skill_from_patterns(...)` and `_propose_skill_updates_from_patterns(...)`.
  - Callable API for other tracks/components: `propose_skill_lifecycle(...)`.
- New skill draft flow:
  - Draft written to `workspace/skills/<slug>.md.pending`.
  - Approval summary includes: name, triggers, tools, steps, dry-run result, and YES/NO/EDIT instructions.
  - YES: atomic promote (`os.replace`) from `.md.pending` to `.md`, reload skills, lifecycle event logged, proposal event marked `completed`.
  - NO: pending file deleted, rejected lifecycle event logged (with optional reason), rejection fingerprint cooldown persisted for 30 days in `self_improvement_events`.
  - EDIT: structured edits supported for `trigger`, `steps`, `tools`, `guardrails`; pending markdown rewritten deterministically and approval cycle repeats.
- Skill iteration flow:
  - Existing skill updates use `workspace/skills/<slug>.md.pending-edit`.
  - Owner gets section-level diff summary.
  - YES: atomic replace of original `.md` with `.pending-edit`.
  - NO: `.pending-edit` deleted and rejection/cooldown event logged.
- Event logging:
  - Lifecycle/proposal/cooldown state is recorded via `self_improvement_events` only.
  - `skill_executions` is not used by lifecycle flow.

## Failure Modes
- No candidate pattern found for explicit owner phrase trigger:
  - Flow skips and sends an owner-facing note to run the workflow more times.
- Proposal has no steps:
  - Flow skips and returns `{"status": "skipped", "reason": "skill has no steps"}`.
- Existing/pending lifecycle files already in progress:
  - Flow skips with `pending skill lifecycle already exists`.
- Cooldown active for previously rejected fingerprint:
  - Flow skips with `rejection cooldown active`; no approval request sent.
- Invalid EDIT payload:
  - Flow keeps pending file and prompts for structured format (`trigger/tools/steps/guardrails`).
- Edit loop saturation (guardrail):
  - After 8 edit iterations without YES/NO completion, pending draft is removed and proposal marked `abandoned`.

## Test Evidence
- Targeted tests:
  - Command: `/Users/brianmeyer/molly/.venv/bin/python -m pytest tests/test_self_improve_skill_lifecycle.py tests/test_agent_approval_runtime.py tests/test_self_improve_suggestions.py -q`
  - Result: `28 passed in 0.89s`
- Full suite:
  - Command: `/Users/brianmeyer/molly/.venv/bin/python -m pytest -q`
  - Result: `151 passed in 1.79s`
- New coverage added in `tests/test_self_improve_skill_lifecycle.py`:
  - pending approve/no/edit flows
  - pending-edit approve/no flows
  - rejection cooldown persistence across engine instances
  - self-improvement event table assertions (and no lifecycle writes to `skill_executions`)
- Approval parsing extension coverage added in `tests/test_agent_approval_runtime.py`:
  - `EDIT trigger: ...` accepted by resolver
  - `NO: reason` returned as reasoned denial for lifecycle consumers
