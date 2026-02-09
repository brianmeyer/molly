# Track C: Skill-Gap Telemetry and Crystallization Analytics

## Branch and scope
- Branch: `codex/track-c-skill-gaps-analytics`
- Worktree: `/private/tmp/molly-tracks/track-c`
- Goal: add robust skill-gap telemetry and analytics on the operational vectorstore DB path.

## What changed

### 1) Vectorstore schema and write path
- Added `skill_gaps` table creation in `memory/vectorstore.py` (same schema path as `tool_calls`/`skill_executions`, i.e. `config.MOLLYGRAPH_PATH`):
  - `id INTEGER PRIMARY KEY`
  - `user_message TEXT`
  - `tools_used TEXT`
  - `session_id TEXT`
  - `created_at TEXT`
  - `addressed INTEGER DEFAULT 0`
- Added `VectorStore.log_skill_gap(...)` helper to insert new rows with JSON `tools_used` payload.

### 2) Per-turn gap detection in `agent.py`
- Added request-scoped tool call attribution (`RequestApprovalState.turn_tool_calls`) and per-tool recording during tool logging.
- Added filtering helpers to exclude non-workflow/meta calls from skill-gap detection:
  - `routing:*`
  - `approval:*`
  - baseline retrieval family (`memory_search`, including namespaced suffix variants)
- Added post-turn gap insertion logic:
  - if **no matched skills** and **>=3 filtered workflow tool calls in that turn**, write one row to `skill_gaps`.
- Attribution is turn-scoped via `RequestApprovalState`, preventing historical leakage across turns.

### 3) New analytics module
- Added `skill_analytics.py` (nearest safe module path because top-level `skills.py` already occupies the `skills` module namespace).
- Implemented:
  - `get_skill_stats(skill_name)`
  - `get_underperforming_skills(min_invocations=5, max_success_rate=0.6)`
  - `get_skill_gap_clusters(days=30)` using deterministic keyword-overlap baseline clustering.

### 4) Test coverage
- Added `tests/test_skill_gap_analytics.py` covering:
  - schema declaration for `skill_gaps`
  - per-turn insertion behavior and non-leakage across turns
  - meta/baseline filtering
  - underperforming-skill selection
  - deterministic clustering behavior

### 5) Stabilization fix found during full pytest
- While running full suite, one existing timeout-coalescing test failed due cancelled inflight approval future behavior.
- Updated `approval.py` to shield pending futures in timeout waits and treat cancelled inflight waits as denials, restoring deterministic timeout coalescing behavior.

## Validation
- New tests: `tests/test_skill_gap_analytics.py` pass.
- Full suite pass (project venv): `149 passed`.
