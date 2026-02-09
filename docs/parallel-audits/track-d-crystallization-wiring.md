# Track D: Crystallization Wiring

## Scope
- Wired weekly self-improvement to consume analytics-style skill-gap clusters.
- Wired nightly contract audit to include deterministic underperforming-skill checks.
- Preserved report-only defaults (no new hard blocking unless explicitly configured).

## Weekly Self-Improve Wiring (`/Users/brianmeyer/molly/self_improve.py`)
- Added `_propose_skill_updates_from_gap_clusters()` and integrated it into `run_weekly_assessment()`.
- Added cluster analytics ingestion via `_query_skill_gap_clusters()` from `skill_gap_clusters` / `skill_gaps`.
- Enforced proposal eligibility rules:
  - minimum cluster size (`>=3` default)
  - no active cooldown window
  - no recent duplicate lifecycle event
- Added Track A hook wiring (`_resolve_track_a_skill_hook`, `_invoke_track_a_skill_hook`) for pending skill drafts.
- Logged lifecycle transitions to `self_improvement_events` (pending/activated), with direct SQLite fallback if vectorstore logging is unavailable.
- Marked skill gaps addressed/proposed/activated with schema-tolerant updates in `_mark_skill_gap_rows_addressed()`.

## Nightly Contract Audit Wiring (`/Users/brianmeyer/molly/contract_audit.py`)
- Added deterministic analytics query `query_underperforming_skills()`:
  - flags skills where `invocations >= 5` and `success_rate < 0.60`
  - supports `skill_analytics` table, with fallback aggregation from `skill_executions`
- Extended nightly deterministic checks with `nightly.underperforming_skills`.
- Added deterministic check output rendering section to markdown report with explicit threshold and summary table.
- Kept model-route fallback behavior intact (`disabled`/`unavailable`/`error` handling unchanged).

## Safety Defaults
- Underperforming-skill finding is `warn` by default.
- Hard fail is only enabled when `CONTRACT_AUDIT_ENFORCE_UNDERPERFORMING_SKILLS=true`.
- No new blocking behavior added by default.

## Tests
- Added weekly proposal trigger coverage in `/Users/brianmeyer/molly/tests/test_self_improve_suggestions.py`.
- Added nightly underperforming deterministic rendering + model fallback coverage in `/Users/brianmeyer/molly/tests/test_contract_audit_config.py`.
