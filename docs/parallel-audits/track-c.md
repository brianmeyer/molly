# Track C: Foundry Pattern Signal Integration

## Change Summary
- Added a Foundry observations adapter (`foundry_adapter.py`) that reads JSONL observation files from `/Users/brianmeyer/.molly/workspace/foundry/observations`.
- Updated workflow pattern detection in `self_improve.py` to merge Foundry sequence support with existing `tool_calls` detection.
- Kept the existing `tool_calls` path as fallback when Foundry data is missing, unreadable, malformed, or stale.
- Filtered low-value primitive steps from pattern detection: `Write`, `Edit`, `Bash`, and `approval:*`.
- Did not modify approval, deploy, restart, or rollback gate logic.

## Failure Modes And Mitigations
1. Failure mode: Foundry observations directory is missing or unreadable.
Mitigation: Adapter returns no Foundry signals and pattern detection continues with `tool_calls` only.

2. Failure mode: Individual observation lines are malformed JSON.
Mitigation: Adapter skips malformed lines and continues parsing the rest of the file.

3. Failure mode: Observation timestamps are invalid or out of retention window.
Mitigation: Adapter ignores invalid/old records and only uses recent entries within the requested day window.

4. Failure mode: Primitive/noise-heavy sequences dominate suggestions.
Mitigation: Low-value tool steps (`Write`, `Edit`, `Bash`, `approval:*`) are filtered before sequence aggregation.

5. Failure mode: Foundry introduces false positives that bypass existing controls.
Mitigation: Foundry only affects ranking/confidence and occurrence aggregation; proposal/deploy paths remain owner-approval gated and rollback-protected.

6. Failure mode: Foundry over-weights unsafe deploy automation recommendations.
Mitigation: No deploy/rollback safety code was changed; all existing approval prompts, deployment gates, and post-deploy health rollback checks are preserved.

## Validation Checklist
- `python3 -m pytest -q /Users/brianmeyer/molly/tests/test_self_improve_suggestions.py`
- `python3 -m pytest -q /Users/brianmeyer/molly/tests/test_self_improve_restart_flow.py`
- `python3 -m pytest -q /Users/brianmeyer/molly/tests/test_foundry_adapter.py`
