# Track B: Skill Hot Reload (Heartbeat Cadence)

## Scope
- `/Users/brianmeyer/molly/skills.py`
- `/Users/brianmeyer/molly/heartbeat.py`
- `/Users/brianmeyer/molly/tests/test_skills_hot_reload.py`
- `/Users/brianmeyer/molly/tests/test_heartbeat_skill_hot_reload.py`

## What Changed

### 1) Atomic hot-reload with rollback
- Added `skills.check_for_changes() -> bool`.
- Added snapshotting of `*.md` filenames + `st_mtime_ns` and diff detection (add/remove/modify).
- On detected changes, reload now:
  - builds a new skill map and trigger pattern list in temporary variables
  - swaps `_skills_cache`, `_trigger_patterns`, and snapshot together only after full success
  - keeps old cache/patterns intact on parse/compile failure
- Added `skills.get_reload_status()` for heartbeat observability.

### 2) Matcher safety for pending artifacts
- Added explicit guards for pending skill names/suffixes:
  - `.pending`
  - `.pending-edit`
- Pending files are excluded from load/compile/match activation.
- Matcher includes a defensive pending-name check even if stale state were ever present.

### 3) Heartbeat integration
- `heartbeat.run_heartbeat()` now calls `skills.check_for_changes()` once per heartbeat cycle.
- Added per-cycle log line:
  - hot-reload status (`skills.get_reload_status()`)
  - cumulative successful reload count in process (`_skill_reload_count`)
- Reload check is exception-guarded to avoid breaking heartbeat flow.

## Behavior Guarantees
- No partial module state activation:
  - no assignment to `_skills_cache` or `_trigger_patterns` occurs until rebuild fully succeeds
  - both structures are swapped together under lock
- Reload failures preserve prior matcher behavior.
- `.pending` and `.pending-edit` files cannot be activated by matcher.

## Validation
- Targeted tests:
  - `pytest -q tests/test_skills_hot_reload.py tests/test_heartbeat_skill_hot_reload.py`
  - Result: `4 passed`
- Additional regression checks:
  - `pytest -q tests/test_audit_fixes.py::TestDynamicTriggers tests/test_preference_signals.py::TestSurfacingMetadata tests/test_skills_hot_reload.py tests/test_heartbeat_skill_hot_reload.py`
  - Result: `14 passed`
- Full suite:
  - `pytest -q`
  - Blocked during collection by missing optional dependencies in environment:
    - `yaml` (`PyYAML`)
    - `neonize`
