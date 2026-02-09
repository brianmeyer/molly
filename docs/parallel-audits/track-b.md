# Track B: Skill Observability

## Scope
- `/Users/brianmeyer/molly/agent.py`
- `/Users/brianmeyer/molly/health.py`
- `/Users/brianmeyer/molly/tests/test_agent_approval_runtime.py`
- `/Users/brianmeyer/molly/tests/test_health_skill_observability.py`

## Before
- Skill matching existed at runtime (`match_skills`) but no per-message execution outcome was captured.
- `skill_executions` table existed but had no guaranteed write path from normal `handle_message` flow.
- Health only checked coarse `skill_executions` presence via `quality.operational_tables` and did not expose:
  - low execution volume vs expected usage
  - drift toward direct `Bash` usage over skill-guided workflows

## After
- Matched skills are logged as execution telemetry on every request where at least one skill matches.
- Each matched skill writes one row to `skill_executions` with:
  - `skill_name`
  - `trigger` (`{source}:{user_message}`)
  - `outcome` (`success` or `failure`)
  - `edits_made` (failure detail when available)
- Logging is best-effort and non-fatal:
  - scheduled asynchronously (not on critical response path)
  - fully exception-guarded so telemetry failures never break user responses

## Health Metrics Design

### 1) Skill Execution Volume
- Check ID: `learning.skill_execution_volume`
- Window: `HEALTH_SKILL_WINDOW_DAYS` (default `7`)
- Data source: `skill_executions`
- Detail payload:
  - total executions
  - success count
  - failure count
  - unknown outcome count
- Status contract:
  - `red`: `executions == 0`
  - `yellow`: `executions < HEALTH_SKILL_LOW_WATERMARK` (default `3`) or `failure > success`
  - `green`: otherwise

### 2) Skill vs Direct Bash Ratio
- Check ID: `learning.skill_vs_direct_bash_ratio`
- Window: `HEALTH_SKILL_WINDOW_DAYS` (default `7`)
- Data sources:
  - `skill_executions` count
  - `tool_calls` where `tool_name='Bash'` (case-insensitive)
- Ratio: `skill_executions / direct_bash_calls`
- Status contract:
  - if `direct_bash_calls == 0`:
    - `yellow` when both are zero
    - `green` when skills > 0
  - otherwise:
    - `red` when ratio `< HEALTH_SKILL_BASH_RATIO_RED` (default `0.30`)
    - `yellow` when ratio `< HEALTH_SKILL_BASH_RATIO_YELLOW` (default `0.75`)
    - `green` otherwise

## Drift Visibility Outcome
- Skill routing is now directly measurable through per-skill outcome logs.
- Drift toward ad-hoc direct Bash usage is visible through an explicit ratio signal.
- Failures in observability plumbing degrade gracefully without user-facing impact.
