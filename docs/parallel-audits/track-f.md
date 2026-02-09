# Track F Rollout Guards

## Purpose
Track F adds pre-prod rollout guards for the skill/tool creation path without introducing hard enforcement by default.

## Default Safety Contract
- `MOLLY_TRACK_F_REPORT_ONLY=true` (default)
- `MOLLY_TRACK_F_ENFORCE_PARSER_COMPAT=false` (default)
- `MOLLY_TRACK_F_ENFORCE_SKILL_TELEMETRY=false` (default)
- `MOLLY_TRACK_F_ENFORCE_FOUNDRY_INGESTION=false` (default)
- `MOLLY_TRACK_F_ENFORCE_PROMOTION_DRIFT=false` (default)

When defaults are used, failed checks are reported as `yellow` and do not hard-block rollout.

## Pre-Prod Audit Command
Run:

```bash
/Users/brianmeyer/molly/.venv/bin/python /Users/brianmeyer/molly/scripts/run_preprod_readiness_audit.py
```

Sample output path:

`/Users/brianmeyer/molly/store/audits/track-f/track-f-preprod-20260209T171327245897Z.md`

Optional strict mode (fail process on red checks):

```bash
/Users/brianmeyer/molly/.venv/bin/python /Users/brianmeyer/molly/scripts/run_preprod_readiness_audit.py --strict
```

## Checks Included
- `trackf.parser_compatibility`
- `trackf.skill_telemetry_presence`
- `trackf.foundry_ingestion_health`
- `trackf.promotion_drift_status`

## Go/No-Go Checklist
- [ ] Audit report generated successfully (markdown output path printed by script)
- [ ] `trackf.parser_compatibility` is green
- [ ] `trackf.skill_telemetry_presence` is green, or yellow with accepted report-only posture
- [ ] `trackf.foundry_ingestion_health` is green, or yellow with accepted report-only posture
- [ ] `trackf.promotion_drift_status` is green, or yellow with accepted report-only posture
- [ ] If any check is red, confirm that red is intentionally enabled via enforcement flags before rollout

## Decision Guidance
- `GO`: no red checks, and any yellow checks are understood/accepted in report-only mode.
- `NO-GO`: red checks present under intentional hard-enforcement or unresolved parser compatibility failures.
