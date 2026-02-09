# Track E — Promotion Drift Gates

## Objective
Add a deterministic contract/drift validation gate before promotion so tool and skill artifacts stay in sync, with explicit audited override support.

## Patched Files
- `/Users/brianmeyer/.molly/workspace/tools/promote-tool.py`
- `/Users/brianmeyer/.molly/workspace/sandbox/tools/promote-tool.py`

## Deterministic Validation Contract
Validation gate id: `tool-skill-contract-v1`

Computed contract payload:
- `tool` (filename)
- `tool_hash` (MD5-12 of sandbox tool file)
- `skill_playbook` (manifest-linked skill or deterministic detection)
- `skill_hash` (MD5-12 of linked skill file)
- `contract_hash` (SHA256-16 over normalized JSON payload)

Core checks (`validation.checks`):
- `sandbox_tool_exists`
- `manifest_skill_matches_detected` (when both exist)
- `manifest_sandbox_path_matches` (when manifest entry exists)
- `manifest_production_path_matches` (when manifest entry exists)
- `skill_file_exists` (when a skill is linked)
- `skill_mentions_tool`
- `skill_playbook_present` (fails when no linked/detected skill)
- `contract_hash_stable` (when prior validation exists for same tool hash)

## Behavior Matrix
| Path | Condition | Exit | File Copy | Manifest Write | Output |
| --- | --- | --- | --- | --- | --- |
| `validate <tool>` | validation passes | `0` | No | No | validation report (`passed: true`) |
| `validate <tool>` | validation fails | `1` | No | No | validation report (`passed: false`) |
| `promote <tool>` | validation passes | `0` | Yes | Yes | promotion success + validation summary |
| `promote <tool>` | validation fails | `1` | No | No | blocked response + hint to use force override |
| `promote <tool> --force` | validation fails and reason missing/short | `1` | No | No | blocked response requiring `--override-reason` |
| `promote <tool> --force --override-reason <reason>` | validation fails | `0` | Yes | Yes | promotion success (`validation.result: override`) |
| `promote <tool> --dry-run` | validation passes | `0` | No | No | dry-run success (`would_promote: true`) |
| `promote <tool> --dry-run` | validation fails, no force | `1` | No | No | blocked dry-run response |
| `promote <tool> --dry-run --force --override-reason <reason>` | validation fails | `0` | No | No | dry-run success (`override_applied: true`) |
| `list` / `status` / `rollback` | normal usage | unchanged | unchanged | unchanged semantics | existing behavior preserved |

## Manifest Metadata (New)
Each promoted tool entry now records:
- `validation.gate_version`
- `validation.validated_at`
- `validation.result` (`pass` or `override`)
- `validation.passed` (raw gate result)
- `validation.overridden`
- `validation.override_reason` (when overridden)
- `validation.checks[]`
- `validation.contract` (`tool_hash`, `skill_playbook`, `skill_hash`, `manifest_skill_playbook`, `detected_skill_playbook`, `contract_hash`)
- `promotion_history[]` with per-promotion validation summary and optional override reason

## Verification Commands Run
- `python3 /Users/brianmeyer/.molly/workspace/tools/promote-tool.py list`
- `python3 /Users/brianmeyer/.molly/workspace/tools/promote-tool.py status react-research.py`
- `python3 /Users/brianmeyer/.molly/workspace/tools/promote-tool.py promote react-research.py --dry-run`
- `python3 /Users/brianmeyer/.molly/workspace/tools/promote-tool.py promote benchmark-research.py --dry-run` (blocked case)
- `python3 /Users/brianmeyer/.molly/workspace/tools/promote-tool.py promote benchmark-research.py --dry-run --force --override-reason "No skill yet; promote for staged integration"`

## Permission/Blocking Notes
Observed one sandbox write block while syncing mirrored script via symlinked workspace path:
- Failed: `cp /Users/brianmeyer/molly/workspace/tools/promote-tool.py /Users/brianmeyer/molly/workspace/sandbox/tools/promote-tool.py` (`Operation not permitted`)
- Resolved by rerunning with escalation approval; parity sync completed.
