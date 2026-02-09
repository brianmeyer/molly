# Track A Audit Note

## Summary
Implemented skill trigger contract compatibility in `skills.py` by supporting both legacy markdown `## Trigger` parsing and YAML front matter `triggers:` lists.

## Risk Note
- Runtime flow risk: Low. Changes are isolated to skill trigger extraction and regex assembly; no execution path outside skill parsing/matching was modified.
- Compatibility risk: Low. Legacy quoted trigger phrases and backticked slash commands continue to work unchanged.
- Mixed-mode risk: Low. Deterministic precedence is now explicit (`YAML triggers` -> `legacy quoted phrases` -> `legacy commands`) with case/whitespace-insensitive de-duplication to prevent pattern drift.
