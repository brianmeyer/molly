# Track D: LLM Audit Layer (Nightly + Weekly)

## What shipped

- Added a contract-audit layer in `/Users/brianmeyer/molly/contract_audit.py`.
- Integrated into nightly maintenance in `/Users/brianmeyer/molly/maintenance.py`.
- Deterministic checks now run first for:
  - Nightly maintenance contract completeness/failures.
  - Weekly assessment schedule/execution contract.
- Optional model audit runs second (report layer only unless explicitly made blocking).

## Routing and config flags

New flags in `/Users/brianmeyer/molly/config.py`:

- `MOLLY_CONTRACT_AUDIT_LLM_ENABLED` (default: `false`)
- `MOLLY_CONTRACT_AUDIT_LLM_BLOCKING` (default: `false`)
- `MOLLY_CONTRACT_AUDIT_NIGHTLY_MODEL` (default: `kimi`) for nightly fast pass
- `MOLLY_CONTRACT_AUDIT_WEEKLY_MODEL` (default: `opus`) for weekly deep pass
- `MOLLY_CONTRACT_AUDIT_MODEL_TIMEOUT_SECONDS` (default: `45`)
- `MOLLY_CONTRACT_AUDIT_KIMI_MODEL` (default: `kimi-k2-0711-preview`)
- `MOLLY_CONTRACT_AUDIT_GEMINI_MODEL` (default: `gemini-2.0-flash`)
- `GEMINI_API_KEY` / `GEMINI_BASE_URL` for Gemini route support

Supported routes: `opus`, `kimi`, `gemini`.

## Safety defaults

- If LLM audit is disabled, model checks are recorded as `disabled by config`.
- If model/API is unavailable, results are recorded as `unavailable (...)`.
- Default mode is non-blocking (`MOLLY_CONTRACT_AUDIT_LLM_BLOCKING=false`), so maintenance still completes and writes reports.
- Deterministic contract failures can still fail/partial the run.

## Persistence

Audit output is written to both existing memory surfaces:

- `/Users/brianmeyer/molly/memory/maintenance/<date>-contract-audit.md`
- `/Users/brianmeyer/molly/memory/health/<date>-contract-audit.md`

Maintenance report rows now include deterministic/model contract audit outcomes and artifact status.

## Rollout recommendation

1. Week 1: keep defaults (`LLM_ENABLED=false`) and observe deterministic audit quality only.
2. Week 2: enable report-only nightly fast pass (`LLM_ENABLED=true`, nightly route `kimi`).
3. Week 3: enable weekly deep pass (`weekly route=opus` or `gemini`) and monitor false positives.
4. After stable operation: optionally set `LLM_BLOCKING=true` only if model uptime and response quality are consistently acceptable.
