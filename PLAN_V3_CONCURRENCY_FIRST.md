# Molly Concurrency-First Stability + Evolution Plan (v3)

Date: 2026-02-15
Owner: Brian + Molly
Scope: Production roadmap for `/Users/brianmeyer/molly` that preserves existing behavior while fixing serialization bottlenecks and enabling safe autonomy/self-evolution.

## 1) Executive Summary

This plan prioritizes the concurrency rewrite first because current Claude SDK constraints serialize too much work and block autonomy.

Core strategy:
1. Add an orchestrator control plane and execution lanes without deleting current subsystems.
2. Preserve current behavior via compatibility adapters, feature flags, shadow runs, and parity gates.
3. Close open bugs/gaps from `master-issues.md` in a phased order, not ad hoc.
4. Keep self-evolution, but run it in a safe lane with hard promotion gates.

Strategic stance:
- This is an \"adult in the room\" migration: strangler pattern first, not a big-bang async rewrite.
- The goal is to prevent rewrite hell (two broken systems, no safe rollback) while unblocking concurrency.

## 2) Non-Negotiable Constraints (Do Not Break)

1. No hard-cut rewrite in early phases.
2. No large deletions (`automations.py`, `health.py`, `maintenance.py`, `self_improve.py`, `issue_registry.py`) until parity is proven.
3. Per-thread response ordering must remain deterministic.
4. Existing approval semantics remain active for high-impact actions.
5. Every migration step is behind a feature flag and has rollback.
6. Existing tests must keep passing; new parity tests are required for each migrated surface.

## 3) Target Architecture (Concurrency-First)

### 3.1 Control Plane

- New `orchestrator/` package manages routing, scheduling, retries, idempotency, and lifecycle.
- Existing modules become adapters called by orchestrator (not immediately replaced).
- Today, much of this logic is implicit in `main.py` request flow; this plan moves it into deterministic code paths.

### 3.2 Execution Lanes

1. Interactive lane:
- User-facing replies.
- Strict per-thread ordering.
- Low latency budget.

2. Background lane:
- Maintenance, automations, triage follow-ups, learning, self-improvement evaluations.
- Parallel workers with bounded concurrency.

### 3.3 Queue + Locking Model

- Per-thread mailbox lock for conversational ordering only.
- Global work queue for independent jobs.
- Idempotency keys for all side-effecting actions (`calendar_create`, `tasks_create`, notifications, proposals).
- Retry policy with dead-letter queue and operator visibility.
- Priority preemption path: owner `stop/cancel` and urgent alert events are handled ahead of long-running background work.

### 3.4 Data Consistency + Graph Write Policy

- Single-writer policy for graph mutations: all Neo4j writes route through a serialized write queue.
- Read/write split: interactive reads can remain parallel, but writes are ordered and auditable.
- Each graph mutation records operation id, source lane, and timestamp for replay/debugging.
- If a write fails after partial downstream effects, compensating action is required before retry.

### 3.5 OpenClaw Patterns Adopted (Now)

- Heartbeat vs cron/job split.
- Isolation boundaries for autonomous jobs.
- Tool policy boundaries by worker role.
- Model failover chain (existing provider stack).

## 4) Migration Strategy

### 4.1 Strangler Pattern

- Keep old entry points active.
- Route selected flows through orchestrator behind feature flags.
- Compare old vs new behavior in shadow mode before cutover.

### 4.2 Feature Flags

- `ORCHESTRATOR_ENABLED`
- `ORCHESTRATOR_INTERACTIVE_ENABLED`
- `ORCHESTRATOR_BACKGROUND_ENABLED`
- `ORCHESTRATOR_SHADOW_MODE`
- `AUTO_ACTIONS_FROM_TRIAGE_ENABLED`
- `SELF_EVOLUTION_SAFE_LANE_ENABLED`

Flag governance:
- Every flag must have an owner, default state, and planned removal milestone.
- No permanent \"temporary\" flags; stale flags are treated as tech debt bugs.

### 4.3 Go/No-Go Gates

Each phase ships only if:
1. Regression tests pass.
2. Parity scenarios pass with functional assertions (effects/outcomes), not brittle wording checks.
3. No increase in critical incidents for 72h.
4. Rollback path tested.

### 4.4 Flag Retirement Plan

- Add a cleanup checkpoint after rollout stabilization to remove:
1. obsolete feature flags,
2. dead adapter paths,
3. shadow-mode-only code.
- Removal requires proof that fallback path is no longer needed in production.

## 5) Detailed Phases

## Phase 0 - Baseline + Safety Net (Week 1)

Goal: Freeze behavior and define parity before architecture changes.

Deliverables:
1. Baseline performance and reliability snapshot:
- Message latency percentiles.
- Queue depth.
- Tool success rates.
- Human intervention rate.

2. Parity test suite from real scenarios:
- Email event detection + calendar creation.
- WhatsApp group passive processing.
- Digest queue behavior.
- Approval flows.
- Followups/commitments pipeline.
- Assertions are functional: state changes, side effects, persisted records, and queue outcomes.
- Avoid conversational string matching except for strict protocol outputs (JSON/tool contracts).

3. Invariant document:
- What must remain unchanged during migration.

Acceptance criteria:
1. Baseline report committed.
2. Parity tests runnable in CI/local.
3. Incident rollback checklist documented.

## Phase 1 - Orchestrator Foundation (Weeks 2-3)

Goal: Remove serialization bottleneck while preserving behavior.

Deliverables:
1. `orchestrator/` skeleton:
- Message intake.
- Per-thread scheduler.
- Background job scheduler.

2. Scoped locking:
- Lock only the user response path per chat thread.
- Run non-dependent background tasks in parallel.
- Ensure an interactive message can preempt waiting behind background workloads.

3. Durable queue primitives:
- Pending/running/completed/failed states.
- Retry with exponential backoff.
- Dead-letter storage.

4. Idempotency framework:
- Action keys for side-effecting tool calls.
- Duplicate suppression across retries/restarts.

Acceptance criteria:
1. Parallel background jobs run with no conversational ordering regressions.
2. No duplicate side effects under forced retries.
3. Load test shows throughput gain vs baseline.
4. `stop/cancel` class messages are acknowledged within SLA even during heavy background load.

## Phase 2 - Critical Bug Closure on New Runtime (Weeks 3-5)

Goal: Fix user-visible breakage and routing gaps first.

Deliverables:
1. Messaging + event automation fixes:
- BUG-28: WhatsApp calendar routing complete for all modes.
- BUG-17: DIGEST_ONLY wired into queue.
- BUG-05: End-of-day wrap reads `followups.md` explicitly.

2. Approval and nightly proposal reliability:
- BUG-06/07/18: proposal queue persisted for morning review; no 2 AM timeout loss.
- BUG-22: approval/metrics instrumentation corrected; distinguish policy denials from tool failures.

3. Maintenance/reporting durability:
- BUG-24: incremental report writes and resume checkpoints.
- BUG-20: standalone GLiNER deploy script with validation.
- BUG-25: GLiNER cooldown policy adjusted + override path.

4. Data/model integrity:
- BUG-11: extractor schema externalization + merge on load.
- BUG-10: relationship audit can auto-fix/reclassify with audit trail.

Acceptance criteria:
1. All listed bugs have reproducible passing tests or runbooks.
2. No lost nightly proposals for 7 consecutive runs.
3. Digest and end-of-day behavior verified in live dry runs.

## Phase 3 - Concurrency Expansion + Operational Gaps (Weeks 5-7)

Goal: Scale concurrency and close medium-priority system gaps.

Deliverables:
1. GAP-11 completion:
- Parallelize independent retrieval/extraction paths under orchestrator.
- Remove avoidable serial waits.

2. Neo4j maintenance fixes:
- BUG-09/GAP-02: CE-safe tx log retention policy and verification checks.
- Enforce graph single-writer queue in production paths.

3. Observability:
- GAP-03: operational metrics consumers (dashboards + anomaly checks).
- SLO alerts: latency, failure, intervention trends.

4. Preference and behavior learning signals:
- GAP-01: expand preference signal capture beyond narrow cases.

5. Restart operations:
- GAP-13: agent-accessible safe restart path with guardrails.

Acceptance criteria:
1. Queue and worker metrics visible daily.
2. Neo4j tx logs remain bounded for 7 days.
3. Restart workflow works without manual shell access.
4. No graph consistency incidents under concurrent read/write stress testing.

## Phase 4 - Foundry + Triage Learning Maturity (Weeks 7-9)

Goal: Improve adaptation quality without unsafe self-rewrites.

Deliverables:
1. Foundry lifecycle completion:
- GAP-12: full OBSERVE/LEARN/CRYSTALLIZE flow.
- GAP-10: increase observation capture volume and quality controls.

2. Triage learning pipeline:
- GAP-15: triage fine-tuning data accumulation/training/eval pipeline.
- GAP-16: logit-constrained classification path.
- Keep existing context enrichment and validate drift.

3. Graph suggestion pipeline hardening:
- GAP-04: ensure real-time hooks + nightly consumption stay healthy.

Acceptance criteria:
1. Foundry cycles produce measurable accepted improvements.
2. Triage latency and precision improve against baseline.
3. No regression in false-positive/false-negative bounds.

## Phase 5 - Safe Self-Evolution Lane (Weeks 9-12)

Goal: Add self-evolution safely, separate from runtime.

Deliverables:
1. Isolated improvement lane:
- Proposals execute in sandbox branch/workspace.
- Runtime lane remains stable until promotion.

2. Evaluator registry + promotion gates:
- Required test suites, policy checks, and benchmark thresholds.
- Human approval required for production code changes.

3. Policy-driven promotion:
- Shadow run -> limited rollout -> full rollout.
- Automatic rollback on SLO breach.

4. DGM-inspired state machine (bounded):
- Keep explicit states and persistence.
- No unbounded self-modification in production runtime.

Acceptance criteria:
1. 100% of proposed code edits pass evaluator gates before promotion.
2. Rollback tested and proven.
3. No direct writes to production runtime from autonomous lane.

## Phase 6 - Product Extensions (After Stability)

Goal: Add non-core capabilities once concurrency + reliability are stable.

Deliverables (priority order):
1. BUG-16: iMessage search reliability investigation/fix.
2. BUG-21/GAP-09: browser automation hardening for JS-heavy + anti-bot sites.
3. GAP-07: Google Maps saved-list integration.
4. GAP-08: dual code-agent workflow formalization.
5. GAP-17: location/travel timezone awareness.

Acceptance criteria:
1. Each extension has dedicated tests, rate limits, and fallback behavior.
2. No regression of core assistant reliability metrics.

## Phase 7 - Cleanup + Simplification (After Stable Cutover)

Goal: Remove migration scaffolding and prevent long-term complexity drift.

Deliverables:
1. Retire rollout flags no longer needed.
2. Remove dead adapter/fallback paths after proven stability windows.
3. Consolidate duplicated orchestration logic into single canonical path.
4. Update runbooks/docs to reflect post-migration architecture only.

Acceptance criteria:
1. Flag inventory reduced to steady-state operational toggles only.
2. No dual-path code remains for migrated surfaces.
3. On-call/debug playbooks reference one active runtime architecture.

## 6) What We Will Not Do Early

1. Big-bang file deletions for line-count reduction.
2. Removing LLM audit paths before equivalent quality checks exist.
3. Full autonomous self-rewriting in production.
4. Architecture success criteria based on `wc -l` targets.

## 7) Master Issues Coverage Matrix

Legend: `P` = planned phase, `V` = verify already-fixed behavior

### Open Bugs

- BUG-05 -> Phase 2
- BUG-06 -> Phase 2
- BUG-07 -> Phase 2
- BUG-09 -> Phase 3
- BUG-10 -> Phase 2
- BUG-11 -> Phase 2
- BUG-15 -> Phase 2 (V: confirm current AUTO behavior remains intact)
- BUG-16 -> Phase 6
- BUG-17 -> Phase 2
- BUG-18 -> Phase 2
- BUG-20 -> Phase 2
- BUG-21 -> Phase 6
- BUG-22 -> Phase 2
- BUG-24 -> Phase 2
- BUG-25 -> Phase 2
- BUG-26 -> Phase 2 (design) + Phase 3 (implementation tuning)
- BUG-28 -> Phase 2

### Gaps

- GAP-01 -> Phase 3
- GAP-02 -> Phase 3
- GAP-03 -> Phase 3
- GAP-04 -> Phase 4
- GAP-07 -> Phase 6
- GAP-08 -> Phase 6
- GAP-09 -> Phase 6
- GAP-10 -> Phase 4
- GAP-11 -> Phase 1 + Phase 3
- GAP-12 -> Phase 4
- GAP-13 -> Phase 3
- GAP-14 -> Phase 4 (V: verify existing enrichment after migration)
- GAP-15 -> Phase 4
- GAP-16 -> Phase 4
- GAP-17 -> Phase 6

## 8) Implementation Rules for Every Phase

1. Start with tests before migration of that surface.
2. Keep old path callable until parity sign-off.
3. Instrument everything before optimizing.
4. Ship small, reversible increments.
5. Run 72h canary before broad enablement for runtime changes.

## 9) Immediate Next Execution Slice (First 10 Working Days)

1. Phase 0 complete baseline/parity suite.
2. Build orchestrator skeleton + per-thread locks (Phase 1).
3. Route one background workflow through orchestrator in shadow mode.
4. Implement BUG-17 + BUG-28 + BUG-05 on orchestrator-compatible path.
5. Stand up proposal queue for BUG-06/07/18.

Success condition at day 10:
- Parallel background processing works,
- no break in user-visible behavior,
- critical routing bugs are closing,
- and migration risk is controlled.

## 10) Change Control

Any proposal to delete or replace major files requires:
1. parity report,
2. rollback proof,
3. sign-off against this plan's invariants,
4. and explicit owner approval.
