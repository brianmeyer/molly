import asyncio
import difflib
import json
import logging
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import config
from foundry_adapter import FoundrySequenceSignal, load_foundry_sequence_signals

log = logging.getLogger(__name__)

_PATCH_HUNK_RE = re.compile(r"^@@\s*-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s*@@")
_GIT_AUTHOR_ENV = {
    "GIT_AUTHOR_NAME": "Molly",
    "GIT_AUTHOR_EMAIL": "molly@local",
    "GIT_COMMITTER_NAME": "Molly",
    "GIT_COMMITTER_EMAIL": "molly@local",
}
_GLINER_BASE_MODEL = "fastino/gliner2-large-v1"
_GLINER_BENCHMARK_SEED = 1337
_GLINER_BENCHMARK_EVAL_RATIO = 0.2
_GLINER_BENCHMARK_THRESHOLD = 0.4
_GLINER_FINETUNE_COOLDOWN_DAYS = 7
_GLINER_TRAINING_SCAN_LIMIT = 4000
_POST_DEPLOY_HEALTH_GRACE_SECONDS = 45
_POST_DEPLOY_HEALTH_RETRY_SECONDS = 60
_TOOL_GAP_MIN_FAILURES = max(1, int(getattr(config, "TOOL_GAP_MIN_FAILURES", 5)))
_TOOL_GAP_WINDOW_DAYS = max(1, int(getattr(config, "TOOL_GAP_WINDOW_DAYS", 7)))
_LOW_VALUE_WORKFLOW_TOOL_NAMES = {"write", "edit", "bash"}


@dataclass
class PatchValidation:
    ok: bool
    reason: str = ""
    touched_files: list[str] | None = None
    changed_lines: int = 0


class SelfImprovementEngine:
    """Phase 7 self-improvement loop.

    This module is intentionally conservative:
    - all code edits are branch + test + approval gated
    - protected files are blocked
    - rollback paths are explicit and logged
    """

    def __init__(self, molly=None):
        self.molly = molly
        self.project_root = config.PROJECT_ROOT
        self.sandbox_root = config.SANDBOX_DIR
        self.skills_dir = self.sandbox_root / "skills"
        self.tools_dir = self.sandbox_root / "tools"
        self.automations_dir = self.sandbox_root / "automations"
        self.patches_dir = self.sandbox_root / "patches"
        self.tests_dir = self.sandbox_root / "tests"
        self.results_dir = self.sandbox_root / "results"
        self.state_path = config.WORKSPACE / "memory" / "self_improve_state.json"
        self._state: dict[str, Any] = {}
        self._initialized = False
        self._last_tick_at: datetime | None = None

    async def initialize(self):
        if self._initialized:
            return
        await asyncio.to_thread(self._ensure_sandbox_dirs)
        await asyncio.to_thread(self._load_state)
        await self._validate_pending_deploy_on_startup()
        self._initialized = True
        log.info("Self-improvement engine initialized")

    async def tick(self):
        if not config.SELF_EDIT_ENABLED:
            return
        await self.initialize()

        now = datetime.now(ZoneInfo(config.TIMEZONE))
        if self._last_tick_at and (now - self._last_tick_at).total_seconds() < 60:
            return
        self._last_tick_at = now

        await self._mark_deploy_stable_if_window_elapsed(now)

        if self._is_weekly_assessment_due(now):
            await self.run_weekly_assessment()

    async def run_memory_optimization(self) -> dict[str, Any]:
        """Nightly memory maintenance extension (Phase 7)."""
        await self.initialize()
        results = await asyncio.to_thread(self._run_memory_optimization_sync)
        samples = results.get("contradiction_samples", [])
        if samples:
            msg = (
                "⚠️ Memory contradiction candidates:\n"
                + "\n".join(f"- {c['entity']}: {', '.join(c['values'])}" for c in samples[:5])
                + "\n\nReply with clarifications and I will update the graph."
            )
            await self._notify_owner(msg)
        return results

    async def propose_core_patch(self, description: str, patch_text: str) -> dict[str, Any]:
        """Capability 3: guarded core code patch workflow."""
        await self.initialize()
        validation = self.validate_patch(patch_text)
        if not validation.ok:
            return {"status": "rejected", "reason": validation.reason}

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        slug = self._slug(description)
        branch = f"molly/improvement-{ts}-{slug}"
        patch_path = self.patches_dir / f"{ts}-{slug}.patch"
        patch_path.write_text(patch_text)

        base_branch = (await self._agit(["rev-parse", "--abbrev-ref", "HEAD"])).stdout.strip()
        baseline_health = ""

        try:
            from health import get_health_doctor

            doctor = get_health_doctor(self.molly)
            baseline_health = await asyncio.to_thread(
                doctor.generate_report, abbreviated=True, trigger="pre-deploy"
            )
        except Exception:
            log.debug("Pre-deploy baseline health probe failed", exc_info=True)

        try:
            await self._agit(["checkout", "-b", branch])
            await self._agit(["apply", str(patch_path)])
            tests_ok, tests_log = await asyncio.to_thread(self._run_test_suite, f"core-{ts}")
            await self._agit(["add", "-A"])
            await self._agit(
                ["commit", "-m", f"[molly-self-edit][proposal] {description.strip()}"],
                env_override=_GIT_AUTHOR_ENV,
            )

            diff_text = (await self._agit(["--no-pager", "show", "--stat", "--oneline", "-1"])).stdout.strip()
            proposal = self._format_core_patch_proposal(
                description=description,
                branch=branch,
                tests_ok=tests_ok,
                tests_log_path=tests_log,
                diff_summary=diff_text,
            )
            await self._notify_owner(proposal)
            self._log_improvement_event(
                event_type="proposal",
                category="core",
                title=description,
                payload=json.dumps(
                    {
                        "branch": branch,
                        "patch_path": str(patch_path),
                        "tests_ok": tests_ok,
                        "tests_log": str(tests_log),
                    },
                    ensure_ascii=True,
                ),
                status="proposed",
            )

            decision = await self._request_owner_decision(
                category="self-improve-core",
                description=proposal,
                required_keyword="DEPLOY",
                allow_edit=False,
            )
            if decision is not True:
                await self._agit(["checkout", base_branch])
                await self._agit(["branch", "-D", branch])
                self._log_improvement_event(
                    event_type="proposal",
                    category="core",
                    title=description,
                    payload=f"branch={branch}",
                    status="rejected",
                )
                return {"status": "rejected", "reason": "owner denied deployment"}

            await self._agit(["checkout", base_branch])
            await self._agit(["merge", "--no-ff", branch, "-m", f"[molly-self-edit] {description.strip()}"])
            merge_commit = (await self._agit(["rev-parse", "HEAD"])).stdout.strip()
            await self._agit(["branch", "-D", branch])
            self._record_pending_deploy(
                commit_hash=merge_commit,
                description=description,
                deployed_at=datetime.now(timezone.utc),
                baseline_health=baseline_health,
            )

            restart_reason = f"self-edit deployed: {description.strip()}"
            if self._request_runtime_restart(restart_reason):
                self._log_improvement_event(
                    event_type="deploy",
                    category="core",
                    title=description,
                    payload=f"commit={merge_commit}",
                    status="restart_requested",
                )
                return {"status": "restart_requested", "branch": branch, "commit": merge_commit}

            # Fallback for non-runtime contexts where Molly cannot restart itself.
            rollback_reason = await self._post_deploy_health_regression_check(
                baseline_health,
                apply_grace_period=True,
            )
            if rollback_reason:
                await self._rollback_commit(merge_commit, rollback_reason)
                return {"status": "rolled_back", "reason": rollback_reason}

            self._log_improvement_event(
                event_type="deploy",
                category="core",
                title=description,
                payload=f"commit={merge_commit}",
                status="deployed",
            )
            return {"status": "deployed", "branch": branch, "commit": merge_commit}
        except Exception as exc:
            log.error("Core patch proposal failed: %s", exc, exc_info=True)
            try:
                await self._agit(["checkout", base_branch], check=False)
            except Exception:
                pass
            return {"status": "failed", "reason": str(exc)}

    async def propose_skill_from_patterns(self, workflow_patterns: list[dict[str, Any]]) -> dict[str, Any]:
        """Capability 1: draft and propose skills from repeated workflow sequences."""
        await self.initialize()
        if not workflow_patterns:
            return {"status": "skipped", "reason": "no patterns"}

        best = sorted(workflow_patterns, key=lambda p: p.get("count", 0), reverse=True)[0]
        skill_name = f"{best.get('name', 'workflow').strip()} skill".strip()
        skill_slug = self._slug(skill_name) or "workflow-skill"
        draft_path = self.skills_dir / f"{skill_slug}.md"

        steps = self._pattern_steps(best)
        if not steps:
            return {"status": "skipped", "reason": "pattern has no steps"}

        trigger_text = best.get("trigger", "When this multi-step workflow repeats.")
        body = [
            f"# {skill_name}",
            "",
            "## Trigger",
            f"- {trigger_text}",
            "",
            "## Required Tools",
        ]
        for step in steps:
            body.append(f"- `{step}`")
        body.extend(
            [
                "",
                "## Steps",
            ]
        )
        for idx, step in enumerate(steps, start=1):
            body.append(f"{idx}. Run `{step}` and capture output.")
        body.extend(
            [
                "",
                "## Guardrails",
                "- Keep outputs concise and factual.",
                "- Ask for confirmation before write/send actions.",
                "",
                "## Examples",
                f"- Trigger example: \"{best.get('example', 'use this workflow')}\"",
            ]
        )
        draft_text = "\n".join(body).strip() + "\n"
        draft_path.write_text(draft_text)

        simulation = self._dry_run_skill(draft_text)
        proposal = (
            f"📝 New skill proposal: \"{skill_name}\"\n\n"
            f"Pattern count: {best.get('count', 0)}\n"
            f"Steps: {' -> '.join(steps)}\n"
            f"Dry-run: {'PASS' if simulation['ok'] else 'FAIL'} ({simulation['reason']})\n\n"
            f"Reply YES to enable, NO to skip, or EDIT: [changes]."
        )
        await self._notify_owner(proposal)
        self._log_improvement_event(
            event_type="proposal",
            category="skill",
            title=skill_name,
            payload=json.dumps({"draft_path": str(draft_path), "pattern": best}, ensure_ascii=True),
            status="proposed",
        )

        decision = await self._request_owner_decision(
            category="self-improve-skill",
            description=proposal,
            required_keyword="YES",
            allow_edit=True,
        )
        if decision is True:
            live_path = config.SKILLS_DIR / draft_path.name
            live_path.parent.mkdir(parents=True, exist_ok=True)
            live_path.write_text(draft_text)
            self._log_improvement_event(
                event_type="proposal",
                category="skill",
                title=skill_name,
                payload=f"installed={live_path}",
                status="approved",
            )
            return {"status": "approved", "path": str(live_path)}
        if isinstance(decision, str):
            return {"status": "edit_requested", "edit": decision}
        self._log_improvement_event(
            event_type="proposal",
            category="skill",
            title=skill_name,
            payload="",
            status="rejected",
        )
        return {"status": "rejected"}

    async def propose_tool(
        self,
        tool_name: str,
        tool_code: str,
        test_code: str,
        read_only: bool = True,
        requires_auth: bool = False,
    ) -> dict[str, Any]:
        """Capability 2: draft tool in sandbox, test in subprocess, propose install."""
        await self.initialize()
        tool_slug = self._slug(tool_name) or "tool"
        tool_file = self.tools_dir / f"{tool_slug}.py"
        test_file = self.tests_dir / f"test_{tool_slug}.py"
        tool_file.write_text(tool_code.rstrip() + "\n")
        test_file.write_text(test_code.rstrip() + "\n")

        tests_ok, tests_log = self._run_pytest_target(test_file, f"tool-{tool_slug}")
        tier = "AUTO" if read_only and not requires_auth else "CONFIRM"
        proposal = (
            f"🔧 New tool proposal: \"{tool_name}\"\n\n"
            f"Tier recommendation: {tier}\n"
            f"Read-only: {read_only}\n"
            f"Requires auth: {requires_auth}\n"
            f"Tests: {'PASS' if tests_ok else 'FAIL'}\n"
            f"Test log: {tests_log}\n\n"
            "Reply YES to install, NO to skip, or EDIT: [changes]."
        )
        await self._notify_owner(proposal)
        self._log_improvement_event(
            event_type="proposal",
            category="tool",
            title=tool_name,
            payload=json.dumps(
                {"tool_file": str(tool_file), "test_file": str(test_file), "tier": tier, "tests_ok": tests_ok},
                ensure_ascii=True,
            ),
            status="proposed",
        )
        decision = await self._request_owner_decision(
            category="self-improve-tool",
            description=proposal,
            required_keyword="YES",
            allow_edit=True,
        )
        if decision is True:
            live_path = self.project_root / "tools" / f"{tool_slug}.py"
            live_path.write_text(tool_file.read_text())
            self._log_improvement_event(
                event_type="proposal",
                category="tool",
                title=tool_name,
                payload=f"installed={live_path};tier={tier}",
                status="approved",
            )
            return {"status": "approved", "path": str(live_path), "tier": tier}
        if isinstance(decision, str):
            return {"status": "edit_requested", "edit": decision}
        self._log_improvement_event(
            event_type="proposal",
            category="tool",
            title=tool_name,
            payload="",
            status="rejected",
        )
        self._log_negative_preference_signal(
            source="self-improve-tool",
            surfaced_summary=f"Tool proposal rejected: {tool_name}",
            sender_pattern=tool_name,
            owner_feedback="owner_rejected_tool_proposal",
        )
        return {"status": "rejected"}

    async def propose_memory_md_update(self) -> dict[str, Any]:
        """Generate MEMORY.md from graph and propose diff."""
        await self.initialize()
        new_text = self._build_memory_md_from_graph(limit=100)
        memory_path = config.WORKSPACE / "MEMORY.md"
        old_text = memory_path.read_text() if memory_path.exists() else ""
        if old_text.strip() == new_text.strip():
            return {"status": "skipped", "reason": "no changes"}

        diff = "\n".join(
            difflib.unified_diff(
                old_text.splitlines(),
                new_text.splitlines(),
                fromfile="MEMORY.md",
                tofile="MEMORY.md",
                lineterm="",
            )
        )
        proposal = (
            "🧠 MEMORY.md weekly update proposal\n\n"
            f"Diff preview:\n{diff[:2500]}\n\n"
            "Reply YES to apply, NO to skip."
        )
        await self._notify_owner(proposal)
        self._log_improvement_event(
            event_type="proposal",
            category="memory",
            title="MEMORY.md weekly update",
            payload=diff[:10000],
            status="proposed",
        )

        decision = await self._request_owner_decision(
            category="self-improve-memory",
            description=proposal,
            required_keyword="YES",
            allow_edit=False,
        )
        if decision is True:
            memory_path.write_text(new_text)
            self._log_improvement_event(
                event_type="proposal",
                category="memory",
                title="MEMORY.md weekly update",
                payload="applied",
                status="approved",
            )
            return {"status": "approved", "path": str(memory_path)}
        self._log_improvement_event(
            event_type="proposal",
            category="memory",
            title="MEMORY.md weekly update",
            payload="",
            status="rejected",
        )
        return {"status": "rejected"}

    async def propose_user_md_update(self) -> dict[str, Any]:
        """Detect user facts in recent owner messages and propose USER.md curation."""
        await self.initialize()
        facts = self._extract_user_facts(days=30)
        if not facts:
            return {"status": "skipped", "reason": "no new facts"}

        user_path = config.WORKSPACE / "USER.md"
        old_text = user_path.read_text() if user_path.exists() else ""
        today = date.today().isoformat()
        addition = [f"## Curated updates ({today})", ""]
        for fact in facts[:12]:
            addition.append(f"- {fact}")
        addition_text = "\n".join(addition).strip() + "\n"
        new_text = old_text.rstrip() + "\n\n" + addition_text if old_text.strip() else addition_text

        diff = "\n".join(
            difflib.unified_diff(
                old_text.splitlines(),
                new_text.splitlines(),
                fromfile="USER.md",
                tofile="USER.md",
                lineterm="",
            )
        )
        proposal = (
            "👤 USER.md update proposal\n\n"
            f"Extracted facts: {len(facts)}\n"
            f"Diff preview:\n{diff[:2500]}\n\n"
            "Reply YES to apply, NO to skip."
        )
        await self._notify_owner(proposal)
        self._log_improvement_event(
            event_type="proposal",
            category="user",
            title="USER.md curation",
            payload=diff[:10000],
            status="proposed",
        )

        decision = await self._request_owner_decision(
            category="self-improve-user",
            description=proposal,
            required_keyword="YES",
            allow_edit=False,
        )
        if decision is True:
            user_path.write_text(new_text)
            self._log_improvement_event(
                event_type="proposal",
                category="user",
                title="USER.md curation",
                payload="applied",
                status="approved",
            )
            return {"status": "approved", "path": str(user_path)}
        self._log_improvement_event(
            event_type="proposal",
            category="user",
            title="USER.md curation",
            payload="",
            status="rejected",
        )
        return {"status": "rejected"}

    async def run_weekly_assessment(self) -> Path:
        """Capability 6: weekly self-assessment report + WhatsApp summary."""
        await self.initialize()
        today = date.today()
        report_path = config.WEEKLY_ASSESSMENT_DIR / f"{today.isoformat()}.md"
        config.WEEKLY_ASSESSMENT_DIR.mkdir(parents=True, exist_ok=True)

        perf = self._performance_metrics(days=7)
        memory_stats = self._memory_stats()
        automation_stats = self._automation_stats()
        workflow_patterns = self._detect_workflow_patterns(days=30, min_occurrences=3)
        health_summary = self._latest_health_summary()
        self_improve_stats = self._self_improvement_stats(days=30)

        lines = [
            f"# Molly — Weekly Self-Assessment ({today.isoformat()})",
            "",
            "## Performance",
            f"- Messages handled: {perf['messages']}",
            f"- Average response latency (tool calls): {perf['avg_latency_ms']}ms",
            f"- Tool calls: {perf['tool_calls']}",
            f"- Sub-agent routing: {perf['routing_distribution']}",
            f"- Extraction quality signal: {perf['extraction_quality']}",
            "",
            "## Memory",
            f"- Entities: {memory_stats['entities']}",
            f"- Relationships: {memory_stats['relationships']}",
            f"- Stale entities (60+ days): {memory_stats['stale_entities']}",
            f"- Orphan entities: {memory_stats['orphans']}",
            "",
            "## Automations",
            f"- Loaded automations: {automation_stats['loaded']}",
            f"- Last run: {automation_stats['last_run']}",
            f"- Failed automations: {automation_stats['failed']}",
            "",
            "## Self-Improvements",
            f"- Proposed (30d): {self_improve_stats['proposed']}",
            f"- Approved (30d): {self_improve_stats['approved']}",
            f"- Rejected (30d): {self_improve_stats['rejected']}",
            f"- {self._gliner_weekly_summary_line()}",
            "",
            "## Workflow Patterns Detected",
        ]
        if workflow_patterns:
            for item in workflow_patterns[:5]:
                steps = self._pattern_steps(item)
                steps_text = str(item.get("steps_text") or " -> ".join(steps)).strip()
                lines.append(
                    f"- {steps_text} (count={item['count']}, confidence={item['confidence']:.2f})"
                )
        else:
            lines.append("- No repeated 3-step sequences found this week.")

        lines.extend(
            [
                "",
                "## Health Summary",
                f"- {health_summary}",
            ]
        )
        report_text = "\n".join(lines).rstrip() + "\n"
        report_path.write_text(report_text)

        self._state["last_weekly_assessment"] = today.isoformat()
        self._save_state()

        summary = (
            f"Weekly self-assessment ready ({today.isoformat()}). "
            f"Messages={perf['messages']}, tool_calls={perf['tool_calls']}, "
            f"patterns={len(workflow_patterns)}, health={health_summary}."
        )
        await self._notify_owner(summary)
        self._log_improvement_event(
            event_type="assessment",
            category="weekly",
            title=f"Weekly self-assessment {today.isoformat()}",
            payload=str(report_path),
            status="generated",
        )

        # Weekly curation proposals.
        await self.propose_memory_md_update()
        await self.propose_user_md_update()
        await self._propose_skill_updates_from_patterns(workflow_patterns)
        await self._propose_tool_updates_from_failures(
            days=_TOOL_GAP_WINDOW_DAYS,
            min_failures=_TOOL_GAP_MIN_FAILURES,
        )
        await self._propose_automation_updates_from_patterns(workflow_patterns)
        return report_path

    async def run_gliner_nightly_cycle(self) -> dict[str, Any]:
        """Nightly GLiNER closed-loop: accumulate data, then trigger fine-tune when ready."""
        await self.initialize()
        accumulation = await asyncio.to_thread(
            self._accumulate_gliner_training_data,
            _GLINER_TRAINING_SCAN_LIMIT,
        )
        total_examples = int(accumulation.get("total_examples", 0))
        required = int(config.GLINER_FINETUNE_MIN_EXAMPLES)
        progress_line = f"GLiNER training data: {total_examples}/{required} examples accumulated"
        log.info(progress_line)

        self._state["gliner_training_examples"] = total_examples
        self._state["gliner_last_result"] = progress_line
        self._state["gliner_last_cycle_status"] = "accumulated"
        self._save_state()

        if total_examples < required:
            self._log_improvement_event(
                event_type="model",
                category="gliner",
                title="GLiNER training data accumulation",
                payload=json.dumps(accumulation, ensure_ascii=True),
                status="insufficient_examples",
            )
            return {
                "status": "insufficient_examples",
                "count": total_examples,
                "required": required,
                "accumulation": accumulation,
                "message": progress_line,
            }

        now_utc = datetime.now(timezone.utc)
        last_run = _parse_datetime(str(self._state.get("gliner_last_finetune_at", "")))
        if last_run and (now_utc - last_run) < timedelta(days=_GLINER_FINETUNE_COOLDOWN_DAYS):
            elapsed = now_utc - last_run
            remaining = timedelta(days=_GLINER_FINETUNE_COOLDOWN_DAYS) - elapsed
            hours_remaining = max(0, int(remaining.total_seconds() // 3600))
            cooldown_line = (
                f"GLiNER fine-tune skipped: last run {last_run.date().isoformat()} "
                f"({hours_remaining}h cooldown remaining)."
            )
            log.info(cooldown_line)
            self._state["gliner_last_result"] = cooldown_line
            self._state["gliner_last_cycle_status"] = "cooldown_active"
            self._save_state()
            self._log_improvement_event(
                event_type="model",
                category="gliner",
                title="GLiNER fine-tune trigger",
                payload=json.dumps(
                    {"count": total_examples, "required": required, "last_run": last_run.isoformat()},
                    ensure_ascii=True,
                ),
                status="cooldown_active",
            )
            return {
                "status": "cooldown_active",
                "count": total_examples,
                "required": required,
                "last_run": last_run.isoformat(),
                "accumulation": accumulation,
                "message": cooldown_line,
            }

        pipeline = await self.run_gliner_finetune_pipeline()
        return {
            "status": "finetune_triggered",
            "accumulation": accumulation,
            "pipeline": pipeline,
        }

    async def run_gliner_finetune_pipeline(self) -> dict[str, Any]:
        """Fine-tune GLiNER2 locally, benchmark on held-out data, and request deploy approval."""
        await self.initialize()
        rows = await asyncio.to_thread(self._load_accumulated_gliner_examples)
        total_rows = len(rows)
        required = int(config.GLINER_FINETUNE_MIN_EXAMPLES)
        if total_rows < required:
            msg = f"GLiNER training data: {total_rows}/{required} examples accumulated"
            self._state["gliner_training_examples"] = total_rows
            self._state["gliner_last_result"] = msg
            self._state["gliner_last_cycle_status"] = "insufficient_examples"
            self._save_state()
            return {
                "status": "insufficient_examples",
                "count": total_rows,
                "required": required,
                "message": msg,
            }

        train_rows, eval_rows = self._split_holdout_rows(rows, eval_ratio=0.2, seed=_GLINER_BENCHMARK_SEED)
        if not train_rows or not eval_rows:
            failure = {
                "status": "split_failed",
                "count": total_rows,
                "train_count": len(train_rows),
                "eval_count": len(eval_rows),
            }
            self._state["gliner_last_cycle_status"] = "split_failed"
            self._state["gliner_last_result"] = "GLiNER fine-tune skipped: invalid train/eval split."
            self._save_state()
            self._log_improvement_event(
                event_type="model",
                category="gliner",
                title="GLiNER2 fine-tune split",
                payload=json.dumps(failure, ensure_ascii=True),
                status="split_failed",
            )
            return failure

        training_strategy = await asyncio.to_thread(self._select_gliner_training_strategy, total_rows)
        strategy_mode = str(training_strategy.get("mode") or "lora").strip().lower()
        if strategy_mode not in {"lora", "full"}:
            strategy_mode = "lora"

        self._state["gliner_last_finetune_at"] = datetime.now(timezone.utc).isoformat()
        self._state["gliner_training_examples"] = total_rows
        self._state["gliner_last_cycle_status"] = "finetune_started"
        self._state["gliner_last_training_strategy"] = strategy_mode
        self._save_state()

        fine_tune = await asyncio.to_thread(self._fine_tune_gliner_candidate, train_rows, strategy_mode)
        if not fine_tune.get("ok", False):
            payload = {
                "count": total_rows,
                "train_count": len(train_rows),
                "eval_count": len(eval_rows),
                "training_strategy": training_strategy,
                "fine_tune": fine_tune,
            }
            self._state["gliner_last_cycle_status"] = "finetune_failed"
            self._state["gliner_last_result"] = (
                f"GLiNER {strategy_mode} fine-tune failed before benchmarking."
            )
            self._save_state()
            self._log_improvement_event(
                event_type="model",
                category="gliner",
                title="GLiNER2 fine-tune run",
                payload=json.dumps(payload, ensure_ascii=True),
                status="finetune_failed",
            )
            return {
                "status": "finetune_failed",
                "count": total_rows,
                "train_count": len(train_rows),
                "eval_count": len(eval_rows),
                "training_strategy": training_strategy,
                "fine_tune": fine_tune,
            }

        candidate_model_ref = str(fine_tune.get("candidate_model") or "").strip()
        benchmark = await asyncio.to_thread(
            self._benchmark_finetune_candidate,
            eval_rows,
            candidate_model_ref,
            len(train_rows),
        )
        payload = {
            "count": total_rows,
            "train_count": len(train_rows),
            "eval_count": len(eval_rows),
            "candidate_model": candidate_model_ref,
            "training_strategy": training_strategy,
            "fine_tune": fine_tune,
            "benchmark": benchmark,
        }

        status = "benchmark_failed"
        if benchmark.get("ok", False):
            if benchmark["improvement"] >= config.GLINER_FINETUNE_BENCHMARK_THRESHOLD:
                status = "proposal_ready"
            else:
                status = "below_threshold"

        if status != "proposal_ready":
            await asyncio.to_thread(
                self._discard_gliner_candidate_model,
                Path(candidate_model_ref) if candidate_model_ref else None,
            )
            if status == "below_threshold":
                self._state["gliner_last_result"] = (
                    f"GLiNER {strategy_mode} fine-tune benchmark below threshold "
                    f"({benchmark.get('improvement', 0.0):+.2%} F1)."
                )
            else:
                self._state["gliner_last_result"] = f"GLiNER {strategy_mode} fine-tune benchmark failed."
            self._state["gliner_last_cycle_status"] = status
            self._save_state()
            await asyncio.to_thread(
                self._record_gliner_benchmark,
                strategy_mode,
                benchmark,
                status,
                total_rows,
            )
            self._log_improvement_event(
                event_type="model",
                category="gliner",
                title="GLiNER2 fine-tune benchmark",
                payload=json.dumps(payload, ensure_ascii=True),
                status=status,
            )
            return {
                "status": status,
                "count": total_rows,
                "train_count": len(train_rows),
                "eval_count": len(eval_rows),
                "candidate_model": candidate_model_ref or None,
                "training_strategy": training_strategy,
                "benchmark": benchmark,
                "fine_tune": fine_tune,
            }

        proposal = self._format_gliner_swap_proposal(benchmark, strategy_mode)
        await self._notify_owner(proposal)
        decision = await self._request_owner_decision(
            category="self-improve-gliner-model-swap",
            description=proposal,
            required_keyword="YES",
            allow_edit=False,
        )
        if decision is True:
            deploy = await asyncio.to_thread(
                self._deploy_gliner_candidate_model,
                Path(candidate_model_ref),
                benchmark,
                fine_tune,
            )
            status = "deployed" if deploy.get("ok", False) else "deploy_failed"
            payload["deploy"] = deploy
            if deploy.get("ok", False):
                self._state["gliner_last_cycle_status"] = "deployed"
                self._state["gliner_last_result"] = (
                    f"{strategy_mode} +{benchmark.get('improvement', 0.0):.2%} F1 "
                    f"(P {benchmark.get('base', {}).get('metrics', {}).get('precision', 0.0):.2f} "
                    f"-> {benchmark.get('candidate', {}).get('metrics', {}).get('precision', 0.0):.2f}), "
                    "approved and deployed."
                )
            else:
                self._state["gliner_last_cycle_status"] = "deploy_failed"
                self._state["gliner_last_result"] = (
                    f"GLiNER {strategy_mode} swap approved but deployment failed."
                )
            self._save_state()
            await asyncio.to_thread(
                self._record_gliner_benchmark,
                strategy_mode,
                benchmark,
                status,
                total_rows,
            )
            self._log_improvement_event(
                event_type="model",
                category="gliner",
                title="GLiNER2 fine-tune benchmark",
                payload=json.dumps(payload, ensure_ascii=True),
                status=status,
            )
            return {
                "status": status,
                "count": total_rows,
                "train_count": len(train_rows),
                "eval_count": len(eval_rows),
                "training_strategy": training_strategy,
                "benchmark": benchmark,
                "fine_tune": fine_tune,
                "deploy": deploy,
            }

        await asyncio.to_thread(
            self._discard_gliner_candidate_model,
            Path(candidate_model_ref) if candidate_model_ref else None,
        )
        self._state["gliner_last_cycle_status"] = "rejected"
        self._state["gliner_last_result"] = (
            f"GLiNER {strategy_mode} candidate rejected; keeping active model."
        )
        self._save_state()
        await asyncio.to_thread(
            self._record_gliner_benchmark,
            strategy_mode,
            benchmark,
            "rejected",
            total_rows,
        )
        self._log_improvement_event(
            event_type="model",
            category="gliner",
            title="GLiNER2 fine-tune benchmark",
            payload=json.dumps(payload, ensure_ascii=True),
            status="rejected",
        )
        return {
            "status": "rejected",
            "count": total_rows,
            "train_count": len(train_rows),
            "eval_count": len(eval_rows),
            "training_strategy": training_strategy,
            "benchmark": benchmark,
            "fine_tune": fine_tune,
        }

    def validate_patch(self, patch_text: str) -> PatchValidation:
        changed_lines = 0
        touched_files: list[str] = []
        for line in patch_text.splitlines():
            if line.startswith("+++ b/"):
                touched_files.append(line[6:].strip())
            if line.startswith("+") or line.startswith("-"):
                if line.startswith("+++") or line.startswith("---"):
                    continue
                changed_lines += 1

        if changed_lines > config.SELF_EDIT_MAX_PATCH_LINES:
            return PatchValidation(
                ok=False,
                reason=f"Patch too large: {changed_lines} lines > {config.SELF_EDIT_MAX_PATCH_LINES}",
                touched_files=touched_files,
                changed_lines=changed_lines,
            )

        for fpath in touched_files:
            base = Path(fpath).name
            if base in config.SELF_EDIT_PROTECTED_FILES:
                return PatchValidation(
                    ok=False,
                    reason=f"Protected file: {fpath}",
                    touched_files=touched_files,
                    changed_lines=changed_lines,
                )
            if base in config.SELF_EDIT_PROTECTED_IDENTITY:
                return PatchValidation(
                    ok=False,
                    reason=f"Protected identity file: {fpath}",
                    touched_files=touched_files,
                    changed_lines=changed_lines,
                )
            if fpath.endswith("/SOUL.md") or fpath == "workspace/SOUL.md":
                return PatchValidation(
                    ok=False,
                    reason="SOUL.md is protected",
                    touched_files=touched_files,
                    changed_lines=changed_lines,
                )

        if "config.py" in touched_files and self._patch_touches_action_tiers_block(patch_text):
            return PatchValidation(
                ok=False,
                reason="config.py ACTION_TIERS block is protected",
                touched_files=touched_files,
                changed_lines=changed_lines,
            )

        return PatchValidation(
            ok=True,
            reason="ok",
            touched_files=touched_files,
            changed_lines=changed_lines,
        )

    def _run_memory_optimization_sync(self) -> dict[str, Any]:
        from memory.graph import get_driver, run_strength_decay

        decayed = run_strength_decay()
        consolidated = self._consolidate_entities()
        stale = self._stale_entities(days=60)
        contradictions = self._detect_contradictions()
        community = self._community_detection()

        # Relationship decay for dormant edges.
        rel_decay = 0
        try:
            driver = get_driver()
            with driver.session() as session:
                rec = session.run(
                    """
                    MATCH ()-[r]->()
                    WHERE r.last_mentioned IS NOT NULL
                    WITH r, duration.between(datetime(r.last_mentioned), datetime()).days AS age_days
                    SET r.strength = CASE
                        WHEN r.strength IS NULL THEN exp(-0.02 * age_days)
                        ELSE r.strength * exp(-0.02 * age_days)
                    END
                    RETURN count(r) AS c
                    """
                ).single()
                rel_decay = int(rec["c"]) if rec else 0
        except Exception:
            log.debug("Relationship decay failed", exc_info=True)

        results = {
            "strength_decay": decayed,
            "entity_consolidations": consolidated,
            "stale_entities": len(stale),
            "contradictions": len(contradictions),
            "contradiction_samples": contradictions[:10],
            "relationship_decay": rel_decay,
            "communities": community,
        }

        self._log_improvement_event(
            event_type="maintenance",
            category="memory",
            title="Nightly memory optimization",
            payload=json.dumps(results, ensure_ascii=True),
            status="completed",
        )
        return results

    def _ensure_sandbox_dirs(self):
        for path in [
            self.sandbox_root,
            self.skills_dir,
            self.tools_dir,
            self.automations_dir,
            self.patches_dir,
            self.tests_dir,
            self.results_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _load_state(self):
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                if isinstance(data, dict):
                    self._state = data
                else:
                    self._state = {}
            except Exception:
                self._state = {}
        else:
            self._state = {}
        self._state.setdefault("pending_deploy", None)
        self._state.setdefault("last_weekly_assessment", "")
        self._state.setdefault("gliner_training_cursor", "")
        self._state.setdefault("gliner_training_examples", 0)
        self._state.setdefault("gliner_last_finetune_at", "")
        self._state.setdefault("gliner_last_deployed_at", "")
        self._state.setdefault("gliner_last_result", "")
        self._state.setdefault("gliner_last_cycle_status", "")
        self._state.setdefault("gliner_active_model_ref", "")
        self._state.setdefault("gliner_last_training_strategy", "lora")
        self._state.setdefault("gliner_benchmark_history", [])

    def _save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, indent=2, default=str))

    def _request_runtime_restart(self, reason: str) -> bool:
        runtime = self.molly
        if runtime is None:
            return False
        restart_fn = getattr(runtime, "request_restart", None)
        if not callable(restart_fn):
            return False
        try:
            restart_fn(reason)
            return True
        except Exception:
            log.error("Restart request failed", exc_info=True)
            return False

    def _git(
        self,
        args: list[str],
        check: bool = True,
        env_override: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        if env_override:
            env.update(env_override)
        result = subprocess.run(
            ["git"] + args,
            cwd=str(self.project_root),
            capture_output=True,
            text=True,
            env=env,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
            )
        return result

    async def _agit(
        self,
        args: list[str],
        check: bool = True,
        env_override: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        """Async wrapper for _git() — avoids blocking the event loop."""
        return await asyncio.to_thread(self._git, args, check, env_override)

    def _run_test_suite(self, label: str) -> tuple[bool, Path]:
        log_path = self.results_dir / f"{label}-pytest.log"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q"],
            cwd=str(self.project_root),
            capture_output=True,
            text=True,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log_path.write_text(output)
        return proc.returncode == 0, log_path

    def _run_pytest_target(self, target: Path, label: str) -> tuple[bool, Path]:
        log_path = self.results_dir / f"{label}-pytest.log"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(target), "-q"],
            cwd=str(self.sandbox_root),
            capture_output=True,
            text=True,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log_path.write_text(output)
        return proc.returncode == 0, log_path

    def _dry_run_skill(self, skill_markdown: str) -> dict[str, Any]:
        missing_tools: list[str] = []
        required = []
        in_required = False
        for line in skill_markdown.splitlines():
            if line.strip().lower() == "## required tools":
                in_required = True
                continue
            if in_required and line.startswith("## "):
                break
            if in_required and line.strip().startswith("-"):
                required.append(line.strip("- ").strip("` "))
        all_known = set(config.ACTION_TIERS["AUTO"]) | set(config.ACTION_TIERS["CONFIRM"]) | set(config.ACTION_TIERS["BLOCKED"])
        for tool in required:
            if tool and tool not in all_known:
                missing_tools.append(tool)
        if missing_tools:
            return {"ok": False, "reason": f"Missing tools: {', '.join(missing_tools)}"}
        return {"ok": True, "reason": "All required tools available"}

    def _slug(self, text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
        return slug[:48] or "change"

    def _patch_touches_action_tiers_block(self, patch_text: str) -> bool:
        action_start, action_end = self._action_tiers_line_range()
        if action_start <= 0 or action_end <= 0:
            return True

        current_file = ""
        old_line = 0
        new_line = 0
        for line in patch_text.splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:].strip()
                continue
            if line.startswith("@@ "):
                m = _PATCH_HUNK_RE.match(line)
                if not m:
                    continue
                old_line = int(m.group(1))
                new_line = int(m.group(2))
                continue
            if current_file != "config.py":
                continue
            if line.startswith("+") and not line.startswith("+++"):
                if action_start <= new_line <= action_end:
                    return True
                new_line += 1
                continue
            if line.startswith("-") and not line.startswith("---"):
                if action_start <= old_line <= action_end:
                    return True
                old_line += 1
                continue
            if line.startswith(" "):
                old_line += 1
                new_line += 1
        return False

    def _action_tiers_line_range(self) -> tuple[int, int]:
        cfg_path = self.project_root / "config.py"
        if not cfg_path.exists():
            return -1, -1
        lines = cfg_path.read_text().splitlines()
        start = -1
        depth = 0
        for idx, line in enumerate(lines, start=1):
            if start < 0 and line.strip().startswith("ACTION_TIERS"):
                start = idx
                depth = line.count("{") - line.count("}")
                if depth <= 0:
                    return start, start
                continue
            if start > 0:
                depth += line.count("{")
                depth -= line.count("}")
                if depth <= 0:
                    return start, idx
        return -1, -1

    async def _request_owner_decision(
        self,
        category: str,
        description: str,
        required_keyword: str = "YES",
        allow_edit: bool = True,
    ) -> bool | str:
        if not self.molly or not getattr(self.molly, "approvals", None) or not getattr(self.molly, "wa", None):
            return False
        owner_jid = self.molly._get_owner_dm_jid()
        if not owner_jid:
            return False
        return await self.molly.approvals.request_custom_approval(
            category=category,
            description=description,
            chat_jid=owner_jid,
            molly=self.molly,
            required_keyword=required_keyword,
            allow_edit=allow_edit,
        )

    async def _notify_owner(self, text: str):
        if not self.molly or not getattr(self.molly, "wa", None):
            return
        owner_jid = self.molly._get_owner_dm_jid()
        if not owner_jid:
            return
        self.molly._track_send(self.molly.wa.send_message(owner_jid, text[:3900]))

    def _log_improvement_event(
        self,
        event_type: str,
        category: str,
        title: str,
        payload: str,
        status: str,
    ):
        try:
            from memory.retriever import get_vectorstore

            vs = get_vectorstore()
            vs.log_self_improvement_event(
                event_type=event_type,
                category=category,
                title=title,
                payload=payload,
                status=status,
            )
        except Exception:
            log.debug("Failed to log self improvement event", exc_info=True)

    def _consolidate_entities(self) -> int:
        from difflib import SequenceMatcher
        from memory.graph import get_driver

        driver = get_driver()
        with driver.session() as session:
            rows = session.run(
                "MATCH (e:Entity) RETURN e.name AS name, e.entity_type AS t ORDER BY e.entity_type, e.name"
            )
            entities = [dict(r) for r in rows]

        by_type: dict[str, list[str]] = {}
        for row in entities:
            by_type.setdefault(row.get("t") or "Unknown", []).append(row.get("name") or "")

        merged = 0
        for etype, names in by_type.items():
            for idx, left in enumerate(names):
                if not left:
                    continue
                for right in names[idx + 1:]:
                    if not right:
                        continue
                    score = SequenceMatcher(None, left.lower(), right.lower()).ratio()
                    if score < 0.9 or left.lower() == right.lower():
                        continue
                    keep, drop = sorted([left, right], key=lambda s: (len(s), s))
                    with driver.session() as session:
                        session.run(
                            """
                            MATCH (keep:Entity {name: $keep}), (drop:Entity {name: $drop})
                            SET keep.mention_count = coalesce(keep.mention_count, 0) + coalesce(drop.mention_count, 0),
                                keep.aliases = coalesce(keep.aliases, []) + coalesce(drop.aliases, []) + [$drop]
                            WITH keep, drop
                            DETACH DELETE drop
                            """,
                            keep=keep,
                            drop=drop,
                        )
                    merged += 1
                    if merged >= 30:
                        return merged
        return merged

    def _stale_entities(self, days: int = 60) -> list[dict[str, Any]]:
        from memory.graph import get_driver

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        driver = get_driver()
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (e:Entity)
                WHERE e.last_mentioned IS NOT NULL AND e.last_mentioned < $cutoff
                RETURN e.name AS name, e.entity_type AS type, e.last_mentioned AS last_mentioned
                ORDER BY e.last_mentioned ASC
                LIMIT 100
                """,
                cutoff=cutoff,
            )
            return [dict(r) for r in rows]

    def _detect_contradictions(self) -> list[dict[str, Any]]:
        from memory.graph import get_driver

        driver = get_driver()
        contradictions: list[dict[str, Any]] = []
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (p:Entity {entity_type: 'Person'})-[:WORKS_AT]->(o:Entity)
                WITH p, collect(DISTINCT o.name) AS orgs
                WHERE size(orgs) > 1
                RETURN p.name AS person, orgs
                LIMIT 25
                """
            )
            for row in rows:
                contradictions.append(
                    {"entity": row["person"], "type": "WORKS_AT", "values": list(row["orgs"])}
                )
        return contradictions

    def _community_detection(self) -> dict[str, int]:
        from memory.graph import get_driver

        driver = get_driver()
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (e:Entity)
                RETURN e.entity_type AS type, count(e) AS c
                ORDER BY c DESC
                """
            )
            return {str(r["type"] or "Unknown"): int(r["c"]) for r in rows}

    def _build_memory_md_from_graph(self, limit: int = 100) -> str:
        from memory.graph import get_driver

        driver = get_driver()
        sections: dict[str, list[str]] = {}
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (e:Entity)
                WHERE e.strength IS NOT NULL
                RETURN e.name AS name, e.entity_type AS type, e.strength AS strength, e.mention_count AS mentions
                ORDER BY e.strength DESC
                LIMIT $limit
                """,
                limit=limit,
            )
            for row in rows:
                etype = str(row["type"] or "Unknown")
                sections.setdefault(etype, []).append(
                    f"- {row['name']} (strength={float(row['strength']):.2f}, mentions={int(row['mentions'] or 0)})"
                )

        today = date.today().isoformat()
        lines = [f"# MEMORY Snapshot ({today})", ""]
        ordered = ["Person", "Project", "Organization", "Technology", "Concept", "Place"]
        seen = set()
        for key in ordered:
            values = sections.get(key)
            if not values:
                continue
            seen.add(key)
            lines.append(f"## {key}s")
            lines.extend(values)
            lines.append("")
        for key, values in sections.items():
            if key in seen:
                continue
            lines.append(f"## {key}")
            lines.extend(values)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _extract_user_facts(self, days: int = 30) -> list[str]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        owner_ids = set(config.OWNER_IDS)
        if not owner_ids:
            return []
        conn = sqlite3.connect(str(config.DATABASE_PATH))
        placeholders = ",".join("?" for _ in owner_ids)
        query = (
            f"SELECT content FROM messages WHERE timestamp > ? AND sender IN ({placeholders}) "
            "ORDER BY timestamp DESC LIMIT 400"
        )
        rows = conn.execute(query, (cutoff, *owner_ids)).fetchall()
        conn.close()
        facts: set[str] = set()
        patterns = [
            re.compile(r"\bI (?:prefer|like|love|hate)\b[^.!\n]{0,120}", re.IGNORECASE),
            re.compile(r"\bI work (?:at|for)\b[^.!\n]{0,120}", re.IGNORECASE),
            re.compile(r"\bI live in\b[^.!\n]{0,120}", re.IGNORECASE),
            re.compile(r"\bmy (?:favorite|preferred)\b[^.!\n]{0,120}", re.IGNORECASE),
        ]
        for (content,) in rows:
            text = str(content or "").strip()
            if not text:
                continue
            for pattern in patterns:
                for match in pattern.findall(text):
                    fact = re.sub(r"\s+", " ", match).strip(" .")
                    if len(fact) >= 8:
                        facts.add(fact)
        return sorted(facts)

    def _performance_metrics(self, days: int = 7) -> dict[str, Any]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        messages = self._count_rows(
            config.DATABASE_PATH,
            "SELECT COUNT(*) FROM messages WHERE timestamp > ?",
            (cutoff,),
        )
        tool_calls = self._count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM tool_calls WHERE created_at > ?",
            (cutoff,),
        )
        avg_latency = self._scalar(
            config.MOLLYGRAPH_PATH,
            "SELECT AVG(latency_ms) FROM tool_calls WHERE created_at > ?",
            (cutoff,),
            default=0.0,
        )
        routing_rows = self._rows(
            config.MOLLYGRAPH_PATH,
            """
            SELECT tool_name, COUNT(*) AS c
            FROM tool_calls
            WHERE created_at > ? AND tool_name LIKE 'routing:subagent_start:%'
            GROUP BY tool_name
            ORDER BY c DESC
            """,
            (cutoff,),
        )
        routing_total = sum(int(r["c"]) for r in routing_rows) or 1
        distribution_parts = []
        for row in routing_rows:
            agent = str(row["tool_name"]).split(":")[-1]
            pct = int((int(row["c"]) / routing_total) * 100)
            distribution_parts.append(f"{agent} {pct}%")
        routing_distribution = ", ".join(distribution_parts) if distribution_parts else "none"

        extraction_count = self._count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM tool_calls WHERE created_at > ? AND tool_name = 'extraction'",
            (cutoff,),
        )
        correction_count = self._count_rows(
            config.MOLLYGRAPH_PATH,
            "SELECT COUNT(*) FROM corrections WHERE created_at > ?",
            (cutoff,),
        )
        extraction_quality = (
            f"extractions={extraction_count}, corrections={correction_count}, "
            f"delta={max(0, extraction_count - correction_count)}"
        )
        return {
            "messages": messages,
            "tool_calls": tool_calls,
            "avg_latency_ms": int(avg_latency or 0),
            "routing_distribution": routing_distribution,
            "extraction_quality": extraction_quality,
        }

    def _memory_stats(self) -> dict[str, Any]:
        from memory.graph import entity_count, relationship_count, get_driver

        entities = entity_count()
        relationships = relationship_count()
        stale = len(self._stale_entities(days=60))
        orphans = 0
        try:
            driver = get_driver()
            with driver.session() as session:
                rec = session.run("MATCH (e:Entity) WHERE NOT (e)--() RETURN count(e) AS c").single()
                orphans = int(rec["c"]) if rec else 0
        except Exception:
            pass
        return {
            "entities": entities,
            "relationships": relationships,
            "stale_entities": stale,
            "orphans": orphans,
        }

    def _automation_stats(self) -> dict[str, Any]:
        path = config.AUTOMATIONS_STATE_FILE
        if not path.exists():
            return {"loaded": 0, "last_run": "-", "failed": 0}
        try:
            data = json.loads(path.read_text())
            automations = data.get("automations", {})
            last_runs = [str(row.get("last_run", "")) for row in automations.values() if row.get("last_run")]
            failed = sum(1 for row in automations.values() if str(row.get("last_status", "")).lower() == "failed")
            return {
                "loaded": len(automations),
                "last_run": max(last_runs) if last_runs else "-",
                "failed": failed,
            }
        except Exception:
            return {"loaded": 0, "last_run": "-", "failed": 0}

    def _latest_health_summary(self) -> str:
        from health import get_health_doctor

        doctor = get_health_doctor(self.molly)
        path = doctor.latest_report_path()
        if not path:
            return "no recent health report"
        payload = {}
        try:
            payload = json.loads(
                doctor.latest_report_text().split("<!-- HEALTH_DATA:", 1)[1].split("-->", 1)[0].strip()
            )
        except Exception:
            payload = {}
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        if summary:
            return f"{summary.get('green', 0)} green / {summary.get('yellow', 0)} yellow / {summary.get('red', 0)} red"
        return f"latest={path.stem}"

    def _self_improvement_stats(self, days: int = 30) -> dict[str, int]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self._rows(
            config.MOLLYGRAPH_PATH,
            """
            SELECT status, COUNT(*) AS c
            FROM self_improvement_events
            WHERE created_at > ?
            GROUP BY status
            """,
            (cutoff,),
        )
        stats = {"proposed": 0, "approved": 0, "rejected": 0}
        for row in rows:
            status = str(row["status"]).lower()
            count = int(row["c"])
            if status in stats:
                stats[status] += count
            if status == "deployed":
                stats["approved"] += count
        return stats

    def _pattern_steps(self, pattern: dict[str, Any]) -> list[str]:
        raw_steps = pattern.get("steps", [])
        if isinstance(raw_steps, str):
            return [step.strip() for step in raw_steps.split("->") if step.strip()]
        if isinstance(raw_steps, list):
            return [str(step).strip() for step in raw_steps if str(step).strip()]
        return []

    def _is_low_value_workflow_tool(self, tool_name: str) -> bool:
        normalized = str(tool_name or "").strip()
        if not normalized:
            return True
        lowered = normalized.lower()
        return lowered in _LOW_VALUE_WORKFLOW_TOOL_NAMES or lowered.startswith("approval:")

    def _load_foundry_sequence_signals(self, days: int = 30) -> dict[str, FoundrySequenceSignal]:
        try:
            return load_foundry_sequence_signals(
                days=days,
                is_low_value_tool=self._is_low_value_workflow_tool,
            )
        except Exception:
            log.debug("Failed to load Foundry observation signals", exc_info=True)
            return {}

    def _has_recent_event(
        self,
        category: str,
        title: str,
        days: int = 30,
    ) -> bool:
        if not title:
            return False
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self._rows(
            config.MOLLYGRAPH_PATH,
            """
            SELECT COUNT(*) AS c
            FROM self_improvement_events
            WHERE category = ?
              AND lower(title) = ?
              AND created_at > ?
            """,
            (category, title.strip().lower(), cutoff),
        )
        if not rows:
            return False
        try:
            return int(rows[0].get("c", 0) or 0) > 0
        except Exception:
            return False

    async def _propose_skill_updates_from_patterns(self, patterns: list[dict[str, Any]]) -> dict[str, Any]:
        if not patterns:
            return {"status": "skipped", "reason": "no patterns"}

        for pattern in patterns:
            steps = self._pattern_steps(pattern)
            if len(steps) < 3:
                continue
            skill_name = f"{str(pattern.get('name', 'workflow')).strip()} skill".strip()
            if self._has_recent_event(category="skill", title=skill_name, days=30):
                continue
            skill_slug = self._slug(skill_name) or "workflow-skill"
            if (config.SKILLS_DIR / f"{skill_slug}.md").exists():
                continue
            return await self.propose_skill_from_patterns([pattern])

        return {"status": "skipped", "reason": "no new skill candidates"}

    def _detect_tool_gap_candidates(
        self,
        days: int = _TOOL_GAP_WINDOW_DAYS,
        min_failures: int = _TOOL_GAP_MIN_FAILURES,
    ) -> list[dict[str, Any]]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self._rows(
            config.MOLLYGRAPH_PATH,
            """
            SELECT tool_name,
                   COUNT(*) AS failures,
                   MAX(created_at) AS last_failed_at,
                   MAX(CASE WHEN trim(coalesce(error_message, '')) <> '' THEN error_message ELSE '' END) AS sample_error
            FROM tool_calls
            WHERE created_at > ?
              AND success = 0
              AND tool_name NOT LIKE 'approval:%'
              AND tool_name NOT LIKE 'routing:%'
            GROUP BY tool_name
            ORDER BY failures DESC, last_failed_at DESC
            """,
            (cutoff,),
        )

        candidates: list[dict[str, Any]] = []
        for row in rows:
            failures = int(row.get("failures", 0) or 0)
            if failures < min_failures:
                continue
            tool_name = str(row.get("tool_name", "")).strip()
            if not tool_name:
                continue
            sample_error = str(row.get("sample_error", "") or "").strip()
            candidates.append(
                {
                    "tool_name": tool_name,
                    "failures": failures,
                    "last_failed_at": str(row.get("last_failed_at", "")),
                    "sample_error": sample_error,
                }
            )
        return candidates

    def _log_negative_preference_signal(
        self,
        source: str,
        surfaced_summary: str,
        sender_pattern: str,
        owner_feedback: str,
    ):
        try:
            from memory.retriever import get_vectorstore

            vs = get_vectorstore()
            vs.log_preference_signal(
                source=source,
                surfaced_summary=surfaced_summary,
                sender_pattern=sender_pattern,
                owner_feedback=owner_feedback,
                signal_type="self_improve_rejection",
            )
        except Exception:
            log.debug("Failed to log negative preference signal", exc_info=True)

    def _build_failure_diagnostic_tool(
        self,
        source_tool_name: str,
        failures: int,
        sample_error: str = "",
    ) -> tuple[str, str, str]:
        source_slug = self._slug(source_tool_name) or "tool"
        source_ident = source_slug.replace("-", "_")
        tool_name = f"Diagnose {source_slug} failures"
        function_name = f"diagnose_{source_ident}_failures"
        db_path = str(config.MOLLYGRAPH_PATH)
        safe_source_tool_name = source_tool_name.replace("\\", "\\\\").replace("'", "\\'")
        safe_db_path = db_path.replace("\\", "\\\\").replace("'", "\\'")
        safe_error = sample_error.replace("\\", "\\\\").replace("'", "\\'")

        tool_code = textwrap.dedent(
            f"""\
            \"\"\"Operational diagnostics helper generated by self-improvement.

            Read-only helper for inspecting recent failed calls for a specific tool.
            \"\"\"

            import sqlite3
            from datetime import datetime, timezone
            from pathlib import Path

            TARGET_TOOL = '{safe_source_tool_name}'
            DB_PATH = Path('{safe_db_path}')


            def {function_name}(limit: int = 20) -> dict:
                capped = max(1, min(int(limit), 200))
                if not DB_PATH.exists():
                    return {{
                        "tool_name": TARGET_TOOL,
                        "count": 0,
                        "failures": [],
                        "error": f"mollygraph_missing:{{DB_PATH}}",
                    }}

                conn = sqlite3.connect(str(DB_PATH))
                conn.row_factory = sqlite3.Row
                try:
                    rows = conn.execute(
                        \"\"\"
                        SELECT created_at, error_message, parameters
                        FROM tool_calls
                        WHERE tool_name = ?
                          AND success = 0
                        ORDER BY created_at DESC
                        LIMIT ?
                        \"\"\",
                        (TARGET_TOOL, capped),
                    ).fetchall()
                finally:
                    conn.close()

                failures = [
                    {{
                        "created_at": str(row["created_at"] or ""),
                        "error_message": str(row["error_message"] or ""),
                        "parameters": str(row["parameters"] or ""),
                    }}
                    for row in rows
                ]
                return {{
                    "tool_name": TARGET_TOOL,
                    "count": len(failures),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "failures": failures,
                }}
            """
        ).strip() + "\n"

        test_code = textwrap.dedent(
            f"""\
            import importlib.util
            from pathlib import Path


            def _resolve_tool_path() -> Path:
                tools_dir = Path(__file__).resolve().parent.parent / "tools"
                for candidate in sorted(tools_dir.glob("*.py")):
                    try:
                        text = candidate.read_text(encoding="utf-8")
                    except Exception:
                        continue
                    if "def {function_name}" in text:
                        return candidate
                raise FileNotFoundError("Generated tool file not found")


            def _load_module():
                tool_path = _resolve_tool_path()
                spec = importlib.util.spec_from_file_location("candidate_tool", str(tool_path))
                module = importlib.util.module_from_spec(spec)
                assert spec is not None and spec.loader is not None
                spec.loader.exec_module(module)
                return module


            def test_{function_name}_returns_structured_payload():
                module = _load_module()
                fn = getattr(module, "{function_name}")
                payload = fn(limit=5)
                assert isinstance(payload, dict)
                assert payload.get("tool_name") == "{safe_source_tool_name}"
                assert "failures" in payload
                assert isinstance(payload["failures"], list)
            """
        ).strip() + "\n"

        if safe_error:
            test_code += f"\n# Observed sample error while generating this candidate: '{safe_error}'\n"
        if failures > 0:
            test_code += f"# Observed failures in the last window: {failures}\n"
        return tool_name, tool_code, test_code

    async def _propose_tool_updates_from_failures(
        self,
        days: int = _TOOL_GAP_WINDOW_DAYS,
        min_failures: int = _TOOL_GAP_MIN_FAILURES,
    ) -> dict[str, Any]:
        candidates = self._detect_tool_gap_candidates(days=days, min_failures=min_failures)
        if not candidates:
            return {"status": "skipped", "reason": "no recurring tool failures"}

        for candidate in candidates:
            source_tool_name = str(candidate.get("tool_name", "")).strip()
            if not source_tool_name:
                continue
            source_slug = self._slug(source_tool_name) or "tool"
            proposal_title = f"Diagnose {source_slug} failures"
            if self._has_recent_event(category="tool", title=proposal_title, days=30):
                continue
            existing_live = self.project_root / "tools" / f"diagnose-{source_slug}-failures.py"
            if existing_live.exists():
                continue

            gate_msg = (
                "🧩 Tool-gap candidate detected\n\n"
                f"Source tool: {source_tool_name}\n"
                f"Failures ({days}d): {int(candidate.get('failures', 0) or 0)}\n"
                f"Last failure: {str(candidate.get('last_failed_at', '')) or '-'}\n"
                f"Sample error: {str(candidate.get('sample_error', ''))[:240] or '(none)'}\n\n"
                "Reply YES to draft a sandbox tool proposal, NO to skip."
            )
            gate_decision = await self._request_owner_decision(
                category="self-improve-tool-gap",
                description=gate_msg,
                required_keyword="YES",
                allow_edit=False,
            )
            gate_status = "approved" if gate_decision is True else "rejected"
            self._log_improvement_event(
                event_type="proposal",
                category="tool-gap",
                title=proposal_title,
                payload=json.dumps(
                    {
                        "source_tool_name": source_tool_name,
                        "failures": int(candidate.get("failures", 0) or 0),
                        "days": int(days),
                        "sample_error": str(candidate.get("sample_error", ""))[:500],
                    },
                    ensure_ascii=True,
                ),
                status=gate_status,
            )
            if gate_decision is not True:
                self._log_negative_preference_signal(
                    source="self-improve-tool-gap",
                    surfaced_summary=proposal_title,
                    sender_pattern=source_tool_name,
                    owner_feedback="owner_rejected_tool_gap_candidate",
                )
                continue

            tool_name, tool_code, test_code = self._build_failure_diagnostic_tool(
                source_tool_name=source_tool_name,
                failures=int(candidate.get("failures", 0) or 0),
                sample_error=str(candidate.get("sample_error", "") or ""),
            )
            return await self.propose_tool(
                tool_name=tool_name,
                tool_code=tool_code,
                test_code=test_code,
                read_only=True,
                requires_auth=False,
            )

        return {"status": "skipped", "reason": "no new tool candidates"}

    def _detect_workflow_patterns(self, days: int = 30, min_occurrences: int = 3) -> list[dict[str, Any]]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self._rows(
            config.MOLLYGRAPH_PATH,
            """
            SELECT tool_name, created_at
            FROM tool_calls
            WHERE created_at > ?
            ORDER BY created_at ASC
            """,
            (cutoff,),
        )
        tools = []
        for row in rows:
            tool_name = str(row.get("tool_name", "")).strip()
            if self._is_low_value_workflow_tool(tool_name):
                continue
            tools.append(tool_name)

        tool_call_counts: dict[str, int] = {}
        for i in range(0, len(tools) - 2):
            seq = tools[i:i + 3]
            if len(set(seq)) == 1:
                continue
            key = " -> ".join(seq)
            tool_call_counts[key] = tool_call_counts.get(key, 0) + 1

        foundry_signals = self._load_foundry_sequence_signals(days=days)
        if not tool_call_counts and not foundry_signals:
            return []

        existing_ids = self._existing_automation_ids()
        patterns = []
        all_sequences = set(tool_call_counts) | set(foundry_signals)
        for seq in sorted(all_sequences):
            sequence_tools = [step.strip() for step in seq.split(" -> ") if step.strip()]
            if len(sequence_tools) != 3:
                continue
            if any(self._is_low_value_workflow_tool(step) for step in sequence_tools):
                continue
            tool_count = int(tool_call_counts.get(seq, 0) or 0)
            foundry_signal = foundry_signals.get(seq)
            foundry_count = int(foundry_signal.count if foundry_signal else 0)
            total_count = tool_count + foundry_count
            if total_count < min_occurrences:
                continue
            overlap = sum(1 for tid in existing_ids if any(t in tid for t in sequence_tools))
            foundry_success_rate = foundry_signal.success_rate if foundry_signal else 0.0
            foundry_boost = min(0.35, (0.08 * foundry_count) + (0.10 * foundry_success_rate))
            confidence = min(0.99, 0.4 + (0.15 * tool_count) + (0.05 * overlap) + foundry_boost)

            source = "tool_calls"
            if tool_count > 0 and foundry_count > 0:
                source = "tool_calls+foundry"
            elif foundry_count > 0:
                source = "foundry"

            pattern = {
                "name": f"Workflow {sequence_tools[0]}",
                "steps": sequence_tools,
                "steps_text": seq,
                "count": total_count,
                "tool_call_count": tool_count,
                "foundry_count": foundry_count,
                "foundry_success_rate": round(foundry_success_rate, 2),
                "confidence": confidence,
                "trigger": "Detected repeated workflow chain",
                "example": seq,
                "source": source,
            }
            if foundry_signal and foundry_signal.latest_at:
                pattern["foundry_latest_at"] = foundry_signal.latest_at
            patterns.append(pattern)
        return sorted(patterns, key=lambda p: (p["count"], p["confidence"]), reverse=True)

    async def _propose_automation_updates_from_patterns(self, patterns: list[dict[str, Any]]):
        if not patterns:
            return
        from automations import propose_automation

        selected = []
        for pattern in patterns:
            steps = self._pattern_steps(pattern)
            if len(steps) < 3:
                continue
            normalized = dict(pattern)
            normalized["steps"] = " -> ".join(steps)
            selected.append(normalized)
            if len(selected) >= 3:
                break
        if not selected:
            return
        skeleton = propose_automation(selected)
        if not skeleton:
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = self.automations_dir / f"proposal-{ts}.yaml"
        path.write_text(skeleton)
        msg = (
            "🤖 Automation proposal from workflow patterns\n\n"
            f"Patterns considered: {len(selected)}\n"
            f"Draft file: {path}\n"
            "Reply YES to keep draft, NO to discard."
        )
        await self._notify_owner(msg)
        decision = await self._request_owner_decision(
            category="self-improve-automation",
            description=msg,
            required_keyword="YES",
            allow_edit=False,
        )
        status = "approved" if decision is True else "rejected"
        self._log_improvement_event(
            event_type="proposal",
            category="automation",
            title="Workflow-derived automation proposal",
            payload=str(path),
            status=status,
        )
        if decision is not True:
            path.unlink(missing_ok=True)

    def _existing_automation_ids(self) -> set[str]:
        ids = set()
        for path in config.AUTOMATIONS_DIR.glob("*.yaml"):
            ids.add(path.stem.lower())
        return ids

    def _gliner_training_dir(self) -> Path:
        return config.WORKSPACE / "memory" / "gliner_training"

    def _gliner_models_dir(self) -> Path:
        return config.WORKSPACE / "models"

    def _gliner_candidate_model_dir(self) -> Path:
        return self._gliner_models_dir() / "gliner_candidate"

    def _gliner_active_model_dir(self) -> Path:
        return self._gliner_models_dir() / "gliner_active"

    def _gliner_training_config_path(self) -> Path:
        return self.project_root / "memory" / "gliner_finetune_config.json"

    def _gliner_weekly_summary_line(self) -> str:
        total = self._count_accumulated_gliner_examples()
        last_finetune = _parse_datetime(str(self._state.get("gliner_last_finetune_at", "")))
        last_finetune_str = last_finetune.date().isoformat() if last_finetune else "never"
        strategy = str(self._state.get("gliner_last_training_strategy", "lora")).strip().lower() or "lora"
        last_result = str(self._state.get("gliner_last_result", "no fine-tune runs yet")).strip()
        if not last_result:
            last_result = "no fine-tune runs yet"
        return (
            f"GLiNER training data: {total} examples. "
            f"Last fine-tune: {last_finetune_str} ({strategy}). Result: {last_result}"
        )

    def _accumulate_gliner_training_data(self, limit: int = _GLINER_TRAINING_SCAN_LIMIT) -> dict[str, Any]:
        from memory.graph import get_driver

        training_dir = self._gliner_training_dir()
        training_dir.mkdir(parents=True, exist_ok=True)
        seen_episode_ids = self._load_existing_gliner_episode_ids(training_dir)
        opus_analysis_text = self._latest_maintenance_analysis_text()
        correction_texts = self._load_recent_correction_texts()
        cursor = str(self._state.get("gliner_training_cursor", "")).strip()

        where_cursor = "AND ep.created_at > $cursor" if cursor else ""
        params: dict[str, Any] = {"limit": int(limit)}
        if cursor:
            params["cursor"] = cursor

        driver = get_driver()
        with driver.session() as session:
            rows = [
                dict(r)
                for r in session.run(
                    f"""
                    MATCH (ep:Episode)
                    WHERE ep.content_preview IS NOT NULL
                      AND trim(ep.content_preview) <> ''
                      AND ep.entities_extracted IS NOT NULL
                      AND size(ep.entities_extracted) > 0
                      {where_cursor}
                    RETURN ep.id AS episode_id,
                           ep.created_at AS created_at,
                           ep.content_preview AS source_text,
                           ep.entities_extracted AS entity_names
                    ORDER BY ep.created_at ASC
                    LIMIT $limit
                    """,
                    **params,
                )
            ]

            new_examples: list[dict[str, Any]] = []
            latest_seen_created_at = cursor
            for row in rows:
                episode_id = str(row.get("episode_id") or "").strip()
                if not episode_id or episode_id in seen_episode_ids:
                    if row.get("created_at"):
                        latest_seen_created_at = str(row["created_at"])
                    continue

                example = self._build_training_example_from_episode(
                    session=session,
                    episode=row,
                    opus_analysis_text=opus_analysis_text,
                    correction_texts=correction_texts,
                )
                if example:
                    new_examples.append(example)
                    seen_episode_ids.add(episode_id)

                if row.get("created_at"):
                    latest_seen_created_at = str(row["created_at"])

        written_path: str | None = None
        if new_examples:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            batch_path = training_dir / f"examples-{ts}.jsonl"
            with open(batch_path, "w", encoding="utf-8") as f:
                for example in new_examples:
                    f.write(json.dumps(example, ensure_ascii=True) + "\n")
            written_path = str(batch_path)
            log.info("GLiNER accumulation: wrote %d examples to %s", len(new_examples), batch_path)
        else:
            log.info("GLiNER accumulation: no new high-confidence examples this run")

        if latest_seen_created_at:
            self._state["gliner_training_cursor"] = latest_seen_created_at
        total_examples = self._count_accumulated_gliner_examples(training_dir)
        self._state["gliner_training_examples"] = total_examples
        self._save_state()

        return {
            "new_examples": len(new_examples),
            "total_examples": total_examples,
            "batch_path": written_path,
            "cursor": self._state.get("gliner_training_cursor", ""),
        }

    def _load_existing_gliner_episode_ids(self, training_dir: Path) -> set[str]:
        seen: set[str] = set()
        for path in sorted(training_dir.glob("*.jsonl")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        episode_id = str(data.get("episode_id") or "").strip()
                        if episode_id:
                            seen.add(episode_id)
            except Exception:
                log.debug("Failed scanning GLiNER training file %s", path, exc_info=True)
        return seen

    def _latest_maintenance_analysis_text(self) -> str:
        maintenance_dir = config.WORKSPACE / "memory" / "maintenance"
        if not maintenance_dir.exists():
            return ""
        candidates = sorted(maintenance_dir.glob("*.md"))
        if not candidates:
            return ""
        latest_path = candidates[-1]
        try:
            content = latest_path.read_text(encoding="utf-8")
        except Exception:
            return ""
        marker = "\n## Analysis\n"
        if marker in content:
            content = content.split(marker, 1)[1]
        return self._normalize_entity_text(content)

    def _load_recent_correction_texts(self, limit: int = 400) -> list[str]:
        rows = self._rows(
            config.MOLLYGRAPH_PATH,
            """
            SELECT context, molly_output, user_correction
            FROM corrections
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        texts = []
        for row in rows:
            merged = " ".join(
                str(row.get(key, "") or "").strip()
                for key in ("context", "molly_output", "user_correction")
            )
            normalized = self._normalize_entity_text(merged)
            if normalized:
                texts.append(normalized)
        return texts

    def _build_training_example_from_episode(
        self,
        session: Any,
        episode: dict[str, Any],
        opus_analysis_text: str,
        correction_texts: list[str],
    ) -> dict[str, Any] | None:
        source_text = str(episode.get("source_text") or "").strip()
        if not source_text:
            return None
        entity_names_raw = episode.get("entity_names") or []
        if not isinstance(entity_names_raw, list):
            return None
        entity_names = sorted(
            {
                str(name).strip()
                for name in entity_names_raw
                if str(name).strip()
            }
        )
        if not entity_names:
            return None

        entity_rows = [
            dict(r)
            for r in session.run(
                """
                MATCH (e:Entity)
                WHERE e.name IN $names
                RETURN e.name AS name,
                       e.entity_type AS entity_type,
                       coalesce(e.mention_count, 0) AS mention_count,
                       EXISTS { MATCH (e)--(:Entity) } AS has_relationship
                """,
                names=entity_names,
            )
        ]
        if not entity_rows:
            return None

        selected_entities: list[dict[str, Any]] = []
        signal_counts = {
            "opus_confirmed": 0,
            "multi_mentions": 0,
            "relationship_backed": 0,
        }
        for entity in entity_rows:
            name = str(entity.get("name") or "").strip()
            label = str(entity.get("entity_type") or "Concept").strip() or "Concept"
            mention_count = int(entity.get("mention_count") or 0)
            has_relationship = bool(entity.get("has_relationship"))
            normalized = self._normalize_entity_text(name)
            if not normalized:
                continue

            corrected = any(normalized in text for text in correction_texts)
            opus_confirmed = bool(opus_analysis_text) and (normalized in opus_analysis_text) and not corrected
            multi_mentions = mention_count >= 2
            relationship_backed = has_relationship

            if not (opus_confirmed or multi_mentions or relationship_backed):
                continue

            if opus_confirmed:
                signal_counts["opus_confirmed"] += 1
            if multi_mentions:
                signal_counts["multi_mentions"] += 1
            if relationship_backed:
                signal_counts["relationship_backed"] += 1

            selected_entities.append(
                {
                    "text": name,
                    "label": label,
                }
            )

        if not selected_entities:
            return None

        selected_names = sorted({str(entity["text"]).strip() for entity in selected_entities})
        relation_rows = [
            dict(r)
            for r in session.run(
                """
                MATCH (h:Entity)-[r]->(t:Entity)
                WHERE h.name IN $names AND t.name IN $names
                RETURN h.name AS head,
                       t.name AS tail,
                       type(r) AS label,
                       coalesce(r.mention_count, 0) AS mention_count
                """,
                names=selected_names,
            )
        ]
        relations: list[dict[str, Any]] = []
        seen_rel_keys: set[tuple[str, str, str]] = set()
        for rel in relation_rows:
            head = str(rel.get("head") or "").strip()
            tail = str(rel.get("tail") or "").strip()
            label = str(rel.get("label") or "").strip()
            if not head or not tail or not label:
                continue
            key = (head, label, tail)
            if key in seen_rel_keys:
                continue
            seen_rel_keys.add(key)
            relations.append(
                {
                    "head": head,
                    "tail": tail,
                    "label": label,
                }
            )

        return {
            "episode_id": str(episode.get("episode_id") or ""),
            "created_at": str(episode.get("created_at") or ""),
            "source_text": source_text,
            "extracted_entities": selected_entities,
            "extracted_relations": relations,
            "quality_signals": signal_counts,
        }

    def _count_accumulated_gliner_examples(self, training_dir: Path | None = None) -> int:
        target_dir = training_dir or self._gliner_training_dir()
        if not target_dir.exists():
            return 0
        total = 0
        for path in target_dir.glob("*.jsonl"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    total += sum(1 for line in f if line.strip())
            except Exception:
                log.debug("Failed counting GLiNER examples in %s", path, exc_info=True)
        return total

    def _load_accumulated_gliner_examples(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        training_dir = self._gliner_training_dir()
        if not training_dir.exists():
            return rows
        for path in sorted(training_dir.glob("*.jsonl")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        row = self._to_benchmark_row(payload)
                        if row:
                            rows.append(row)
            except Exception:
                log.debug("Failed loading GLiNER examples from %s", path, exc_info=True)
        return rows

    def _to_benchmark_row(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        text = str(payload.get("source_text") or payload.get("text") or "").strip()
        entities = payload.get("extracted_entities", payload.get("entities", []))
        relations = payload.get("extracted_relations", payload.get("relations", []))
        if not text or not isinstance(entities, list) or not entities:
            return None
        return {
            "episode_id": str(payload.get("episode_id") or ""),
            "created_at": str(payload.get("created_at") or ""),
            "text": text,
            "entities": entities,
            "relations": relations if isinstance(relations, list) else [],
        }

    def _to_gliner_training_record(self, row: dict[str, Any]) -> dict[str, Any] | None:
        text = str(row.get("text") or "").strip()
        if not text:
            return None
        text_norm = self._normalize_entity_text(text)

        entities_map: dict[str, list[str]] = {}
        entity_names: set[str] = set()
        for ent in row.get("entities") or []:
            if isinstance(ent, str):
                name = ent.strip()
                label = "Concept"
            elif isinstance(ent, dict):
                name = str(ent.get("text") or ent.get("name") or "").strip()
                label = str(ent.get("label") or ent.get("type") or "Concept").strip() or "Concept"
            else:
                continue
            if not name:
                continue
            if self._normalize_entity_text(name) not in text_norm:
                continue
            entities_map.setdefault(label, [])
            if name not in entities_map[label]:
                entities_map[label].append(name)
                entity_names.add(name)

        relations_out: list[dict[str, Any]] = []
        for rel in row.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            head = str(rel.get("head") or "").strip()
            tail = str(rel.get("tail") or "").strip()
            label = str(rel.get("label") or "").strip().lower().replace("_", " ")
            if not head or not tail or not label:
                continue
            if head not in entity_names or tail not in entity_names:
                continue
            if self._normalize_entity_text(head) not in text_norm:
                continue
            if self._normalize_entity_text(tail) not in text_norm:
                continue
            relations_out.append({label: {"head": head, "tail": tail}})

        output: dict[str, Any] = {}
        if entities_map:
            output["entities"] = entities_map
        if relations_out:
            output["relations"] = relations_out
        if not output:
            return None
        return {"input": text, "output": output}

    def _active_gliner_model_ref(self) -> str:
        active_dir = self._gliner_active_model_dir()
        if active_dir.exists():
            return str(active_dir)
        active_state_ref = str(self._state.get("gliner_active_model_ref", "")).strip()
        if active_state_ref and Path(active_state_ref).exists():
            return active_state_ref
        return _GLINER_BASE_MODEL

    def _select_gliner_training_strategy(self, total_examples: int) -> dict[str, Any]:
        full_min_examples = max(1, int(config.GLINER_FULL_FINETUNE_MIN_EXAMPLES))
        plateau_window = max(1, int(config.GLINER_LORA_PLATEAU_WINDOW))
        plateau_epsilon = max(0.0, float(config.GLINER_LORA_PLATEAU_EPSILON))
        if total_examples < full_min_examples:
            return {
                "mode": "lora",
                "reason": "insufficient_examples_for_full_finetune",
                "full_min_examples": full_min_examples,
                "total_examples": total_examples,
            }

        history = self._state.get("gliner_benchmark_history")
        if not isinstance(history, list):
            history = []
        recent_lora = [
            row for row in history
            if isinstance(row, dict)
            and str(row.get("strategy", "")).lower() == "lora"
            and bool(row.get("benchmark_ok"))
        ]
        recent_lora = recent_lora[-plateau_window:]
        if len(recent_lora) < plateau_window:
            return {
                "mode": "lora",
                "reason": "not_enough_lora_history",
                "required_runs": plateau_window,
                "available_runs": len(recent_lora),
                "total_examples": total_examples,
            }

        improvements = [
            float(row.get("improvement", 0.0) or 0.0)
            for row in recent_lora
        ]
        max_gain = max(improvements) if improvements else 0.0
        plateaued = max_gain <= plateau_epsilon
        if plateaued:
            return {
                "mode": "full",
                "reason": "lora_plateau_detected",
                "plateau_window": plateau_window,
                "plateau_epsilon": plateau_epsilon,
                "recent_improvements": improvements,
                "total_examples": total_examples,
            }
        return {
            "mode": "lora",
            "reason": "lora_still_improving",
            "plateau_window": plateau_window,
            "plateau_epsilon": plateau_epsilon,
            "recent_improvements": improvements,
            "total_examples": total_examples,
        }

    def _record_gliner_benchmark(
        self,
        strategy: str,
        benchmark: dict[str, Any],
        status: str,
        total_examples: int,
    ):
        history = self._state.get("gliner_benchmark_history")
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "strategy": strategy,
                "status": status,
                "total_examples": int(total_examples),
                "benchmark_ok": bool(benchmark.get("ok", False)),
                "improvement": float(benchmark.get("improvement", 0.0) or 0.0),
                "base_score": float(benchmark.get("base_score", 0.0) or 0.0),
                "candidate_score": float(benchmark.get("candidate_score", 0.0) or 0.0),
                "eval_count": int(benchmark.get("split", {}).get("eval_count", 0) or 0),
            }
        )
        self._state["gliner_benchmark_history"] = history[-20:]
        self._save_state()

    def _fine_tune_gliner_candidate(self, train_rows: list[dict[str, Any]], mode: str = "lora") -> dict[str, Any]:
        train_records = [r for r in (self._to_gliner_training_record(row) for row in train_rows) if r]
        if not train_records:
            return {"ok": False, "error": "no_valid_train_records"}

        from gliner2.training.trainer import train_gliner2

        models_dir = self._gliner_models_dir()
        candidate_dir = self._gliner_candidate_model_dir()
        runs_dir = models_dir / "gliner_runs"
        splits_dir = self._gliner_training_dir() / "splits"
        models_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        train_split_path = splits_dir / f"train-{ts}.jsonl"
        with open(train_split_path, "w", encoding="utf-8") as f:
            for record in train_records:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

        output_dir = runs_dir / ts
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_size = min(4, max(1, len(train_records)))
        normalized_mode = "full" if str(mode).strip().lower() == "full" else "lora"
        use_lora = normalized_mode == "lora"
        num_epochs = 1 if use_lora else 2

        try:
            result = train_gliner2(
                model_path=self._active_gliner_model_ref(),
                train_data=train_records,
                output_dir=str(output_dir),
                num_epochs=num_epochs,
                batch_size=batch_size,
                eval_strategy="no",
                fp16=False,
                bf16=False,
                num_workers=0,
                logging_steps=max(1, min(25, len(train_records))),
                use_lora=use_lora,
                save_adapter_only=False,
                seed=_GLINER_BENCHMARK_SEED,
            )
        except Exception as exc:
            log.error("GLiNER fine-tuning failed", exc_info=True)
            return {"ok": False, "error": str(exc)}

        final_dir = output_dir / "final"
        if not final_dir.exists():
            best_dir = output_dir / "best"
            if best_dir.exists():
                final_dir = best_dir
            else:
                return {"ok": False, "error": "trained_model_not_found", "output_dir": str(output_dir)}

        self._discard_gliner_candidate_model(candidate_dir)
        shutil.copytree(final_dir, candidate_dir, dirs_exist_ok=False)
        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "base_model": self._active_gliner_model_ref(),
            "train_examples": len(train_records),
            "batch_size": batch_size,
            "mode": normalized_mode,
            "num_epochs": num_epochs,
            "output_dir": str(output_dir),
            "result": result,
        }
        metadata_path = candidate_dir / "fine_tune_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True))

        return {
            "ok": True,
            "candidate_model": str(candidate_dir),
            "train_split_path": str(train_split_path),
            "output_dir": str(output_dir),
            "metadata_path": str(metadata_path),
            "mode": normalized_mode,
            "result": result,
        }

    def _discard_gliner_candidate_model(self, candidate_path: Path | None):
        if not candidate_path:
            return
        if candidate_path.exists():
            shutil.rmtree(candidate_path, ignore_errors=True)

    def _format_gliner_swap_proposal(self, benchmark: dict[str, Any], mode: str = "lora") -> str:
        base_metrics = benchmark.get("base", {}).get("metrics", {})
        candidate_metrics = benchmark.get("candidate", {}).get("metrics", {})
        base_precision = float(base_metrics.get("precision", 0.0) or 0.0)
        cand_precision = float(candidate_metrics.get("precision", 0.0) or 0.0)
        eval_count = int(benchmark.get("split", {}).get("eval_count", 0) or 0)
        strategy_label = "full fine-tune" if str(mode).strip().lower() == "full" else "LoRA fine-tune"
        return (
            f"GLiNER {strategy_label} ready: precision improved "
            f"from {base_precision:.2f} to {cand_precision:.2f} "
            f"on {eval_count} held-out examples. Approve model swap?\n\n"
            "Reply YES to deploy or NO to keep the current model."
        )

    def _deploy_gliner_candidate_model(
        self,
        candidate_path: Path,
        benchmark: dict[str, Any],
        fine_tune: dict[str, Any],
    ) -> dict[str, Any]:
        if not candidate_path.exists():
            return {"ok": False, "error": "candidate_path_missing"}

        active_dir = self._gliner_active_model_dir()
        active_dir.parent.mkdir(parents=True, exist_ok=True)
        previous_active_ref = self._active_gliner_model_ref()

        backup_dir = None
        if active_dir.exists():
            backup_root = self._gliner_models_dir() / "gliner_backups"
            backup_root.mkdir(parents=True, exist_ok=True)
            backup_dir = backup_root / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            shutil.copytree(active_dir, backup_dir, dirs_exist_ok=False)

        if active_dir.exists():
            shutil.rmtree(active_dir, ignore_errors=True)
        shutil.copytree(candidate_path, active_dir, dirs_exist_ok=False)

        deployed_at = datetime.now(timezone.utc).isoformat()
        self._state["gliner_active_model_ref"] = str(active_dir)
        self._state["gliner_last_deployed_at"] = deployed_at
        self._save_state()

        config_payload = {
            "updated_at": deployed_at,
            "active_model_ref": str(active_dir),
            "previous_model_ref": previous_active_ref,
            "backup_model_ref": str(backup_dir) if backup_dir else None,
            "benchmark": benchmark,
            "fine_tune": {
                "mode": fine_tune.get("mode", "lora"),
                "train_split_path": fine_tune.get("train_split_path"),
                "output_dir": fine_tune.get("output_dir"),
                "metadata_path": fine_tune.get("metadata_path"),
            },
        }
        config_path = self._gliner_training_config_path()
        config_path.write_text(json.dumps(config_payload, indent=2, ensure_ascii=True))
        commit = self._commit_gliner_training_config(config_path)

        return {
            "ok": True,
            "active_model": str(active_dir),
            "backup_model": str(backup_dir) if backup_dir else None,
            "training_config": str(config_path),
            "git": commit,
        }

    def _commit_gliner_training_config(self, config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            return {"ok": False, "error": "config_missing"}
        try:
            rel_path = config_path.relative_to(self.project_root)
        except ValueError:
            rel_path = config_path
        rel = str(rel_path)

        try:
            self._git(["add", "--", rel])
            staged = self._git(["diff", "--cached", "--name-only", "--", rel], check=False)
            if not staged.stdout.strip():
                return {"ok": True, "status": "no_changes"}
            commit_msg = f"[molly-self-improve] GLiNER fine-tune config update ({date.today().isoformat()})"
            commit = self._git(
                ["commit", "-m", commit_msg, "--", rel],
                check=False,
                env_override=_GIT_AUTHOR_ENV,
            )
            if commit.returncode != 0:
                return {
                    "ok": False,
                    "status": "commit_failed",
                    "error": (commit.stderr or commit.stdout).strip()[:1000],
                }
            commit_hash = self._git(["rev-parse", "HEAD"], check=False).stdout.strip()
            return {"ok": True, "status": "committed", "commit": commit_hash}
        except Exception as exc:
            return {"ok": False, "status": "commit_exception", "error": str(exc)}

    def _split_holdout_rows(
        self,
        rows: list[dict[str, Any]],
        eval_ratio: float = _GLINER_BENCHMARK_EVAL_RATIO,
        seed: int = _GLINER_BENCHMARK_SEED,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not rows:
            return [], []
        indices = list(range(len(rows)))
        random.Random(seed).shuffle(indices)
        if len(rows) == 1:
            eval_count = 1
        else:
            eval_count = max(1, int(round(len(rows) * eval_ratio)))
            eval_count = min(eval_count, len(rows) - 1)
        eval_idx = set(indices[:eval_count])
        train_rows = [rows[i] for i in range(len(rows)) if i not in eval_idx]
        eval_rows = [rows[i] for i in range(len(rows)) if i in eval_idx]
        return train_rows, eval_rows

    def _normalize_entity_text(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        return re.sub(r"\s+", " ", text)

    def _extract_expected_entity_set(self, row: dict[str, Any]) -> set[str]:
        entities = row.get("entities") or []
        if not isinstance(entities, list):
            return set()
        normalized: set[str] = set()
        for ent in entities:
            if isinstance(ent, str):
                name = self._normalize_entity_text(ent)
            elif isinstance(ent, dict):
                name = self._normalize_entity_text(
                    ent.get("text") or ent.get("name") or ent.get("entity")
                )
            else:
                name = ""
            if name:
                normalized.add(name)
        return normalized

    def _extract_predicted_entity_set(self, result: Any) -> set[str]:
        if not isinstance(result, dict):
            return set()
        entity_dict = result.get("entities", result)
        if not isinstance(entity_dict, dict):
            return set()
        predicted: set[str] = set()
        for items in entity_dict.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, str):
                    name = self._normalize_entity_text(item)
                elif isinstance(item, dict):
                    name = self._normalize_entity_text(item.get("text"))
                else:
                    name = ""
                if name:
                    predicted.add(name)
        return predicted

    def _compute_prf_metrics(self, tp: int, fp: int, fn: int) -> dict[str, float]:
        precision = (tp / (tp + fp)) if (tp + fp) else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    def _load_gliner_entity_model(self, model_ref: str) -> tuple[Any, Any]:
        from gliner2 import GLiNER2
        from memory.extractor import ENTITY_SCHEMA

        model = GLiNER2.from_pretrained(model_ref)
        schema = model.create_schema().entities(ENTITY_SCHEMA)
        return model, schema

    def _evaluate_model_on_rows(
        self,
        model_ref: str,
        rows: list[dict[str, Any]],
        threshold: float = _GLINER_BENCHMARK_THRESHOLD,
    ) -> dict[str, Any]:
        if not rows:
            return {
                "ok": False,
                "error": "empty_eval_set",
                "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "counts": {"tp": 0, "fp": 0, "fn": 0},
                "rows_total": 0,
                "rows_evaluated": 0,
                "rows_failed": 0,
                "latency_ms_avg": 0.0,
            }
        try:
            model, schema = self._load_gliner_entity_model(model_ref)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"model_load_failed: {exc}",
                "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "counts": {"tp": 0, "fp": 0, "fn": 0},
                "rows_total": len(rows),
                "rows_evaluated": 0,
                "rows_failed": len(rows),
                "latency_ms_avg": 0.0,
            }

        tp = 0
        fp = 0
        fn = 0
        rows_evaluated = 0
        rows_failed = 0
        latency_sum_ms = 0.0
        failure_samples: list[dict[str, Any]] = []

        for idx, row in enumerate(rows):
            text = str(row.get("text") or "").strip()
            if not text:
                rows_failed += 1
                if len(failure_samples) < 5:
                    failure_samples.append({"row_index": idx, "error": "missing_text"})
                continue

            expected = self._extract_expected_entity_set(row)
            try:
                t0 = time.monotonic()
                result = model.extract(
                    text,
                    schema,
                    threshold=threshold,
                    include_confidence=True,
                )
                latency_sum_ms += (time.monotonic() - t0) * 1000.0
            except Exception as exc:
                rows_failed += 1
                if len(failure_samples) < 5:
                    failure_samples.append({"row_index": idx, "error": str(exc)[:300]})
                continue

            predicted = self._extract_predicted_entity_set(result)
            tp += len(predicted & expected)
            fp += len(predicted - expected)
            fn += len(expected - predicted)
            rows_evaluated += 1

        if rows_evaluated == 0:
            return {
                "ok": False,
                "error": "all_inference_failed",
                "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "counts": {"tp": tp, "fp": fp, "fn": fn},
                "rows_total": len(rows),
                "rows_evaluated": 0,
                "rows_failed": rows_failed,
                "latency_ms_avg": 0.0,
                "failure_samples": failure_samples,
            }

        metrics = self._compute_prf_metrics(tp=tp, fp=fp, fn=fn)
        return {
            "ok": rows_failed == 0,
            "error": "" if rows_failed == 0 else "partial_inference_failures",
            "metrics": metrics,
            "counts": {"tp": tp, "fp": fp, "fn": fn},
            "rows_total": len(rows),
            "rows_evaluated": rows_evaluated,
            "rows_failed": rows_failed,
            "latency_ms_avg": round(latency_sum_ms / rows_evaluated, 2),
            "failure_samples": failure_samples,
        }

    def _benchmark_finetune_candidate(
        self,
        rows: list[dict[str, Any]],
        candidate_model_ref: str | None = None,
        train_count: int | None = None,
    ) -> dict[str, Any]:
        eval_rows = rows
        resolved_train_count = int(train_count or 0)
        if train_count is None:
            split_train, split_eval = self._split_holdout_rows(rows)
            eval_rows = split_eval
            resolved_train_count = len(split_train)

        if not eval_rows:
            return {
                "ok": False,
                "base_score": 0.0,
                "candidate_score": 0.0,
                "improvement": 0.0,
                "split": {"seed": _GLINER_BENCHMARK_SEED, "train_count": resolved_train_count, "eval_count": 0},
                "failure": {"reason": "no_eval_rows"},
            }

        base_model_ref = self._active_gliner_model_ref()
        candidate_model_ref = (candidate_model_ref or "").strip()
        if not candidate_model_ref:
            candidate_path = self._gliner_candidate_model_dir()
            if candidate_path.exists():
                candidate_model_ref = str(candidate_path)
        if not candidate_model_ref:
            return {
                "ok": False,
                "base_score": 0.0,
                "candidate_score": 0.0,
                "improvement": 0.0,
                "split": {
                    "seed": _GLINER_BENCHMARK_SEED,
                    "train_count": resolved_train_count,
                    "eval_count": len(eval_rows),
                },
                "failure": {"reason": "candidate_model_missing"},
            }

        base_eval = self._evaluate_model_on_rows(base_model_ref, eval_rows)
        candidate_eval = self._evaluate_model_on_rows(candidate_model_ref, eval_rows)

        base_score = float(base_eval.get("metrics", {}).get("f1", 0.0) or 0.0)
        candidate_score = float(candidate_eval.get("metrics", {}).get("f1", 0.0) or 0.0)
        benchmark_ok = bool(base_eval.get("ok")) and bool(candidate_eval.get("ok"))
        improvement = (candidate_score - base_score) if benchmark_ok else 0.0

        failure_details = []
        if not base_eval.get("ok"):
            failure_details.append(
                {"model": "base", "error": base_eval.get("error", "unknown")}
            )
        if not candidate_eval.get("ok"):
            failure_details.append(
                {"model": "candidate", "error": candidate_eval.get("error", "unknown")}
            )

        return {
            "ok": benchmark_ok,
            "split": {
                "seed": _GLINER_BENCHMARK_SEED,
                "train_count": resolved_train_count,
                "eval_count": len(eval_rows),
            },
            "base_model": base_model_ref,
            "candidate_model": candidate_model_ref or None,
            "base": base_eval,
            "candidate": candidate_eval,
            "base_score": round(base_score, 4),
            "candidate_score": round(candidate_score, 4),
            "improvement": round(improvement, 4),
            "failure": None if benchmark_ok else {"reason": "model_evaluation_failed", "details": failure_details},
        }

    def _count_rows(self, db_path: Path, sql: str, params: tuple[Any, ...]) -> int:
        try:
            conn = sqlite3.connect(str(db_path))
            value = conn.execute(sql, params).fetchone()[0]
            conn.close()
            return int(value or 0)
        except Exception:
            return 0

    def _scalar(
        self,
        db_path: Path,
        sql: str,
        params: tuple[Any, ...],
        default: float = 0.0,
    ) -> float:
        try:
            conn = sqlite3.connect(str(db_path))
            value = conn.execute(sql, params).fetchone()[0]
            conn.close()
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _rows(self, db_path: Path, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
            conn.close()
            return rows
        except Exception:
            return []

    def _format_core_patch_proposal(
        self,
        description: str,
        branch: str,
        tests_ok: bool,
        tests_log_path: Path,
        diff_summary: str,
    ) -> str:
        return textwrap.dedent(
            f"""\
            🔬 Code change proposal: "{description.strip()}"

            Branch: {branch}
            Tests: {"PASS" if tests_ok else "FAIL"}
            Test log: {tests_log_path}

            {diff_summary[:2000]}

            Reply DEPLOY to merge, NO to discard.
            """
        ).strip()

    def _record_pending_deploy(
        self,
        commit_hash: str,
        description: str,
        deployed_at: datetime,
        baseline_health: str,
    ):
        self._state["pending_deploy"] = {
            "commit_hash": commit_hash,
            "description": description,
            "deployed_at": deployed_at.isoformat(),
            "healthy": False,
            "baseline_health": baseline_health[:20000],
            "startup_validation": "pending",
        }
        self._save_state()

    @staticmethod
    def _format_health_state_for_log(status_map: dict[str, str]) -> str:
        if not status_map:
            return "(no component data)"
        return json.dumps(dict(sorted(status_map.items())), ensure_ascii=True)

    @staticmethod
    def _format_health_regressions_for_log(regressions: list[dict[str, str]]) -> str:
        if not regressions:
            return "(none)"
        labels: list[str] = []
        for row in regressions:
            check_id = str(row.get("id", "")).strip()
            before = str(row.get("before", "")).strip()
            after = str(row.get("after", "")).strip()
            if check_id:
                labels.append(f"{check_id}:{before}->{after}")
        return ", ".join(labels[:10]) if labels else "(none)"

    async def _post_deploy_health_regression_check(
        self,
        baseline_health: str,
        *,
        apply_grace_period: bool = False,
    ) -> str | None:
        if not config.HEALTH_POST_DEPLOY_CHECK:
            return None
        try:
            from health import get_health_doctor

            doctor = get_health_doctor(self.molly)
            baseline_state = doctor.extract_status_map(baseline_health)
            log.info(
                "Post-deploy health pre-edit baseline by component: %s",
                self._format_health_state_for_log(baseline_state),
            )

            if apply_grace_period:
                # Freshly deployed edits can briefly trigger noise (DB reconnect/model warmup/token refresh).
                await asyncio.sleep(_POST_DEPLOY_HEALTH_GRACE_SECONDS)

            first_candidate = await asyncio.to_thread(
                doctor.generate_report, abbreviated=True, trigger="post-deploy"
            )
            first_state = doctor.extract_status_map(first_candidate)
            first_regressions = doctor.compare_worsened_components(
                baseline_report=baseline_health,
                candidate_report=first_candidate,
            )
            log.info(
                "Post-deploy health first check by component: %s",
                self._format_health_state_for_log(first_state),
            )
            log.info(
                "Post-deploy health first-check regressions: %s",
                self._format_health_regressions_for_log(first_regressions),
            )
            if not first_regressions:
                log.info("Post-deploy rollback decision: keep (no components worsened from baseline)")
                return None

            await asyncio.sleep(_POST_DEPLOY_HEALTH_RETRY_SECONDS)
            retry_candidate = await asyncio.to_thread(
                doctor.generate_report, abbreviated=True, trigger="post-deploy-retry"
            )
            retry_state = doctor.extract_status_map(retry_candidate)
            retry_regressions = doctor.compare_worsened_components(
                baseline_report=baseline_health,
                candidate_report=retry_candidate,
            )
            log.info(
                "Post-deploy health retry check by component: %s",
                self._format_health_state_for_log(retry_state),
            )
            log.info(
                "Post-deploy health retry-check regressions: %s",
                self._format_health_regressions_for_log(retry_regressions),
            )

            first_by_id = {str(row.get("id", "")).strip(): row for row in first_regressions if row.get("id")}
            retry_by_id = {str(row.get("id", "")).strip(): row for row in retry_regressions if row.get("id")}
            persistent_ids = sorted(check_id for check_id in first_by_id if check_id in retry_by_id)
            if not persistent_ids:
                log.info(
                    "Post-deploy rollback decision: keep (first-check regressions cleared on retry)"
                )
                return None

            persistent = [retry_by_id[check_id] for check_id in persistent_ids]
            reason_detail = self._format_health_regressions_for_log(persistent)
            log.warning(
                "Post-deploy rollback decision: rollback (persistent component regressions: %s)",
                reason_detail,
            )
            return f"post-deploy health regression persisted after retry: {reason_detail}"
        except Exception:
            log.debug("Post-deploy health check failed", exc_info=True)
        return None

    async def _validate_pending_deploy_on_startup(self):
        pending = self._state.get("pending_deploy") or {}
        if not pending or pending.get("healthy"):
            return

        commit = str(pending.get("commit_hash", "")).strip()
        if not commit:
            self._state["pending_deploy"] = None
            self._save_state()
            return

        baseline = str(pending.get("baseline_health", "") or "")
        regression = None
        if baseline:
            regression = await self._post_deploy_health_regression_check(
                baseline,
                apply_grace_period=True,
            )
        else:
            log.warning(
                "Pending deploy %s missing baseline health; skipping startup regression check.",
                commit,
            )

        if regression:
            pending["startup_validation"] = "failed"
            self._state["pending_deploy"] = pending
            self._save_state()
            await self._rollback_commit(commit, f"startup health regression: {regression}")
            self._request_runtime_restart("rollback applied after startup health validation")
            return

        pending["healthy"] = True
        pending["startup_validation"] = "passed"
        pending["startup_validated_at"] = datetime.now(timezone.utc).isoformat()
        self._state["pending_deploy"] = pending
        self._save_state()
        log.info("Startup deploy validation passed for commit %s", commit)

    async def _mark_deploy_stable_if_window_elapsed(self, now_local: datetime):
        pending = self._state.get("pending_deploy") or {}
        if not pending or pending.get("healthy"):
            return
        deployed_at = _parse_datetime(pending.get("deployed_at", ""))
        if not deployed_at:
            return
        now_utc = now_local.astimezone(timezone.utc)
        age = (now_utc - deployed_at).total_seconds()
        if age < config.SELF_EDIT_AUTO_ROLLBACK_WINDOW:
            return
        # Active health check before marking stable
        commit = str(pending.get("commit_hash", "")).strip()
        baseline = pending.get("baseline_health", "")
        if commit and baseline:
            regression = await self._post_deploy_health_regression_check(baseline)
            if regression:
                await self._rollback_commit(
                    commit, f"stability-window health regression: {regression}"
                )
                return
        pending["healthy"] = True
        self._state["pending_deploy"] = pending
        self._save_state()

    async def _rollback_commit(self, commit_hash: str, reason: str):
        try:
            await self._agit(["revert", "--no-edit", commit_hash], env_override=_GIT_AUTHOR_ENV)
            self._log_improvement_event(
                event_type="deploy",
                category="core",
                title="Auto rollback",
                payload=f"commit={commit_hash};reason={reason}",
                status="rolled_back",
            )
            await self._notify_owner(
                f"⚠️ Self-edit rollback executed.\nCommit: {commit_hash}\nReason: {reason}"
            )
        except Exception as exc:
            log.error("Auto rollback failed: %s", exc, exc_info=True)
            await self._notify_owner(
                f"⚠️ Rollback attempt failed for commit {commit_hash}: {exc}"
            )
        finally:
            self._state["pending_deploy"] = None
            self._save_state()

    def _is_weekly_assessment_due(self, now_local: datetime) -> bool:
        target_day = config.WEEKLY_ASSESSMENT_DAY.strip().lower()
        weekday_map = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        if now_local.weekday() != weekday_map.get(target_day, 6):
            return False
        if now_local.hour != config.WEEKLY_ASSESSMENT_HOUR:
            return False
        last = str(self._state.get("last_weekly_assessment", ""))
        return last != now_local.date().isoformat()


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None
