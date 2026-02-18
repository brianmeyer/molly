import asyncio
import difflib
import json
import logging
import os
import tempfile
import textwrap
from contextlib import suppress
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from utils import atomic_write

import config
from evolution._base import PatchValidation
from evolution.context import EngineContext
from evolution.pattern_helpers import pattern_steps as _pattern_steps_fn

log = logging.getLogger(__name__)


def _atomic_write(path: Path, text: str) -> None:
    """Write text to *path* atomically via tempfile + os.replace."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, text.encode())
        os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            os.close(fd)
        with suppress(OSError):
            os.unlink(tmp)
        raise


from evolution.infra import (
    _parse_datetime,
    _GIT_AUTHOR_ENV,
    _POST_DEPLOY_HEALTH_GRACE_SECONDS,
    _POST_DEPLOY_HEALTH_RETRY_SECONDS,
    _SKILL_GAP_MIN_CLUSTER_SIZE,
    _TOOL_GAP_MIN_FAILURES,
    _TOOL_GAP_WINDOW_DAYS,
)

# Service imports (deferred construction in __init__)
from evolution.infra import InfraService
from evolution.owner_comms import OwnerCommsService
from evolution.graph_ops import GraphOpsService
from evolution.stats import StatsService
from evolution.automation_patterns import AutomationPatternsService
from evolution.skill_lifecycle import SkillLifecycleService
from evolution.skill_gaps import SkillGapsService
from evolution.tool_gaps import ToolGapsService
from evolution.gliner_training import GLiNERTrainingService
from evolution.qwen_training import QwenTrainingService


class SelfImprovementEngine:
    """Phase 7 self-improvement loop.

    This module is intentionally conservative:
    - all code edits are branch + test + approval gated
    - protected files are blocked
    - rollback paths are explicit and logged

    Architecture: composed services with explicit dependency injection.
    The public API is unchanged from the original design.
    """

    def __init__(self, molly=None):
        # Shared context replaces implicit self.* attribute contract
        self.ctx = EngineContext(molly=molly)

        # Composed services — explicit dependency graph (acyclic)
        self.infra = InfraService(self.ctx)
        self.comms = OwnerCommsService(self.ctx)
        self.graph = GraphOpsService(self.ctx, self.comms)
        self.stats = StatsService(self.ctx, self.infra, self.graph)
        self.auto_patterns = AutomationPatternsService(self.ctx, self.infra, self.comms)
        self.skill_lifecycle = SkillLifecycleService(self.ctx, self.infra, self.comms, self.auto_patterns)
        self.skill_gaps = SkillGapsService(self.ctx, self.infra, self.comms, self.skill_lifecycle)
        self.tool_gaps = ToolGapsService(self.ctx, self.infra, self.comms, self.skill_gaps)
        self.gliner = GLiNERTrainingService(self.ctx, self.infra, self.comms)
        self.qwen = QwenTrainingService(self.ctx, self.infra, self.comms)

        self._initialized = False
        self._last_tick_at: datetime | None = None

    # ------------------------------------------------------------------
    # Public API delegators (facade contract)
    # ------------------------------------------------------------------

    async def propose_skill_from_patterns(self, *args, **kwargs):
        return await self.skill_lifecycle.propose_skill_from_patterns(*args, **kwargs)

    async def propose_skill_lifecycle(self, *args, **kwargs):
        return await self.skill_lifecycle.propose_skill_lifecycle(*args, **kwargs)

    def should_trigger_owner_skill_phrase(self, *args, **kwargs):
        return self.skill_lifecycle.should_trigger_owner_skill_phrase(*args, **kwargs)

    async def propose_skill_from_owner_phrase(self, *args, **kwargs):
        return await self.skill_lifecycle.propose_skill_from_owner_phrase(*args, **kwargs)

    async def run_gliner_nightly_cycle(self):
        return await self.gliner.run_gliner_nightly_cycle()

    async def run_qwen_nightly_cycle(self):
        return await self.qwen.run_nightly_cycle()

    async def run_gliner_finetune_pipeline(self):
        return await self.gliner.run_gliner_finetune_pipeline()

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        if self._initialized:
            return
        await asyncio.to_thread(self.infra.ensure_sandbox_dirs)
        await asyncio.to_thread(self.ctx.load_state)
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
        last = str(self.ctx.state.get("last_weekly_assessment", ""))
        return last != now_local.date().isoformat()

    async def run_memory_optimization(self) -> dict[str, Any]:
        """Nightly memory maintenance extension (Phase 7)."""
        await self.initialize()
        results = await asyncio.to_thread(self.graph.run_memory_optimization_sync)
        samples = results.get("contradiction_samples", [])
        if samples:
            msg = (
                "⚠️ Memory contradiction candidates:\n"
                + "\n".join(f"- {c['entity']}: {', '.join(c['values'])}" for c in samples[:5])
                + "\n\nReply with clarifications and I will update the graph."
            )
            await self.comms.notify_owner(msg)
        return results

    async def propose_core_patch(self, description: str, patch_text: str) -> dict[str, Any]:
        """Capability 3: guarded core code patch workflow."""
        await self.initialize()
        validation = self.validate_patch(patch_text)
        if not validation.ok:
            return {"status": "rejected", "reason": validation.reason}

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        slug = self.infra.slug(description)
        branch = f"molly/improvement-{ts}-{slug}"
        patch_path = self.ctx.patches_dir / f"{ts}-{slug}.patch"
        patch_path.write_text(patch_text)

        base_branch = (await self.infra.agit(["rev-parse", "--abbrev-ref", "HEAD"])).stdout.strip()
        baseline_health = ""

        try:
            from monitoring import get_health_doctor

            doctor = get_health_doctor(self.ctx.molly)
            baseline_health = await asyncio.to_thread(
                doctor.generate_report, abbreviated=True, trigger="pre-deploy"
            )
        except Exception:
            log.debug("Pre-deploy baseline health probe failed", exc_info=True)

        try:
            await self.infra.agit(["checkout", "-b", branch])
            await self.infra.agit(["apply", str(patch_path)])
            tests_ok, tests_log = await asyncio.to_thread(self.infra.run_test_suite, f"core-{ts}")
            await self.infra.agit(["add", "-A"])
            await self.infra.agit(
                ["commit", "-m", f"[molly-self-edit][proposal] {description.strip()}"],
                env_override=_GIT_AUTHOR_ENV,
            )

            diff_text = (await self.infra.agit(["--no-pager", "show", "--stat", "--oneline", "-1"])).stdout.strip()
            proposal = self._format_core_patch_proposal(
                description=description,
                branch=branch,
                tests_ok=tests_ok,
                tests_log_path=tests_log,
                diff_summary=diff_text,
            )
            await self.comms.notify_owner(proposal)
            self.comms.log_improvement_event(
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

            decision = await self.comms.request_owner_decision(
                category="self-improve-core",
                description=proposal,
                required_keyword="DEPLOY",
                allow_edit=False,
            )
            if decision is not True:
                await self.infra.agit(["checkout", base_branch])
                await self.infra.agit(["branch", "-D", branch])
                self.comms.log_improvement_event(
                    event_type="proposal",
                    category="core",
                    title=description,
                    payload=f"branch={branch}",
                    status="rejected",
                )
                return {"status": "rejected", "reason": "owner denied deployment"}

            await self.infra.agit(["checkout", base_branch])
            await self.infra.agit(["merge", "--no-ff", branch, "-m", f"[molly-self-edit] {description.strip()}"])
            merge_commit = (await self.infra.agit(["rev-parse", "HEAD"])).stdout.strip()
            await self.infra.agit(["branch", "-D", branch])

            # Record causal edges for dependency tracking
            try:
                from evolution.causal import record_consequences
                post_tests_ok, _post_log = await asyncio.to_thread(
                    self.infra.run_test_suite, f"post-deploy-{ts}",
                )
                changed_files = patch_text.split("---")
                for cf in changed_files:
                    if cf.strip():
                        record_consequences(
                            commit_hash=merge_commit,
                            file_changed=cf.strip()[:200],
                            baseline={"suite": "pass" if tests_ok else "fail"},
                            post={"suite": "pass" if post_tests_ok else "fail"},
                        )
            except Exception:
                log.debug("Causal edge recording failed", exc_info=True)

            self._record_pending_deploy(
                commit_hash=merge_commit,
                description=description,
                deployed_at=datetime.now(timezone.utc),
                baseline_health=baseline_health,
            )

            restart_reason = f"self-edit deployed: {description.strip()}"
            if self.infra.request_runtime_restart(restart_reason):
                self.comms.log_improvement_event(
                    event_type="deploy",
                    category="core",
                    title=description,
                    payload=f"commit={merge_commit}",
                    status="restart_requested",
                )
                return {"status": "restart_requested", "branch": branch, "commit": merge_commit}

            # Fallback for non-runtime contexts where Molly cannot restart herself.
            rollback_reason = await self._post_deploy_health_regression_check(
                baseline_health,
                apply_grace_period=True,
            )
            if rollback_reason:
                await self._rollback_commit(merge_commit, rollback_reason)
                return {"status": "rolled_back", "reason": rollback_reason}

            self.comms.log_improvement_event(
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
                await self.infra.agit(["checkout", base_branch], check=False)
            except Exception:
                pass
            return {"status": "failed", "reason": str(exc)}

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
        return await self.tool_gaps.propose_tool(
            tool_name=tool_name,
            tool_code=tool_code,
            test_code=test_code,
            read_only=read_only,
            requires_auth=requires_auth,
        )

    async def propose_memory_md_update(self) -> dict[str, Any]:
        """Generate MEMORY.md from graph and propose diff."""
        await self.initialize()
        new_text = self.graph.build_memory_md_from_graph(limit=100)
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
        await self.comms.notify_owner(proposal)
        self.comms.log_improvement_event(
            event_type="proposal",
            category="memory",
            title="MEMORY.md weekly update",
            payload=diff[:10000],
            status="proposed",
        )

        decision = await self.comms.request_owner_decision(
            category="self-improve-memory",
            description=proposal,
            required_keyword="YES",
            allow_edit=False,
        )
        if decision is True:
            _atomic_write(memory_path, new_text)
            self.comms.log_improvement_event(
                event_type="proposal",
                category="memory",
                title="MEMORY.md weekly update",
                payload="applied",
                status="approved",
            )
            return {"status": "approved", "path": str(memory_path)}
        self.comms.log_improvement_event(
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
        facts = self.graph.extract_user_facts(days=30)
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
        await self.comms.notify_owner(proposal)
        self.comms.log_improvement_event(
            event_type="proposal",
            category="user",
            title="USER.md curation",
            payload=diff[:10000],
            status="proposed",
        )

        decision = await self.comms.request_owner_decision(
            category="self-improve-user",
            description=proposal,
            required_keyword="YES",
            allow_edit=False,
        )
        if decision is True:
            _atomic_write(user_path, new_text)
            self.comms.log_improvement_event(
                event_type="proposal",
                category="user",
                title="USER.md curation",
                payload="applied",
                status="approved",
            )
            return {"status": "approved", "path": str(user_path)}
        self.comms.log_improvement_event(
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

        perf = self.stats.performance_metrics(days=7)
        memory_stats = self.stats.memory_stats()
        automation_stats = self.stats.automation_stats()
        workflow_patterns = self.auto_patterns.detect_workflow_patterns(days=30, min_occurrences=3)
        health_summary = self.stats.latest_health_summary()
        self_improve_stats = self.stats.self_improvement_stats(days=30)

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
            f"- {self.gliner.gliner_weekly_summary_line()}",
            "",
            "## Workflow Patterns Detected",
        ]
        if workflow_patterns:
            for item in workflow_patterns[:5]:
                steps = _pattern_steps_fn(item)
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
        atomic_write(report_path, report_text)

        self.ctx.state["last_weekly_assessment"] = today.isoformat()
        self.ctx.save_state()

        summary = (
            f"Weekly self-assessment ready ({today.isoformat()}). "
            f"Messages={perf['messages']}, tool_calls={perf['tool_calls']}, "
            f"patterns={len(workflow_patterns)}, health={health_summary}."
        )
        await self.comms.notify_owner(summary)
        self.comms.log_improvement_event(
            event_type="assessment",
            category="weekly",
            title=f"Weekly self-assessment {today.isoformat()}",
            payload=str(report_path),
            status="generated",
        )

        # Weekly curation proposals.
        await self.propose_memory_md_update()
        await self.propose_user_md_update()
        await self.skill_gaps.propose_skill_updates_from_gap_clusters(min_cluster_size=_SKILL_GAP_MIN_CLUSTER_SIZE)
        await self.skill_gaps.propose_skill_updates_from_patterns(workflow_patterns)
        await self.tool_gaps.propose_tool_updates_from_failures(
            days=_TOOL_GAP_WINDOW_DAYS,
            min_failures=_TOOL_GAP_MIN_FAILURES,
        )
        await self.auto_patterns.propose_automation_updates_from_patterns(workflow_patterns)
        return report_path

    async def propose_skill_updates(self, patterns: list[dict]) -> dict:
        """Public API for nightly skill-pattern proposals."""
        await self.initialize()
        return await self.skill_gaps.propose_skill_updates_from_patterns(patterns)

    async def propose_tool_updates(self, days: int = 7, min_failures: int = 5) -> dict:
        """Public API for nightly tool-gap proposals."""
        await self.initialize()
        return await self.tool_gaps.propose_tool_updates_from_failures(days=days, min_failures=min_failures)

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

        if "config.py" in touched_files and self.infra.patch_touches_action_tiers_block(patch_text):
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
        self.ctx.state["pending_deploy"] = {
            "commit_hash": commit_hash,
            "description": description,
            "deployed_at": deployed_at.isoformat(),
            "healthy": False,
            "baseline_health": baseline_health[:20000],
            "startup_validation": "pending",
        }
        self.ctx.save_state()

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
            from monitoring import get_health_doctor

            doctor = get_health_doctor(self.ctx.molly)
            baseline_state = doctor.extract_status_map(baseline_health)
            log.info(
                "Post-deploy health pre-edit baseline by component: %s",
                self._format_health_state_for_log(baseline_state),
            )

            if apply_grace_period:
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
        pending = self.ctx.state.get("pending_deploy") or {}
        if not pending or pending.get("healthy"):
            return

        commit = str(pending.get("commit_hash", "")).strip()
        if not commit:
            self.ctx.state["pending_deploy"] = None
            self.ctx.save_state()
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
            self.ctx.state["pending_deploy"] = pending
            self.ctx.save_state()
            await self._rollback_commit(commit, f"startup health regression: {regression}")
            self.infra.request_runtime_restart("rollback applied after startup health validation")
            return

        pending["healthy"] = True
        pending["startup_validation"] = "passed"
        pending["startup_validated_at"] = datetime.now(timezone.utc).isoformat()
        self.ctx.state["pending_deploy"] = pending
        self.ctx.save_state()
        log.info("Startup deploy validation passed for commit %s", commit)

    async def _mark_deploy_stable_if_window_elapsed(self, now_local: datetime):
        pending = self.ctx.state.get("pending_deploy") or {}
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
        self.ctx.state["pending_deploy"] = pending
        self.ctx.save_state()

    async def _rollback_commit(self, commit_hash: str, reason: str):
        try:
            await self.infra.agit(["revert", "--no-edit", commit_hash], env_override=_GIT_AUTHOR_ENV)
            self.comms.log_improvement_event(
                event_type="deploy",
                category="core",
                title="Auto rollback",
                payload=f"commit={commit_hash};reason={reason}",
                status="rolled_back",
            )
            await self.comms.notify_owner(
                f"⚠️ Self-edit rollback executed.\nCommit: {commit_hash}\nReason: {reason}"
            )
        except Exception as exc:
            log.error("Auto rollback failed: %s", exc, exc_info=True)
            await self.comms.notify_owner(
                f"⚠️ Rollback attempt failed for commit {commit_hash}: {exc}"
            )
        finally:
            self.ctx.state["pending_deploy"] = None
            self.ctx.save_state()

