import asyncio
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import health
import monitoring.maintenance as maintenance


class _FrozenDateTime:
    current: datetime = datetime(2026, 2, 8, 23, 0, 0)

    @classmethod
    def now(cls):
        return cls.current


class _ImmediateImprover:
    async def run_memory_optimization(self):
        return {
            "entity_consolidations": 0,
            "stale_entities": 0,
            "contradictions": 0,
        }

    async def run_gliner_nightly_cycle(self):
        return {"status": "insufficient_examples", "message": "not-ready"}

    async def run_qwen_nightly_cycle(self):
        return {"status": "skipped", "message": "not-ready"}

    async def run_weekly_assessment(self):
        return "2026-02-08.md"

    async def propose_skill_updates(self, patterns):
        return {"proposed": 0}

    async def propose_tool_updates(self, **kwargs):
        return {"proposed": 0}


class _BlockingImprover:
    def __init__(self):
        self.first_started = asyncio.Event()
        self.second_started = asyncio.Event()
        self.release = asyncio.Event()
        self.call_count = 0

    async def run_memory_optimization(self):
        self.call_count += 1
        if self.call_count == 1:
            self.first_started.set()
        elif self.call_count == 2:
            self.second_started.set()
        await self.release.wait()
        return {
            "entity_consolidations": 0,
            "stale_entities": 0,
            "contradictions": 0,
        }

    async def run_gliner_nightly_cycle(self):
        return {"status": "insufficient_examples", "message": "not-ready"}

    async def propose_skill_updates(self, patterns):
        return {"proposed": 0}

    async def propose_tool_updates(self, **kwargs):
        return {"proposed": 0}

    async def run_weekly_assessment(self):
        return "2026-02-08.md"


def _fake_graph_module() -> types.ModuleType:
    module = types.ModuleType("memory.graph")
    module.get_graph_summary = lambda: {
        "entity_count": 10,
        "relationship_count": 20,
        "top_connected": [],
        "recent": [],
    }
    module.entity_count = lambda: 10
    module.relationship_count = lambda: 20
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = MagicMock(single=MagicMock(return_value=None))
    module.get_driver = lambda: mock_driver
    return module


def _fake_health_module() -> types.ModuleType:
    module = types.ModuleType("health")

    class _Doctor:
        def run_daily(self):
            return "ok"

    module.get_health_doctor = lambda molly=None: _Doctor()
    return module


class TestMaintenanceSchedulingContract(unittest.TestCase):
    def test_daily_maintenance_is_catch_up_safe_once_per_day(self):
        with patch.object(maintenance, "_now_local", lambda: datetime(2026, 2, 8, 23, 0, 0)):
            self.assertTrue(maintenance.should_run_maintenance(None))

            self.assertFalse(
                maintenance.should_run_maintenance(datetime(2026, 2, 8, 0, 1, 0))
            )

            self.assertTrue(
                maintenance.should_run_maintenance(datetime(2026, 2, 7, 23, 59, 0))
            )

        with patch.object(maintenance, "_now_local", lambda: datetime(2026, 2, 8, 22, 0, 0)):
            self.assertFalse(
                maintenance.should_run_maintenance(datetime(2026, 2, 7, 23, 59, 0))
            )


class TestMaintenanceRunContracts(unittest.IsolatedAsyncioTestCase):
    def _runtime_harness(self, temp_root: Path, improver, strength_side_effect=None):
        stack = ExitStack()

        # Build a fake relationship_audit module
        fake_rel_audit = types.ModuleType("memory.relationship_audit")

        async def _fake_run_rel_audit(**kwargs):
            return {
                "auto_fixes_applied": 0,
                "quarantined_count": 0,
                "deterministic_result": {"status": "pass"},
            }

        fake_rel_audit.run_relationship_audit = _fake_run_rel_audit

        # Build a fake retriever with a conn-bearing vectorstore
        fake_retriever = types.ModuleType("memory.retriever")
        _mock_conn = MagicMock()
        _mock_conn.execute.return_value.fetchall.return_value = []
        _mock_vs = MagicMock()
        _mock_vs.conn = _mock_conn
        fake_retriever.get_vectorstore = lambda: _mock_vs

        # Build a fake foundry_adapter (needs FoundrySequenceSignal for self_improve import)
        from dataclasses import dataclass

        fake_foundry = types.ModuleType("foundry_adapter")
        fake_foundry.load_foundry_sequence_signals = lambda days=7: {}

        @dataclass(frozen=True)
        class _FakeSignal:
            steps: tuple = ()
            count: int = 0
            successes: int = 0
            latest_at: str = ""

            @property
            def success_rate(self):
                return 0.0

        fake_foundry.FoundrySequenceSignal = _FakeSignal

        # Build a fake graph_suggestions
        fake_gs = types.ModuleType("memory.graph_suggestions")
        fake_gs.build_suggestion_digest = lambda: ""

        # Build fake job modules for new maintenance steps
        fake_entity_audit = types.ModuleType("monitoring.jobs.entity_audit")

        async def _fake_run_entity_audit():
            return {
                "entities_audited": 0,
                "relationships_audited": 0,
                "gliner_examples_written": 0,
                "types_adopted": 0,
            }

        fake_entity_audit.run_entity_audit = _fake_run_entity_audit

        fake_graph_maint = types.ModuleType("monitoring.jobs.graph_maintenance")
        fake_graph_maint.run_neo4j_checkpoint = lambda: "community edition, skipped"
        fake_graph_maint.run_strength_decay = AsyncMock(return_value=11)
        fake_graph_maint.run_dedup_sweep = lambda: 3
        fake_graph_maint.run_orphan_cleanup = lambda: 2

        fake_analysis = types.ModuleType("monitoring.jobs.analysis_jobs")
        fake_analysis.compute_operational_insights = lambda: {
            "tool_count_24h": 0,
            "failing_tools": [],
            "unused_skills": [],
        }
        fake_analysis.run_graph_suggestions_digest = lambda: "no suggestions today"

        fake_learning = types.ModuleType("monitoring.jobs.learning_jobs")
        fake_learning.run_foundry_skill_scan = AsyncMock(return_value="0 patterns")
        fake_learning.run_tool_gap_scan = AsyncMock(return_value="0 gaps")
        fake_learning.run_correction_patterns = lambda: "0 corrections"

        fake_audit_jobs = types.ModuleType("monitoring.jobs.audit_jobs")
        fake_audit_jobs.record_maintenance_issues = lambda results, status: (0, 0)

        fake_self_improve = types.ModuleType("monitoring.jobs.self_improve_jobs")

        async def _fake_mem_opt(impr):
            mem_opt = await impr.run_memory_optimization()
            return (
                f"consolidated={mem_opt.get('entity_consolidations', 0)}, "
                f"stale={mem_opt.get('stale_entities', 0)}, "
                f"contradictions={mem_opt.get('contradictions', 0)}"
            )

        fake_self_improve.run_memory_optimization = _fake_mem_opt

        async def _fake_gliner(impr):
            gliner_cycle = await impr.run_gliner_nightly_cycle()
            return str(gliner_cycle.get("message") or gliner_cycle.get("status", "unknown"))

        fake_self_improve.run_gliner_loop = _fake_gliner

        async def _fake_qwen(impr):
            qwen_cycle = await impr.run_qwen_nightly_cycle()
            return str(qwen_cycle.get("message") or qwen_cycle.get("status", "unknown"))

        fake_self_improve.run_qwen_loop = _fake_qwen

        async def _fake_weekly(impr, now_local):
            return False, "not due"

        fake_self_improve.run_weekly_assessment = _fake_weekly

        fake_cleanup = types.ModuleType("monitoring.jobs.cleanup_jobs")
        fake_cleanup.prune_daily_logs = lambda: "nothing to prune"

        stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "memory.graph": _fake_graph_module(),
                    "health": _fake_health_module(),
                    "memory.relationship_audit": fake_rel_audit,
                    "memory.retriever": fake_retriever,
                    "foundry_adapter": fake_foundry,
                    "memory.graph_suggestions": fake_gs,
                    "monitoring.jobs.entity_audit": fake_entity_audit,
                    "monitoring.jobs.graph_maintenance": fake_graph_maint,
                    "monitoring.jobs.analysis_jobs": fake_analysis,
                    "monitoring.jobs.learning_jobs": fake_learning,
                    "monitoring.jobs.audit_jobs": fake_audit_jobs,
                    "monitoring.jobs.self_improve_jobs": fake_self_improve,
                    "monitoring.jobs.cleanup_jobs": fake_cleanup,
                },
            )
        )
        stack.enter_context(
            patch.object(maintenance, "MAINTENANCE_DIR", temp_root / "memory" / "maintenance")
        )
        stack.enter_context(
            patch.object(config, "MAINTENANCE_STATE_FILE", temp_root / "store" / "maintenance_state.json")
        )
        stack.enter_context(
            patch.object(
                maintenance,
                "HEALTH_LOG_PATH",
                temp_root / "memory" / "maintenance" / "health-log.md",
            )
        )
        stack.enter_context(
            patch.object(config, "HEALTH_REPORT_DIR", temp_root / "memory" / "health")
        )
        stack.enter_context(patch.object(maintenance, "write_health_check"))
        if strength_side_effect is None:
            stack.enter_context(patch.object(maintenance, "_run_strength_decay", return_value="11 entities updated"))
        else:
            stack.enter_context(
                patch.object(
                    maintenance,
                    "_run_strength_decay",
                    side_effect=strength_side_effect,
                )
            )
        stack.enter_context(patch.object(maintenance, "_run_dedup_sweep", return_value="3 entities merged"))
        stack.enter_context(patch.object(maintenance, "_run_orphan_cleanup", return_value="orphans=2, self_refs=1, blocklisted=0"))
        stack.enter_context(patch.object(maintenance, "_run_self_ref_cleanup", return_value=1))
        stack.enter_context(patch.object(maintenance, "_run_blocklist_cleanup", return_value=0))
        stack.enter_context(patch.object(maintenance, "_prune_daily_logs", return_value="archived 4 daily log(s)"))
        stack.enter_context(
            patch.object(maintenance, "_run_opus_analysis", new=AsyncMock(return_value=""))
        )
        stack.enter_context(
            patch.object(
                maintenance,
                "run_contract_audits",
                new=AsyncMock(return_value={
                    "nightly_deterministic": {"status": "pass", "summary": "pass", "checks": []},
                    "weekly_deterministic": {"status": "pass", "summary": "pass", "checks": []},
                    "nightly_model": {"status": "disabled", "summary": "disabled", "route": "kimi", "output": ""},
                    "weekly_model": {"status": "disabled", "summary": "disabled", "route": "opus", "output": ""},
                    "artifacts": {"maintenance": "", "health": "", "error": ""},
                    "markdown": "# audit\n",
                }),
            )
        )
        molly = SimpleNamespace(self_improvement=improver, wa=None)
        return stack, molly

    async def test_single_flight_blocks_second_maintenance_start(self):
        temp_root = Path(tempfile.mkdtemp(prefix="maintenance-single-flight-"))
        improver = _BlockingImprover()
        stack, molly = self._runtime_harness(temp_root=temp_root, improver=improver)

        with stack:
            first = asyncio.create_task(maintenance.run_maintenance(molly=molly))
            await asyncio.wait_for(improver.first_started.wait(), timeout=1.0)
            second = asyncio.create_task(maintenance.run_maintenance(molly=molly))

            overlapped = False
            try:
                await asyncio.wait_for(improver.second_started.wait(), timeout=0.3)
                overlapped = True
            except asyncio.TimeoutError:
                overlapped = False
            finally:
                improver.release.set()
                await asyncio.wait_for(asyncio.gather(first, second), timeout=2.0)

        self.assertFalse(
            overlapped,
            "Contract: a second maintenance run must not start while one is active.",
        )

    async def test_step_failure_does_not_abort_whole_run(self):
        temp_root = Path(tempfile.mkdtemp(prefix="maintenance-step-failure-"))
        improver = _ImmediateImprover()
        stack, molly = self._runtime_harness(
            temp_root=temp_root,
            improver=improver,
            strength_side_effect=RuntimeError("intentional failure"),
        )

        with stack:
            await maintenance.run_maintenance(molly=molly)

        report_path = (
            temp_root / "memory" / "maintenance" / f"{date.today().isoformat()}.md"
        )
        self.assertTrue(report_path.exists())
        report = report_path.read_text()
        self.assertIn("| Strength decay | failed |", report)
        self.assertIn("| Deduplication | 3 entities merged |", report)
        self.assertIn("| Daily log pruning |", report)

    async def test_final_report_includes_late_step_outcomes(self):
        temp_root = Path(tempfile.mkdtemp(prefix="maintenance-report-contract-"))
        improver = _ImmediateImprover()
        stack, molly = self._runtime_harness(temp_root=temp_root, improver=improver)

        with stack:
            await maintenance.run_maintenance(molly=molly)

        report_path = (
            temp_root / "memory" / "maintenance" / f"{date.today().isoformat()}.md"
        )
        report = report_path.read_text()
        self.assertIn("| GLiNER loop |", report)
        self.assertIn("| Health Doctor |", report)

    async def test_contract_model_audit_unavailable_is_non_blocking_by_default(self):
        temp_root = Path(tempfile.mkdtemp(prefix="maintenance-contract-nonblocking-"))
        improver = _ImmediateImprover()
        stack, molly = self._runtime_harness(temp_root=temp_root, improver=improver)
        audit_bundle = {
            "nightly_deterministic": {"status": "pass", "summary": "pass", "checks": []},
            "weekly_deterministic": {"status": "pass", "summary": "pass", "checks": []},
            "nightly_model": {
                "status": "unavailable",
                "summary": "unavailable (missing API key)",
                "route": "kimi",
                "output": "",
            },
            "weekly_model": {
                "status": "disabled",
                "summary": "disabled by config",
                "route": "opus",
                "output": "",
            },
            "artifacts": {
                "maintenance": str(temp_root / "memory" / "maintenance" / "audit.md"),
                "health": str(temp_root / "memory" / "health" / "audit.md"),
                "error": "",
            },
            "markdown": "# audit\n",
        }
        stack.enter_context(
            patch.object(
                maintenance,
                "run_contract_audits",
                new=AsyncMock(return_value=audit_bundle),
            )
        )
        stack.enter_context(patch.object(config, "CONTRACT_AUDIT_LLM_BLOCKING", False))

        with stack:
            state = await maintenance.run_maintenance(molly=molly)

        self.assertEqual(state["status"], "success")
        self.assertNotIn("Contract audit nightly (model)", state["failed_steps"])
        report_path = (
            temp_root / "memory" / "maintenance" / f"{date.today().isoformat()}.md"
        )
        report = report_path.read_text()
        self.assertIn("| Contract audit nightly (model) |", report)
        self.assertIn("unavailable (missing API key)", report)

    async def test_contract_model_audit_can_be_blocking_when_enabled(self):
        temp_root = Path(tempfile.mkdtemp(prefix="maintenance-contract-blocking-"))
        improver = _ImmediateImprover()
        stack, molly = self._runtime_harness(temp_root=temp_root, improver=improver)
        audit_bundle = {
            "nightly_deterministic": {"status": "pass", "summary": "pass", "checks": []},
            "weekly_deterministic": {"status": "pass", "summary": "pass", "checks": []},
            "nightly_model": {
                "status": "error",
                "summary": "error (timeout)",
                "route": "gemini",
                "output": "",
            },
            "weekly_model": {
                "status": "completed",
                "summary": "completed via opus",
                "route": "opus",
                "output": "ok",
            },
            "artifacts": {
                "maintenance": str(temp_root / "memory" / "maintenance" / "audit.md"),
                "health": str(temp_root / "memory" / "health" / "audit.md"),
                "error": "",
            },
            "markdown": "# audit\n",
        }
        stack.enter_context(
            patch.object(
                maintenance,
                "run_contract_audits",
                new=AsyncMock(return_value=audit_bundle),
            )
        )
        stack.enter_context(patch.object(config, "CONTRACT_AUDIT_LLM_BLOCKING", True))

        with stack:
            state = await maintenance.run_maintenance(molly=molly)

        self.assertEqual(state["status"], "partial")
        self.assertIn("Contract audit nightly (model)", state["failed_steps"])


class TestHealthExpectationContract(unittest.TestCase):
    def test_daily_maintenance_report_expectation_is_enforced(self):
        from monitoring.agents.data_quality import _maintenance_log_check

        temp_root = Path(tempfile.mkdtemp(prefix="health-maintenance-contract-"))
        maintenance_dir = temp_root / "memory" / "maintenance"
        health_dir = temp_root / "memory" / "health"

        with patch.object(config, "WORKSPACE", temp_root), patch.object(
            config, "HEALTH_REPORT_DIR", health_dir
        ):
            maintenance_dir.mkdir(parents=True, exist_ok=True)

            status, detail = _maintenance_log_check()
            self.assertEqual(status, "red")
            self.assertIn("today/yesterday", detail)

            yesterday = (date.today() - timedelta(days=1)).isoformat()
            (maintenance_dir / f"{yesterday}.md").write_text("# Maintenance report\n")

            status, detail = _maintenance_log_check()
            self.assertEqual(status, "green")
            self.assertIn(yesterday, detail)


if __name__ == "__main__":
    unittest.main()
