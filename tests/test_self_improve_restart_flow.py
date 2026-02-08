import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from self_improve import PatchValidation, SelfImprovementEngine


def _agit_result(stdout: str = ""):
    return SimpleNamespace(stdout=stdout, returncode=0, stderr="")


class TestSelfImproveRestartFlow(unittest.IsolatedAsyncioTestCase):
    async def test_propose_core_patch_requests_restart(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="self-improve-restart-"))
        runtime = SimpleNamespace(request_restart=MagicMock())
        engine = SelfImprovementEngine(molly=runtime)
        engine.patches_dir = temp_dir

        with patch.object(engine, "initialize", new=AsyncMock()), \
                patch.object(engine, "validate_patch", return_value=PatchValidation(ok=True)), \
                patch.object(engine, "_run_test_suite", return_value=(True, temp_dir / "tests.log")), \
                patch.object(engine, "_notify_owner", new=AsyncMock()), \
                patch.object(engine, "_request_owner_decision", new=AsyncMock(return_value=True)), \
                patch.object(engine, "_record_pending_deploy") as pending_mock, \
                patch.object(engine, "_post_deploy_health_regression_check", new=AsyncMock()) as post_check_mock, \
                patch.object(engine, "_rollback_commit", new=AsyncMock()) as rollback_mock:
            async def fake_agit(args, check=True, env_override=None):
                if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
                    return _agit_result("main\n")
                if args == ["--no-pager", "show", "--stat", "--oneline", "-1"]:
                    return _agit_result("abc123 diff")
                if args == ["rev-parse", "HEAD"]:
                    return _agit_result("abc123\n")
                return _agit_result()

            with patch.object(engine, "_agit", side_effect=fake_agit):
                result = await engine.propose_core_patch(
                    description="restartable deploy",
                    patch_text="@@ -1 +1 @@\n-old\n+new\n",
                )

        self.assertEqual(result["status"], "restart_requested")
        runtime.request_restart.assert_called_once()
        pending_mock.assert_called_once()
        post_check_mock.assert_not_awaited()
        rollback_mock.assert_not_awaited()

    async def test_startup_validation_passes_and_marks_healthy(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="self-improve-startup-pass-"))
        engine = SelfImprovementEngine()
        engine.state_path = temp_dir / "state.json"
        engine._state = {
            "pending_deploy": {
                "commit_hash": "abc123",
                "healthy": False,
                "baseline_health": "baseline-report",
                "startup_validation": "pending",
            }
        }

        with patch.object(engine, "_post_deploy_health_regression_check", new=AsyncMock(return_value=None)), \
                patch.object(engine, "_rollback_commit", new=AsyncMock()) as rollback_mock, \
                patch.object(engine, "_request_runtime_restart", return_value=False) as restart_mock:
            await engine._validate_pending_deploy_on_startup()

        pending = engine._state["pending_deploy"]
        self.assertTrue(pending["healthy"])
        self.assertEqual(pending["startup_validation"], "passed")
        self.assertTrue(pending.get("startup_validated_at"))
        rollback_mock.assert_not_awaited()
        restart_mock.assert_not_called()

    async def test_startup_validation_rolls_back_and_requests_restart(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="self-improve-startup-fail-"))
        engine = SelfImprovementEngine()
        engine.state_path = temp_dir / "state.json"
        engine._state = {
            "pending_deploy": {
                "commit_hash": "abc123",
                "healthy": False,
                "baseline_health": "baseline-report",
                "startup_validation": "pending",
            }
        }

        with patch.object(
            engine,
            "_post_deploy_health_regression_check",
            new=AsyncMock(return_value="component_x:red->red"),
        ), patch.object(engine, "_rollback_commit", new=AsyncMock()) as rollback_mock, patch.object(
            engine, "_request_runtime_restart", return_value=True
        ) as restart_mock:
            await engine._validate_pending_deploy_on_startup()

        rollback_mock.assert_awaited_once()
        rollback_args = rollback_mock.await_args.args
        self.assertEqual(rollback_args[0], "abc123")
        self.assertIn("startup health regression", rollback_args[1])
        restart_mock.assert_called_once()
        self.assertEqual(engine._state["pending_deploy"]["startup_validation"], "failed")

    async def test_intentional_restart_path_skips_immediate_rollback(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="self-improve-intentional-restart-"))
        runtime = SimpleNamespace(request_restart=MagicMock())
        engine = SelfImprovementEngine(molly=runtime)
        engine.patches_dir = temp_dir

        with patch.object(engine, "initialize", new=AsyncMock()), \
                patch.object(engine, "validate_patch", return_value=PatchValidation(ok=True)), \
                patch.object(engine, "_run_test_suite", return_value=(True, temp_dir / "tests.log")), \
                patch.object(engine, "_notify_owner", new=AsyncMock()), \
                patch.object(engine, "_request_owner_decision", new=AsyncMock(return_value=True)), \
                patch.object(engine, "_record_pending_deploy"), \
                patch.object(engine, "_post_deploy_health_regression_check", new=AsyncMock()) as post_check_mock, \
                patch.object(engine, "_rollback_commit", new=AsyncMock()) as rollback_mock:
            async def fake_agit(args, check=True, env_override=None):
                if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
                    return _agit_result("main\n")
                if args == ["--no-pager", "show", "--stat", "--oneline", "-1"]:
                    return _agit_result("abc123 diff")
                if args == ["rev-parse", "HEAD"]:
                    return _agit_result("abc123\n")
                return _agit_result()

            with patch.object(engine, "_agit", side_effect=fake_agit):
                await engine.propose_core_patch(
                    description="skip immediate rollback",
                    patch_text="@@ -1 +1 @@\n-old\n+new\n",
                )

        post_check_mock.assert_not_awaited()
        rollback_mock.assert_not_awaited()
        runtime.request_restart.assert_called_once()


if __name__ == "__main__":
    unittest.main()
