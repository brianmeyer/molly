import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import contract_audit


def _baseline_results() -> dict[str, str]:
    return {
        "Health check": "completed",
        "Strength decay": "11 entities updated",
        "Deduplication": "3 entities merged",
        "Orphan cleanup": "2 orphans, 1 self-refs, 0 blocklisted",
        "Memory optimization": "consolidated=0, stale=0, contradictions=0",
        "Daily log pruning": "4 logs archived",
        "GLiNER loop": "not-ready",
        "Weekly assessment": "not due",
        "Health Doctor": "completed",
    }


class TestContractAuditConfig(unittest.IsolatedAsyncioTestCase):
    async def test_deterministic_checks_run_before_model_audits(self):
        temp_root = Path(tempfile.mkdtemp(prefix="contract-audit-order-"))
        order: list[str] = []

        def _nightly(_task_results):
            order.append("nightly_deterministic")
            return {"status": "pass", "summary": "pass", "checks": []}

        def _weekly(*, weekly_due, weekly_result):
            _ = weekly_due, weekly_result
            order.append("weekly_deterministic")
            return {"status": "pass", "summary": "pass", "checks": []}

        async def _model(*, audit_pass, deterministic_result, context):
            _ = deterministic_result, context
            order.append(f"{audit_pass}_model")
            return {
                "status": "disabled",
                "route": "disabled",
                "summary": "disabled by config",
                "output": "",
            }

        with patch.object(contract_audit, "run_nightly_deterministic_checks", side_effect=_nightly), patch.object(
            contract_audit, "run_weekly_deterministic_checks", side_effect=_weekly
        ), patch.object(contract_audit, "run_model_audit", side_effect=_model):
            await contract_audit.run_contract_audits(
                today="2026-02-09",
                task_results=_baseline_results(),
                weekly_due=False,
                weekly_result="not due",
                maintenance_dir=temp_root / "memory" / "maintenance",
                health_dir=temp_root / "memory" / "health",
            )

        self.assertEqual(
            order,
            [
                "nightly_deterministic",
                "weekly_deterministic",
                "nightly_model",
                "weekly_model",
            ],
        )

    async def test_model_routes_use_nightly_and_weekly_flags(self):
        temp_root = Path(tempfile.mkdtemp(prefix="contract-audit-routing-"))
        called_routes: list[str] = []

        async def _invoke(route: str, prompt: str):
            _ = prompt
            called_routes.append(route)
            return "Verdict: pass"

        with patch.object(config, "CONTRACT_AUDIT_LLM_ENABLED", True), patch.object(
            config, "CONTRACT_AUDIT_NIGHTLY_MODEL", "gemini"
        ), patch.object(config, "CONTRACT_AUDIT_WEEKLY_MODEL", "opus"), patch.object(
            contract_audit, "_invoke_model_route", side_effect=_invoke
        ):
            result = await contract_audit.run_contract_audits(
                today="2026-02-09",
                task_results=_baseline_results(),
                weekly_due=True,
                weekly_result="generated 2026-02-09.md",
                maintenance_dir=temp_root / "memory" / "maintenance",
                health_dir=temp_root / "memory" / "health",
            )

        self.assertEqual(called_routes, ["gemini", "opus"])
        self.assertEqual(result["nightly_model"]["status"], "completed")
        self.assertEqual(result["weekly_model"]["status"], "completed")

    async def test_model_layer_disabled_by_default(self):
        temp_root = Path(tempfile.mkdtemp(prefix="contract-audit-disabled-"))

        with patch.object(config, "CONTRACT_AUDIT_LLM_ENABLED", False):
            result = await contract_audit.run_contract_audits(
                today="2026-02-09",
                task_results=_baseline_results(),
                weekly_due=False,
                weekly_result="not due",
                maintenance_dir=temp_root / "memory" / "maintenance",
                health_dir=temp_root / "memory" / "health",
            )

        self.assertEqual(result["nightly_model"]["status"], "disabled")
        self.assertEqual(result["weekly_model"]["status"], "disabled")
        self.assertTrue((temp_root / "memory" / "maintenance" / "2026-02-09-contract-audit.md").exists())
        self.assertTrue((temp_root / "memory" / "health" / "2026-02-09-contract-audit.md").exists())

    async def test_model_route_fallback_advances_to_next_available_provider(self):
        attempts = []

        async def _invoke(route: str, prompt: str):
            _ = prompt
            attempts.append(route)
            if route == "kimi":
                raise contract_audit.AuditModelUnavailable("MOONSHOT_API_KEY not configured")
            if route == "opus":
                return "Verdict: pass"
            raise AssertionError(f"unexpected route: {route}")

        with patch.object(config, "CONTRACT_AUDIT_LLM_ENABLED", True), patch.object(
            config, "CONTRACT_AUDIT_NIGHTLY_MODEL", "kimi,opus,gemini"
        ), patch.object(contract_audit, "_invoke_model_route", side_effect=_invoke):
            result = await contract_audit.run_model_audit(
                audit_pass="nightly",
                deterministic_result={"status": "pass", "summary": "ok", "checks": []},
                context={},
            )

        self.assertEqual(attempts, ["kimi", "opus"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["route"], "opus")
        parsed_attempts = json.loads(result["attempts"])
        self.assertEqual(parsed_attempts[0]["status"], "unavailable")
        self.assertEqual(parsed_attempts[1]["status"], "completed")

    async def test_model_route_fallback_reports_unavailable_when_all_unavailable(self):
        async def _invoke(route: str, prompt: str):
            _ = prompt
            raise contract_audit.AuditModelUnavailable(f"{route} key missing")

        with patch.object(config, "CONTRACT_AUDIT_LLM_ENABLED", True), patch.object(
            config, "CONTRACT_AUDIT_WEEKLY_MODEL", "kimi,gemini"
        ), patch.object(contract_audit, "_invoke_model_route", side_effect=_invoke):
            result = await contract_audit.run_model_audit(
                audit_pass="weekly",
                deterministic_result={"status": "pass", "summary": "ok", "checks": []},
                context={},
            )

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("kimi key missing", result["summary"])
        self.assertIn("gemini key missing", result["summary"])
        parsed_attempts = json.loads(result["attempts"])
        self.assertEqual([row["route"] for row in parsed_attempts], ["kimi", "gemini"])


if __name__ == "__main__":
    unittest.main()
