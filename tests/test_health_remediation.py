import unittest
from dataclasses import dataclass

from health_remediation import (
    ACTION_AUTO_FIX,
    ACTION_ESCALATE_OWNER,
    ACTION_OBSERVE_ONLY,
    ACTION_PROPOSE_CORE_PATCH,
    ACTION_PROPOSE_SKILL,
    ACTION_PROPOSE_TOOL,
    HealthSignal,
    YELLOW_ESCALATION_DAYS,
    build_remediation_plan,
    resolve_router_row,
    route_health_signal,
    router_table_rows,
)


@dataclass
class FakeHealthCheck:
    check_id: str
    status: str
    yellow_streak_days: int = 1


class TestHealthRemediation(unittest.TestCase):
    def test_green_routes_to_observe_only(self):
        plan = route_health_signal("component.disk_space", "green")
        self.assertEqual(plan.action, ACTION_OBSERVE_ONLY)
        self.assertFalse(plan.escalation.escalate_owner_now)
        self.assertFalse(plan.escalation.immediate_investigation_candidate)

    def test_mapping_covers_all_action_types(self):
        cases = [
            ("component.disk_space", "yellow", ACTION_AUTO_FIX),
            ("learning.self_improvement_proposals", "yellow", ACTION_PROPOSE_SKILL),
            ("component.whatsapp", "yellow", ACTION_PROPOSE_TOOL),
            ("pipeline.message_to_embedding", "yellow", ACTION_PROPOSE_CORE_PATCH),
            ("component.neo4j", "yellow", ACTION_ESCALATE_OWNER),
            ("pipeline.entity_to_relationship", "yellow", ACTION_OBSERVE_ONLY),
        ]
        for check_id, severity, expected in cases:
            with self.subTest(check_id=check_id, severity=severity):
                self.assertEqual(
                    route_health_signal(check_id, severity).action,
                    expected,
                )

    def test_red_is_immediate_investigation_candidate(self):
        plan = route_health_signal("component.whatsapp", "red")
        self.assertEqual(plan.action, ACTION_ESCALATE_OWNER)
        self.assertEqual(plan.suggested_action, ACTION_PROPOSE_TOOL)
        self.assertTrue(plan.escalation.escalate_owner_now)
        self.assertTrue(plan.escalation.immediate_investigation_candidate)

    def test_yellow_three_day_contract_escalates_owner(self):
        day_two = route_health_signal(
            "component.whatsapp",
            "yellow",
            yellow_streak_days=YELLOW_ESCALATION_DAYS - 1,
        )
        self.assertEqual(day_two.action, ACTION_PROPOSE_TOOL)
        self.assertFalse(day_two.escalation.escalate_owner_now)
        self.assertEqual(day_two.escalation.yellow_days_until_escalation, 1)

        day_three = route_health_signal(
            "component.whatsapp",
            "yellow",
            yellow_streak_days=YELLOW_ESCALATION_DAYS,
        )
        self.assertEqual(day_three.action, ACTION_ESCALATE_OWNER)
        self.assertEqual(day_three.suggested_action, ACTION_PROPOSE_TOOL)
        self.assertTrue(day_three.escalation.escalate_owner_now)
        self.assertFalse(day_three.escalation.immediate_investigation_candidate)
        self.assertEqual(day_three.escalation.yellow_days_until_escalation, 0)

    def test_build_plan_accepts_mixed_signal_shapes(self):
        signals = [
            HealthSignal("component.disk_space", "yellow"),
            {"check_id": "learning.preference_signals", "status": "yellow"},
            FakeHealthCheck("component.whatsapp", "red"),
        ]
        plans = build_remediation_plan(
            signals,
            yellow_streak_by_check={"component.disk_space": YELLOW_ESCALATION_DAYS},
        )
        self.assertEqual(len(plans), 3)
        self.assertEqual(plans[0].action, ACTION_ESCALATE_OWNER)
        self.assertEqual(plans[0].suggested_action, ACTION_AUTO_FIX)
        self.assertEqual(plans[1].action, ACTION_PROPOSE_SKILL)
        self.assertEqual(plans[2].action, ACTION_ESCALATE_OWNER)
        self.assertEqual(plans[2].suggested_action, ACTION_PROPOSE_TOOL)

    def test_router_table_rows_include_exact_prefix_and_default(self):
        rows = {row["scope"]: row for row in router_table_rows()}
        self.assertEqual(rows["component.whatsapp"]["yellow"], ACTION_PROPOSE_TOOL)
        self.assertEqual(rows["pipeline.*"]["yellow"], ACTION_PROPOSE_CORE_PATCH)
        self.assertEqual(rows["*"]["yellow"], ACTION_OBSERVE_ONLY)
        self.assertEqual(rows["*"]["red"], ACTION_ESCALATE_OWNER)

    def test_resolve_router_row_for_unknown_check_uses_prefix_fallback(self):
        row = resolve_router_row("quality.new_signal")
        self.assertEqual(row["green"], ACTION_OBSERVE_ONLY)
        self.assertEqual(row["yellow"], ACTION_PROPOSE_CORE_PATCH)
        self.assertEqual(row["red"], ACTION_ESCALATE_OWNER)

    def test_unsupported_severity_raises(self):
        with self.assertRaises(ValueError):
            route_health_signal("component.disk_space", "orange")


if __name__ == "__main__":
    unittest.main()
