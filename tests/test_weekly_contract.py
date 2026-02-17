import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from evolution.skills import SelfImprovementEngine


class TestWeeklyAssessmentContract(unittest.TestCase):
    def test_weekly_assessment_is_due_in_sunday_window_only(self):
        engine = SelfImprovementEngine()
        engine.ctx.state = {}

        sunday_target = datetime(2026, 2, 8, 3, 0, 0)
        sunday_wrong_hour = datetime(2026, 2, 8, 2, 0, 0)
        monday_target = datetime(2026, 2, 9, 3, 0, 0)

        with patch.object(config, "WEEKLY_ASSESSMENT_DAY", "sunday"), patch.object(
            config, "WEEKLY_ASSESSMENT_HOUR", 3
        ):
            self.assertTrue(engine._is_weekly_assessment_due(sunday_target))
            self.assertFalse(engine._is_weekly_assessment_due(sunday_wrong_hour))
            self.assertFalse(engine._is_weekly_assessment_due(monday_target))

            engine.ctx.state["last_weekly_assessment"] = "2026-02-08"
            self.assertFalse(engine._is_weekly_assessment_due(sunday_target))


if __name__ == "__main__":
    unittest.main()
