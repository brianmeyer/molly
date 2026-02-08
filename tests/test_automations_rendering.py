import unittest
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path("/Users/brianmeyer/molly")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from automations import AutomationEngine


class TestAutomationRenderString(unittest.TestCase):
    def setUp(self):
        self.engine = AutomationEngine(molly=None)
        self.run_context = {
            "now": datetime(2026, 2, 8, 12, 34, 56, tzinfo=timezone.utc),
            "trigger_payload": {
                "email": {
                    "snippet": 'Alert payload {"kind":"digest","count":2}',
                }
            },
        }
        self.outputs = {
            "triage": {"output": 'JSON body {"items":[1,2,3],"meta":{"ok":true}}'},
        }

    def test_replaces_known_tokens_and_keeps_brace_content(self):
        text = (
            "Date: {date}\n"
            "When: {datetime}\n"
            "Triage(short): {triage}\n"
            "Triage(full): {triage.output}\n"
            "Snippet: {trigger.email.snippet}\n"
        )
        rendered = self.engine._render_string(text, self.outputs, self.run_context)

        self.assertIn("Date: 2026-02-08", rendered)
        self.assertIn("When: 2026-02-08T12:34:56+00:00", rendered)
        self.assertIn('Triage(short): JSON body {"items":[1,2,3],"meta":{"ok":true}}', rendered)
        self.assertIn('Triage(full): JSON body {"items":[1,2,3],"meta":{"ok":true}}', rendered)
        self.assertIn('Snippet: Alert payload {"kind":"digest","count":2}', rendered)

    def test_unknown_tokens_and_literal_braces_are_unchanged(self):
        text = "Keep {unknown.var} {not_defined} raw JSON {\"a\":1} and lone { brace"
        rendered = self.engine._render_string(text, self.outputs, self.run_context)
        self.assertEqual(rendered, text)


if __name__ == "__main__":
    unittest.main()
