import json
import tempfile
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from automation_triggers import EmailTrigger


class TestEmailTriggerHighWater(unittest.IsolatedAsyncioTestCase):
    async def test_uses_high_water_after_query_and_filters_boundary_duplicates(self):
        high_water_ts_ms = 1_700_000_000_000
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "automations": {
                            "email-triage": {
                                "trigger_state": {
                                    "email": {
                                        "high_water_internal_ts_ms": high_water_ts_ms,
                                        "high_water_ids": ["19c3ed6c543dbde4"],
                                    }
                                }
                            }
                        }
                    }
                )
            )

            trigger = EmailTrigger(
                {
                    "type": "email",
                    "query": "is:unread newer_than:20m",
                    "max_results": 10,
                    "_automation_id": "email-triage",
                    "_state_path": str(state_path),
                }
            )

            captured_query = {"value": ""}

            async def fake_search(query: str, max_results: int) -> list[dict]:
                captured_query["value"] = query
                return [
                    {"id": "19c3ed6c543dbde4", "internal_ts_ms": high_water_ts_ms},
                    {"id": "19c3ef100100a755", "internal_ts_ms": high_water_ts_ms + 1000},
                ]

            trigger._search_emails = fake_search  # type: ignore[method-assign]

            fired = await trigger.should_fire({})

            self.assertTrue(fired)
            self.assertIn("is:unread", captured_query["value"])
            self.assertIn(f"after:{(high_water_ts_ms // 1000) - 1}", captured_query["value"])
            self.assertNotIn("newer_than:", captured_query["value"])

            payload_messages = trigger.last_payload.get("messages", [])
            self.assertEqual(len(payload_messages), 1)
            self.assertEqual(payload_messages[0].get("id"), "19c3ef100100a755")


if __name__ == "__main__":
    unittest.main()
