import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from foundry_adapter import load_foundry_sequence_signals


def _is_low_value_tool(tool_name: str) -> bool:
    lowered = str(tool_name or "").strip().lower()
    return lowered in {"write", "edit", "bash"} or lowered.startswith("approval:")


class TestFoundryAdapter(unittest.TestCase):
    def test_load_foundry_sequence_signals_aggregates_and_filters(self):
        with tempfile.TemporaryDirectory(prefix="foundry-observations-") as tmp:
            observations_dir = Path(tmp)
            payload_path = observations_dir / "2026-02-09.jsonl"
            payload_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "timestamp": "2026-02-09T13:49:27+00:00",
                                "tool_sequence": ["WebSearch", "kimi_research", "worker_agent"],
                                "outcome": "success",
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-02-09T13:55:00+00:00",
                                "tool_sequence": ["Write", "Edit", "Bash"],
                                "outcome": "success",
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-02-09T13:58:00+00:00",
                                "tool_sequence": ["WebSearch", "kimi_research", "worker_agent"],
                                "outcome": "failed",
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2025-01-01T00:00:00+00:00",
                                "tool_sequence": ["old", "stale", "sequence"],
                                "outcome": "success",
                            }
                        ),
                        "{not-json",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            signals = load_foundry_sequence_signals(
                days=30,
                observations_dir=observations_dir,
                is_low_value_tool=_is_low_value_tool,
                now_utc=datetime(2026, 2, 9, 18, 0, tzinfo=timezone.utc),
            )

        key = "WebSearch -> kimi_research -> worker_agent"
        self.assertIn(key, signals)
        signal = signals[key]
        self.assertEqual(signal.count, 2)
        self.assertEqual(signal.successes, 1)
        self.assertAlmostEqual(signal.success_rate, 0.5)
        self.assertEqual(signal.latest_at, "2026-02-09T13:58:00+00:00")
        self.assertNotIn("Write -> Edit -> Bash", signals)
        self.assertNotIn("old -> stale -> sequence", signals)

    def test_load_foundry_sequence_signals_supports_sliding_windows(self):
        with tempfile.TemporaryDirectory(prefix="foundry-observations-windows-") as tmp:
            observations_dir = Path(tmp)
            payload_path = observations_dir / "2026-02-09.jsonl"
            payload_path.write_text(
                json.dumps(
                    {
                        "timestamp": "2026-02-09T13:49:27+00:00",
                        "tool_sequence": ["alpha", "beta", "gamma", "delta"],
                        "outcome": "success",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            signals = load_foundry_sequence_signals(
                days=30,
                observations_dir=observations_dir,
                is_low_value_tool=_is_low_value_tool,
                now_utc=datetime(2026, 2, 9, 18, 0, tzinfo=timezone.utc),
            )

        self.assertIn("alpha -> beta -> gamma", signals)
        self.assertIn("beta -> gamma -> delta", signals)
        self.assertEqual(signals["alpha -> beta -> gamma"].count, 1)
        self.assertEqual(signals["beta -> gamma -> delta"].count, 1)

    def test_load_foundry_sequence_signals_missing_directory(self):
        missing = Path("/tmp/foundry-observations-missing-does-not-exist")
        signals = load_foundry_sequence_signals(
            days=30,
            observations_dir=missing,
            is_low_value_tool=_is_low_value_tool,
            now_utc=datetime(2026, 2, 9, 18, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(signals, {})


if __name__ == "__main__":
    unittest.main()
