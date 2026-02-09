import unittest
from datetime import datetime, timezone

from database import normalize_timestamp
from health import _parse_iso


class TestTimestampNormalization(unittest.TestCase):
    def test_normalize_epoch_units_to_same_instant(self):
        base = 1770598413  # 2026-02-09T00:53:33+00:00
        variants = [
            str(base),  # seconds
            str(base * 1_000),  # milliseconds
            str(base * 1_000_000),  # microseconds
            str(base * 1_000_000_000),  # nanoseconds
        ]

        normalized = [normalize_timestamp(v) for v in variants]
        parsed = [datetime.fromisoformat(v) for v in normalized]

        for dt in parsed:
            self.assertEqual(int(dt.timestamp()), base)
            self.assertEqual(dt.tzinfo, timezone.utc)

    def test_health_parser_accepts_epoch_units(self):
        base = 1770598413
        variants = [
            str(base),
            str(base * 1_000),
            str(base * 1_000_000),
            str(base * 1_000_000_000),
        ]

        for ts in variants:
            parsed = _parse_iso(ts)
            self.assertIsNotNone(parsed)
            self.assertEqual(int(parsed.timestamp()), base)


if __name__ == "__main__":
    unittest.main()
