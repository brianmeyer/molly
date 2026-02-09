#!/usr/bin/env python3
"""Run Track F pre-prod readiness checks and emit a markdown report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import health


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Molly Track F pre-prod readiness audit and write a markdown report.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory for markdown report (defaults to config.TRACK_F_AUDIT_DIR).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero when any red check is present. "
            "Default behavior is report-only success unless strict mode is enabled."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    doctor = health.get_health_doctor()

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    report_path = doctor.run_track_f_preprod_audit(output_dir=output_dir)
    checks = doctor.track_f_preprod_checks()

    red = sum(1 for c in checks if c.status == "red")
    yellow = sum(1 for c in checks if c.status == "yellow")
    green = sum(1 for c in checks if c.status == "green")

    print(f"Track F pre-prod audit report: {report_path}")
    print(f"Summary: green={green} yellow={yellow} red={red}")

    if args.strict and red > 0:
        print("Strict mode: red checks detected, failing.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
