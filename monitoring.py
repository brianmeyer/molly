import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import health_legacy
import health_remediation_legacy
import maintenance_legacy

log = logging.getLogger(__name__)


@dataclass
class HealthDoctor:
    """Narrow health facade used by runtime call sites.

    The heavy implementation remains in ``health_legacy`` for rollback safety.
    This wrapper intentionally exposes the stable methods the runtime depends on
    while still delegating unknown attributes for compatibility during transition.
    """

    _delegate: health_legacy.HealthDoctor

    def __init__(self, molly=None):
        self._delegate = health_legacy.HealthDoctor(molly=molly)

    def run_daily(self) -> str:
        return self._delegate.run_daily()

    def run_abbreviated_preflight(self) -> str:
        return self._delegate.run_abbreviated_preflight()

    def generate_report(self, abbreviated: bool = False, trigger: str = "manual") -> str:
        return self._delegate.generate_report(abbreviated=abbreviated, trigger=trigger)

    def latest_report_path(self) -> Path | None:
        return self._delegate.latest_report_path()

    def latest_report_text(self) -> str | None:
        return self._delegate.latest_report_text()

    def extract_status_map(self, report_text: str) -> dict[str, str]:
        return self._delegate.extract_status_map(report_text)

    def __getattr__(self, name: str) -> Any:
        # Compatibility bridge while legacy health code is kept for rollback.
        return getattr(self._delegate, name)


_default_doctor: HealthDoctor | None = None


def get_health_doctor(molly=None) -> HealthDoctor:
    """Factory used by commands/main/self_improve.

    When ``molly`` is provided we create a scoped doctor to preserve per-instance
    behavior; otherwise we reuse a shared singleton for low-overhead checks.
    """

    global _default_doctor
    if molly is not None:
        return HealthDoctor(molly=molly)
    if _default_doctor is None:
        _default_doctor = HealthDoctor(molly=None)
    return _default_doctor


async def run_maintenance(molly=None) -> dict[str, Any]:
    """Nightly maintenance entrypoint consumed by ``main.py``."""

    return await maintenance_legacy.run_maintenance(molly=molly)


def should_run_maintenance(last_run: datetime | None) -> bool:
    """Nightly maintenance scheduling gate consumed by ``main.py``."""

    return maintenance_legacy.should_run_maintenance(last_run)


def route_health_signal(check_id: str, severity: str, **kwargs):
    """Remediation router exported for runtime and diagnostics."""

    return health_remediation_legacy.route_health_signal(check_id, severity, **kwargs)


__all__ = [
    "HealthDoctor",
    "get_health_doctor",
    "run_maintenance",
    "should_run_maintenance",
    "route_health_signal",
]
