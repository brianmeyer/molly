"""Resource monitoring with three tiers (GREEN, YELLOW, RED).

Uses ``psutil`` for CPU and memory usage.  Tiers determine how aggressively
the evolution engine should operate:

- GREEN  (<60% CPU, <70% RAM) — full judges, full proposals
- YELLOW (60-80% CPU, 70-85% RAM) — skip diversity judge, defer non-urgent
- RED    (>80% CPU, >85% RAM) — defer all proposals, Haiku only
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Tier thresholds
_CPU_YELLOW = 60.0
_CPU_RED = 80.0
_RAM_YELLOW = 70.0
_RAM_RED = 85.0


def get_resource_tier() -> str:
    """Return 'GREEN', 'YELLOW', or 'RED' based on current system load."""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory().percent
    except ImportError:
        log.debug("psutil not available — assuming GREEN")
        return "GREEN"
    except Exception:
        log.warning("Failed to read system resources", exc_info=True)
        return "YELLOW"  # conservative fallback

    if cpu > _CPU_RED or ram > _RAM_RED:
        tier = "RED"
    elif cpu > _CPU_YELLOW or ram > _RAM_YELLOW:
        tier = "YELLOW"
    else:
        tier = "GREEN"

    log.debug("Resource tier: %s (cpu=%.1f%%, ram=%.1f%%)", tier, cpu, ram)
    return tier


def should_defer() -> bool:
    """Return True if the system is under heavy load (RED tier)."""
    return get_resource_tier() == "RED"
