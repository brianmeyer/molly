"""GLiNER LoRA fine-tune pipeline — thin facade.

The heavy implementation lives inside ``SelfImprovementEngine`` methods
(evolution/skills.py, lines ~1212-4102).  This module provides a
package-level entry point so maintenance jobs can import::

    from evolution.gliner_training import run_gliner_finetune_pipeline

Label *generation* stays in ``monitoring/jobs/entity_audit.py``.  This
module is the training *consumer* only.

Constants re-exported for discoverability:
    GLINER_BASE_MODEL, GLINER_BENCHMARK_SEED, GLINER_BENCHMARK_EVAL_RATIO,
    GLINER_BENCHMARK_THRESHOLD, GLINER_FINETUNE_COOLDOWN_DAYS
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# Re-export constants used by the pipeline (originally module-level in
# self_improve.py, now in evolution/skills.py).
GLINER_BASE_MODEL = "fastino/gliner2-large-v1"
GLINER_BENCHMARK_SEED = 1337
GLINER_BENCHMARK_EVAL_RATIO = 0.2
GLINER_BENCHMARK_THRESHOLD = 0.4
GLINER_FINETUNE_COOLDOWN_DAYS = 7
GLINER_TRAINING_SCAN_LIMIT = 4000


async def run_gliner_finetune_pipeline(
    engine=None,
    molly=None,
) -> dict[str, Any]:
    """Run the full GLiNER nightly cycle: accumulate → split → train → benchmark → deploy.

    Parameters
    ----------
    engine : SelfImprovementEngine | None
        Pre-initialised engine.  If *None*, one is created from *molly*.
    molly : object | None
        Molly agent instance (used to construct engine if needed).

    Returns
    -------
    dict
        Pipeline result with keys like ``status``, ``count``, ``benchmark``, etc.
    """
    if engine is None:
        from evolution.skills import SelfImprovementEngine
        engine = SelfImprovementEngine(molly=molly)
        await engine.initialize()

    return await engine.run_gliner_nightly_cycle()


async def run_gliner_accumulation(
    engine=None,
    molly=None,
    limit: int = GLINER_TRAINING_SCAN_LIMIT,
) -> dict[str, Any]:
    """Accumulate GLiNER training data without triggering fine-tune.

    Useful for nightly maintenance when you want accumulation only.
    """
    import asyncio

    if engine is None:
        from evolution.skills import SelfImprovementEngine
        engine = SelfImprovementEngine(molly=molly)
        await engine.initialize()

    return await asyncio.to_thread(engine._accumulate_gliner_training_data, limit)


def get_gliner_stats(engine=None) -> dict[str, Any]:
    """Return current GLiNER training statistics from engine state."""
    if engine is None:
        return {
            "status": "no_engine",
            "training_examples": 0,
            "last_result": "",
            "last_cycle_status": "",
        }
    state = getattr(engine, "_state", {})
    return {
        "training_examples": int(state.get("gliner_training_examples", 0)),
        "last_result": str(state.get("gliner_last_result", "")),
        "last_cycle_status": str(state.get("gliner_last_cycle_status", "")),
        "last_finetune_at": str(state.get("gliner_last_finetune_at", "")),
        "last_training_strategy": str(state.get("gliner_last_training_strategy", "")),
    }
