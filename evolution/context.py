"""Shared context for the evolution engine's composed services.

``EngineContext`` replaces the implicit ``self.*`` attribute contract from
the former monolithic architecture.  Every field that was previously set
in ``SelfImprovementEngine.__init__`` now lives here, and each service
receives this as its first constructor argument.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config
from utils import atomic_write_json

log = logging.getLogger(__name__)


@dataclass
class EngineContext:
    """Shared read/write context passed to all composed services."""

    molly: Any = None
    project_root: Path = field(default_factory=lambda: config.PROJECT_ROOT)
    sandbox_root: Path = field(default_factory=lambda: config.SANDBOX_DIR)
    state_path: Path = field(
        default_factory=lambda: config.WORKSPACE / "memory" / "self_improve_state.json"
    )

    # Derived paths — set in __post_init__
    skills_dir: Path = field(init=False)
    tools_dir: Path = field(init=False)
    automations_dir: Path = field(init=False)
    patches_dir: Path = field(init=False)
    tests_dir: Path = field(init=False)
    results_dir: Path = field(init=False)

    # Mutable shared state (was self._state on the old God Object)
    state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.skills_dir = self.sandbox_root / "skills"
        self.tools_dir = self.sandbox_root / "tools"
        self.automations_dir = self.sandbox_root / "automations"
        self.patches_dir = self.sandbox_root / "patches"
        self.tests_dir = self.sandbox_root / "tests"
        self.results_dir = self.sandbox_root / "results"

    # ------------------------------------------------------------------
    # State persistence (moved from EngineInfra._load_state / _save_state)
    # ------------------------------------------------------------------

    def load_state(self) -> None:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                if isinstance(data, dict):
                    self.state = data
                else:
                    self.state = {}
            except Exception:
                log.warning("State file %s corrupt or unreadable, resetting to defaults", self.state_path, exc_info=True)
                self.state = {}
        else:
            self.state = {}
        # Ensure default keys
        _defaults: list[tuple[str, Any]] = [
            ("pending_deploy", None),
            ("last_weekly_assessment", ""),
            ("gliner_training_cursor", ""),
            ("gliner_training_examples", 0),
            ("gliner_last_finetune_at", ""),
            ("gliner_last_deployed_at", ""),
            ("gliner_last_result", ""),
            ("gliner_last_cycle_status", ""),
            ("gliner_active_model_ref", ""),
            ("gliner_last_training_strategy", "lora"),
            ("gliner_benchmark_history", []),
        ]
        for key, default in _defaults:
            self.state.setdefault(key, default)

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.state_path, self.state, indent=2)
