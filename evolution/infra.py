"""Infrastructure service for SelfImprovementEngine.

Provides low-level helpers: sandbox directory management, persistent state
I/O, git wrappers, test-suite runners, slug generation, patch analysis,
and lightweight DB query helpers.  These are split out so the main
``skills.py`` file stays focused on business logic.
"""

import asyncio
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
import db_pool
from foundry_adapter import FoundrySequenceSignal, load_foundry_sequence_signals
from utils import atomic_write_json

log = logging.getLogger(__name__)

_PATCH_HUNK_RE = re.compile(r"^@@\s*-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s*@@")
_GIT_AUTHOR_ENV = {
    "GIT_AUTHOR_NAME": "Molly",
    "GIT_AUTHOR_EMAIL": "molly@local",
    "GIT_COMMITTER_NAME": "Molly",
    "GIT_COMMITTER_EMAIL": "molly@local",
}
_GLINER_BASE_MODEL = "fastino/gliner2-large-v1"
_GLINER_BENCHMARK_SEED = 1337
_GLINER_BENCHMARK_EVAL_RATIO = 0.2
_GLINER_BENCHMARK_THRESHOLD = 0.4
_GLINER_FINETUNE_COOLDOWN_DAYS = 7
_GLINER_TRAINING_SCAN_LIMIT = 4000
_POST_DEPLOY_HEALTH_GRACE_SECONDS = 45
_POST_DEPLOY_HEALTH_RETRY_SECONDS = 60
_TOOL_GAP_MIN_FAILURES = max(1, int(getattr(config, "TOOL_GAP_MIN_FAILURES", 5)))
_TOOL_GAP_WINDOW_DAYS = max(1, int(getattr(config, "TOOL_GAP_WINDOW_DAYS", 7)))
_LOW_VALUE_WORKFLOW_TOOL_NAMES = {"write", "edit", "bash"}
_SKILL_REJECTION_COOLDOWN_DAYS = max(1, int(getattr(config, "SKILL_REJECTION_COOLDOWN_DAYS", 30)))
_OWNER_SKILL_PHRASE_RE = re.compile(
    r"\b(?:make\s+a\s+skill\s+for\s+that|save\s+that\s+as\s+a\s+skill)\b",
    re.IGNORECASE,
)
_SKILL_GAP_MIN_CLUSTER_SIZE = max(1, int(getattr(config, "SKILL_GAP_MIN_CLUSTER_SIZE", 3)))
_SKILL_GAP_RECENT_PROPOSAL_DAYS = max(1, int(getattr(config, "SKILL_GAP_RECENT_PROPOSAL_DAYS", 14)))
_SKILL_GAP_PROPOSAL_COOLDOWN_DAYS = max(1, int(getattr(config, "SKILL_GAP_PROPOSAL_COOLDOWN_DAYS", 7)))


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None



class InfraService:
    """Infrastructure service — sandbox dirs, git, tests, DB helpers.

    Receives an ``EngineContext`` instead of relying on implicit ``self.*``
    attributes from a God Object.
    """

    def __init__(self, ctx):
        from evolution.context import EngineContext  # deferred to avoid circular
        self.ctx: EngineContext = ctx

    def ensure_sandbox_dirs(self) -> None:
        for path in [
            self.ctx.sandbox_root,
            self.ctx.skills_dir,
            self.ctx.tools_dir,
            self.ctx.automations_dir,
            self.ctx.patches_dir,
            self.ctx.tests_dir,
            self.ctx.results_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def request_runtime_restart(self, reason: str) -> bool:
        runtime = self.ctx.molly
        if runtime is None:
            return False
        restart_fn = getattr(runtime, "request_restart", None)
        if not callable(restart_fn):
            return False
        try:
            restart_fn(reason)
            return True
        except Exception:
            log.error("Restart request failed", exc_info=True)
            return False

    def git(
        self,
        args: list[str],
        check: bool = True,
        env_override: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        if env_override:
            env.update(env_override)
        result = subprocess.run(
            ["git"] + args,
            cwd=str(self.ctx.project_root),
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
            )
        return result

    async def agit(
        self,
        args: list[str],
        check: bool = True,
        env_override: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        """Async wrapper for git() — avoids blocking the event loop."""
        return await asyncio.to_thread(self.git, args, check, env_override)

    def run_test_suite(self, label: str) -> tuple[bool, Path]:
        log_path = self.ctx.results_dir / f"{label}-pytest.log"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q"],
            cwd=str(self.ctx.project_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log_path.write_text(output)
        return proc.returncode == 0, log_path

    def run_pytest_target(self, target: Path, label: str) -> tuple[bool, Path]:
        log_path = self.ctx.results_dir / f"{label}-pytest.log"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(target), "-q"],
            cwd=str(self.ctx.sandbox_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log_path.write_text(output)
        return proc.returncode == 0, log_path

    def get_sandbox(self):
        """Return the best available sandbox for isolated code execution.

        Phase 5C: Prefers Docker sandbox when available, falls back to
        subprocess isolation.
        """
        try:
            from evolution.docker_sandbox import get_sandbox
            return get_sandbox()
        except Exception:
            log.debug("Docker sandbox unavailable, using subprocess", exc_info=True)
            return None

    def dry_run_skill(self, skill_markdown: str) -> dict[str, Any]:
        missing_tools: list[str] = []
        required = []
        in_required = False
        for line in skill_markdown.splitlines():
            if line.strip().lower() == "## required tools":
                in_required = True
                continue
            if in_required and line.startswith("## "):
                break
            if in_required and line.strip().startswith("-"):
                required.append(line.strip("- ").strip("` "))
        all_known = (
            set(config.ACTION_TIERS["AUTO"])
            | set(config.ACTION_TIERS["CONFIRM"])
            | set(config.ACTION_TIERS["BLOCKED"])
        )
        for tool in required:
            if tool and tool not in all_known:
                missing_tools.append(tool)
        if missing_tools:
            return {"ok": False, "reason": f"Missing tools: {', '.join(missing_tools)}"}
        return {"ok": True, "reason": "All required tools available"}

    def slug(self, text: str) -> str:
        s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
        return s[:48] or "change"

    def patch_touches_action_tiers_block(self, patch_text: str) -> bool:
        action_start, action_end = self.action_tiers_line_range()
        if action_start <= 0 or action_end <= 0:
            return True

        current_file = ""
        old_line = 0
        new_line = 0
        for line in patch_text.splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:].strip()
                continue
            if line.startswith("@@ "):
                m = _PATCH_HUNK_RE.match(line)
                if not m:
                    continue
                old_line = int(m.group(1))
                new_line = int(m.group(2))
                continue
            if current_file != "config.py":
                continue
            if line.startswith("+") and not line.startswith("+++"):
                if action_start <= new_line <= action_end:
                    return True
                new_line += 1
                continue
            if line.startswith("-") and not line.startswith("---"):
                if action_start <= old_line <= action_end:
                    return True
                old_line += 1
                continue
            if line.startswith(" "):
                old_line += 1
                new_line += 1
        return False

    def action_tiers_line_range(self) -> tuple[int, int]:
        cfg_path = self.ctx.project_root / "config.py"
        if not cfg_path.exists():
            return -1, -1
        lines = cfg_path.read_text().splitlines()
        start = -1
        depth = 0
        for idx, line in enumerate(lines, start=1):
            if start < 0 and line.strip().startswith("ACTION_TIERS"):
                start = idx
                depth = line.count("{") - line.count("}")
                if depth <= 0:
                    return start, start
                continue
            if start > 0:
                depth += line.count("{")
                depth -= line.count("}")
                if depth <= 0:
                    return start, idx
        return -1, -1

    def count_rows(self, db_path: Path, sql: str, params: tuple[Any, ...]) -> int:
        try:
            conn = db_pool.sqlite_connect(str(db_path))
            try:
                value = conn.execute(sql, params).fetchone()[0]
                return int(value or 0)
            finally:
                conn.close()
        except Exception:
            return 0

    def scalar(
        self,
        db_path: Path,
        sql: str,
        params: tuple[Any, ...],
        default: float = 0.0,
    ) -> float:
        try:
            conn = db_pool.sqlite_connect(str(db_path))
            try:
                value = conn.execute(sql, params).fetchone()[0]
                if value is None:
                    return default
                return float(value)
            finally:
                conn.close()
        except Exception:
            return default

    def rows(self, db_path: Path, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        try:
            conn = db_pool.sqlite_connect(str(db_path))
            try:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute(sql, params).fetchall()]
            finally:
                conn.close()
        except Exception:
            return []

    def table_columns(self, db_path: Path, table_name: str) -> list[str]:
        try:
            conn = db_pool.sqlite_connect(str(db_path))
            try:
                result = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                return [str(row[1]) for row in result if len(row) > 1]
            finally:
                conn.close()
        except Exception:
            return []
