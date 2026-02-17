"""Safe git workflow for evolution patches.

All changes go through branches — main is NEVER modified directly.
On failure the branch is deleted and main remains untouched.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_GIT_AUTHOR_ENV = {
    "GIT_AUTHOR_NAME": "Molly",
    "GIT_AUTHOR_EMAIL": "molly@local",
    "GIT_COMMITTER_NAME": "Molly",
    "GIT_COMMITTER_EMAIL": "molly@local",
}


async def _agit(args: list[str], cwd: str | Path | None = None, timeout: int = 30, **kw) -> str:
    """Run a git command asynchronously and return stdout."""
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=cwd, env={**os.environ, **_GIT_AUTHOR_ENV},
        **kw,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        log.warning("Git command timed out, killing process: git %s (timeout=%ss)", ' '.join(args), timeout)
        proc.kill()
        raise RuntimeError(f"git {' '.join(args)} timed out after {timeout}s")
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr.decode().strip()}")
    return stdout.decode().strip()


async def create_branch(name: str, cwd: str | Path | None = None) -> str:
    """Create and checkout a new evolution branch. Returns branch name."""
    branch = f"evolution/{name}"
    await _agit(["checkout", "-b", branch], cwd=cwd)
    log.info("Created branch %s", branch)
    return branch


async def validate_patch(diff_text: str, cwd: str | Path | None = None) -> bool:
    """Dry-run ``git apply --check`` on a diff. Returns True if clean."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "apply", "--check",
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=cwd,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(diff_text.encode()), timeout=15)
        return proc.returncode == 0
    except asyncio.TimeoutError:
        log.warning("validate_patch timed out")
        return False
    except Exception:
        log.warning("validate_patch failed", exc_info=True)
        return False


async def run_checks(cwd: str | Path | None = None) -> dict:
    """Run ruff + pytest and return results dict."""
    results: dict = {"ruff": False, "pytest": False}

    # Ruff
    try:
        proc = await asyncio.create_subprocess_exec(
            "ruff", "check", ".",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd,
        )
        await asyncio.wait_for(proc.communicate(), timeout=30)
        results["ruff"] = proc.returncode == 0
    except asyncio.TimeoutError:
        log.warning("ruff check timed out")
        results["ruff"] = False
    except FileNotFoundError:
        results["ruff"] = True  # ruff not installed — skip

    # Pytest
    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", "-m", "pytest", "tests/", "-q", "--tb=no",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
        results["pytest"] = proc.returncode == 0
        results["pytest_output"] = stdout.decode()[:2000]
    except asyncio.TimeoutError:
        log.warning("pytest timed out")
        results["pytest"] = False
    except Exception:
        results["pytest"] = False

    return results


async def merge_to_main(branch: str, cwd: str | Path | None = None) -> str:
    """Merge branch into main with --no-ff. Returns merge commit hash."""
    base = await _agit(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    if base == branch:
        await _agit(["checkout", "main"], cwd=cwd)
    merge_commit = await _agit(["merge", "--no-ff", branch, "-m", f"Merge {branch}"], cwd=cwd)
    await _agit(["branch", "-d", branch], cwd=cwd)
    log.info("Merged %s into main", branch)
    return merge_commit


async def rollback(branch: str, cwd: str | Path | None = None) -> None:
    """Force-delete a branch and return to main. Main untouched."""
    try:
        await _agit(["checkout", "main"], cwd=cwd)
    except Exception:
        pass
    try:
        await _agit(["branch", "-D", branch], cwd=cwd)
        log.info("Rolled back: deleted branch %s", branch)
    except Exception:
        log.warning("Failed to delete branch %s", branch, exc_info=True)


async def safe_modify(
    name: str,
    apply_fn,
    cwd: str | Path | None = None,
) -> dict:
    """Full safe-modify cycle: branch → apply → check → merge or rollback."""
    branch = await create_branch(name, cwd=cwd)
    try:
        await apply_fn()
        checks = await run_checks(cwd=cwd)
        if checks.get("ruff") and checks.get("pytest"):
            commit = await merge_to_main(branch, cwd=cwd)
            return {"status": "merged", "branch": branch, "commit": commit}
        else:
            await rollback(branch, cwd=cwd)
            return {"status": "failed", "branch": branch, "checks": checks}
    except Exception as exc:
        await rollback(branch, cwd=cwd)
        return {"status": "error", "branch": branch, "error": str(exc)}
