"""Code generation agent — Claude Code CLI + OpenAI Codex backends.

Delegates complex code changes to purpose-built coding agents instead
of Molly crafting raw diffs.  All output goes through ``patch_enforcer``
before application.

Rate limit: max 5 codegen calls per hour.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field

from evolution._base import EnforcedPatch

log = logging.getLogger(__name__)

_RATE_LIMIT = 5
_RATE_WINDOW = 3600  # 1 hour
_call_timestamps: list[float] = []


@dataclass
class CodegenRequest:
    task_description: str
    target_files: list[str] = field(default_factory=list)
    context_files: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    max_tokens: int = 8000
    timeout_seconds: int = 300


@dataclass
class CodegenResult:
    success: bool
    patches: list[EnforcedPatch] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    tokens_used: int = 0
    latency_seconds: float = 0.0
    backend_used: str = ""


def is_available(backend: str = "claude_code") -> bool:
    """Check if a codegen backend is installed and configured."""
    if backend == "claude_code":
        try:
            proc = subprocess.run(
                ["claude", "--version"],
                capture_output=True, timeout=5,
            )
            return proc.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    if backend == "codex":
        try:
            import config
            return bool(getattr(config, "OPENAI_API_KEY", ""))
        except Exception:
            return False

    return False


def _check_rate_limit() -> bool:
    """Return True if we're within the rate limit."""
    now = time.time()
    # Prune old timestamps
    _call_timestamps[:] = [t for t in _call_timestamps if now - t < _RATE_WINDOW]
    return len(_call_timestamps) < _RATE_LIMIT


def _record_call() -> None:
    """Record a codegen call for rate limiting."""
    _call_timestamps.append(time.time())


async def generate_code(
    request: CodegenRequest,
    backend: str | None = None,
) -> CodegenResult:
    """Generate code using the specified (or auto-selected) backend.

    All output goes through ``patch_enforcer`` for salvage.
    """
    if not _check_rate_limit():
        log.warning("Codegen rate limit reached (%d/%d per hour)", _RATE_LIMIT, _RATE_LIMIT)
        return CodegenResult(success=False, stderr="rate limit exceeded", backend_used="")

    # Auto-select backend
    if backend is None:
        if is_available("claude_code"):
            backend = "claude_code"
        elif is_available("codex"):
            backend = "codex"
        else:
            return CodegenResult(success=False, stderr="no backend available", backend_used="")

    _record_call()

    if backend == "claude_code":
        return await _run_claude_code(request)
    elif backend == "codex":
        return await _run_codex(request)
    else:
        return CodegenResult(success=False, stderr=f"unknown backend: {backend}", backend_used=backend)


async def _run_claude_code(request: CodegenRequest) -> CodegenResult:
    """Run Claude Code CLI subprocess."""
    start = time.time()

    prompt = _build_prompt(request)
    cmd = ["claude", "--print", "--dangerously-skip-permissions", "-p", prompt]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=request.timeout_seconds,
        )
        stdout = stdout_bytes.decode() if stdout_bytes else ""
        stderr = stderr_bytes.decode() if stderr_bytes else ""
        elapsed = time.time() - start

        # Parse output through patch enforcer
        patches = _parse_patches(stdout)

        return CodegenResult(
            success=proc.returncode == 0 and bool(patches),
            patches=patches,
            stdout=stdout[:5000],
            stderr=stderr[:2000],
            latency_seconds=elapsed,
            backend_used="claude_code",
        )
    except asyncio.TimeoutError:
        return CodegenResult(
            success=False, stderr="timeout", latency_seconds=request.timeout_seconds,
            backend_used="claude_code",
        )
    except FileNotFoundError:
        return CodegenResult(success=False, stderr="claude CLI not found", backend_used="claude_code")


async def _run_codex(request: CodegenRequest) -> CodegenResult:
    """Run OpenAI Codex API call (stub — wired in Batch 8)."""
    start = time.time()

    # Stub: actual API call will be wired during integration
    log.debug("Codex backend called (stub)")
    elapsed = time.time() - start

    return CodegenResult(
        success=False,
        stderr="codex backend not yet wired",
        latency_seconds=elapsed,
        backend_used="codex",
    )


def _build_prompt(request: CodegenRequest) -> str:
    """Build a prompt for the codegen backend."""
    parts = [f"Task: {request.task_description}"]
    if request.target_files:
        parts.append(f"Target files: {', '.join(request.target_files)}")
    if request.constraints:
        parts.append(f"Constraints: {'; '.join(request.constraints)}")
    return "\n".join(parts)


def _parse_patches(output: str) -> list[EnforcedPatch]:
    """Parse codegen output into EnforcedPatch objects via patch_enforcer."""
    from evolution.patch_enforcer import enforce
    result = enforce(output)
    return [result] if result else []
