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
import threading
import time
from dataclasses import dataclass, field

from evolution._base import EnforcedPatch
from evolution.db import get_connection

log = logging.getLogger(__name__)

_RATE_LIMIT = 5
_RATE_WINDOW = 3600  # 1 hour
_call_timestamps: list[float] = []
_rate_lock = threading.Lock()


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
    with _rate_lock:
        _call_timestamps[:] = [t for t in _call_timestamps if now - t < _RATE_WINDOW]
        return len(_call_timestamps) < _RATE_LIMIT


def _record_call() -> None:
    """Record a codegen call for rate limiting."""
    with _rate_lock:
        _call_timestamps.append(time.time())


def _log_codegen_result(
    result: CodegenResult, proposal_id: str = "", patch_strategy: str = "",
) -> None:
    """Persist codegen result to evolution.db for backend success tracking."""
    try:
        conn = get_connection()
        try:
            conn.execute(
                """INSERT INTO codegen_results
                   (proposal_id, backend, success, latency_seconds, tokens_used,
                    patch_strategy, error, patch_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    proposal_id,
                    result.backend_used,
                    int(result.success),
                    result.latency_seconds,
                    result.tokens_used,
                    patch_strategy,
                    result.stderr[:500] if not result.success else "",
                    len(result.patches),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        log.debug("Failed to log codegen result", exc_info=True)


async def generate_code(
    request: CodegenRequest,
    backend: str | None = None,
    proposal_id: str = "",
) -> CodegenResult:
    """Generate code using the specified (or auto-selected) backend.

    All output goes through ``patch_enforcer`` for salvage.
    Results are logged to ``codegen_results`` in evolution.db.
    """
    if not _check_rate_limit():
        log.warning("Codegen rate limit reached (%d/%d per hour)", _RATE_LIMIT, _RATE_LIMIT)
        result = CodegenResult(success=False, stderr="rate limit exceeded", backend_used="")
        _log_codegen_result(result, proposal_id)
        return result

    # Auto-select backend
    if backend is None:
        if is_available("claude_code"):
            backend = "claude_code"
        elif is_available("codex"):
            backend = "codex"
        else:
            result = CodegenResult(success=False, stderr="no backend available", backend_used="")
            _log_codegen_result(result, proposal_id)
            return result

    _record_call()

    if backend == "claude_code":
        result = await _run_claude_code(request)
    elif backend == "codex":
        result = await _run_codex(request)
    else:
        result = CodegenResult(success=False, stderr=f"unknown backend: {backend}", backend_used=backend)

    _log_codegen_result(result, proposal_id)
    return result


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
        log.warning("Codegen timed out for backend=claude_code timeout=%ss", request.timeout_seconds)
        return CodegenResult(
            success=False, stderr="timeout", latency_seconds=request.timeout_seconds,
            backend_used="claude_code",
        )
    except FileNotFoundError:
        return CodegenResult(success=False, stderr="claude CLI not found", backend_used="claude_code")


async def _run_codex(request: CodegenRequest) -> CodegenResult:
    """Run OpenAI API call for code generation."""
    start = time.time()

    try:
        import config
        import openai

        client = openai.AsyncOpenAI(api_key=getattr(config, "OPENAI_API_KEY", ""))
        prompt = _build_prompt(request)

        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o",
                max_tokens=request.max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a code generation assistant. Output ONLY unified diff patches "
                            "that can be applied with `git apply`. Use standard diff format with "
                            "--- a/file and +++ b/file headers."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            ),
            timeout=request.timeout_seconds,
        )

        elapsed = time.time() - start
        text = resp.choices[0].message.content or "" if resp.choices else ""
        tokens_used = resp.usage.total_tokens if resp.usage else 0

        patches = _parse_patches(text)

        return CodegenResult(
            success=bool(patches),
            patches=patches,
            stdout=text[:5000],
            tokens_used=tokens_used,
            latency_seconds=elapsed,
            backend_used="codex",
        )
    except asyncio.TimeoutError:
        log.warning("Codegen timed out for backend=codex timeout=%ss", request.timeout_seconds)
        return CodegenResult(
            success=False, stderr="OpenAI timeout",
            latency_seconds=request.timeout_seconds, backend_used="codex",
        )
    except Exception as exc:
        elapsed = time.time() - start
        log.debug("Codex backend failed: %s", exc, exc_info=True)
        return CodegenResult(
            success=False, stderr=str(exc)[:500],
            latency_seconds=elapsed, backend_used="codex",
        )


def _read_file_safe(path: str, max_lines: int = 300) -> str:
    """Read a file's contents, truncating at max_lines. Returns empty string on failure."""
    try:
        from pathlib import Path as _Path
        p = _Path(path)
        if not p.is_file():
            return ""
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines truncated)"]
        return "\n".join(lines)
    except Exception:
        return ""


def _build_prompt(request: CodegenRequest) -> str:
    """Build a prompt for the codegen backend, including file contents when available."""
    parts = [f"Task: {request.task_description}"]
    if request.target_files:
        parts.append(f"Target files: {', '.join(request.target_files)}")
    if request.constraints:
        parts.append(f"Constraints: {'; '.join(request.constraints)}")

    if request.context_files:
        parts.append("\n--- File Contents ---")
        for fpath in request.context_files:
            content = _read_file_safe(fpath)
            if content:
                parts.append(f"\n### {fpath}\n```\n{content}\n```")
            else:
                parts.append(f"\n### {fpath}\n(file not readable)")

    return "\n".join(parts)


def _parse_patches(output: str) -> list[EnforcedPatch]:
    """Parse codegen output into EnforcedPatch objects via patch_enforcer."""
    from evolution.patch_enforcer import enforce
    result = enforce(output)
    return [result] if result else []
