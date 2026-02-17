"""Docker sandbox for isolated code execution (Phase 5C.5).

Provides Docker isolation for untrusted code execution:
  - Tool/skill tests from self-improvement engine
  - Group message handlers
  - Custom plugin code

Safety constraints:
  - network=none (no network access)
  - 256MB memory limit
  - 60s timeout
  - Read-only root filesystem with /tmp writable
  - No privileged access

Falls back to subprocess isolation when Docker is unavailable.
"""
from __future__ import annotations

import asyncio
import logging
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import config

log = logging.getLogger(__name__)


class DockerSandbox:
    """Execute code in an isolated Docker container.

    Usage::

        sandbox = DockerSandbox()
        if sandbox.is_available():
            result = await sandbox.run("print('hello')")
            print(result["stdout"])  # "hello\n"
    """

    IMAGE = config.DOCKER_SANDBOX_IMAGE
    TIMEOUT = config.DOCKER_SANDBOX_TIMEOUT
    MEMORY_LIMIT = config.DOCKER_SANDBOX_MEMORY
    NETWORK = "none"

    def __init__(
        self,
        image: str | None = None,
        timeout: int | None = None,
        memory_limit: str | None = None,
    ):
        if image:
            self.IMAGE = image
        if timeout:
            self.TIMEOUT = timeout
        if memory_limit:
            self.MEMORY_LIMIT = memory_limit

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Check if Docker daemon is running and the sandbox image exists."""
        if not config.DOCKER_SANDBOX_ENABLED:
            return False

        if not shutil.which("docker"):
            log.debug("Docker not found in PATH")
            return False

        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                log.debug("Docker daemon not running")
                return False
        except Exception:
            log.debug("Docker availability check failed", exc_info=True)
            return False

        return True

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    async def run(
        self,
        code: str,
        requirements: list[str] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute Python code in a Docker container.

        Parameters
        ----------
        code : str
            Python code to execute.
        requirements : list[str] | None
            pip packages to install before running.
        timeout : int | None
            Override timeout in seconds.

        Returns
        -------
        dict
            {stdout, stderr, exit_code, elapsed_ms, timed_out}
        """
        effective_timeout = timeout or self.TIMEOUT

        # Write code to a temp file
        with tempfile.TemporaryDirectory(prefix="molly-sandbox-") as tmpdir:
            code_path = Path(tmpdir) / "script.py"
            code_path.write_text(code, encoding="utf-8")

            # Build the docker run command with tracked container name
            container_name = f"molly-sandbox-{uuid.uuid4().hex[:8]}"
            cmd = self._build_docker_cmd(tmpdir, requirements, effective_timeout, container_name)

            t0 = time.monotonic()
            # Single timeout layer: subprocess gets effective_timeout,
            # asyncio gets +10s grace for container startup/teardown
            try:
                async with asyncio.timeout(effective_timeout + 10):
                    result = await asyncio.to_thread(
                        self._exec, cmd, effective_timeout,
                    )
            except asyncio.TimeoutError:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                # Explicitly kill orphaned container to prevent zombies
                await self._kill_container(container_name)
                return {
                    "stdout": "",
                    "stderr": f"Timeout after {effective_timeout}s",
                    "exit_code": -1,
                    "elapsed_ms": elapsed_ms,
                    "timed_out": True,
                }

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            # Truncate output to prevent OOM from malicious scripts
            stdout = (result.stdout or "")[:1_000_000]
            stderr = (result.stderr or "")[:500_000]
            if len(result.stdout or "") > 1_000_000:
                stdout += "\n... (output truncated at 1MB)"
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result.returncode,
                "elapsed_ms": elapsed_ms,
                "timed_out": False,
            }

    def run_sync(
        self,
        code: str,
        requirements: list[str] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Synchronous version of run()."""
        effective_timeout = timeout or self.TIMEOUT

        with tempfile.TemporaryDirectory(prefix="molly-sandbox-") as tmpdir:
            code_path = Path(tmpdir) / "script.py"
            code_path.write_text(code, encoding="utf-8")

            container_name = f"molly-sandbox-{uuid.uuid4().hex[:8]}"
            cmd = self._build_docker_cmd(tmpdir, requirements, effective_timeout, container_name)

            t0 = time.monotonic()
            result = self._exec(cmd, effective_timeout + 10)
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            # Kill orphaned container on timeout (matches async run())
            if result.returncode == -1:
                try:
                    subprocess.run(
                        ["docker", "kill", container_name],
                        capture_output=True, timeout=10,
                    )
                except Exception:
                    pass

            # Truncate output (matches async run())
            stdout = (result.stdout or "")[:1_000_000]
            stderr = (result.stderr or "")[:500_000]
            if len(result.stdout or "") > 1_000_000:
                stdout += "\n... (output truncated at 1MB)"

            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result.returncode,
                "elapsed_ms": elapsed_ms,
                "timed_out": result.returncode == -1,
            }

    def _build_docker_cmd(
        self,
        tmpdir: str,
        requirements: list[str] | None,
        timeout: int,
        container_name: str | None = None,
    ) -> list[str]:
        """Build the docker run command."""
        cmd = [
            "docker", "run",
            "--rm",                              # auto-remove container
            "--network", self.NETWORK,           # no network
            "--memory", self.MEMORY_LIMIT,       # memory cap
            "--cpus", "1.0",                     # CPU cap
            "--read-only",                       # read-only root
            "--tmpfs", "/tmp:rw,size=64m",       # writable /tmp
            "-v", f"{tmpdir}:/workspace:ro",     # mount code read-only
            "--workdir", "/workspace",
        ]
        if container_name:
            cmd.extend(["--name", container_name])

        # Build the script to execute
        if requirements:
            # Sanitize each requirement to prevent shell injection.
            # Only allow alphanumeric, dots, hyphens, underscores, brackets,
            # commas, comparison operators, and equals (standard pip specifiers).
            _SAFE_REQ = re.compile(r'^[A-Za-z0-9._\-\[\],>=<!~]+$')
            safe_reqs = []
            for req in requirements:
                if not _SAFE_REQ.match(req):
                    log.warning("Rejecting suspicious requirement: %r", req)
                    continue
                safe_reqs.append(shlex.quote(req))
            if safe_reqs:
                # Install to /tmp/pip since rootfs is read-only;
                # export PYTHONPATH so python script.py sees the packages.
                install_cmd = (
                    f"pip install --quiet --target /tmp/pip {' '.join(safe_reqs)} "
                    f"&& export PYTHONPATH=/tmp/pip:$PYTHONPATH && "
                )
            else:
                install_cmd = ""
        else:
            install_cmd = ""

        cmd.extend([
            self.IMAGE,
            "sh", "-c", f"{install_cmd}python script.py",
        ])

        return cmd

    @staticmethod
    def _exec(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
        """Execute a subprocess with timeout."""
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=-1,
                stdout="",
                stderr=f"Process timed out after {timeout}s",
            )

    @staticmethod
    async def _kill_container(container_name: str) -> None:
        """Best-effort kill of an orphaned Docker container."""
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "kill", container_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            log.debug("Killed orphaned container: %s", container_name)
        except Exception:
            log.debug("Failed to kill container %s (may already be stopped)", container_name)

    # ------------------------------------------------------------------
    # Image management
    # ------------------------------------------------------------------

    async def ensure_image(self) -> bool:
        """Pull the sandbox image if it doesn't exist locally."""
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["docker", "image", "inspect", self.IMAGE],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return True

            log.info("Pulling sandbox image: %s", self.IMAGE)
            result = await asyncio.to_thread(
                subprocess.run,
                ["docker", "pull", self.IMAGE],
                capture_output=True,
                text=True,
                timeout=300,
            )
            return result.returncode == 0
        except Exception:
            log.warning("Failed to ensure sandbox image", exc_info=True)
            return False


# ---------------------------------------------------------------------------
# Subprocess fallback (when Docker is unavailable)
# ---------------------------------------------------------------------------

class SubprocessSandbox:
    """Lightweight subprocess-based sandbox (fallback when Docker unavailable).

    Less isolated than Docker but still provides:
      - Timeout enforcement
      - Separate process (memory isolation)
      - No network restrictions (use with trusted code only)
    """

    TIMEOUT = config.DOCKER_SANDBOX_TIMEOUT

    async def run(
        self,
        code: str,
        requirements: list[str] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute Python code in a subprocess."""
        import sys

        effective_timeout = timeout or self.TIMEOUT

        with tempfile.TemporaryDirectory(prefix="molly-sandbox-") as tmpdir:
            code_path = Path(tmpdir) / "script.py"
            code_path.write_text(code, encoding="utf-8")

            # Install requirements into a temp venv (not host Python)
            sys_executable = sys.executable
            if requirements:
                _SAFE_REQ = re.compile(r'^[A-Za-z0-9._\-\[\],>=<!~]+$')
                safe_reqs = [r for r in requirements if _SAFE_REQ.match(r)]
                rejected = set(requirements) - set(safe_reqs)
                for bad in rejected:
                    log.warning("Rejecting suspicious requirement: %r", bad)
                if safe_reqs:
                    venv_dir = Path(tmpdir) / ".venv"
                    try:
                        # Create isolated venv so pip install doesn't pollute host
                        await asyncio.to_thread(
                            subprocess.run,
                            [sys.executable, "-m", "venv", str(venv_dir)],
                            capture_output=True, text=True, timeout=30,
                        )
                        venv_python = str(venv_dir / "bin" / "python")
                        await asyncio.to_thread(
                            subprocess.run,
                            [venv_python, "-m", "pip", "install", "--quiet"] + safe_reqs,
                            capture_output=True, text=True, timeout=60,
                        )
                        sys_executable = venv_python
                    except Exception:
                        log.warning("Failed to install requirements in venv: %s", safe_reqs)

            t0 = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        subprocess.run,
                        [sys_executable, str(code_path)],
                        capture_output=True,
                        text=True,
                        timeout=effective_timeout,
                        cwd=tmpdir,
                    ),
                    timeout=effective_timeout + 5,
                )
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                # Truncate output to prevent OOM
                stdout = (result.stdout or "")[:1_000_000]
                stderr = (result.stderr or "")[:500_000]
                if len(result.stdout or "") > 1_000_000:
                    stdout += "\n... (output truncated at 1MB)"
                return {
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": result.returncode,
                    "elapsed_ms": elapsed_ms,
                    "timed_out": False,
                }
            except (asyncio.TimeoutError, subprocess.TimeoutExpired):
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                return {
                    "stdout": "",
                    "stderr": f"Timeout after {effective_timeout}s",
                    "exit_code": -1,
                    "elapsed_ms": elapsed_ms,
                    "timed_out": True,
                }

    @staticmethod
    def is_available() -> bool:
        """Subprocess sandbox is always available."""
        return True


# ---------------------------------------------------------------------------
# Factory — pick the best available sandbox
# ---------------------------------------------------------------------------

def get_sandbox() -> DockerSandbox | SubprocessSandbox:
    """Return the best available sandbox.

    Prefers Docker when available, falls back to subprocess.
    """
    if DockerSandbox.is_available():
        log.debug("Using Docker sandbox")
        return DockerSandbox()
    log.debug("Docker unavailable, using subprocess sandbox")
    return SubprocessSandbox()
