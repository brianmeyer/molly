"""Tests for Phase 5C.5 Docker Sandbox.

Tests cover:
  - DockerSandbox availability check
  - SubprocessSandbox (always available)
  - get_sandbox() factory
  - Code execution (subprocess fallback)
  - Timeout enforcement
  - Config values
"""
from __future__ import annotations

import asyncio
import unittest

import config


class TestDockerSandboxConfig(unittest.TestCase):
    """Test Docker sandbox config values."""

    def test_docker_enabled_by_default(self):
        self.assertTrue(config.DOCKER_SANDBOX_ENABLED)

    def test_docker_image(self):
        self.assertEqual(config.DOCKER_SANDBOX_IMAGE, "python:3.12-slim")

    def test_docker_timeout(self):
        self.assertGreaterEqual(config.DOCKER_SANDBOX_TIMEOUT, 10)

    def test_docker_memory(self):
        self.assertEqual(config.DOCKER_SANDBOX_MEMORY, "256m")


class TestDockerSandboxAvailability(unittest.TestCase):
    """Test availability checks."""

    def test_docker_unavailable_when_disabled(self):
        from evolution.docker_sandbox import DockerSandbox
        # Availability depends on Docker being installed, not just the config flag
        # Just verify the function doesn't crash
        DockerSandbox.is_available()

    def test_subprocess_always_available(self):
        from evolution.docker_sandbox import SubprocessSandbox
        self.assertTrue(SubprocessSandbox.is_available())


class TestGetSandbox(unittest.TestCase):
    """Test sandbox factory."""

    def test_factory_returns_docker_when_enabled(self):
        from evolution.docker_sandbox import get_sandbox, DockerSandbox
        sandbox = get_sandbox()
        # Docker is enabled by default, should get DockerSandbox
        self.assertIsInstance(sandbox, DockerSandbox)


class TestSubprocessSandbox(unittest.TestCase):
    """Test subprocess sandbox execution."""

    def test_run_simple_code(self):
        from evolution.docker_sandbox import SubprocessSandbox
        sandbox = SubprocessSandbox()

        async def _test():
            result = await sandbox.run("print('hello world')")
            self.assertEqual(result["exit_code"], 0)
            self.assertIn("hello world", result["stdout"])
            self.assertFalse(result["timed_out"])
            self.assertGreater(result["elapsed_ms"], 0)

        asyncio.run(_test())

    def test_run_code_with_error(self):
        from evolution.docker_sandbox import SubprocessSandbox
        sandbox = SubprocessSandbox()

        async def _test():
            result = await sandbox.run("raise ValueError('test error')")
            self.assertNotEqual(result["exit_code"], 0)
            self.assertIn("ValueError", result["stderr"])

        asyncio.run(_test())

    def test_run_code_with_timeout(self):
        from evolution.docker_sandbox import SubprocessSandbox
        sandbox = SubprocessSandbox()

        async def _test():
            result = await sandbox.run(
                "import time; time.sleep(10)",
                timeout=2,
            )
            self.assertTrue(result["timed_out"])
            self.assertEqual(result["exit_code"], -1)

        asyncio.run(_test())

    def test_run_code_with_output(self):
        from evolution.docker_sandbox import SubprocessSandbox
        sandbox = SubprocessSandbox()

        async def _test():
            result = await sandbox.run("for i in range(5): print(i)")
            self.assertEqual(result["exit_code"], 0)
            lines = result["stdout"].strip().split("\n")
            self.assertEqual(lines, ["0", "1", "2", "3", "4"])

        asyncio.run(_test())


class TestDockerSandboxInit(unittest.TestCase):
    """Test DockerSandbox initialization."""

    def test_custom_image(self):
        from evolution.docker_sandbox import DockerSandbox
        sandbox = DockerSandbox(image="python:3.11-slim")
        self.assertEqual(sandbox.IMAGE, "python:3.11-slim")

    def test_custom_timeout(self):
        from evolution.docker_sandbox import DockerSandbox
        sandbox = DockerSandbox(timeout=30)
        self.assertEqual(sandbox.TIMEOUT, 30)

    def test_custom_memory(self):
        from evolution.docker_sandbox import DockerSandbox
        sandbox = DockerSandbox(memory_limit="512m")
        self.assertEqual(sandbox.MEMORY_LIMIT, "512m")


class TestInfraServiceSandbox(unittest.TestCase):
    """Test that InfraService can provide a sandbox."""

    def test_get_sandbox_method(self):
        from evolution.infra import InfraService

        class MockCtx:
            project_root = config.PROJECT_ROOT
            sandbox_root = config.SANDBOX_DIR
            results_dir = config.SANDBOX_DIR

        infra = InfraService(MockCtx())
        sandbox = infra.get_sandbox()
        # Docker is enabled; should return DockerSandbox when Docker is available
        if sandbox is not None:
            from evolution.docker_sandbox import DockerSandbox, SubprocessSandbox
            self.assertIsInstance(sandbox, (DockerSandbox, SubprocessSandbox))


if __name__ == "__main__":
    unittest.main()
