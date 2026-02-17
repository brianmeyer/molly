"""Parallel Claude SDK workers for orchestrator subtasks.

Workers use Claude Agent SDK on Max subscription.  Each worker gets a
profile (model + tools) and handles one subtask.  Independent subtasks
run in parallel via asyncio.gather; dependent subtasks execute sequentially.

Key design:
  - Config-driven profiles (WORKER_PROFILES) — model + tool list per domain
  - Semaphore(MAX_CONCURRENT_WORKERS) limits parallel SDK sessions
  - Circuit breaker: 3 consecutive failures → skip profile for 5 minutes
  - Per-worker timeout (default 30s, research 60s)
  - Result aggregator combines outputs into single coherent response
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)

import threading

import config
from orchestrator import Subtask, TriageResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker profiles — model + tools per domain
# ---------------------------------------------------------------------------

# Maps profile name → {model config key, MCP servers to include, tool names}
# MCP servers reference keys from agent.py's _MCP_SERVER_SPECS
WORKER_PROFILES: dict[str, dict[str, Any]] = {
    "calendar": {
        "model_key": "WORKER_MODEL_FAST",
        "mcp_servers": ["google-calendar"],
        "timeout": "WORKER_TIMEOUT_DEFAULT",
        "prompt": (
            "You are a calendar assistant. Handle Google Calendar operations: "
            "create, read, update, delete events. Be precise with dates and times."
        ),
    },
    "email": {
        "model_key": "WORKER_MODEL_DEFAULT",
        "mcp_servers": ["gmail"],
        "timeout": "WORKER_TIMEOUT_DEFAULT",
        "prompt": (
            "You are an email assistant. Handle Gmail operations: search, read, "
            "draft, send, reply. Be professional and concise."
        ),
    },
    "contacts": {
        "model_key": "WORKER_MODEL_FAST",
        "mcp_servers": ["google-people"],
        "timeout": "WORKER_TIMEOUT_DEFAULT",
        "prompt": (
            "You are a contacts assistant. Look up people in Google Contacts."
        ),
    },
    "tasks": {
        "model_key": "WORKER_MODEL_FAST",
        "mcp_servers": ["google-tasks"],
        "timeout": "WORKER_TIMEOUT_DEFAULT",
        "prompt": (
            "You are a task management assistant. Handle Google Tasks: "
            "list, create, complete, delete tasks."
        ),
    },
    "research": {
        "model_key": "WORKER_MODEL_DEEP",
        "mcp_servers": ["kimi", "grok", "groq"],
        "timeout": "WORKER_TIMEOUT_RESEARCH",
        "prompt": (
            "You are a research analyst. Use web search and knowledge models to "
            "find information, synthesize findings, and provide thorough answers."
        ),
    },
    "writer": {
        "model_key": "WORKER_MODEL_DEFAULT",
        "mcp_servers": [],
        "timeout": "WORKER_TIMEOUT_DEFAULT",
        "prompt": (
            "You are a writing assistant. Draft text, compose messages, "
            "format documents, and help with creative writing."
        ),
    },
    "files": {
        "model_key": "WORKER_MODEL_DEFAULT",
        "mcp_servers": ["google-drive"],
        "timeout": "WORKER_TIMEOUT_DEFAULT",
        "prompt": (
            "You are a file assistant. Read, search, and navigate files "
            "in Google Drive and local storage."
        ),
    },
    "imessage": {
        "model_key": "WORKER_MODEL_DEFAULT",
        "mcp_servers": ["imessage"],
        "timeout": "WORKER_TIMEOUT_DEFAULT",
        "prompt": (
            "You are an iMessage assistant. Search, read, and manage iMessages."
        ),
    },
    "browser": {
        "model_key": "WORKER_MODEL_DEFAULT",
        "mcp_servers": ["browser-mcp"],
        "timeout": "WORKER_TIMEOUT_RESEARCH",
        "prompt": (
            "You are a browser automation assistant. Navigate web pages, "
            "interact with elements, and extract content."
        ),
    },
    "general": {
        "model_key": "WORKER_MODEL_DEFAULT",
        "mcp_servers": [],  # Gets ALL servers loaded at runtime
        "timeout": "WORKER_TIMEOUT_DEFAULT",
        "prompt": (
            "You are a capable general assistant. Handle any task that doesn't "
            "fit a specific domain. Be helpful and thorough."
        ),
    },
}


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

@dataclass
class _CircuitState:
    """Track consecutive failures per profile for circuit breaking.

    States: CLOSED (healthy) → OPEN (tripped) → HALF_OPEN (test call) → CLOSED
    """
    failures: int = 0
    tripped_at: float = 0.0  # monotonic time when circuit opened
    half_open: bool = False  # True when allowing a single test call

    def record_failure(self) -> None:
        self.half_open = False
        self.failures += 1
        if self.failures >= 3:
            self.tripped_at = time.monotonic()

    def record_success(self) -> None:
        self.failures = 0
        self.tripped_at = 0.0
        self.half_open = False

    def is_open(self) -> bool:
        """True if circuit is open (tripped and not yet recovered).

        Pure read-only query — does NOT transition state.  Use
        ``try_half_open()`` when you intend to make a real test call.
        """
        if self.failures < 3:
            return False
        if time.monotonic() - self.tripped_at > 300:
            if not self.half_open:
                return False  # Cooldown expired, eligible for test call
            return True  # Half-open test call already in progress
        return True

    def try_half_open(self) -> bool:
        """Attempt to claim the half-open slot for a test call.

        Returns True if the caller may proceed (circuit closed *or*
        half-open slot successfully claimed).  Returns False if the
        circuit is open and no test call is allowed.
        """
        if self.failures < 3:
            return True  # Circuit closed — always allow
        if time.monotonic() - self.tripped_at > 300 and not self.half_open:
            self.half_open = True
            log.debug("Circuit breaker entering half-open state")
            return True  # Caller gets the one test call
        return False


_circuit_states: dict[str, _CircuitState] = {}
_circuit_lock = threading.Lock()


def _get_circuit(profile: str) -> _CircuitState:
    """Thread-safe circuit state retrieval."""
    with _circuit_lock:
        if profile not in _circuit_states:
            _circuit_states[profile] = _CircuitState()
        return _circuit_states[profile]


# ---------------------------------------------------------------------------
# Worker result
# ---------------------------------------------------------------------------

@dataclass
class WorkerResult:
    """Result from a single worker execution."""
    subtask_id: str
    profile: str
    text: str
    success: bool
    latency_ms: float = 0.0
    model: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# MCP server loading (reuses agent.py patterns)
# ---------------------------------------------------------------------------

def _load_mcp_servers_for_profile(profile_name: str) -> dict[str, object]:
    """Load MCP servers needed for a specific worker profile.

    For 'general' profile, loads ALL available servers.
    Otherwise, loads only the servers listed in the profile config.
    """
    from agent import _MCP_SERVER_SPECS

    disabled_servers = set(getattr(config, "DISABLED_MCP_SERVERS", set()))
    profile = WORKER_PROFILES.get(profile_name, WORKER_PROFILES["general"])
    server_names = profile.get("mcp_servers", [])

    # General profile gets all servers
    if profile_name == "general" or not server_names:
        server_names = list(_MCP_SERVER_SPECS.keys())

    servers: dict[str, object] = {}
    for name in server_names:
        if name in disabled_servers:
            continue
        spec = _MCP_SERVER_SPECS.get(name)
        if spec is None:
            continue
        try:
            if isinstance(spec, tuple):
                module_name, attr_name = spec
                module = importlib.import_module(module_name)
                servers[name] = getattr(module, attr_name)
            elif isinstance(spec, dict):
                server_cfg = {"command": spec["command"]}
                if spec.get("args"):
                    server_cfg["args"] = list(spec["args"])
                if spec.get("env"):
                    server_cfg["env"] = dict(spec["env"])
                servers[name] = server_cfg
        except Exception:
            log.debug("Worker: failed to load MCP server %s for profile %s",
                      name, profile_name, exc_info=True)

    return servers


def _get_allowed_tools(profile_name: str) -> list[str]:
    """Get AUTO-tier tool names for a worker profile's MCP servers."""
    from agent import _MCP_SERVER_TOOL_NAMES

    auto_tools = set(config.ACTION_TIERS.get("AUTO", set()))
    disabled_tools = set(getattr(config, "DISABLED_TOOL_NAMES", set()))
    profile = WORKER_PROFILES.get(profile_name, WORKER_PROFILES["general"])
    server_names = profile.get("mcp_servers", [])

    if profile_name == "general" or not server_names:
        # General gets all AUTO tools
        return sorted(t for t in auto_tools if t not in disabled_tools)

    # Filter to tools from this profile's MCP servers
    profile_tools: set[str] = set()
    for name in server_names:
        profile_tools.update(_MCP_SERVER_TOOL_NAMES.get(name, set()))

    return sorted(t for t in profile_tools if t in auto_tools and t not in disabled_tools)


# ---------------------------------------------------------------------------
# Single worker execution
# ---------------------------------------------------------------------------

_worker_semaphores: dict[int, asyncio.Semaphore] = {}
_semaphore_lock = threading.Lock()


def _get_semaphore() -> asyncio.Semaphore:
    """Return a per-event-loop semaphore (avoids 'bound to different loop' errors)."""
    loop_id = id(asyncio.get_running_loop())
    with _semaphore_lock:
        if loop_id not in _worker_semaphores:
            _worker_semaphores[loop_id] = asyncio.Semaphore(config.MAX_CONCURRENT_WORKERS)
        return _worker_semaphores[loop_id]


async def _execute_worker(
    subtask: Subtask,
    context: str = "",
) -> WorkerResult:
    """Execute a single worker for a subtask.

    Acquires the worker semaphore, creates a Claude SDK client with the
    profile-specific tools, runs the query, and returns the result.
    """
    profile_name = subtask.profile
    if profile_name not in WORKER_PROFILES:
        profile_name = "general"

    circuit = _get_circuit(profile_name)
    with _circuit_lock:
        if not circuit.try_half_open():
            return WorkerResult(
                subtask_id=subtask.id,
                profile=profile_name,
                text="",
                success=False,
                error=f"Circuit breaker open for profile '{profile_name}' — skipping",
            )

    profile = WORKER_PROFILES[profile_name]
    model_key = profile.get("model_key", "WORKER_MODEL_DEFAULT")
    model = getattr(config, model_key, config.WORKER_MODEL_DEFAULT)
    timeout_key = profile.get("timeout", "WORKER_TIMEOUT_DEFAULT")
    timeout = float(getattr(config, timeout_key, config.WORKER_TIMEOUT_DEFAULT))

    prompt = profile.get("prompt", "You are a helpful assistant.")
    worker_prompt = f"{prompt}\n\nTask: {subtask.description}"
    if context:
        worker_prompt += f"\n\nContext from previous steps:\n{context}"

    start = time.monotonic()
    sem = _get_semaphore()

    async with sem:
        client: ClaudeSDKClient | None = None
        try:
            mcp_servers = _load_mcp_servers_for_profile(profile_name)
            allowed_tools = _get_allowed_tools(profile_name)

            options = ClaudeAgentOptions(
                system_prompt=prompt,
                model=model,
                allowed_tools=allowed_tools,
                mcp_servers=mcp_servers,
                cwd=str(getattr(config, "WORKSPACE", ".")),
                permission_mode="auto",
                max_turns=10,
            )

            client = ClaudeSDKClient(options=options)
            await client.connect()

            # Timeout covers both query() and receive_response()
            response_parts: list[str] = []
            async with asyncio.timeout(timeout):
                await client.query(worker_prompt)

                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                response_parts.append(block.text)
                    elif isinstance(msg, ResultMessage):
                        break

            response_text = "\n".join(response_parts).strip()
            elapsed = (time.monotonic() - start) * 1000

            with _circuit_lock:
                circuit.record_success()

            return WorkerResult(
                subtask_id=subtask.id,
                profile=profile_name,
                text=response_text,
                success=True,
                latency_ms=elapsed,
                model=model,
            )

        except asyncio.TimeoutError:
            elapsed = (time.monotonic() - start) * 1000
            with _circuit_lock:
                circuit.record_failure()
            log.warning("Worker %s timed out after %.0fms", profile_name, elapsed)
            return WorkerResult(
                subtask_id=subtask.id,
                profile=profile_name,
                text="",
                success=False,
                latency_ms=elapsed,
                model=model,
                error=f"Timed out after {timeout}s",
            )

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            with _circuit_lock:
                circuit.record_failure()
            log.error("Worker %s failed: %s", profile_name, exc, exc_info=True)
            return WorkerResult(
                subtask_id=subtask.id,
                profile=profile_name,
                text="",
                success=False,
                latency_ms=elapsed,
                model=model,
                error=str(exc),
            )

        finally:
            if client is not None:
                try:
                    await client.disconnect()
                except Exception:
                    log.warning("Worker disconnect failed for %s", profile_name, exc_info=True)


# ---------------------------------------------------------------------------
# Parallel execution with dependency resolution
# ---------------------------------------------------------------------------

async def execute_subtasks(
    triage: TriageResult,
    original_message: str = "",
) -> list[WorkerResult]:
    """Execute all subtasks from a triage result, respecting dependencies.

    Independent subtasks run in parallel (up to MAX_CONCURRENT_WORKERS).
    Dependent subtasks wait for their dependencies to complete first.
    """
    if not triage.subtasks:
        return []

    results: dict[str, WorkerResult] = {}
    subtask_map = {st.id: st for st in triage.subtasks}

    # Build dependency graph
    remaining = set(subtask_map.keys())
    completed = set()

    while remaining:
        # Find subtasks whose dependencies are all completed
        ready = []
        for st_id in remaining:
            st = subtask_map[st_id]
            deps = set(st.depends_on) & set(subtask_map.keys())
            if deps.issubset(completed):
                ready.append(st)

        if not ready:
            # Deadlock — break circular deps by running all remaining
            log.warning(
                "Dependency deadlock detected for subtasks: %s. Running all.",
                remaining,
            )
            ready = [subtask_map[st_id] for st_id in remaining]

        # Execute ready subtasks in parallel
        tasks = []
        for st in ready:
            # Build context from completed dependencies
            dep_context = ""
            if st.depends_on:
                dep_parts = []
                for dep_id in st.depends_on:
                    dep_result = results.get(dep_id)
                    if dep_result and dep_result.success:
                        dep_parts.append(f"[{dep_id}] {dep_result.text[:500]}")
                if dep_parts:
                    dep_context = "\n".join(dep_parts)

            tasks.append(_execute_worker(st, context=dep_context))

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, res in enumerate(batch_results):
            st = ready[i]
            if isinstance(res, Exception):
                results[st.id] = WorkerResult(
                    subtask_id=st.id,
                    profile=st.profile,
                    text="",
                    success=False,
                    error=str(res),
                )
            else:
                results[st.id] = res
            remaining.discard(st.id)
            completed.add(st.id)

    # Return in original subtask order
    return [results[st.id] for st in triage.subtasks if st.id in results]


# ---------------------------------------------------------------------------
# Result aggregator
# ---------------------------------------------------------------------------

def aggregate_results(
    triage: TriageResult,
    results: list[WorkerResult],
    original_message: str = "",
) -> str:
    """Combine worker results into a single coherent response.

    For simple tasks (1 worker): return the worker's response directly.
    For complex tasks (2+ workers): format a combined response,
    masking partial failures unless all workers failed.
    """
    if not results:
        return "I wasn't able to process that request. Could you try again?"

    # Filter to successful results
    successes = [r for r in results if r.success and r.text]
    failures = [r for r in results if not r.success]

    # All failed — report the error
    if not successes:
        error_summary = "; ".join(
            f"{r.profile}: {r.error or 'unknown error'}"
            for r in failures[:3]
        )
        return (
            "I ran into issues processing your request. "
            f"Here's what happened: {error_summary}. "
            "Could you try again or rephrase?"
        )

    # Single worker — return directly
    if len(results) == 1 and len(successes) == 1:
        return successes[0].text

    # Multiple workers — combine responses with profile labels
    parts: list[str] = []

    if len(successes) == len(results):
        # All succeeded — clean combination with labels
        for r in successes:
            parts.append(f"**{r.profile.title()}:**\n{r.text}")
    else:
        # Partial success — include results but note incomplete work
        for r in successes:
            parts.append(f"**{r.profile.title()}:**\n{r.text}")

        # Only mention failures if they seem critical
        critical_failures = [
            r for r in failures
            if r.profile not in ("writer",)  # Writer failures are cosmetic
        ]
        if critical_failures:
            failed_profiles = ", ".join(r.profile for r in critical_failures[:3])
            parts.append(
                f"\n_(Note: I wasn't able to complete the {failed_profiles} "
                f"part{'s' if len(critical_failures) > 1 else ''} of your request.)_"
            )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

async def run_workers(
    triage: TriageResult,
    original_message: str = "",
) -> str:
    """Execute triage subtasks through workers and return aggregated response.

    This is the main entry point called by agent.py when orchestrator is enabled.
    """
    start = time.monotonic()

    results = await execute_subtasks(triage, original_message)

    elapsed = (time.monotonic() - start) * 1000
    success_count = sum(1 for r in results if r.success)
    log.info(
        "Workers completed: %d/%d succeeded in %.0fms (profiles: %s)",
        success_count,
        len(results),
        elapsed,
        ", ".join(r.profile for r in results),
    )

    # Return empty string when ALL workers failed so the caller can
    # fall back to the serial Claude SDK path.
    if not any(r.success and r.text for r in results):
        log.warning("All workers failed — returning empty for serial fallback")
        return ""

    return aggregate_results(triage, results, original_message)


# ---------------------------------------------------------------------------
# Stats / introspection
# ---------------------------------------------------------------------------

def get_worker_stats() -> dict:
    """Return worker circuit breaker and profile status."""
    with _circuit_lock:
        circuit_info = {
            name: {
                "failures": state.failures,
                "open": state.is_open(),
            }
            for name, state in _circuit_states.items()
        }
    return {
        "max_concurrent": config.MAX_CONCURRENT_WORKERS,
        "profiles": list(WORKER_PROFILES.keys()),
        "circuit_breakers": circuit_info,
    }
