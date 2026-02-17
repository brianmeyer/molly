import asyncio
import importlib
import json
import logging
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    CLIJSONDecodeError,
    ClaudeSDKClient,
    ClaudeAgentOptions,
    CLIConnectionError,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
)

import config
from approval import RequestApprovalState, get_action_tier, is_auto_approved_path
from memory.processor import process_conversation
from memory.retriever import retrieve_context
from skills import get_skill_context, match_skills

log = logging.getLogger(__name__)

_MCP_SERVER_SPECS = {
    "google-calendar": ("tools.calendar", "calendar_server"),
    "gmail": ("tools.gmail", "gmail_server"),
    "google-people": ("tools.google_people", "people_server"),
    "google-tasks": ("tools.google_tasks", "tasks_server"),
    "google-drive": ("tools.google_drive", "drive_server"),
    "google-meet": ("tools.google_meet", "meet_server"),
    "apple-mcp": {
        "command": "bunx",
        "args": ["--no-cache", "apple-mcp@latest"],
    },
    "imessage": ("tools.imessage", "imessage_server"),
    "whatsapp-history": ("tools.whatsapp", "whatsapp_server"),
    "kimi": ("tools.kimi", "kimi_server"),
    "grok": ("tools.grok", "grok_server"),
    "groq": ("tools.groq", "groq_server"),
}

# Conditionally add browser MCP server when enabled
if config.BROWSER_MCP_ENABLED:
    _MCP_SERVER_SPECS["browser-mcp"] = {
        "command": "npx",
        "args": ["-y", "@anthropic-ai/browser-mcp@latest"],
        "env": {
            "BROWSER_PROFILE_DIR": str(config.BROWSER_PROFILE_DIR),
        },
    }

_MCP_SERVER_TOOL_NAMES = {
    "google-calendar": {
        "calendar_list", "calendar_get", "calendar_search",
        "calendar_create", "calendar_update", "calendar_delete",
    },
    "gmail": {"gmail_search", "gmail_read", "gmail_draft", "gmail_send", "gmail_reply"},
    "google-people": {"people_search", "people_get", "people_list"},
    "google-tasks": {"tasks_list", "tasks_list_tasks", "tasks_create", "tasks_complete", "tasks_delete"},
    "google-drive": {"drive_search", "drive_get", "drive_read"},
    "google-meet": {"meet_list", "meet_get", "meet_transcripts", "meet_recordings"},
    "apple-mcp": {
        "contacts", "notes", "messages", "mail", "reminders", "calendar", "maps",
    },
    "imessage": {"imessage_search", "imessage_recent", "imessage_thread", "imessage_unread"},
    "whatsapp-history": {"whatsapp_search"},
    "kimi": {"kimi_research"},
    "grok": {"grok_reason"},
    "groq": {"groq_reason"},
}

_CHAT_RUNTIME_LOCK = asyncio.Lock()
_CHAT_RUNTIME_IDLE_SECONDS = 30 * 60
_CHAT_RUNTIME_MAX_ENTRIES = 200
_SKILL_GAP_META_PREFIXES = ("routing:", "approval:")
_SKILL_GAP_BASELINE_TOOLS = {
    "memory_search",
}
_SKILL_GAP_BASELINE_SUFFIXES = (
    "__memory_search",
    ":memory_search",
    ".memory_search",
)
_AUTO_CREATE_NOTIFY_TOOLS = {"calendar_create", "tasks_create"}


@dataclass
class _ChatRuntime:
    """State for one chat's Claude SDK client lifecycle."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    client: ClaudeSDKClient | None = None
    system_prompt: str = ""
    session_id: str | None = None
    approval_manager: object | None = None
    molly_instance: object | None = None
    request_state: RequestApprovalState | None = None
    last_used_monotonic: float = field(default_factory=time.monotonic)


_CHAT_RUNTIMES: dict[str, _ChatRuntime] = {}


def _load_mcp_servers() -> dict[str, object]:
    """Load enabled MCP servers, skipping any disabled by startup preflight."""
    disabled_servers = set(getattr(config, "DISABLED_MCP_SERVERS", set()))
    disabled_tools = set(getattr(config, "DISABLED_TOOL_NAMES", set()))
    servers: dict[str, object] = {}

    for server_name, spec in _MCP_SERVER_SPECS.items():
        if server_name in disabled_servers:
            continue
        try:
            if isinstance(spec, tuple):
                module_name, attr_name = spec
                module = importlib.import_module(module_name)
                servers[server_name] = getattr(module, attr_name)
            elif isinstance(spec, dict):
                server_cfg = {"command": spec["command"]}
                if spec.get("args"):
                    server_cfg["args"] = list(spec["args"])
                if spec.get("env"):
                    server_cfg["env"] = dict(spec["env"])
                servers[server_name] = server_cfg
            else:
                raise TypeError(f"Unsupported MCP server spec type: {type(spec)!r}")
        except Exception as e:
            detail = spec[0] if isinstance(spec, tuple) else str(spec)
            log.warning(
                "Disabling MCP server '%s' — failed to load %s (%s)",
                server_name,
                detail,
                e,
            )
            disabled_servers.add(server_name)
            disabled_tools.update(_MCP_SERVER_TOOL_NAMES.get(server_name, set()))

    with config._runtime_lock:
        config.DISABLED_MCP_SERVERS = disabled_servers
        config.DISABLED_TOOL_NAMES = disabled_tools
    return servers


_MAX_DAILY_LOG_BYTES = 35_000


def load_identity_stack() -> str:
    """Read and concatenate all identity files into system prompt."""
    parts = []
    for path in config.IDENTITY_FILES:
        if path.exists():
            parts.append(f"<!-- {path.name} -->\n{path.read_text()}")

    # Add today's and yesterday's daily logs (tail-truncated to cap context size)
    today = date.today()
    for d in [today, today - timedelta(days=1)]:
        log_path = config.WORKSPACE / "memory" / f"{d.isoformat()}.md"
        if log_path.exists():
            content = log_path.read_text()
            if len(content) > _MAX_DAILY_LOG_BYTES:
                content = (
                    f"[... truncated {len(content) - _MAX_DAILY_LOG_BYTES} bytes ...]\n"
                    + content[-_MAX_DAILY_LOG_BYTES:]
                )
            parts.append(f"<!-- Daily Log: {d.isoformat()} -->\n{content}")

    return "\n\n---\n\n".join(parts)


def _normalize_tool_name(tool_name: str) -> str:
    """Normalize MCP-prefixed tool names to short names for tier lookup.

    The CLI sends tool names like 'mcp__gmail__gmail_send' for MCP tools
    but our ACTION_TIERS use short names like 'gmail_send'.
    Built-in tools ('Bash', 'Write') are passed through unchanged.
    """
    if tool_name.startswith("mcp__"):
        # mcp__servername__toolname → toolname
        parts = tool_name.split("__")
        if len(parts) >= 3:
            return "__".join(parts[2:])
    return tool_name


def make_tool_checker(
    approval_manager=None,
    molly=None,
    chat_jid: str = "",
    request_state: RequestApprovalState | None = None,
):
    """Create a can_use_tool callback for code-enforced action tiers.

    AUTO    → PermissionResultAllow (immediate)
    CONFIRM → WhatsApp approval flow → Allow/Deny (or deny if no approval_manager)
    BLOCKED → PermissionResultDeny (immediate)

    Within one handle_message request, CONFIRM-tier decisions are cached and
    coalesced to avoid repeated prompts.

    This callback only fires for tools NOT in allowed_tools (i.e., CONFIRM and
    BLOCKED tier tools). AUTO-tier tools are pre-approved via allowed_tools and
    never reach this callback.

    This callback MUST always be set so the SDK uses the stdio control protocol
    instead of interactive terminal prompts. When no approval_manager is available
    (heartbeat, terminal), CONFIRM-tier tools are denied with a message.
    """

    async def can_use_tool(tool_name: str, tool_input: dict, _context) -> PermissionResultAllow | PermissionResultDeny:
        short_name = _normalize_tool_name(tool_name)
        tier = get_action_tier(short_name, tool_input)
        t0 = time.time()
        log.info("can_use_tool fired: %s (normalized: %s) → tier=%s", tool_name, short_name, tier)

        if tier == "AUTO":
            _log_tool(tool_name, tool_input, True, t0, request_state=request_state)
            return PermissionResultAllow()

        if tier == "BLOCKED":
            log.warning("BLOCKED tool call: %s", tool_name)
            _log_tool(tool_name, tool_input, False, t0, "BLOCKED", request_state=request_state)
            return PermissionResultDeny(
                message=(
                    f"This action ({tool_name}) is blocked. "
                    f"It's classified as destructive or irreversible. "
                    f"Brian needs to do this himself."
                )
            )

        # CONFIRM tier
        # Check if this is a file write to an auto-approved path
        if is_auto_approved_path(short_name, tool_input):
            log.info("Auto-approved %s to memory path: %s", tool_name, tool_input.get("file_path", ""))
            _log_tool(tool_name, tool_input, True, t0, request_state=request_state)
            return PermissionResultAllow()

        # No approval manager (headless/heartbeat) — deny CONFIRM-tier
        if not approval_manager or not molly:
            log.info("No approval manager — denying CONFIRM tool: %s", tool_name)
            _log_tool(
                tool_name,
                tool_input,
                False,
                t0,
                "no_approval_manager",
                request_state=request_state,
            )
            return PermissionResultDeny(
                message=(
                    f"This action ({tool_name}) requires Brian's approval, "
                    f"but no approval channel is available right now."
                )
            )

        # Request approval via WhatsApp and wait
        log.info("CONFIRM tier — requesting approval for %s", tool_name)
        result = await approval_manager.request_tool_approval(
            tool_name, tool_input, chat_jid, molly,
            request_state=request_state,
        )

        if result is True:
            log.info("Tool approved: %s", tool_name)
            _log_tool(tool_name, tool_input, True, t0, request_state=request_state)
            return PermissionResultAllow()
        elif isinstance(result, str):
            # Edit instruction from Brian — deny with modification guidance
            log.info("Tool edit requested: %s → %s", tool_name, result)
            _log_tool(
                tool_name,
                tool_input,
                False,
                t0,
                "edit_requested",
                request_state=request_state,
            )
            return PermissionResultDeny(
                message=(
                    f"Brian wants you to modify this action before retrying. "
                    f"His edit instructions: {result}"
                )
            )
        else:
            log.info("Tool denied: %s", tool_name)
            _log_tool(
                tool_name,
                tool_input,
                False,
                t0,
                "denied/timed out",
                request_state=request_state,
            )
            return PermissionResultDeny(
                message="Brian denied this action or it timed out."
            )

    return can_use_tool


def _record_turn_tool_call(
    request_state: RequestApprovalState | None,
    tool_name: str,
):
    if request_state is None:
        return
    name = str(tool_name or "").strip()
    if not name:
        return
    request_state.turn_tool_calls.append(name)


def _record_executed_tool_call(
    request_state: RequestApprovalState | None,
    tool_name: str,
):
    if request_state is None:
        return
    name = str(tool_name or "").strip()
    if not name:
        return
    request_state.executed_tool_calls.append(name)


def _is_excluded_skill_gap_tool(tool_name: str) -> bool:
    name = str(tool_name or "").strip().lower()
    if not name:
        return True
    if any(name.startswith(prefix) for prefix in _SKILL_GAP_META_PREFIXES):
        return True
    if name in _SKILL_GAP_BASELINE_TOOLS:
        return True
    if any(name.endswith(suffix) for suffix in _SKILL_GAP_BASELINE_SUFFIXES):
        return True
    return False


def _filter_workflow_tool_calls(tool_names: list[str]) -> list[str]:
    return [name for name in tool_names if not _is_excluded_skill_gap_tool(name)]


def _log_tool(
    tool_name: str,
    tool_input: dict,
    success: bool,
    t0: float,
    error: str = "",
    request_state: RequestApprovalState | None = None,
):
    """Log a tool call to operational memory (best-effort)."""
    _record_turn_tool_call(request_state, tool_name)
    try:
        from memory.retriever import get_vectorstore
        vs = get_vectorstore()
        params = json.dumps(tool_input, default=str)[:500]
        latency_ms = int((time.time() - t0) * 1000)
        vs.log_tool_call(tool_name, params, success, latency_ms, error)
    except Exception:
        log.debug("Failed to log tool call: %s", tool_name, exc_info=True)


def _skill_name_for_log(skill: object) -> str:
    name = str(getattr(skill, "name", "") or "").strip()
    if name:
        return name
    return str(skill).strip()


def _log_skill_executions_best_effort(
    skill_names: list[str],
    trigger: str,
    outcome: str,
    detail: str = "",
):
    """Persist skill execution outcomes without interrupting response flow."""
    if not skill_names:
        return
    try:
        from memory.retriever import get_vectorstore

        vs = get_vectorstore()
        for skill_name in skill_names:
            vs.log_skill_execution(
                skill_name=skill_name,
                trigger=trigger,
                outcome=outcome,
                edits_made=detail,
            )
    except Exception:
        log.debug("Failed to log skill executions", exc_info=True)


def _schedule_skill_execution_logs(
    matched_skills: list[object],
    trigger: str,
    outcome: str,
    detail: str = "",
):
    """Schedule non-blocking skill execution telemetry writes."""
    seen: set[str] = set()
    skill_names: list[str] = []
    for skill in matched_skills:
        name = _skill_name_for_log(skill)
        if name and name not in seen:
            seen.add(name)
            skill_names.append(name)
    if not skill_names:
        return

    async def _emit():
        # Yield once so response sending is never blocked by telemetry.
        await asyncio.sleep(0)
        _log_skill_executions_best_effort(skill_names, trigger, outcome, detail)

    task = asyncio.create_task(
        _emit(),
        name=f"skill-telemetry:{','.join(skill_names)[:60]}",
    )
    task.add_done_callback(
        lambda t: log.debug("Skill telemetry task failed", exc_info=t.exception())
        if not t.cancelled() and t.exception() else None
    )


async def _get_chat_runtime(chat_id: str) -> _ChatRuntime:
    await _evict_stale_chat_runtimes()

    async with _CHAT_RUNTIME_LOCK:
        runtime = _CHAT_RUNTIMES.get(chat_id)
        if runtime is None:
            runtime = _ChatRuntime()
            _CHAT_RUNTIMES[chat_id] = runtime
        runtime.last_used_monotonic = time.monotonic()
        return runtime


async def _evict_stale_chat_runtimes() -> int:
    now = time.monotonic()
    victims: list[tuple[str, _ChatRuntime, str]] = []

    async with _CHAT_RUNTIME_LOCK:
        for chat_id, runtime in list(_CHAT_RUNTIMES.items()):
            idle_for_s = now - runtime.last_used_monotonic
            if runtime.lock.locked() or idle_for_s < _CHAT_RUNTIME_IDLE_SECONDS:
                continue
            _CHAT_RUNTIMES.pop(chat_id, None)
            victims.append((chat_id, runtime, f"idle>{_CHAT_RUNTIME_IDLE_SECONDS}s"))

        over_by = len(_CHAT_RUNTIMES) - _CHAT_RUNTIME_MAX_ENTRIES
        if over_by > 0:
            candidates = sorted(
                (
                    (chat_id, runtime)
                    for chat_id, runtime in _CHAT_RUNTIMES.items()
                    if not runtime.lock.locked()
                ),
                key=lambda pair: pair[1].last_used_monotonic,
            )
            for chat_id, runtime in candidates[:over_by]:
                _CHAT_RUNTIMES.pop(chat_id, None)
                victims.append((chat_id, runtime, "lru_capacity"))

    for chat_id, runtime, reason in victims:
        await _disconnect_runtime(runtime)
        log.info("Evicted chat runtime for %s (%s)", chat_id, reason)

    return len(victims)


def _build_turn_prompt(
    user_message: str,
    memory_context: str,
    skill_context: str,
    chat_context: str | None,
    response_guidance: str | None,
) -> str:
    sections: list[str] = []

    # Always inject current date/time in Brian's timezone so the agent never
    # has to guess "today" from UTC message timestamps.  (BUG-27 root cause fix)
    try:
        now_local = datetime.now(ZoneInfo(config.TIMEZONE))
        date_line = now_local.strftime("Current date/time: %A, %B %d, %Y %I:%M %p %Z")
        sections.append(date_line)
    except Exception:
        pass  # degrade gracefully — don't block the turn

    if memory_context:
        sections.append(f"Relevant memory:\n{memory_context}")
    if skill_context:
        sections.append(f"Skill guidance:\n{skill_context}")
    if chat_context:
        sections.append(f"Chat context:\n{chat_context}")
    if response_guidance:
        sections.append(f"Response format:\n{response_guidance}")

    if not sections:
        return user_message

    return (
        "Use this runtime context for this turn.\n\n"
        + "\n\n---\n\n".join(sections)
        + f"\n\n---\n\nUser request:\n{user_message}"
    )


def _response_guidance_for_source(source: str) -> str | None:
    if not config.WHATSAPP_PROMPT_GUARDRAILS:
        return None

    normalized = (source or "").strip().lower()
    if normalized not in {"whatsapp", "heartbeat", "automation", "imessage-mention"}:
        return None

    return (
        "The final response will be read in WhatsApp. "
        "Use plain text formatting that is easy to scan.\n"
        "- Do not use markdown tables.\n"
        "- Do not use # headings.\n"
        "- Do not use fenced code blocks unless explicitly requested.\n"
        "- Prefer short bullets with '-' and clear labels."
    )


def _iter_exceptions(exc: BaseException):
    if isinstance(exc, ExceptionGroup):
        for item in exc.exceptions:
            yield from _iter_exceptions(item)
    else:
        yield exc


def _is_recoverable_transport_error(exc: BaseException) -> bool:
    for item in _iter_exceptions(exc):
        if isinstance(item, (CLIConnectionError, CLIJSONDecodeError)):
            return True
        message = str(item).lower()
        if (
            "stream closed" in message
            or "not ready for writing" in message
            or "processtransport" in message
            or "buffer size" in message
        ):
            return True
    return False


def _handle_sdk_stderr(line: str):
    msg = line.strip()
    if not msg:
        return

    noisy_patterns = (
        "/$bunfs/root/claude:",
        "Error in hook callback",
        "Claude Code has been suspended.",
        "ctrl + z now suspends Claude Code",
        "error: Stream closed",
    )
    if any(pattern in msg for pattern in noisy_patterns):
        log.debug("Suppressed noisy Claude CLI stderr: %s", msg[:200])
        return

    log.warning("Claude CLI stderr: %s", msg[:500])


def _is_cross_task_cancel_scope_error(exc: BaseException) -> bool:
    """AnyIO raises this when query.close() runs in a different asyncio task."""
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return "cancel scope" in message and "different task" in message


async def _disconnect_runtime(runtime: _ChatRuntime):
    if runtime.client is None:
        return
    client = runtime.client
    runtime.client = None  # clear ref first so retries don't reuse a broken client
    try:
        await asyncio.wait_for(client.disconnect(), timeout=5.0)
    except asyncio.TimeoutError:
        log.warning("Claude SDK disconnect timed out — killing subprocess")
        _force_kill_sdk_client(client)
    except RuntimeError as exc:
        if _is_cross_task_cancel_scope_error(exc):
            # Expected when rotating a runtime across handler tasks.
            log.info("Claude SDK disconnect crossed task boundary; forcing transport shutdown")
            await _force_close_sdk_transport(client)
            _force_kill_sdk_client(client)
            return
        log.warning("Claude SDK disconnect failed — killing subprocess", exc_info=True)
        _force_kill_sdk_client(client)
    except Exception:
        log.warning("Claude SDK disconnect failed — killing subprocess", exc_info=True)
        _force_kill_sdk_client(client)


async def _force_close_sdk_transport(client: ClaudeSDKClient):
    """Best-effort close for SDK transport when query.close() can't run safely."""
    transport = getattr(client, "_transport", None)
    if transport is None:
        query = getattr(client, "_query", None)
        transport = getattr(query, "transport", None) if query is not None else None
    if transport is None:
        return

    try:
        await asyncio.wait_for(transport.close(), timeout=2.0)
    except Exception:
        log.debug("Could not close Claude SDK transport in fallback path", exc_info=True)
    finally:
        with suppress(Exception):
            setattr(client, "_query", None)
        with suppress(Exception):
            setattr(client, "_transport", None)


def _force_kill_sdk_client(client: ClaudeSDKClient):
    """Best-effort kill of the underlying SDK subprocess to prevent zombies."""
    try:
        proc = getattr(client, "_process", None) or getattr(client, "process", None)
        if proc is None:
            transport = getattr(client, "_transport", None)
            proc = getattr(transport, "_process", None) if transport is not None else None
        if proc is None:
            query = getattr(client, "_query", None)
            transport = getattr(query, "transport", None) if query is not None else None
            proc = getattr(transport, "_process", None) if transport is not None else None
        if proc is not None and getattr(proc, "returncode", None) is None:
            proc.kill()
            log.info("Force-killed lingering Claude SDK subprocess (pid=%s)", getattr(proc, "pid", "?"))
    except Exception:
        log.debug("Could not force-kill SDK subprocess", exc_info=True)


async def _ensure_connected_runtime(
    runtime: _ChatRuntime,
    chat_id: str,
    system_prompt: str,
) -> None:
    if runtime.client is not None and runtime.system_prompt == system_prompt:
        return

    if runtime.client is not None:
        await _disconnect_runtime(runtime)

    disabled_tools = set(getattr(config, "DISABLED_TOOL_NAMES", set()))
    auto_tools = sorted(t for t in config.ACTION_TIERS["AUTO"] if t not in disabled_tools)

    async def can_use_tool(tool_name: str, tool_input: dict, context):
        checker = make_tool_checker(
            approval_manager=runtime.approval_manager,
            molly=runtime.molly_instance,
            chat_jid=chat_id,
            request_state=runtime.request_state,
        )
        return await checker(tool_name, tool_input, context)

    async def _on_post_tool_use(input, tool_use_id, context):
        if not isinstance(input, dict):
            return {"continue_": True}
        tool_name = _normalize_tool_name(str(input.get("tool_name", "")))
        tool_input = input.get("tool_input", {})
        if not isinstance(tool_input, dict):
            tool_input = {}
        _record_executed_tool_call(runtime.request_state, tool_name)
        _maybe_notify_auto_created(
            runtime=runtime,
            chat_id=chat_id,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_response=input.get("tool_response"),
        )
        return {"continue_": True}

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        model=config.CLAUDE_MODEL,
        allowed_tools=auto_tools,
        agents=_build_agents(),
        hooks={
            "SubagentStart": [HookMatcher(hooks=[_on_subagent_start])],
            "SubagentStop": [HookMatcher(hooks=[_on_subagent_stop])],
            "PostToolUse": [HookMatcher(hooks=[_on_post_tool_use])],
        },
        mcp_servers=_load_mcp_servers(),
        cwd=str(config.WORKSPACE),
        stderr=_handle_sdk_stderr,
        can_use_tool=can_use_tool,
        max_buffer_size=10 * 1024 * 1024,  # 10MB (default 1MB too small for long sessions)
    )
    if runtime.session_id:
        options.resume = runtime.session_id

    runtime.client = ClaudeSDKClient(options=options)
    await runtime.client.connect()
    runtime.system_prompt = system_prompt


async def _query_with_client(
    runtime: _ChatRuntime,
    turn_prompt: str,
) -> tuple[str, str | None]:
    if runtime.client is None:
        raise RuntimeError("Claude client is not connected")

    response_text = ""
    new_session_id = None
    active_session = runtime.session_id or "default"

    await runtime.client.query(turn_prompt, session_id=active_session)

    async for message in runtime.client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_text += block.text
        elif isinstance(message, ResultMessage):
            new_session_id = message.session_id
            if message.is_error:
                log.error("Claude returned error: %s", message.result)

    return response_text, new_session_id


def _emit_approval_metrics(request_state: RequestApprovalState):
    if request_state.tool_asks <= 0:
        return
    log.info(
        "[approval METRIC] - request_id=%s | tool_asks=%d | prompts_sent=%d | "
        "auto_approved=%d | all_grant=%s",
        request_state.request_id,
        request_state.tool_asks,
        request_state.prompts_sent,
        request_state.auto_approved,
        str(request_state.approved_all_confirm).lower(),
    )


def _maybe_log_skill_gap(
    user_message: str,
    matched_skills: list,
    request_state: RequestApprovalState | None,
    session_id: str | None,
):
    if matched_skills or request_state is None:
        return

    workflow_calls = _filter_workflow_tool_calls(request_state.turn_tool_calls)
    if len(workflow_calls) < 3:
        return

    try:
        from memory.retriever import get_vectorstore

        vs = get_vectorstore()
        gap_id = vs.log_skill_gap(
            user_message=user_message,
            tools_used=workflow_calls,
            session_id=session_id or "",
        )
        log.info(
            "Skill gap logged: id=%s tools=%d session=%s",
            gap_id,
            len(workflow_calls),
            session_id or "",
        )
    except Exception:
        log.debug("Failed to log skill gap", exc_info=True)


def _build_agents() -> dict[str, AgentDefinition]:
    """Define sub-agents for intelligent task routing.

    Opus orchestrates and delegates to these agents via the Task tool.
    Sub-agents inherit all available tools (tools=None).
    """
    return {
        "quick": AgentDefinition(
            description="Fast lookups, formatting, trivial subtasks",
            prompt=(
                "You are a fast helper. Handle simple lookups, formatting, "
                "unit conversions, quick calculations, and other mechanical "
                "subtasks. Be concise and direct."
            ),
            model="haiku",
        ),
        "worker": AgentDefinition(
            description="Email drafts, research synthesis, multi-step tool use",
            prompt=(
                "You are a capable worker. Handle substantial tasks like "
                "drafting emails, synthesizing research across multiple sources, "
                "multi-step tool workflows, and data gathering. Be thorough "
                "but efficient."
            ),
            model="sonnet",
        ),
        "analyst": AgentDefinition(
            description="Deep analysis, strategic thinking, complex reasoning",
            prompt=(
                "You are a deep analyst. Handle tasks requiring careful reasoning, "
                "strategic analysis, nuanced judgment, and complex multi-factor "
                "decisions. Take your time and be thorough."
            ),
            model="opus",
        ),
    }


async def _on_subagent_start(input, tool_use_id, context):
    """Log routing decision when a sub-agent is started."""
    agent_type = input.get("agent_type", "unknown") if isinstance(input, dict) else "unknown"
    try:
        from memory.retriever import get_vectorstore
        vs = get_vectorstore()
        vs.log_tool_call(
            f"routing:subagent_start:{agent_type}",
            json.dumps({"agent": agent_type}),
            True, 0, "",
        )
    except Exception:
        log.debug("Failed to log subagent start: %s", agent_type, exc_info=True)
    return {"continue_": True}


async def _on_subagent_stop(input, tool_use_id, context):
    """Log when a sub-agent completes."""
    agent_type = input.get("agent_type", "unknown") if isinstance(input, dict) else "unknown"
    try:
        from memory.retriever import get_vectorstore
        vs = get_vectorstore()
        vs.log_tool_call(
            f"routing:subagent_stop:{agent_type}",
            json.dumps({"agent": agent_type}),
            True, 0, "",
        )
    except Exception:
        log.debug("Failed to log subagent stop: %s", agent_type, exc_info=True)
    return {"continue_": True}


def _maybe_notify_auto_created(
    runtime: _ChatRuntime,
    chat_id: str,
    tool_name: str,
    tool_input: dict,
    tool_response,
):
    if tool_name not in _AUTO_CREATE_NOTIFY_TOOLS:
        return
    molly = runtime.molly_instance
    if not molly:
        return
    notifier = getattr(molly, "notify_auto_created_tool_result", None)
    if not callable(notifier):
        return
    try:
        notifier(chat_id, tool_name, tool_input, tool_response)
    except Exception:
        log.warning("Auto-create notification failed for %s", tool_name, exc_info=True)


async def handle_message(
    user_message: str,
    chat_id: str,
    session_id: str | None = None,
    approval_manager=None,
    molly_instance=None,
    source: str = "unknown",
    chat_context: str | None = None,
) -> tuple[str, str | None]:
    """Send a message through Claude and return (response_text, new_session_id)."""
    runtime = await _get_chat_runtime(chat_id)

    # Pre-processing: run independent steps concurrently
    # retrieve_context is async (runs its own executor threads internally),
    # while load_identity_stack and match_skills are sync → run in executor.
    # return_exceptions=True lets us degrade gracefully if retrieval fails.
    loop = asyncio.get_running_loop()
    results = await asyncio.gather(
        asyncio.wait_for(loop.run_in_executor(None, load_identity_stack), timeout=15.0),
        retrieve_context(user_message),
        asyncio.wait_for(loop.run_in_executor(None, match_skills, user_message), timeout=15.0),
        return_exceptions=True,
    )
    # Unpack with graceful degradation: system_prompt is required,
    # but memory context and skills can fall back to empty defaults.
    # Use BaseException for system_prompt (re-raise KeyboardInterrupt/SystemExit),
    # but Exception for optional results (let shutdown signals propagate).
    system_prompt = results[0]
    if isinstance(system_prompt, BaseException):
        raise system_prompt  # Identity stack is required — can't respond without it
    if isinstance(results[1], Exception):
        log.warning("Memory retrieval failed, continuing without context: %s", results[1])
        memory_context_raw = ""
    else:
        memory_context_raw = results[1]
    if isinstance(results[2], Exception):
        log.warning("Skill matching failed, continuing without skills: %s", results[2])
        matched_skills = []
    else:
        matched_skills = results[2]
    memory_context = memory_context_raw or ""
    skill_context = get_skill_context(matched_skills) or ""
    response_guidance = _response_guidance_for_source(source)
    turn_prompt = _build_turn_prompt(
        user_message=user_message,
        memory_context=memory_context,
        skill_context=skill_context,
        chat_context=chat_context,
        response_guidance=response_guidance,
    )

    response_text = ""
    new_session_id = None
    request_state = RequestApprovalState()
    query_succeeded = False
    failure_detail = ""

    # --- Orchestrator path (Phase 5A) ---
    # When enabled, route through Kimi K2.5 triage + parallel workers.
    # Falls back to serial Claude SDK path on ANY failure.
    orchestrator_handled = False
    if getattr(config, "ORCHESTRATOR_ENABLED", False):
        try:
            from orchestrator import classify_message
            from workers import run_workers

            triage = await classify_message(user_message)
            if triage.classification in ("simple", "complex") and triage.subtasks:
                response_text = await run_workers(triage, original_message=user_message)
                if response_text:
                    orchestrator_handled = True
                    query_succeeded = True
                    # Orchestrator uses its own worker sessions; don't carry
                    # forward a stale session_id from a previous serial call.
                    new_session_id = session_id or None
                    log.info(
                        "Orchestrator handled message: %s (%s, %.0fms, %d workers)",
                        triage.classification, triage.model_used,
                        triage.latency_ms, len(triage.subtasks),
                    )
        except Exception as exc:
            log.warning(
                "Orchestrator path failed, falling back to serial: %s", exc,
                exc_info=True,
            )
            orchestrator_handled = False

    if not orchestrator_handled:
        async with runtime.lock:
            runtime.last_used_monotonic = time.monotonic()

            if session_id and session_id != runtime.session_id:
                runtime.session_id = session_id
                await _disconnect_runtime(runtime)
            elif session_id and runtime.session_id is None:
                runtime.session_id = session_id

            runtime.approval_manager = approval_manager
            runtime.molly_instance = molly_instance
            runtime.request_state = request_state

            try:
                for attempt in range(2):
                    try:
                        await _ensure_connected_runtime(runtime, chat_id, system_prompt)
                        response_text, new_session_id = await _query_with_client(runtime, turn_prompt)
                        query_succeeded = True
                        failure_detail = ""
                        break
                    except Exception as exc:
                        failure_detail = type(exc).__name__
                        recoverable = _is_recoverable_transport_error(exc)
                        if recoverable:
                            # If buffer overflow, the session itself is too large — reset it
                            is_overflow = any(
                                "buffer size" in str(e).lower()
                                for e in _iter_exceptions(exc)
                            )
                            if is_overflow:
                                log.warning(
                                    "Session buffer overflow in %s — resetting session (attempt %d/2)",
                                    chat_id,
                                    attempt + 1,
                                )
                                runtime.session_id = None
                            else:
                                log.warning(
                                    "Recoverable Claude transport error in %s (attempt %d/2): %s",
                                    chat_id,
                                    attempt + 1,
                                    exc,
                                )
                            await _disconnect_runtime(runtime)
                            if approval_manager and hasattr(approval_manager, "cancel_pending"):
                                with suppress(Exception):
                                    approval_manager.cancel_pending(chat_id)
                            request_state.reset_for_retry()
                            if attempt == 0:
                                continue

                        log.error("Claude query failed for chat %s", chat_id, exc_info=True)
                        if not response_text:
                            response_text = "Something went wrong on my end. Try again in a moment."
                        break
            finally:
                runtime.request_state = None

            if new_session_id:
                runtime.session_id = new_session_id
            else:
                new_session_id = runtime.session_id
            runtime.last_used_monotonic = time.monotonic()

    _emit_approval_metrics(request_state)
    _maybe_log_skill_gap(
        user_message=user_message,
        matched_skills=matched_skills,
        request_state=request_state,
        session_id=new_session_id,
    )

    if matched_skills:
        outcome = "success"
        detail = ""
        if not query_succeeded or response_text.startswith("Something went wrong"):
            outcome = "failure"
            detail = failure_detail or "response_error"
        trigger = f"{source}:{user_message}"
        _schedule_skill_execution_logs(matched_skills, trigger=trigger, outcome=outcome, detail=detail)

    # Best-effort: write foundry observation for multi-step executed tool sequences
    executed_calls: list[str] = []
    if request_state:
        executed_calls = list(request_state.executed_tool_calls)
        if not executed_calls:
            executed_calls = list(request_state.turn_tool_calls)

    if len(executed_calls) >= 3:
        try:
            from foundry_adapter import write_observation
            write_observation(
                tool_sequence=executed_calls,
                outcome="success" if query_succeeded else "failure",
                context=user_message[:200],
            )
        except Exception:
            log.debug("Failed to write foundry observation", exc_info=True)

    # Async post-processing: embed + store conversation turn
    if response_text and not response_text.startswith("Something went wrong"):
        task = asyncio.create_task(
            process_conversation(user_message, response_text, chat_id, source=source),
            name=f"post-process:{chat_id[:20]}",
        )
        task.add_done_callback(
            lambda t: log.error("Post-processing failed: %s", t.exception(), exc_info=t.exception())
            if not t.cancelled() and t.exception() else None
        )

    log.info(
        "Response for %s: %d chars, session=%s",
        chat_id,
        len(response_text),
        new_session_id,
    )
    return response_text, new_session_id
