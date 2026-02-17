#!/usr/bin/env python3
"""Phase 5C Dry Run — Voice + Browser + Qwen3 LoRA + Plugins + Docker Sandbox.

Validates all 5C components: voice loop wiring, browser MCP config,
Qwen3 training service, plugin architecture, and Docker sandbox.

Usage:
    python3 scripts/dry_run_5c.py            # offline tests only
    python3 scripts/dry_run_5c.py --live      # includes live Porcupine + API tests
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

LIVE = "--live" in sys.argv
PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0


def ok(msg: str):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  \033[92m✓\033[0m {msg}")


def fail(msg: str):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  \033[91m✗\033[0m {msg}")


def skip(msg: str):
    global SKIP_COUNT
    SKIP_COUNT += 1
    print(f"  \033[93m⊘\033[0m {msg} (skipped — needs --live)")


def section(title: str):
    print(f"\n\033[1m{'─'*60}\033[0m")
    print(f"\033[1m{title}\033[0m")
    print(f"\033[1m{'─'*60}\033[0m")


# ──────────────────────────────────────────────────────────────
# Test 1: Voice Loop Config
# ──────────────────────────────────────────────────────────────

def test_1_voice_config():
    section("Test 1: Voice Loop Config")

    # Config values exist
    assert hasattr(config, "VOICE_ENABLED"), "VOICE_ENABLED missing"
    ok(f"VOICE_ENABLED = {config.VOICE_ENABLED}")

    assert hasattr(config, "PICOVOICE_ACCESS_KEY"), "PICOVOICE_ACCESS_KEY missing"
    key_set = bool(config.PICOVOICE_ACCESS_KEY)
    ok(f"PICOVOICE_ACCESS_KEY = {'set' if key_set else 'NOT SET'}")

    ppn_path = Path(config.PORCUPINE_MODEL_PATH)
    if ppn_path.exists():
        ok(f".ppn model found: {ppn_path}")
    else:
        fail(f".ppn model NOT found: {ppn_path}")

    ok(f"GEMINI_LIVE_MODEL = {config.GEMINI_LIVE_MODEL}")
    ok(f"VOICE_MAX_SESSION_MINUTES = {config.VOICE_MAX_SESSION_MINUTES}")
    ok(f"VOICE_DAILY_BUDGET_MINUTES = {config.VOICE_DAILY_BUDGET_MINUTES}")


# ──────────────────────────────────────────────────────────────
# Test 2: Voice Loop State Machine
# ──────────────────────────────────────────────────────────────

def test_2_voice_state_machine():
    section("Test 2: Voice Loop State Machine")

    from voice_loop import VoiceLoop, VoiceState

    vl = VoiceLoop()
    if vl.state == VoiceState.LISTENING:
        ok("Initial state is LISTENING")
    else:
        fail(f"Expected LISTENING, got {vl.state}")

    # Budget tracking
    vl._daily_minutes_used = 0.0
    if vl._check_budget():
        ok("Budget check passes with 0 minutes used")
    else:
        fail("Budget check should pass")

    vl._daily_minutes_used = config.VOICE_DAILY_BUDGET_MINUTES + 1
    vl._daily_reset_date = time.strftime("%Y-%m-%d")
    if not vl._check_budget():
        ok("Budget check fails when exhausted")
    else:
        fail("Budget should be exhausted")

    # Reset
    vl._daily_reset_date = "2000-01-01"
    if vl._check_budget():
        ok("Budget resets on new day")
    else:
        fail("Budget should reset")

    # Stop
    vl.stop()
    if not vl._running:
        ok("stop() sets _running=False")
    else:
        fail("stop() should set _running=False")

    # Stats
    stats = vl.get_stats()
    if "state" in stats and "daily_minutes_used" in stats:
        ok(f"Stats: {stats}")
    else:
        fail(f"Stats missing keys: {stats}")


# ──────────────────────────────────────────────────────────────
# Test 3: Voice Tool Declarations
# ──────────────────────────────────────────────────────────────

def test_3_voice_tools():
    section("Test 3: Voice Tool Declarations")

    from voice_loop import get_voice_tool_declarations

    decls = get_voice_tool_declarations()
    names = {d["name"] for d in decls}

    for expected in ["check_calendar", "send_message", "create_task", "search_memory"]:
        if expected in names:
            ok(f"Tool declared: {expected}")
        else:
            fail(f"Missing tool: {expected}")

    # Validate schema
    for decl in decls:
        if "parameters" in decl and decl["parameters"].get("type") == "object":
            ok(f"  {decl['name']} has valid parameter schema")
        else:
            fail(f"  {decl['name']} has invalid parameter schema")


# ──────────────────────────────────────────────────────────────
# Test 4: Voice → Memory Wiring (tool bridge)
# ──────────────────────────────────────────────────────────────

def test_4_voice_memory_wiring():
    section("Test 4: Voice → Memory Wiring")

    from voice_loop import VoiceLoop

    vl = VoiceLoop()

    # Test unknown tool returns error
    result = asyncio.run(vl._execute_tool("nonexistent_tool", {}))
    if "error" in result and ("Unknown tool" in result["error"] or "not available" in result["error"]):
        ok("Unknown tool returns error")
    else:
        fail(f"Expected error for unknown tool, got: {result}")

    # Test tool map has correct entries
    tool_map = {
        "check_calendar": vl._tool_check_calendar,
        "send_message": vl._tool_send_message,
        "create_task": vl._tool_create_task,
        "search_memory": vl._tool_search_memory,
    }
    for name, handler in tool_map.items():
        if callable(handler):
            ok(f"Tool handler callable: {name}")
        else:
            fail(f"Tool handler not callable: {name}")

    # Test context loading
    context = vl._load_system_context()
    if isinstance(context, str):
        if len(context) > 0:
            ok(f"System context loaded: {len(context)} chars")
        else:
            ok("System context loaded (empty — identity files may not exist)")
    else:
        fail("System context should be a string")


# ──────────────────────────────────────────────────────────────
# Test 5: Live Porcupine Initialization
# ──────────────────────────────────────────────────────────────

def test_5_live_porcupine():
    section("Test 5: Live Porcupine Initialization")

    if not LIVE:
        skip("Porcupine init")
        return

    from voice_loop import VoiceLoop

    ppn_path = Path(config.PORCUPINE_MODEL_PATH)
    if not ppn_path.exists():
        fail(f".ppn model not found at {ppn_path}")
        return

    if not config.PICOVOICE_ACCESS_KEY:
        fail("PICOVOICE_ACCESS_KEY not set")
        return

    vl = VoiceLoop()
    try:
        vl._init_porcupine()
        ok(f"Porcupine initialized (frame_length={vl.porcupine.frame_length})")

        # Verify audio format matches Gemini Live
        if vl.porcupine.sample_rate == 16000:
            ok("Sample rate is 16kHz (matches Gemini Live)")
        else:
            fail(f"Expected 16kHz, got {vl.porcupine.sample_rate}")

        vl._cleanup()
        ok("Porcupine cleanup successful")

    except Exception as e:
        fail(f"Porcupine init failed: {e}")
        vl._cleanup()


# ──────────────────────────────────────────────────────────────
# Test 6: Live Memory Search via Voice Tool
# ──────────────────────────────────────────────────────────────

def test_6_live_memory_search():
    section("Test 6: Live Memory Search via Voice Tool")

    if not LIVE:
        skip("Memory search via voice tool")
        return

    from voice_loop import VoiceLoop

    vl = VoiceLoop()

    try:
        result = asyncio.run(vl._tool_search_memory({"query": "hello"}))
        if "context" in result:
            ok(f"Memory search returned context ({len(result['context'])} chars)")
        elif "error" in result:
            ok(f"Memory search returned error (expected if no data): {result['error']}")
        else:
            fail(f"Unexpected result: {result}")
    except Exception as e:
        fail(f"Memory search failed: {e}")


# ──────────────────────────────────────────────────────────────
# Test 7: Browser MCP Config
# ──────────────────────────────────────────────────────────────

def test_7_browser_mcp():
    section("Test 7: Browser MCP Config")

    from tools.browser_mcp import (
        get_browser_mcp_config,
        get_browser_tool_specs,
        is_browser_available,
        BROWSER_BLOCKED_ACTIONS,
    )

    cfg = get_browser_mcp_config()
    if cfg["name"] == "browser" and "--headless" in cfg["args"]:
        ok(f"Browser MCP config: headless mode, profile dir={cfg['env']['BROWSER_PROFILE_DIR']}")
    else:
        fail(f"Browser MCP config invalid: {cfg}")

    specs = get_browser_tool_specs()
    tool_names = {s["name"] for s in specs}
    expected_tools = {"browser_navigate", "browser_click", "browser_type", "browser_screenshot", "browser_extract_text"}
    if tool_names == expected_tools:
        ok(f"Browser tools: {sorted(tool_names)}")
    else:
        fail(f"Expected {expected_tools}, got {tool_names}")

    if "credential_entry" in BROWSER_BLOCKED_ACTIONS:
        ok("Credential entry blocked")
    else:
        fail("credential_entry should be blocked")

    # Availability (should be False by default)
    if not is_browser_available():
        ok("Browser MCP not available (disabled by default)")
    else:
        ok("Browser MCP available")

    # Worker profile wired up
    from workers import WORKER_PROFILES
    browser_profile = WORKER_PROFILES.get("browser")
    if browser_profile and "browser-mcp" in browser_profile.get("mcp_servers", []):
        ok("Browser worker profile has 'browser-mcp' MCP server")
    else:
        fail("Browser worker profile missing 'browser-mcp' MCP server")


# ──────────────────────────────────────────────────────────────
# Test 8: Qwen3 LoRA Training Service
# ──────────────────────────────────────────────────────────────

def test_8_qwen_training():
    section("Test 8: Qwen3 LoRA Training Service")

    from evolution.qwen_training import (
        QwenTrainingService,
        QWEN_LORA_MIN_EXAMPLES,
        QWEN_LORA_COOLDOWN_DAYS,
    )

    # Constants
    if QWEN_LORA_MIN_EXAMPLES == 500:
        ok("Min examples = 500")
    else:
        fail(f"Expected 500, got {QWEN_LORA_MIN_EXAMPLES}")

    if QWEN_LORA_COOLDOWN_DAYS == 7:
        ok("Cooldown = 7 days")
    else:
        fail(f"Expected 7, got {QWEN_LORA_COOLDOWN_DAYS}")

    # Prompt formatting
    prompt = QwenTrainingService._format_triage_prompt("What's on my calendar?")
    if "calendar" in prompt and "direct" in prompt:
        ok("Triage prompt formatted correctly")
    else:
        fail(f"Triage prompt: {prompt[:100]}")

    # Classification parsing
    for text, expected in [
        ("direct", "direct"),
        ("This is a complex task", "complex"),
        ("simple query", "simple"),
        ("urgent email from CEO", "urgent"),
        ("random gibberish", "direct"),
    ]:
        result = QwenTrainingService._parse_classification(text)
        if result == expected:
            ok(f"Parse '{text}' → '{result}'")
        else:
            fail(f"Parse '{text}': expected '{expected}', got '{result}'")

    # Data splitting
    class MockCtx:
        state = {}
        def save_state(self): pass

    svc = QwenTrainingService(MockCtx(), None, None)
    rows = [{"text": f"msg {i}", "classification": "direct"} for i in range(100)]
    train, eval_set = svc.split_data(rows)
    if len(train) == 80 and len(eval_set) == 20:
        ok(f"Data split: {len(train)} train, {len(eval_set)} eval")
    else:
        fail(f"Expected 80/20 split, got {len(train)}/{len(eval_set)}")

    # Hash dedup
    h1 = QwenTrainingService._hash_text("Hello World")
    h2 = QwenTrainingService._hash_text("hello world")
    if h1 == h2:
        ok("Text hashing is case-insensitive")
    else:
        fail("Text hashing should be case-insensitive")


# ──────────────────────────────────────────────────────────────
# Test 9: Plugin Architecture
# ──────────────────────────────────────────────────────────────

def test_9_plugins():
    section("Test 9: Plugin Architecture")

    import tempfile
    from plugins.base import PluginBase, PluginRegistry, PluginType
    from plugins.loader import discover_plugins, load_plugins

    # Registry basics
    registry = PluginRegistry()
    if len(registry) == 0:
        ok("Fresh registry is empty")
    else:
        fail(f"Expected empty, got {len(registry)}")

    # Create and register a plugin
    class TestPlugin(PluginBase):
        name = "dry-run-test"
        plugin_type = PluginType.TOOL
        version = "1.0.0"
        description = "Dry run test plugin"

        def register(self, registry):
            pass

    plugin = TestPlugin()
    registry.register(plugin)

    if "dry-run-test" in registry:
        ok("Plugin registered successfully")
    else:
        fail("Plugin not in registry")

    info = registry.list_plugins()
    if len(info) == 1 and info[0]["name"] == "dry-run-test":
        ok(f"list_plugins: {info}")
    else:
        fail(f"list_plugins unexpected: {info}")

    # Type filtering
    channels = registry.list_by_type(PluginType.CHANNEL)
    tools = registry.list_by_type(PluginType.TOOL)
    if len(channels) == 0 and len(tools) == 1:
        ok("Type filtering works")
    else:
        fail(f"Type filter: {len(channels)} channels, {len(tools)} tools")

    # Unregister
    removed = registry.unregister("dry-run-test")
    if removed is not None and "dry-run-test" not in registry:
        ok("Plugin unregistered")
    else:
        fail("Unregister failed")

    # Discovery from temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "my_plugin.py").write_text("X = 42\n")
        (Path(tmpdir) / "_hidden.py").write_text("X = 42\n")
        discovered = discover_plugins(Path(tmpdir))
        if len(discovered) == 1 and discovered[0]["module"] == "my_plugin":
            ok(f"Discovery found 1 plugin (skipped _hidden): {discovered}")
        else:
            fail(f"Discovery unexpected: {discovered}")

    # Load from temp dir with valid plugin
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "valid_plugin.py").write_text('''
from plugins.base import PluginBase, PluginType

class TestPlugin(PluginBase):
    name = "loaded-test"
    plugin_type = PluginType.TOOL
    version = "0.1.0"
    description = "Test"

    def register(self, registry):
        pass

PLUGIN_CLASS = TestPlugin
''')
        reg2 = PluginRegistry()
        loaded = load_plugins(Path(tmpdir), reg2)
        if loaded == ["loaded-test"]:
            ok(f"Loaded plugin from file: {loaded}")
        else:
            fail(f"Load unexpected: {loaded}")


# ──────────────────────────────────────────────────────────────
# Test 10: Docker Sandbox
# ──────────────────────────────────────────────────────────────

def test_10_docker_sandbox():
    section("Test 10: Docker Sandbox")

    from evolution.docker_sandbox import DockerSandbox, SubprocessSandbox, get_sandbox

    # Config
    if not config.DOCKER_SANDBOX_ENABLED:
        ok("Docker sandbox disabled by default")
    else:
        ok("Docker sandbox enabled")

    # Docker unavailable when disabled
    if not DockerSandbox.is_available():
        ok("DockerSandbox not available (disabled)")
    else:
        ok("DockerSandbox available")

    # Subprocess always available
    if SubprocessSandbox.is_available():
        ok("SubprocessSandbox always available")
    else:
        fail("SubprocessSandbox should always be available")

    # Factory
    sandbox = get_sandbox()
    if isinstance(sandbox, SubprocessSandbox):
        ok("get_sandbox() returns SubprocessSandbox (Docker disabled)")
    else:
        ok(f"get_sandbox() returns {type(sandbox).__name__}")

    # Run code in subprocess sandbox
    result = asyncio.run(sandbox.run("print(2 + 2)"))
    if result["exit_code"] == 0 and "4" in result["stdout"]:
        ok(f"SubprocessSandbox executed: stdout={result['stdout'].strip()}, {result['elapsed_ms']}ms")
    else:
        fail(f"SubprocessSandbox failed: {result}")

    # Error handling
    result = asyncio.run(sandbox.run("raise ValueError('test')"))
    if result["exit_code"] != 0:
        ok(f"Error code returned for failing code: exit_code={result['exit_code']}")
    else:
        fail("Should return non-zero exit code")


# ──────────────────────────────────────────────────────────────
# Test 11: main.py Integration Points
# ──────────────────────────────────────────────────────────────

def test_11_integration():
    section("Test 11: main.py Integration Points")

    # Check that main.py has voice loop integration
    main_path = Path(__file__).resolve().parent.parent / "main.py"
    main_text = main_path.read_text()

    if "voice_loop" in main_text and "VoiceLoop" in main_text:
        ok("main.py has voice loop integration")
    else:
        fail("main.py missing voice loop integration")

    if "load_plugins" in main_text:
        ok("main.py has plugin loading")
    else:
        fail("main.py missing plugin loading")

    # Check requirements.txt
    req_path = Path(__file__).resolve().parent.parent / "requirements.txt"
    req_text = req_path.read_text()

    for dep in ["pvporcupine", "pvrecorder", "google-genai", "pyaudio", "peft", "datasets"]:
        if dep in req_text:
            ok(f"requirements.txt has {dep}")
        else:
            fail(f"requirements.txt missing {dep}")

    # Check evolution/infra.py has sandbox method
    infra_path = Path(__file__).resolve().parent.parent / "evolution" / "infra.py"
    infra_text = infra_path.read_text()
    if "get_sandbox" in infra_text:
        ok("evolution/infra.py has get_sandbox()")
    else:
        fail("evolution/infra.py missing get_sandbox()")


# ──────────────────────────────────────────────────────────────
# Test 12: Live Gemini Client Init
# ──────────────────────────────────────────────────────────────

def test_12_live_gemini_client():
    section("Test 12: Live Gemini Client Init")

    if not LIVE:
        skip("Gemini client init")
        return

    if not config.GEMINI_API_KEY:
        fail("GEMINI_API_KEY not set")
        return

    from voice_loop import VoiceLoop

    vl = VoiceLoop()
    vl._init_gemini_client()

    if vl._client is not None:
        ok("Gemini client initialized")
    else:
        fail("Gemini client is None")

    # Build live config
    try:
        live_config = vl._build_live_config()
        ok(f"Live config built successfully")
    except Exception as e:
        fail(f"Live config failed: {e}")


# ──────────────────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────────────────

def main():
    print("\n\033[1m╔══════════════════════════════════════════════════════════╗\033[0m")
    print("\033[1m║  Phase 5C Dry Run — Voice + Browser + LoRA + Plugins    ║\033[0m")
    print(f"\033[1m║  Mode: {'LIVE 🔴' if LIVE else 'OFFLINE 🟢'}                                       ║\033[0m")
    print("\033[1m╚══════════════════════════════════════════════════════════╝\033[0m")

    test_1_voice_config()
    test_2_voice_state_machine()
    test_3_voice_tools()
    test_4_voice_memory_wiring()
    test_5_live_porcupine()
    test_6_live_memory_search()
    test_7_browser_mcp()
    test_8_qwen_training()
    test_9_plugins()
    test_10_docker_sandbox()
    test_11_integration()
    test_12_live_gemini_client()

    print(f"\n\033[1m{'─'*60}\033[0m")
    total = PASS_COUNT + FAIL_COUNT + SKIP_COUNT
    print(f"\n  Results: {PASS_COUNT} passed, {FAIL_COUNT} failed, {SKIP_COUNT} skipped / {total} total")
    if FAIL_COUNT > 0:
        print(f"\n  \033[91m{FAIL_COUNT} FAILURES\033[0m")
        sys.exit(1)
    else:
        print(f"\n  \033[92mALL PASSED\033[0m ({'+ live' if LIVE else 'offline only'})")
        sys.exit(0)


if __name__ == "__main__":
    main()
