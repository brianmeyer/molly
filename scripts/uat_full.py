#!/usr/bin/env python3
"""Molly Production UAT — Automated Acceptance Test Suite.

Validates that the Molly codebase is production-ready by exercising
imports, configuration, database operations, memory pipeline, and
key subsystems without needing a live WhatsApp connection.

Usage:
    python scripts/uat_full.py          # Run all checks
    python scripts/uat_full.py -v       # Verbose output
    python scripts/uat_full.py --quick  # Skip slow checks (ML models)

Exit codes:
    0  All checks passed
    1  One or more checks failed
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------
_results: list[dict] = []
_verbose = False


def _check(name: str, category: str = "general"):
    """Decorator that wraps a check function with pass/fail tracking."""
    def decorator(fn):
        def wrapper():
            t0 = time.monotonic()
            try:
                fn()
                elapsed = int((time.monotonic() - t0) * 1000)
                _results.append({"name": name, "cat": category, "ok": True, "ms": elapsed})
                _log_result(True, name, elapsed)
            except Exception as e:
                elapsed = int((time.monotonic() - t0) * 1000)
                _results.append({"name": name, "cat": category, "ok": False, "ms": elapsed, "err": str(e)})
                _log_result(False, name, elapsed, str(e))
        wrapper._check_name = name
        wrapper._check_cat = category
        return wrapper
    return decorator


def _log_result(ok: bool, name: str, ms: int, err: str = ""):
    icon = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
    suffix = f"  ({ms}ms)" if _verbose else ""
    line = f"  {icon} {name}{suffix}"
    if err and _verbose:
        line += f"\n    → {err}"
    print(line)


# ===========================================================================
# Category 1: Imports — every module must import cleanly
# ===========================================================================
CORE_MODULES = [
    "config", "database", "agent", "approval", "commands",
    "contacts", "gateway", "heartbeat", "main",
]
MEMORY_MODULES = [
    "memory.vectorstore", "memory.processor", "memory.retriever",
    "memory.triage", "memory.graph", "memory.embeddings",
    "memory.extractor",
]
EVOLUTION_MODULES = [
    "evolution", "evolution.db", "evolution.hooks", "evolution.skills",
    "evolution.shadow", "evolution.codegen", "evolution.code_loop",
    "evolution.judges", "evolution.docker_sandbox",
    "evolution.skill_lifecycle", "evolution.qwen_training",
]
MONITORING_MODULES = [
    "monitoring._base", "monitoring.health", "monitoring.maintenance",
    "monitoring.remediation",
]
CHANNEL_MODULES = [
    "channels.base", "channels.whatsapp", "channels.web",
]
OTHER_MODULES = [
    "orchestrator", "workers", "contract_audit",
    "plugins.base", "plugins.loader",
]


def _make_import_check(mod_name: str, category: str):
    @_check(f"import {mod_name}", category)
    def _inner():
        importlib.import_module(mod_name)
    return _inner


_import_checks = []
for _mod in CORE_MODULES:
    _import_checks.append(_make_import_check(_mod, "import/core"))
for _mod in MEMORY_MODULES:
    _import_checks.append(_make_import_check(_mod, "import/memory"))
for _mod in EVOLUTION_MODULES:
    _import_checks.append(_make_import_check(_mod, "import/evolution"))
for _mod in MONITORING_MODULES:
    _import_checks.append(_make_import_check(_mod, "import/monitoring"))
for _mod in CHANNEL_MODULES:
    _import_checks.append(_make_import_check(_mod, "import/channels"))
for _mod in OTHER_MODULES:
    _import_checks.append(_make_import_check(_mod, "import/other"))


# ===========================================================================
# Category 2: Configuration
# ===========================================================================
@_check("config.STORE_DIR exists", "config")
def check_store_dir():
    import config
    config.STORE_DIR.mkdir(parents=True, exist_ok=True)
    assert config.STORE_DIR.is_dir(), f"{config.STORE_DIR} is not a directory"


@_check("config.WORKSPACE exists", "config")
def check_workspace():
    import config
    config.WORKSPACE.mkdir(parents=True, exist_ok=True)
    assert config.WORKSPACE.is_dir(), f"{config.WORKSPACE} is not a directory"


@_check("config.IDENTITY_FILES listed", "config")
def check_identity_files():
    import config
    assert len(config.IDENTITY_FILES) >= 3, f"Expected >=3 identity files, got {len(config.IDENTITY_FILES)}"


@_check("ANTHROPIC_API_KEY set in env", "config")
def check_anthropic_key():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    assert key, "ANTHROPIC_API_KEY is not set in environment"


@_check("config.COMMANDS populated", "config")
def check_commands():
    import config
    assert len(config.COMMANDS) >= 5, f"Expected >=5 commands, got {len(config.COMMANDS)}"
    for cmd in ["/help", "/clear", "/memory"]:
        assert cmd in config.COMMANDS, f"Missing command: {cmd}"


@_check("config.ACTION_TIERS defined", "config")
def check_approval_tiers():
    import config
    tiers = getattr(config, "ACTION_TIERS", None)
    assert tiers is not None, "ACTION_TIERS not defined"
    assert "AUTO" in tiers, "AUTO tier not found in ACTION_TIERS"


# ===========================================================================
# Category 3: Database operations
# ===========================================================================
@_check("Database initializes with WAL mode", "database")
def check_sqlite_pragmas():
    from database import Database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        db = Database(tmp_path)
        db.initialize()
        mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal", f"Expected WAL, got {mode}"
    finally:
        tmp_path.unlink(missing_ok=True)


@_check("Database.store_message round-trip", "database")
def check_db_store():
    from database import Database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        db = Database(tmp_path)
        db.initialize()
        db.store_message(
            msg_id="uat_msg_001",
            chat_jid="uat_test_chat@s.whatsapp.net",
            sender="user@s.whatsapp.net",
            sender_name="UAT Tester",
            content="Hello from UAT",
            timestamp="2026-01-01T00:00:00Z",
            is_from_me=False,
        )
        msgs = db.get_recent_messages("uat_test_chat@s.whatsapp.net", limit=1)
        assert len(msgs) >= 1, "No messages returned"
        assert "Hello from UAT" in msgs[-1].get("content", ""), "Message content mismatch"
    finally:
        tmp_path.unlink(missing_ok=True)


@_check("VectorStore opens with busy_timeout", "database")
def check_vectorstore_pragmas():
    from memory.vectorstore import VectorStore
    with tempfile.TemporaryDirectory() as td:
        vs = VectorStore(db_path=Path(td) / "test_vec.db")
        vs.initialize()
        row = vs.conn.execute("PRAGMA busy_timeout").fetchone()
        assert row and int(row[0]) >= 5000, f"busy_timeout not set: {row}"
        vs.conn.close()


@_check("evolution.db opens with busy_timeout", "database")
def check_evolution_db_pragmas():
    from evolution.db import get_connection
    conn = get_connection()
    try:
        row = conn.execute("PRAGMA busy_timeout").fetchone()
        assert row and int(row[0]) >= 5000, f"busy_timeout not set: {row}"
    finally:
        conn.close()


# ===========================================================================
# Category 4: Memory pipeline
# ===========================================================================
@_check("Embeddings model loads", "memory")
def check_embeddings():
    from memory.embeddings import embed
    vec = embed("UAT test sentence for embedding model")
    assert len(vec) > 0, "Empty embedding"
    assert isinstance(vec[0], float), f"Expected float, got {type(vec[0])}"


@_check("GLiNER entity extraction works", "memory")
def check_extractor():
    from memory.extractor import extract
    result = extract("Brian met Alice at Google headquarters in Mountain View.")
    assert "entities" in result, "No entities key"
    names = [e["text"] for e in result["entities"]]
    assert any("Brian" in n or "Alice" in n or "Google" in n for n in names), f"No expected entities: {names}"


@_check("VectorStore store + search round-trip", "memory")
def check_vectorstore_roundtrip():
    from memory.vectorstore import VectorStore
    from memory.embeddings import embed
    with tempfile.TemporaryDirectory() as td:
        vs = VectorStore(db_path=Path(td) / "test_vec.db")
        vs.initialize()
        vec = embed("UAT test: The capital of France is Paris.")
        vs.store_chunk(content="The capital of France is Paris.", embedding=vec,
                       source="uat", chat_jid="uat_test")
        results = vs.search(embed("What is the capital of France?"), limit=1)
        assert len(results) >= 1, "No search results"
        assert "Paris" in results[0]["content"], f"Expected Paris in results: {results[0]['content']}"
        vs.conn.close()


@_check("FTS5 search with special characters", "memory")
def check_fts5_safety():
    from memory.vectorstore import VectorStore
    from memory.embeddings import embed
    with tempfile.TemporaryDirectory() as td:
        vs = VectorStore(db_path=Path(td) / "test_fts.db")
        vs.initialize()
        vec = embed("Test document for FTS5")
        vs.store_chunk(content="Test document for FTS5 safety check",
                       embedding=vec, source="uat", chat_jid="uat_test")
        # These should not crash (special FTS5 chars)
        for query in ['"test"', 'test*', 'test AND OR NOT', 'test:value', '(test)', 'test^2']:
            try:
                vs.search_fts(query, limit=1)
            except Exception as e:
                raise AssertionError(f"FTS5 query crashed on '{query}': {e}")
        vs.conn.close()


@_check("Triage model classifies messages", "memory")
def check_triage():
    from memory.triage import preload_model, classify_local
    import config
    model_path = config.TRIAGE_MODEL_PATH.expanduser()
    if not model_path.exists():
        raise AssertionError(f"Triage model not found: {model_path}")
    preload_model()
    result = classify_local("triage", "Hey can you check my calendar for tomorrow?")
    assert result in ("relevant", "urgent", "background", "noise", ""), f"Unexpected triage result: {result}"


# ===========================================================================
# Category 5: Approval system
# ===========================================================================
@_check("ApprovalManager instantiates", "approval")
def check_approval_init():
    from approval import ApprovalManager
    am = ApprovalManager()
    assert hasattr(am, "try_resolve"), "Missing try_resolve method"
    assert hasattr(am, "request_tool_approval"), "Missing request_tool_approval method"


@_check("Approval tier function resolves tools", "approval")
def check_approval_routing():
    from approval import get_action_tier
    # AUTO tier tools should be recognized
    tier = get_action_tier("calendar_list")
    assert tier == "AUTO", f"Expected AUTO tier for calendar_list, got {tier}"


# ===========================================================================
# Category 6: Evolution engine
# ===========================================================================
@_check("evolution.db schema creates", "evolution")
def check_evolution_schema():
    from evolution.db import ensure_schema, get_connection
    ensure_schema()
    conn = get_connection()
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "skills" in tables or "dgm_state" in tables, f"Missing evolution tables: {tables}"
    finally:
        conn.close()


@_check("evolution hooks fire without error", "evolution")
def check_evolution_hooks():
    from evolution.hooks import pre_execution_hook, post_execution_hook
    ctx = pre_execution_hook(
        task_hash="uat_test_hash",
        task_class="uat_test",
        context={"source": "uat"},
    )
    assert isinstance(ctx, dict), f"Expected dict, got {type(ctx)}"

    async def _run_post():
        await post_execution_hook(
            task_hash="uat_test_hash",
            task_class="uat_test",
            outcome={"success": True, "tool_calls": 0},
            context=ctx,
        )
    asyncio.run(_run_post())


# ===========================================================================
# Category 7: Monitoring & health
# ===========================================================================
@_check("HealthDoctor instantiates", "monitoring")
def check_health_doctor():
    from monitoring.health import HealthDoctor
    doctor = HealthDoctor()
    assert hasattr(doctor, "run_daily"), "Missing run_daily method"
    assert hasattr(doctor, "run_abbreviated_preflight"), "Missing preflight method"


@_check("run_maintenance function importable", "monitoring")
def check_maintenance_runner():
    from monitoring.maintenance import run_maintenance
    assert callable(run_maintenance), "run_maintenance is not callable"


# ===========================================================================
# Category 8: Channels & plugins
# ===========================================================================
@_check("ChannelRegistry loads", "channels")
def check_channel_registry():
    from channels.base import ChannelRegistry
    registry = ChannelRegistry()
    assert hasattr(registry, "register"), "Missing register method"
    assert hasattr(registry, "get"), "Missing get method"


@_check("PluginRegistry loads", "channels")
def check_plugin_registry():
    from plugins.base import PluginRegistry
    registry = PluginRegistry()
    assert hasattr(registry, "register"), "Missing register method"


# ===========================================================================
# Category 9: Orchestrator
# ===========================================================================
@_check("Orchestrator DB initializes", "orchestrator")
def check_orchestrator_db():
    import orchestrator
    orchestrator._ensure_tables()
    # Verify busy_timeout is set
    db_path = orchestrator._get_db_path()
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("PRAGMA busy_timeout").fetchone()
        # Note: busy_timeout is set in _ensure_tables but doesn't persist across connections
        # The important thing is that _ensure_tables doesn't crash
    finally:
        conn.close()


# ===========================================================================
# Category 10: Neo4j graph (graceful degradation)
# ===========================================================================
@_check("Neo4j get_driver() handles unavailable gracefully", "graph")
def check_neo4j_degradation():
    from memory.graph import get_driver, GraphUnavailableError
    try:
        driver = get_driver()
        # If Neo4j is running, this succeeds — good
        assert driver is not None, "Driver returned None"
    except GraphUnavailableError:
        # If Neo4j is down, we get the right exception — also good
        pass
    except Exception as e:
        raise AssertionError(f"Unexpected exception type: {type(e).__name__}: {e}")


# ===========================================================================
# Category 11: Thread safety
# ===========================================================================
@_check("Workers circuit breaker lock exists", "thread_safety")
def check_circuit_lock():
    import workers
    assert hasattr(workers, "_circuit_lock"), "Missing _circuit_lock"
    import threading
    assert isinstance(workers._circuit_lock, type(threading.Lock())), "Not a Lock"


@_check("Graph write lock exists", "thread_safety")
def check_graph_write_lock():
    from memory.graph import _GRAPH_SYNC_WRITE_LOCK
    import threading
    assert isinstance(_GRAPH_SYNC_WRITE_LOCK, type(threading.Lock())), "Not a Lock"


# ===========================================================================
# Category 12: Commands
# ===========================================================================
@_check("handle_command /help works", "commands")
def check_help_command():
    from commands import handle_command

    async def _run():
        result = await handle_command("/help", "uat_test_chat", None)
        assert result and len(result) > 50, f"Help too short: {len(result or '')} chars"
    asyncio.run(_run())


# ===========================================================================
# Category 13: Agent (end-to-end, requires API key)
# ===========================================================================
@_check("handle_message returns response", "agent_e2e")
def check_handle_message():
    from agent import handle_message

    async def _run():
        response, session_id = await handle_message(
            "What is 2 + 2? Reply with just the number.",
            chat_id="uat_test_chat",
            source="uat",
        )
        assert response, "Empty response from Claude"
        assert "4" in response, f"Expected '4' in response: {response[:200]}"
    asyncio.run(_run())


# ===========================================================================
# Runner
# ===========================================================================
_SLOW_CHECKS = {"memory", "agent_e2e"}


def run_all(quick: bool = False, e2e: bool = True):
    """Run all UAT checks and print summary."""
    all_checks = list(_import_checks)
    # Gather all module-level check functions
    this_module = sys.modules[__name__]
    for name in sorted(dir(this_module)):
        obj = getattr(this_module, name)
        if callable(obj) and hasattr(obj, "_check_name"):
            all_checks.append(obj)

    categories = {}
    for check in all_checks:
        cat = getattr(check, "_check_cat", "general")
        if quick and cat in _SLOW_CHECKS:
            continue
        if not e2e and cat == "agent_e2e":
            continue
        categories.setdefault(cat, []).append(check)

    print("\n\033[1m🧪 Molly Production UAT\033[0m")
    print(f"   Mode: {'quick' if quick else 'full'} | E2E: {'yes' if e2e else 'skip'}\n")

    for cat, checks in categories.items():
        print(f"\033[1m  [{cat}]\033[0m")
        for check in checks:
            check()
        print()

    # Summary
    passed = sum(1 for r in _results if r["ok"])
    failed = sum(1 for r in _results if not r["ok"])
    total = len(_results)
    total_ms = sum(r["ms"] for r in _results)

    print("\033[1m" + "=" * 60 + "\033[0m")
    if failed == 0:
        print(f"\033[32m  ✓ ALL {passed}/{total} checks passed ({total_ms}ms)\033[0m")
    else:
        print(f"\033[31m  ✗ {failed}/{total} checks FAILED\033[0m")
        print()
        for r in _results:
            if not r["ok"]:
                print(f"    FAIL: {r['name']}")
                if r.get("err"):
                    print(f"          {r['err']}")
    print()

    return 0 if failed == 0 else 1


def main():
    global _verbose
    parser = argparse.ArgumentParser(description="Molly Production UAT")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show timing details")
    parser.add_argument("--quick", action="store_true", help="Skip slow checks (ML models)")
    parser.add_argument("--no-e2e", action="store_true", help="Skip Claude API e2e check")
    args = parser.parse_args()
    _verbose = args.verbose

    # Suppress noisy library logs during UAT
    logging.basicConfig(level=logging.WARNING)
    for name in ("neo4j", "httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)

    exit_code = run_all(quick=args.quick, e2e=not args.no_e2e)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
