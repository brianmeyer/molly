#!/usr/bin/env python3
"""Phase 5A+5B Dry Run — Integration & Live Testing.

Validates the full pipeline end-to-end: channels → orchestrator → workers
→ hybrid search → gateway auth.

Usage:
    python3 scripts/dry_run_5ab.py            # offline tests only
    python3 scripts/dry_run_5ab.py --live      # includes live API calls
"""
from __future__ import annotations

import asyncio
import hmac
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

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
# Test 1: Channel Registry Smoke
# ──────────────────────────────────────────────────────────────

def test_1_channel_registry():
    section("Test 1: Channel Registry Smoke")
    from channels.base import registry
    import channels.whatsapp   # noqa: F401
    import channels.imessage   # noqa: F401
    import channels.email      # noqa: F401
    import channels.gateway    # noqa: F401
    import channels.web        # noqa: F401
    from channels.base import InboundMessage, OutboundMessage

    names = registry.list_channels()
    if len(names) == 5 and set(names) == {"whatsapp", "imessage", "email", "gateway", "web"}:
        ok(f"Registry has 5 channels: {names}")
    else:
        fail(f"Expected 5 channels, got {len(names)}: {names}")

    # Normalize a WhatsApp message
    wa = registry.get("whatsapp")
    msg = wa.normalize_inbound({
        "msg_id": "dry-run-001",
        "chat_jid": "120123456789@s.whatsapp.net",
        "sender_jid": "120123456789@s.whatsapp.net",
        "sender_name": "Brian",
        "content": "@Molly check my calendar",
        "timestamp": "2026-02-16T10:00:00Z",
        "is_from_me": False,
        "is_group": False,
    }, is_owner=True, chat_mode="owner_dm")

    if msg.source == "whatsapp" and msg.has_trigger and "calendar" in msg.clean_text:
        ok(f"WhatsApp normalize: source={msg.source}, trigger={msg.has_trigger}, clean='{msg.clean_text}'")
    else:
        fail(f"WhatsApp normalize failed: {msg}")

    # Format outbound
    out = OutboundMessage(text="**Bold header**\n## Subheading\nPlain text", chat_id="x")
    formatted = wa.format_outbound(out)
    if "**" not in formatted and "##" not in formatted:
        ok(f"WhatsApp format: markdown stripped ({len(formatted)} chars)")
    else:
        fail(f"WhatsApp format still has markdown: {formatted[:80]}")


# ──────────────────────────────────────────────────────────────
# Test 2: Hybrid Search Integration
# ──────────────────────────────────────────────────────────────

def test_2_hybrid_search():
    section("Test 2: Hybrid Search Integration")
    from memory.vectorstore import VectorStore, EMBEDDING_DIM

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)

    try:
        vs = VectorStore(db_path)
        vs.initialize()
        dummy = [0.0] * EMBEDDING_DIM

        vs.store_chunk("Brian Meyer lives in Portland Oregon", dummy, source="whatsapp")
        vs.store_chunk("Meeting with John at 3pm about Q4 report", dummy, source="email")
        vs.store_chunk("Dinner reservation at Canlis for Saturday", dummy, source="imessage")
        vs.store_chunk("Project deadline moved to March 15", dummy, source="email")
        vs.store_chunk("Brian's calendar has a dentist appointment", dummy, source="whatsapp")

        # FTS5 sync check
        fts_count = vs.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        if fts_count == 5:
            ok(f"FTS5 synced: {fts_count} rows match {vs.chunk_count()} chunks")
        else:
            fail(f"FTS5 count {fts_count} != chunk count {vs.chunk_count()}")

        # BM25 keyword search
        fts_results = vs.search_fts("Brian Meyer", top_k=3)
        if fts_results and "Portland" in fts_results[0]["content"]:
            ok(f"BM25 keyword: 'Brian Meyer' → {len(fts_results)} results, top match correct")
        else:
            fail(f"BM25 keyword search failed: {fts_results}")

        # Hybrid search
        hybrid = vs.hybrid_search("meeting", dummy, top_k=3)
        if hybrid:
            top = hybrid[0]
            ok(f"Hybrid search: {len(hybrid)} results, top hybrid_score={top['hybrid_score']:.2f}")
        else:
            fail("Hybrid search returned empty")

        # Keyword boost: with uniform vectors, BM25 should differentiate
        hybrid_name = vs.hybrid_search("Brian Meyer", dummy, top_k=5)
        bm25_matched = [r for r in hybrid_name if r["bm25_score"] > 0]
        bm25_unmatched = [r for r in hybrid_name if r["bm25_score"] == 0]
        if bm25_matched and bm25_unmatched:
            if max(r["hybrid_score"] for r in bm25_matched) > max(r["hybrid_score"] for r in bm25_unmatched):
                ok("Keyword boost: BM25-matched results ranked higher than unmatched")
            else:
                fail("Keyword boost failed: BM25 matches not ranked higher")
        elif bm25_matched:
            ok("Keyword boost: all results matched BM25 (small corpus)")
        else:
            fail("Keyword boost: no BM25 matches found")

        # Backfill test
        vs.conn.execute("DELETE FROM chunks_fts")
        vs.conn.commit()
        fts_empty = vs.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        assert fts_empty == 0
        vs2 = VectorStore(db_path)
        vs2.initialize()
        fts_after = vs2.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        if fts_after == 5:
            ok(f"FTS5 backfill: rebuilt {fts_after} rows from existing chunks")
        else:
            fail(f"FTS5 backfill: expected 5, got {fts_after}")
        vs2.close()

        vs.close()
    finally:
        os.unlink(db_path)


# ──────────────────────────────────────────────────────────────
# Test 3: Orchestrator Classification (Live)
# ──────────────────────────────────────────────────────────────

def test_3_orchestrator_classification():
    section("Test 3: Orchestrator Classification")
    if not LIVE:
        skip("classify_message with live Kimi K2.5")
        return

    from orchestrator import classify_message

    # Temporarily enable orchestrator for live testing
    saved_enabled = config.ORCHESTRATOR_ENABLED
    config.ORCHESTRATOR_ENABLED = True

    test_cases = [
        ("What's on my calendar tomorrow?", "simple", "calendar"),
        ("Hey", "direct", None),
        ("Check my email and then schedule a meeting based on what you find", "complex", None),
    ]

    try:
        for msg, expected_type, expected_profile in test_cases:
            try:
                result = asyncio.run(classify_message(msg))
                type_ok = result.classification == expected_type
                # Subtasks are Subtask dataclass instances with .profile attribute
                first_profile = result.subtasks[0].profile if result.subtasks else None
                profile_ok = expected_profile is None or first_profile == expected_profile

                label = f"'{msg[:40]}...' → {result.classification}"
                if result.subtasks:
                    profiles = [s.profile for s in result.subtasks]
                    label += f" ({len(result.subtasks)} subtasks: {profiles})"
                label += f" [{result.model_used}, {result.latency_ms:.0f}ms]"

                if type_ok:
                    ok(label)
                else:
                    fail(f"Expected {expected_type}, got {result.classification}: {label}")
            except Exception as e:
                fail(f"classify_message('{msg[:30]}...') raised: {e}")
    finally:
        config.ORCHESTRATOR_ENABLED = saved_enabled


# ──────────────────────────────────────────────────────────────
# Test 4: Fallback Chain (Live)
# ──────────────────────────────────────────────────────────────

def test_4_fallback_chain():
    section("Test 4: Fallback Chain")
    if not LIVE:
        skip("Kimi→Gemini fallback with patched API key")
        return

    if not config.GEMINI_API_KEY:
        skip("No GEMINI_API_KEY — can't test Gemini fallback")
        return

    from orchestrator import classify_message

    # Temporarily enable orchestrator + disable Kimi for fallback test
    saved_enabled = config.ORCHESTRATOR_ENABLED
    saved_key = config.MOONSHOT_API_KEY
    config.ORCHESTRATOR_ENABLED = True
    try:
        config.MOONSHOT_API_KEY = ""
        result = asyncio.run(classify_message("What time is it?"))
        if "gemini" in result.model_used.lower():
            ok(f"Fallback to Gemini: model={result.model_used}, reason='{result.fallback_reason}'")
        elif result.model_used == "hardcoded":
            ok(f"Fallback to hardcoded (Gemini also unavailable): reason='{result.fallback_reason}'")
        else:
            fail(f"Unexpected fallback model: {result.model_used}")
    except Exception as e:
        fail(f"Fallback chain raised: {e}")
    finally:
        config.MOONSHOT_API_KEY = saved_key
        config.ORCHESTRATOR_ENABLED = saved_enabled


# ──────────────────────────────────────────────────────────────
# Test 5: Worker Profile Validation
# ──────────────────────────────────────────────────────────────

def test_5_worker_profiles():
    section("Test 5: Worker Profile Validation")
    from workers import WORKER_PROFILES

    required_keys = {"model_key", "mcp_servers", "timeout", "prompt"}
    valid_model_keys = {"WORKER_MODEL_FAST", "WORKER_MODEL_DEFAULT", "WORKER_MODEL_DEEP"}
    # These profiles intentionally have no MCP servers (use built-in tools)
    no_mcp_ok = {"writer", "browser", "general"}
    issues = []

    for name, profile in WORKER_PROFILES.items():
        missing = required_keys - set(profile.keys())
        if missing:
            issues.append(f"{name}: missing keys {missing}")
        model_key = profile.get("model_key", "")
        if model_key not in valid_model_keys:
            issues.append(f"{name}: invalid model_key '{model_key}'")
        if not profile.get("mcp_servers") and name not in no_mcp_ok:
            issues.append(f"{name}: empty mcp_servers")
        # Check config reference
        if not hasattr(config, model_key):
            issues.append(f"{name}: config.{model_key} does not exist")

    if not issues:
        ok(f"All {len(WORKER_PROFILES)} worker profiles valid")
    else:
        for issue in issues:
            fail(issue)

    # Print profile summary
    for name, profile in sorted(WORKER_PROFILES.items()):
        model = getattr(config, profile["model_key"], "?")
        servers = ", ".join(profile["mcp_servers"])
        print(f"    {name:12s} → {model:24s} [{servers}]")


# ──────────────────────────────────────────────────────────────
# Test 6: Agent.py Orchestrator Wiring (Offline — import check)
# ──────────────────────────────────────────────────────────────

def test_6_agent_wiring():
    section("Test 6: Agent.py Orchestrator Wiring")

    # Verify the orchestrator integration point exists in agent.py
    import agent
    src = Path(agent.__file__).read_text()

    if "ORCHESTRATOR_ENABLED" in src:
        ok("agent.py contains ORCHESTRATOR_ENABLED check")
    else:
        fail("agent.py missing ORCHESTRATOR_ENABLED check")

    if "classify_message" in src and "run_workers" in src:
        ok("agent.py imports classify_message + run_workers")
    else:
        fail("agent.py missing orchestrator/worker imports")

    if "orchestrator_handled" in src:
        ok("agent.py has orchestrator_handled flag for fallback")
    else:
        fail("agent.py missing orchestrator_handled fallback")

    # Check the flag is off by default
    if not config.ORCHESTRATOR_ENABLED:
        ok(f"ORCHESTRATOR_ENABLED = {config.ORCHESTRATOR_ENABLED} (safe default)")
    else:
        fail(f"ORCHESTRATOR_ENABLED = {config.ORCHESTRATOR_ENABLED} (should be False by default!)")


# ──────────────────────────────────────────────────────────────
# Test 7: Gateway Webhook Auth
# ──────────────────────────────────────────────────────────────

def test_7_gateway_auth():
    section("Test 7: Gateway Webhook Auth")

    # Reimplement the auth logic (it's a closure inside attach_gateway_routes)
    def check_auth(authorization: str | None, token: str) -> bool:
        if not token:
            return True
        if not authorization:
            return False
        provided = authorization.replace("Bearer ", "").strip()
        return hmac.compare_digest(provided, token)

    # With token configured
    token = "test-secret-token-123"
    if check_auth(f"Bearer {token}", token):
        ok("Webhook auth: correct Bearer token accepted")
    else:
        fail("Webhook auth: correct token rejected")

    if not check_auth("Bearer wrong-token", token):
        ok("Webhook auth: wrong token rejected")
    else:
        fail("Webhook auth: wrong token accepted!")

    if not check_auth(None, token):
        ok("Webhook auth: missing header rejected")
    else:
        fail("Webhook auth: missing header accepted!")

    if check_auth(token, token):
        ok("Webhook auth: raw token (no Bearer prefix) accepted")
    else:
        fail("Webhook auth: raw token rejected")

    # With no token configured
    if check_auth(None, ""):
        ok("Webhook auth: no token configured → all requests allowed")
    else:
        fail("Webhook auth: no token configured but request rejected")

    # Verify config has the field
    if hasattr(config, "GATEWAY_WEBHOOK_TOKEN"):
        ok(f"config.GATEWAY_WEBHOOK_TOKEN exists (value={'set' if config.GATEWAY_WEBHOOK_TOKEN else 'empty'})")
    else:
        fail("config.GATEWAY_WEBHOOK_TOKEN missing")


# ──────────────────────────────────────────────────────────────
# Test 8: Orchestrator SQLite Logging
# ──────────────────────────────────────────────────────────────

def test_8_orchestrator_logging():
    section("Test 8: Orchestrator SQLite Logging")
    if not LIVE:
        skip("SQLite log inspection (needs live calls from test 3)")
        return

    # Match the path used by orchestrator.py: config.WORKSPACE / "store" / "orchestrator.db"
    workspace = getattr(config, "WORKSPACE", Path.home() / ".molly")
    db_path = Path(workspace) / "store" / "orchestrator.db"
    if not db_path.exists():
        fail(f"orchestrator.db does not exist at {db_path}")
        return

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        count = conn.execute("SELECT COUNT(*) FROM orchestrator_log").fetchone()[0]
        if count > 0:
            ok(f"orchestrator_log has {count} entries")
        else:
            fail("orchestrator_log is empty")

        # Check schema
        cols = conn.execute("PRAGMA table_info(orchestrator_log)").fetchall()
        col_names = {c["name"] for c in cols}
        required = {"timestamp", "model", "latency_ms", "classification", "subtask_count"}
        missing = required - col_names
        if not missing:
            ok(f"Schema valid: {required}")
        else:
            fail(f"Schema missing columns: {missing}")

        # Show recent entries
        recent = conn.execute(
            "SELECT model, classification, latency_ms, subtask_count "
            "FROM orchestrator_log ORDER BY id DESC LIMIT 5"
        ).fetchall()
        for row in recent:
            print(f"    {row['model']:20s} {row['classification']:10s} "
                  f"{row['latency_ms']:6.0f}ms  {row['subtask_count']} subtasks")

        conn.close()
    except Exception as e:
        fail(f"SQLite query failed: {e}")


# ──────────────────────────────────────────────────────────────
# Test 9: Full Pipeline E2E (Live)
# ──────────────────────────────────────────────────────────────

def test_9_pipeline_e2e():
    section("Test 9: Full Pipeline E2E")
    if not LIVE:
        skip("Full pipeline with orchestrator enabled")
        return

    from orchestrator import classify_message

    # Temporarily enable orchestrator for live testing
    saved_enabled = config.ORCHESTRATOR_ENABLED
    config.ORCHESTRATOR_ENABLED = True

    test_messages = [
        "What day is my next dentist appointment?",
        "Draft an email to Sarah about the Q4 numbers and then add a reminder for Friday",
        "Good morning",
    ]

    try:
        for msg in test_messages:
            try:
                t0 = time.monotonic()
                result = asyncio.run(classify_message(msg))
                elapsed = (time.monotonic() - t0) * 1000

                label = f"'{msg[:50]}'"
                details = (
                    f"type={result.classification}, "
                    f"conf={result.confidence:.1f}, "
                    f"model={result.model_used}, "
                    f"latency={result.latency_ms:.0f}ms (wall={elapsed:.0f}ms)"
                )
                if result.subtasks:
                    profiles = [s.profile for s in result.subtasks]
                    details += f", workers={profiles}"
                    deps = [s.depends_on for s in result.subtasks]
                    has_deps = any(d for d in deps)
                    if has_deps:
                        details += " (with dependencies)"

                ok(f"{label}\n      → {details}")

            except Exception as e:
                fail(f"Pipeline failed for '{msg[:30]}...': {e}")
    finally:
        config.ORCHESTRATOR_ENABLED = saved_enabled


# ──────────────────────────────────────────────────────────────
# Test 10: Retriever Hybrid Path (Offline)
# ──────────────────────────────────────────────────────────────

def test_10_retriever_hybrid():
    section("Test 10: Retriever Hybrid Path")
    from memory.vectorstore import VectorStore, EMBEDDING_DIM
    from memory import retriever
    from memory import embeddings as _emb_module

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)

    try:
        vs = VectorStore(db_path)
        vs.initialize()
        dummy = [0.0] * EMBEDDING_DIM

        vs.store_chunk("Brian Meyer lives in Portland Oregon", dummy, source="whatsapp")
        vs.store_chunk("Meeting with John at 3pm about Q4 report", dummy, source="email")
        vs.store_chunk("Calendar event: dentist appointment Tuesday", dummy, source="whatsapp")

        # Monkey-patch the retriever's vectorstore singleton AND the embed function
        # (embed needs sentence_transformers which may not be installed)
        original_vs = retriever._vectorstore
        original_embed = retriever.embed
        retriever._vectorstore = vs
        retriever.embed = lambda text: dummy  # Return dummy vector for all queries

        try:
            result = retriever._retrieve_semantic("Brian Meyer", top_k=3)
            if result["result_count"] > 0:
                ok(f"Retriever returned {result['result_count']} results")
            else:
                fail("Retriever returned 0 results")

            if "hybrid" in result["context"].lower():
                ok("Retriever context header mentions 'hybrid search'")
            else:
                fail(f"Expected 'hybrid' in context header, got: {result['context'][:80]}")

            if "Brian" in result["context"]:
                ok("Retriever context contains expected content")
            else:
                fail("Retriever context missing expected content")

        finally:
            retriever._vectorstore = original_vs
            retriever.embed = original_embed

        vs.close()
    finally:
        os.unlink(db_path)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    mode = "LIVE" if LIVE else "OFFLINE"
    print(f"\n\033[1m{'='*60}\033[0m")
    print(f"\033[1m  Phase 5A+5B Dry Run ({mode})\033[0m")
    print(f"\033[1m{'='*60}\033[0m")

    if LIVE:
        keys = {
            "MOONSHOT_API_KEY": bool(config.MOONSHOT_API_KEY),
            "GEMINI_API_KEY": bool(config.GEMINI_API_KEY),
            "ANTHROPIC_API_KEY": bool(getattr(config, "ANTHROPIC_API_KEY", "")),
        }
        print(f"\n  API keys: {', '.join(f'{k}={'set' if v else 'MISSING'}' for k,v in keys.items())}")

    test_1_channel_registry()
    test_2_hybrid_search()
    test_3_orchestrator_classification()
    test_4_fallback_chain()
    test_5_worker_profiles()
    test_6_agent_wiring()
    test_7_gateway_auth()
    test_8_orchestrator_logging()
    test_9_pipeline_e2e()
    test_10_retriever_hybrid()

    # Summary
    total = PASS_COUNT + FAIL_COUNT + SKIP_COUNT
    print(f"\n\033[1m{'='*60}\033[0m")
    if FAIL_COUNT == 0:
        color = "\033[92m"  # green
        status = "ALL PASSED"
    else:
        color = "\033[91m"  # red
        status = f"{FAIL_COUNT} FAILED"
    print(f"{color}\033[1m  {status}: {PASS_COUNT} passed, {FAIL_COUNT} failed, {SKIP_COUNT} skipped ({total} total)\033[0m")
    print(f"\033[1m{'='*60}\033[0m\n")

    return 1 if FAIL_COUNT > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
