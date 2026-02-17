#!/usr/bin/env python3
"""Phase 5A concurrency stress test + live model smoke tests.

Tests:
  1. Concurrent SQLite writes to orchestrator.db (5 writers)
  2. Concurrent orchestrator classification (3 parallel)
  3. Circuit breaker behavior (synthetic failures)
  4. Live Kimi K2.5 API call (if MOONSHOT_API_KEY present)
  5. Live Gemini API call (if GEMINI_API_KEY present)
  6. End-to-end orchestrator → classify_message (live, if keys present)

Usage:
  python3 scripts/stress_concurrency.py           # Run all tests
  python3 scripts/stress_concurrency.py --live     # Include live API calls
  python3 scripts/stress_concurrency.py --quick    # Skip slow tests
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def section(msg: str) -> None:
    print(f"\n{BOLD}{'─' * 50}{RESET}")
    print(f"{BOLD}{msg}{RESET}")
    print(f"{BOLD}{'─' * 50}{RESET}")


# ---------------------------------------------------------------------------
# Test 1: Concurrent SQLite writes
# ---------------------------------------------------------------------------

async def test_concurrent_sqlite_writes() -> bool:
    """Hammer orchestrator.db with 5 concurrent writers."""
    from orchestrator import _get_db_path, _ensure_tables

    _ensure_tables()
    db_path = str(_get_db_path())
    errors = []

    async def writer(writer_id: int) -> None:
        for i in range(20):
            try:
                conn = sqlite3.connect(db_path, timeout=10)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """INSERT INTO orchestrator_log
                       (timestamp, model, latency_ms, classification,
                        subtask_count, confidence, fallback_reason,
                        message_preview, raw_response)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (time.time(), f"test-writer-{writer_id}", i * 10,
                     "simple", 1, 0.9, "", f"stress test {writer_id}-{i}", ""),
                )
                conn.commit()
                conn.close()
                await asyncio.sleep(0.01)
            except Exception as exc:
                errors.append(f"Writer {writer_id}, iteration {i}: {exc}")

    tasks = [writer(i) for i in range(5)]
    await asyncio.gather(*tasks)

    if errors:
        for e in errors[:5]:
            fail(e)
        return False

    # Verify writes
    conn = sqlite3.connect(db_path)
    count = conn.execute(
        "SELECT COUNT(*) FROM orchestrator_log WHERE model LIKE 'test-writer-%'"
    ).fetchone()[0]
    conn.close()

    if count >= 100:  # 5 writers × 20 writes
        ok(f"SQLite: {count} concurrent writes, zero errors")
        return True
    else:
        fail(f"SQLite: expected 100 writes, got {count}")
        return False


# ---------------------------------------------------------------------------
# Test 2: Concurrent orchestrator classification
# ---------------------------------------------------------------------------

async def test_concurrent_classification() -> bool:
    """Run 3 classify_message calls in parallel (hardcoded fallback)."""
    from orchestrator import classify_message

    # With ORCHESTRATOR_ENABLED=False, this uses the disabled path
    # Enable temporarily for testing
    original = getattr(config, "ORCHESTRATOR_ENABLED", False)

    messages = [
        "What's on my calendar today?",
        "Send an email to John about the project update and then schedule a meeting for tomorrow",
        "Hey what's up?",
    ]

    try:
        # Test with orchestrator disabled — should return "disabled" path
        config.ORCHESTRATOR_ENABLED = False
        results = await asyncio.gather(
            *[classify_message(m) for m in messages]
        )
        for r in results:
            if r.model_used != "disabled":
                fail(f"Expected 'disabled' model, got '{r.model_used}'")
                return False
        ok(f"Concurrent disabled classification: {len(results)} calls OK")
        return True
    finally:
        config.ORCHESTRATOR_ENABLED = original


# ---------------------------------------------------------------------------
# Test 3: Circuit breaker
# ---------------------------------------------------------------------------

async def test_circuit_breaker() -> bool:
    """Verify circuit breaker trips after 3 failures and recovers."""
    from workers import _get_circuit, _CircuitState

    circuit = _CircuitState()

    # Should start closed
    if circuit.is_open():
        fail("Circuit should start closed")
        return False

    # 2 failures — still closed
    circuit.record_failure()
    circuit.record_failure()
    if circuit.is_open():
        fail("Circuit should still be closed after 2 failures")
        return False

    # 3rd failure — should open
    circuit.record_failure()
    if not circuit.is_open():
        fail("Circuit should be open after 3 failures")
        return False

    # Success resets
    circuit.record_success()
    if circuit.is_open():
        fail("Circuit should close after success")
        return False

    ok("Circuit breaker: trips at 3, resets on success")
    return True


# ---------------------------------------------------------------------------
# Test 4: Live Kimi K2.5 API call
# ---------------------------------------------------------------------------

async def test_live_kimi() -> bool:
    """Make a live Kimi K2.5 API call to verify connectivity and response format."""
    import httpx

    if not config.MOONSHOT_API_KEY:
        warn("MOONSHOT_API_KEY not set — skipping live Kimi test")
        return True

    model = getattr(config, "KIMI_TRIAGE_MODEL", "kimi-k2.5")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Respond with ONLY a JSON object: {\"status\": \"ok\", \"model\": \"your-model-name\"}"},
            {"role": "user", "content": "ping"},
        ],
        "thinking": {"type": "disabled"},
    }
    headers = {
        "Authorization": f"Bearer {config.MOONSHOT_API_KEY}",
        "Content-Type": "application/json",
    }

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{config.MOONSHOT_BASE_URL}/chat/completions",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        elapsed = (time.monotonic() - start) * 1000
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})

        if content:
            ok(f"Kimi K2.5 live: {elapsed:.0f}ms, {usage.get('total_tokens', '?')} tokens")
            ok(f"  Response preview: {content[:100]}")
            return True
        else:
            fail(f"Kimi K2.5 returned empty response after {elapsed:.0f}ms")
            return False

    except httpx.HTTPStatusError as exc:
        fail(f"Kimi K2.5 HTTP {exc.response.status_code}: {exc.response.text[:200]}")
        return False
    except Exception as exc:
        fail(f"Kimi K2.5 error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Test 5: Live Gemini API call
# ---------------------------------------------------------------------------

async def test_live_gemini() -> bool:
    """Make a live Gemini API call to verify connectivity."""
    import httpx

    if not config.GEMINI_API_KEY:
        warn("GEMINI_API_KEY not set — skipping live Gemini test")
        return True

    model = getattr(config, "GEMINI_TRIAGE_FALLBACK", "gemini-2.5-flash-lite")
    url = f"{config.GEMINI_BASE_URL}/models/{model}:generateContent?key={config.GEMINI_API_KEY}"

    payload = {
        "contents": [
            {"parts": [{"text": "Respond with ONLY: {\"status\": \"ok\"}"}]},
        ],
        "generationConfig": {"temperature": 0.1},
    }

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        elapsed = (time.monotonic() - start) * 1000
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = parts[0].get("text", "") if parts else ""

        if text:
            ok(f"Gemini live: {elapsed:.0f}ms, model={model}")
            ok(f"  Response preview: {text[:100]}")
            return True
        else:
            fail(f"Gemini returned empty response after {elapsed:.0f}ms")
            return False

    except Exception as exc:
        fail(f"Gemini error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Test 6: End-to-end orchestrator classify (live)
# ---------------------------------------------------------------------------

async def test_live_orchestrator_classify() -> bool:
    """Run classify_message with live API calls."""
    from orchestrator import classify_message

    if not config.MOONSHOT_API_KEY and not config.GEMINI_API_KEY:
        warn("No API keys — skipping live orchestrator test")
        return True

    original = getattr(config, "ORCHESTRATOR_ENABLED", False)
    config.ORCHESTRATOR_ENABLED = True

    test_messages = [
        ("What's on my calendar tomorrow?", "simple"),
        ("Hey", "direct"),
        ("Email John the meeting notes and then add a reminder for Friday", "complex"),
    ]

    all_passed = True
    try:
        for msg, expected_type in test_messages:
            start = time.monotonic()
            result = await classify_message(msg)
            elapsed = (time.monotonic() - start) * 1000

            if result.model_used in ("disabled", "hardcoded"):
                warn(f"  '{msg[:40]}' → {result.classification} via {result.model_used} ({elapsed:.0f}ms) [no live model]")
            elif result.classification:
                ok(f"  '{msg[:40]}' → {result.classification} (confidence={result.confidence:.2f}) via {result.model_used} ({elapsed:.0f}ms)")
                if result.subtasks:
                    for st in result.subtasks:
                        ok(f"    [{st.id}] {st.profile}: {st.description[:60]}")
            else:
                fail(f"  '{msg[:40]}' → empty classification")
                all_passed = False
    finally:
        config.ORCHESTRATOR_ENABLED = original

    return all_passed


# ---------------------------------------------------------------------------
# Test 7: Orchestrator JSON parsing
# ---------------------------------------------------------------------------

async def test_json_parsing() -> bool:
    """Test the JSON extraction from various LLM response formats."""
    from orchestrator import _extract_json, _parse_triage_response

    test_cases = [
        # Clean JSON
        ('{"type": "direct", "confidence": 0.95, "subtasks": []}', "direct"),
        # Markdown code block
        ('```json\n{"type": "simple", "confidence": 0.8, "subtasks": [{"id": "t1", "profile": "calendar", "description": "check cal"}]}\n```', "simple"),
        # Leading text
        ('Here is the classification:\n{"type": "complex", "confidence": 0.7, "subtasks": []}', "complex"),
        # Thinking mode with reasoning prefix
        ('I need to analyze this message...\n\n{"type": "direct", "confidence": 0.9, "subtasks": []}', "direct"),
    ]

    all_passed = True
    for raw, expected_type in test_cases:
        result = _parse_triage_response(raw)
        if result is None:
            fail(f"Failed to parse: {raw[:50]}")
            all_passed = False
        elif result.classification != expected_type:
            fail(f"Expected {expected_type}, got {result.classification} from: {raw[:50]}")
            all_passed = False

    if all_passed:
        ok(f"JSON parsing: {len(test_cases)} test cases passed")
    return all_passed


# ---------------------------------------------------------------------------
# Test 8: Worker profile validation
# ---------------------------------------------------------------------------

async def test_worker_profiles() -> bool:
    """Validate all worker profiles have required fields."""
    from workers import WORKER_PROFILES

    required_keys = {"model_key", "mcp_servers", "timeout", "prompt"}
    all_passed = True

    for name, profile in WORKER_PROFILES.items():
        missing = required_keys - set(profile.keys())
        if missing:
            fail(f"Profile '{name}' missing keys: {missing}")
            all_passed = False

        # Verify model_key references a real config attr
        model_key = profile.get("model_key", "")
        if not hasattr(config, model_key):
            fail(f"Profile '{name}' references non-existent config.{model_key}")
            all_passed = False

    if all_passed:
        ok(f"Worker profiles: {len(WORKER_PROFILES)} profiles valid")
    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    args = set(sys.argv[1:])
    run_live = "--live" in args or "--all" in args
    quick = "--quick" in args

    print(f"\n{BOLD}Phase 5A Concurrency Stress Test{RESET}")
    print(f"Mode: {'live + stress' if run_live else 'quick' if quick else 'standard'}")

    passed = 0
    failed = 0
    total = 0

    # --- Offline tests (always run) ---
    section("Offline Tests")

    total += 1
    if await test_json_parsing():
        passed += 1
    else:
        failed += 1

    total += 1
    if await test_worker_profiles():
        passed += 1
    else:
        failed += 1

    total += 1
    if await test_circuit_breaker():
        passed += 1
    else:
        failed += 1

    if not quick:
        total += 1
        if await test_concurrent_sqlite_writes():
            passed += 1
        else:
            failed += 1

        total += 1
        if await test_concurrent_classification():
            passed += 1
        else:
            failed += 1

    # --- Live API tests ---
    if run_live:
        section("Live API Smoke Tests")

        total += 1
        if await test_live_kimi():
            passed += 1
        else:
            failed += 1

        total += 1
        if await test_live_gemini():
            passed += 1
        else:
            failed += 1

        section("Live Orchestrator End-to-End")

        total += 1
        if await test_live_orchestrator_classify():
            passed += 1
        else:
            failed += 1

    # --- Summary ---
    section("Results")
    color = GREEN if failed == 0 else RED
    print(f"  {color}{passed}/{total} passed, {failed} failed{RESET}")

    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
