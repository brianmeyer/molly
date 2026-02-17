"""Tests for Phase 5A orchestrator + workers.

Unit tests (mocked): Always run.
Live smoke tests: Only when MOONSHOT_API_KEY or GEMINI_API_KEY present.
"""
from __future__ import annotations

import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import config
from orchestrator import (
    TriageResult,
    Subtask,
    _extract_json,
    _hardcoded_classification,
    _parse_triage_response,
    classify_message,
    get_orchestrator_stats,
)
from workers import (
    WorkerResult,
    WORKER_PROFILES,
    _CircuitState,
    aggregate_results,
)


class TestExtractJson(unittest.TestCase):
    """Test JSON extraction from various LLM response formats."""

    def test_clean_json(self):
        data = _extract_json('{"type": "direct", "confidence": 0.9}')
        self.assertIsNotNone(data)
        self.assertEqual(data["type"], "direct")

    def test_markdown_fenced(self):
        raw = '```json\n{"type": "simple", "confidence": 0.8}\n```'
        data = _extract_json(raw)
        self.assertIsNotNone(data)
        self.assertEqual(data["type"], "simple")

    def test_leading_text(self):
        raw = 'Here is the result:\n{"type": "complex", "confidence": 0.7}'
        data = _extract_json(raw)
        self.assertIsNotNone(data)
        self.assertEqual(data["type"], "complex")

    def test_empty_string(self):
        self.assertIsNone(_extract_json(""))
        self.assertIsNone(_extract_json(None))

    def test_no_json(self):
        self.assertIsNone(_extract_json("This is plain text without JSON"))

    def test_nested_json(self):
        raw = '{"type": "complex", "subtasks": [{"id": "t1", "profile": "calendar"}]}'
        data = _extract_json(raw)
        self.assertIsNotNone(data)
        self.assertEqual(len(data["subtasks"]), 1)

    def test_stray_closing_brace_before_json(self):
        """Regression: stray } before valid JSON should not prevent parsing."""
        raw = 'foo } {"type":"simple","confidence":0.8}'
        data = _extract_json(raw)
        self.assertIsNotNone(data)
        self.assertEqual(data["type"], "simple")

    def test_multiple_stray_braces(self):
        raw = '}} text } {"type": "direct"}'
        data = _extract_json(raw)
        self.assertIsNotNone(data)
        self.assertEqual(data["type"], "direct")


class TestParseTriageResponse(unittest.TestCase):
    """Test triage response parsing into TriageResult."""

    def test_direct_classification(self):
        raw = '{"type": "direct", "confidence": 0.95, "subtasks": []}'
        result = _parse_triage_response(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.classification, "direct")
        self.assertAlmostEqual(result.confidence, 0.95)
        self.assertEqual(len(result.subtasks), 0)

    def test_simple_with_subtask(self):
        raw = json.dumps({
            "type": "simple",
            "confidence": 0.85,
            "subtasks": [
                {"id": "t1", "profile": "calendar", "description": "Check calendar"}
            ],
        })
        result = _parse_triage_response(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.classification, "simple")
        self.assertEqual(len(result.subtasks), 1)
        self.assertEqual(result.subtasks[0].profile, "calendar")

    def test_complex_with_dependencies(self):
        raw = json.dumps({
            "type": "complex",
            "confidence": 0.7,
            "subtasks": [
                {"id": "t1", "profile": "research", "description": "Find info"},
                {"id": "t2", "profile": "email", "description": "Draft email", "depends_on": ["t1"]},
            ],
        })
        result = _parse_triage_response(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.classification, "complex")
        self.assertEqual(len(result.subtasks), 2)
        self.assertEqual(result.subtasks[1].depends_on, ["t1"])

    def test_depends_on_string_normalized_to_list(self):
        """Regression: depends_on as string 't1' should become ['t1'] not ['t','1']."""
        raw = json.dumps({
            "type": "complex",
            "confidence": 0.8,
            "subtasks": [
                {"id": "t1", "profile": "research", "description": "Research"},
                {"id": "t2", "profile": "email", "description": "Email", "depends_on": "t1"},
            ],
        })
        result = _parse_triage_response(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.subtasks[1].depends_on, ["t1"])

    def test_unknown_type_normalizes(self):
        raw = '{"type": "unknown_type", "confidence": 0.5}'
        result = _parse_triage_response(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.classification, "simple")

    def test_confidence_clamped(self):
        raw = '{"type": "direct", "confidence": 1.5}'
        result = _parse_triage_response(raw)
        self.assertAlmostEqual(result.confidence, 1.0)

        raw = '{"type": "direct", "confidence": -0.5}'
        result = _parse_triage_response(raw)
        self.assertAlmostEqual(result.confidence, 0.0)

    def test_max_5_subtasks(self):
        raw = json.dumps({
            "type": "complex",
            "confidence": 0.8,
            "subtasks": [
                {"id": f"t{i}", "profile": "general", "description": f"task {i}"}
                for i in range(10)
            ],
        })
        result = _parse_triage_response(raw)
        self.assertEqual(len(result.subtasks), 5)


class TestHardcodedClassification(unittest.TestCase):
    """Test regex-based fallback classification."""

    def test_greeting(self):
        result = _hardcoded_classification("hello")
        self.assertEqual(result.classification, "direct")
        self.assertEqual(result.model_used, "hardcoded")

    def test_short_message(self):
        result = _hardcoded_classification("ok")
        self.assertEqual(result.classification, "direct")

    def test_calendar_keyword(self):
        result = _hardcoded_classification("What's on my calendar today?")
        self.assertEqual(result.classification, "simple")
        self.assertEqual(result.subtasks[0].profile, "calendar")

    def test_email_keyword(self):
        result = _hardcoded_classification("Send an email to John")
        self.assertEqual(result.classification, "simple")
        self.assertEqual(result.subtasks[0].profile, "email")

    def test_multi_domain(self):
        result = _hardcoded_classification("Check my calendar and send an email about it")
        self.assertEqual(result.classification, "complex")
        self.assertGreater(len(result.subtasks), 1)

    def test_no_keywords_general(self):
        result = _hardcoded_classification("What's the meaning of life and how does it relate to pancakes?")
        self.assertEqual(result.classification, "simple")
        self.assertEqual(result.subtasks[0].profile, "general")


class TestClassifyMessageDisabled(unittest.TestCase):
    """Test classify_message with orchestrator disabled."""

    def test_disabled_returns_general(self):
        with patch.object(config, "ORCHESTRATOR_ENABLED", False):
            result = asyncio.run(
                classify_message("anything")
            )
            self.assertEqual(result.classification, "simple")
            self.assertEqual(result.model_used, "disabled")
            self.assertEqual(result.fallback_reason, "orchestrator disabled")
            self.assertEqual(len(result.subtasks), 1)
            self.assertEqual(result.subtasks[0].profile, "general")


class TestClassifyMessageFallback(unittest.TestCase):
    """Test classify_message fallback chain."""

    def test_all_models_fail_uses_hardcoded(self):
        """When all API models fail, falls back to hardcoded regex."""
        with patch.object(config, "ORCHESTRATOR_ENABLED", True), \
             patch.object(config, "MOONSHOT_API_KEY", ""), \
             patch.object(config, "GEMINI_API_KEY", ""):
            result = asyncio.run(
                classify_message("Check my calendar")
            )
            # Should fall through to hardcoded since both API keys are empty
            self.assertEqual(result.model_used, "hardcoded")
            self.assertEqual(result.classification, "simple")
            self.assertEqual(result.subtasks[0].profile, "calendar")


class TestCircuitBreaker(unittest.TestCase):
    """Test worker circuit breaker behavior."""

    def test_starts_closed(self):
        cb = _CircuitState()
        self.assertFalse(cb.is_open())

    def test_opens_after_3_failures(self):
        cb = _CircuitState()
        cb.record_failure()
        cb.record_failure()
        self.assertFalse(cb.is_open())
        cb.record_failure()
        self.assertTrue(cb.is_open())

    def test_success_resets(self):
        cb = _CircuitState()
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        self.assertTrue(cb.is_open())
        cb.record_success()
        self.assertFalse(cb.is_open())

    def test_auto_recover_after_timeout(self):
        cb = _CircuitState()
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        self.assertTrue(cb.is_open())
        # Simulate 5+ minutes passing
        cb.tripped_at = time.monotonic() - 301
        self.assertFalse(cb.is_open())


class TestAggregateResults(unittest.TestCase):
    """Test result aggregation."""

    def test_single_success(self):
        triage = TriageResult(classification="simple", confidence=0.9)
        results = [WorkerResult(subtask_id="t1", profile="calendar", text="You have a meeting at 3pm.", success=True)]
        output = aggregate_results(triage, results)
        self.assertEqual(output, "You have a meeting at 3pm.")

    def test_all_failed(self):
        triage = TriageResult(classification="complex", confidence=0.7)
        results = [
            WorkerResult(subtask_id="t1", profile="calendar", text="", success=False, error="timeout"),
            WorkerResult(subtask_id="t2", profile="email", text="", success=False, error="API error"),
        ]
        output = aggregate_results(triage, results)
        self.assertIn("issues", output.lower())
        self.assertIn("calendar", output)

    def test_partial_success(self):
        triage = TriageResult(classification="complex", confidence=0.7)
        results = [
            WorkerResult(subtask_id="t1", profile="calendar", text="Meeting at 3pm.", success=True),
            WorkerResult(subtask_id="t2", profile="email", text="", success=False, error="timeout"),
        ]
        output = aggregate_results(triage, results)
        self.assertIn("Meeting at 3pm", output)
        self.assertIn("email", output.lower())

    def test_empty_results(self):
        triage = TriageResult(classification="simple", confidence=0.9)
        output = aggregate_results(triage, [])
        self.assertIn("wasn't able", output.lower())

    def test_multiple_successes(self):
        triage = TriageResult(classification="complex", confidence=0.8)
        results = [
            WorkerResult(subtask_id="t1", profile="calendar", text="No meetings today.", success=True),
            WorkerResult(subtask_id="t2", profile="email", text="Email sent to John.", success=True),
        ]
        output = aggregate_results(triage, results)
        self.assertIn("No meetings", output)
        self.assertIn("Email sent", output)


class TestWorkerProfiles(unittest.TestCase):
    """Validate worker profile configuration."""

    def test_all_profiles_have_required_keys(self):
        required = {"model_key", "mcp_servers", "timeout", "prompt"}
        for name, profile in WORKER_PROFILES.items():
            with self.subTest(profile=name):
                self.assertTrue(
                    required.issubset(profile.keys()),
                    f"Profile '{name}' missing: {required - set(profile.keys())}",
                )

    def test_model_keys_exist_in_config(self):
        for name, profile in WORKER_PROFILES.items():
            with self.subTest(profile=name):
                model_key = profile["model_key"]
                self.assertTrue(
                    hasattr(config, model_key),
                    f"config.{model_key} not found (used by profile '{name}')",
                )

    def test_timeout_keys_exist_in_config(self):
        for name, profile in WORKER_PROFILES.items():
            with self.subTest(profile=name):
                timeout_key = profile["timeout"]
                self.assertTrue(
                    hasattr(config, timeout_key),
                    f"config.{timeout_key} not found (used by profile '{name}')",
                )

    def test_known_profiles(self):
        expected = {
            "calendar", "email", "contacts", "tasks", "research",
            "writer", "files", "imessage", "browser", "general",
        }
        self.assertEqual(set(WORKER_PROFILES.keys()), expected)


class TestOrchestratorStats(unittest.TestCase):
    """Test orchestrator stats retrieval."""

    def test_stats_returns_dict(self):
        stats = get_orchestrator_stats(hours=1)
        self.assertIsInstance(stats, dict)
        self.assertIn("enabled", stats)


# ---------------------------------------------------------------------------
# Live API smoke tests — skipped if API keys not present
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    config.MOONSHOT_API_KEY,
    "MOONSHOT_API_KEY not set — skipping live Kimi test",
)
class TestLiveKimi(unittest.TestCase):
    """Live Kimi K2.5 API smoke test."""

    def test_kimi_ping(self):
        """Verify Kimi K2.5 responds to a simple ping."""
        import httpx

        async def _test():
            body = {
                "model": getattr(config, "KIMI_TRIAGE_MODEL", "kimi-k2.5"),
                "messages": [
                    {"role": "system", "content": "Respond with: {\"status\": \"ok\"}"},
                    {"role": "user", "content": "ping"},
                ],
                "thinking": {"type": "disabled"},
            }
            headers = {
                "Authorization": f"Bearer {config.MOONSHOT_API_KEY}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{config.MOONSHOT_BASE_URL}/chat/completions",
                    headers=headers,
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            self.assertTrue(len(content) > 0, "Kimi returned empty response")

        asyncio.run(_test())

    def test_kimi_json_classification(self):
        """Verify Kimi K2.5 can return structured JSON classification."""
        import httpx

        async def _test():
            from orchestrator import ORCHESTRATOR_PROMPT, _parse_triage_response

            body = {
                "model": getattr(config, "KIMI_TRIAGE_MODEL", "kimi-k2.5"),
                "messages": [
                    {"role": "system", "content": ORCHESTRATOR_PROMPT},
                    {"role": "user", "content": "What's on my calendar today?"},
                ],
                "thinking": {"type": "disabled"},
            }
            headers = {
                "Authorization": f"Bearer {config.MOONSHOT_API_KEY}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{config.MOONSHOT_BASE_URL}/chat/completions",
                    headers=headers,
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            result = _parse_triage_response(content)
            self.assertIsNotNone(result, f"Failed to parse Kimi response: {content[:200]}")
            self.assertIn(result.classification, ("direct", "simple", "complex"))

        asyncio.run(_test())


@unittest.skipUnless(
    config.GEMINI_API_KEY,
    "GEMINI_API_KEY not set — skipping live Gemini test",
)
class TestLiveGemini(unittest.TestCase):
    """Live Gemini API smoke test."""

    def test_gemini_ping(self):
        """Verify Gemini Flash-Lite responds."""
        import httpx

        async def _test():
            model = getattr(config, "GEMINI_TRIAGE_FALLBACK", "gemini-2.5-flash-lite")
            url = f"{config.GEMINI_BASE_URL}/models/{model}:generateContent?key={config.GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": "Respond with: ok"}]}],
            }
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = parts[0].get("text", "") if parts else ""
            self.assertTrue(len(text) > 0, "Gemini returned empty response")

        asyncio.run(_test())


if __name__ == "__main__":
    unittest.main()
