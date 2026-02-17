"""Tests for Phase 5C.3 Qwen3 LoRA Training.

Tests cover:
  - Training service initialization
  - Prompt formatting
  - Classification parsing
  - Data splitting
  - Text hashing / dedup
  - Config values
"""
from __future__ import annotations

import unittest

import config


class TestQwenConfig(unittest.TestCase):
    """Test Qwen LoRA config values."""

    def test_qwen_lora_disabled_by_default(self):
        self.assertFalse(config.QWEN_LORA_ENABLED)

    def test_min_examples(self):
        self.assertEqual(config.QWEN_LORA_MIN_EXAMPLES, 500)


class TestQwenPromptFormatting(unittest.TestCase):
    """Test triage prompt formatting."""

    def test_format_triage_prompt(self):
        from evolution.qwen_training import QwenTrainingService
        prompt = QwenTrainingService._format_triage_prompt("What's on my calendar?")
        self.assertIn("Classify this message", prompt)
        self.assertIn("What's on my calendar?", prompt)
        self.assertIn("direct", prompt)
        self.assertIn("simple", prompt)
        self.assertIn("complex", prompt)

    def test_format_triage_response(self):
        from evolution.qwen_training import QwenTrainingService
        response = QwenTrainingService._format_triage_response(
            "simple",
            [{"profile": "calendar"}, {"profile": "email"}],
        )
        self.assertIn("simple", response)
        self.assertIn("calendar", response)
        self.assertIn("email", response)

    def test_format_triage_response_empty_subtasks(self):
        from evolution.qwen_training import QwenTrainingService
        response = QwenTrainingService._format_triage_response("direct", [])
        self.assertIn("direct", response)
        self.assertIn("general", response)


class TestQwenClassificationParsing(unittest.TestCase):
    """Test parsing classification from model output."""

    def test_parse_direct(self):
        from evolution.qwen_training import QwenTrainingService
        self.assertEqual(QwenTrainingService._parse_classification("direct"), "direct")

    def test_parse_simple(self):
        from evolution.qwen_training import QwenTrainingService
        self.assertEqual(QwenTrainingService._parse_classification("simple"), "simple")

    def test_parse_complex(self):
        from evolution.qwen_training import QwenTrainingService
        self.assertEqual(QwenTrainingService._parse_classification("This is complex task"), "complex")

    def test_parse_urgent_email(self):
        from evolution.qwen_training import QwenTrainingService
        self.assertEqual(QwenTrainingService._parse_classification("urgent email"), "urgent")

    def test_parse_relevant(self):
        from evolution.qwen_training import QwenTrainingService
        self.assertEqual(QwenTrainingService._parse_classification("relevant"), "relevant")

    def test_parse_background(self):
        from evolution.qwen_training import QwenTrainingService
        self.assertEqual(QwenTrainingService._parse_classification("background noise"), "background")

    def test_parse_fallback(self):
        from evolution.qwen_training import QwenTrainingService
        self.assertEqual(QwenTrainingService._parse_classification("unknown gibberish"), "direct")


class TestQwenDataSplitting(unittest.TestCase):
    """Test train/eval data splitting."""

    def test_split_basic(self):
        from evolution.qwen_training import QwenTrainingService

        # Use a minimal mock context
        class MockCtx:
            state = {}
            def save_state(self): pass

        svc = QwenTrainingService(MockCtx(), None, None)
        rows = [{"text": f"msg {i}", "classification": "direct"} for i in range(100)]
        train, eval_set = svc.split_data(rows)
        self.assertEqual(len(train) + len(eval_set), 100)
        self.assertGreater(len(train), len(eval_set))
        # Default 80/20 split
        self.assertEqual(len(eval_set), 20)

    def test_split_empty(self):
        from evolution.qwen_training import QwenTrainingService

        class MockCtx:
            state = {}
            def save_state(self): pass

        svc = QwenTrainingService(MockCtx(), None, None)
        train, eval_set = svc.split_data([])
        self.assertEqual(train, [])
        self.assertEqual(eval_set, [])

    def test_split_deterministic(self):
        from evolution.qwen_training import QwenTrainingService

        class MockCtx:
            state = {}
            def save_state(self): pass

        svc = QwenTrainingService(MockCtx(), None, None)
        rows = [{"text": f"msg {i}"} for i in range(50)]
        train1, eval1 = svc.split_data(rows, seed=42)
        train2, eval2 = svc.split_data(rows, seed=42)
        self.assertEqual(train1, train2)
        self.assertEqual(eval1, eval2)


class TestQwenTextHashing(unittest.TestCase):
    """Test text hashing for dedup."""

    def test_hash_text(self):
        from evolution.qwen_training import QwenTrainingService
        h1 = QwenTrainingService._hash_text("Hello World")
        h2 = QwenTrainingService._hash_text("hello world")
        self.assertEqual(h1, h2)  # Case-insensitive

    def test_hash_different_texts(self):
        from evolution.qwen_training import QwenTrainingService
        h1 = QwenTrainingService._hash_text("Hello")
        h2 = QwenTrainingService._hash_text("World")
        self.assertNotEqual(h1, h2)


class TestQwenConstants(unittest.TestCase):
    """Test module-level constants."""

    def test_constants(self):
        from evolution.qwen_training import (
            QWEN_LORA_MIN_EXAMPLES,
            QWEN_LORA_COOLDOWN_DAYS,
            QWEN_LORA_EVAL_RATIO,
            QWEN_LORA_SEED,
            QWEN_LORA_AB_EVAL_SIZE,
        )
        self.assertEqual(QWEN_LORA_MIN_EXAMPLES, 500)
        self.assertEqual(QWEN_LORA_COOLDOWN_DAYS, 7)
        self.assertAlmostEqual(QWEN_LORA_EVAL_RATIO, 0.2)
        self.assertEqual(QWEN_LORA_SEED, 42)
        self.assertEqual(QWEN_LORA_AB_EVAL_SIZE, 100)


if __name__ == "__main__":
    unittest.main()
