import unittest
from pathlib import Path
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from evolution.skills import SelfImprovementEngine


class TestSelfImproveBenchmark(unittest.TestCase):
    def setUp(self):
        self.engine = SelfImprovementEngine()

    def test_split_holdout_rows_is_deterministic(self):
        rows = [
            {"text": f"message {i}", "entities": [f"entity-{i}"]}
            for i in range(10)
        ]
        train_a, eval_a = self.engine._split_holdout_rows(rows, eval_ratio=0.2, seed=1337)
        train_b, eval_b = self.engine._split_holdout_rows(rows, eval_ratio=0.2, seed=1337)

        self.assertEqual([r["text"] for r in train_a], [r["text"] for r in train_b])
        self.assertEqual([r["text"] for r in eval_a], [r["text"] for r in eval_b])
        self.assertEqual(len(eval_a), 2)
        self.assertEqual(len(train_a), 8)

    def test_compute_prf_metrics_math(self):
        metrics = self.engine._compute_prf_metrics(tp=2, fp=1, fn=1)
        self.assertEqual(metrics["precision"], 0.6667)
        self.assertEqual(metrics["recall"], 0.6667)
        self.assertEqual(metrics["f1"], 0.6667)

    def test_benchmark_returns_failure_metadata_when_candidate_eval_fails(self):
        rows = [
            {"text": f"message {i}", "entities": [{"text": f"entity-{i}", "label": "Person"}]}
            for i in range(5)
        ]
        base_eval = {
            "ok": True,
            "error": "",
            "metrics": {"precision": 0.8, "recall": 0.7, "f1": 0.75},
            "counts": {"tp": 3, "fp": 1, "fn": 1},
            "rows_total": 1,
            "rows_evaluated": 1,
            "rows_failed": 0,
            "latency_ms_avg": 12.5,
            "failure_samples": [],
        }
        candidate_eval = {
            "ok": False,
            "error": "all_inference_failed",
            "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "counts": {"tp": 0, "fp": 0, "fn": 0},
            "rows_total": 1,
            "rows_evaluated": 0,
            "rows_failed": 1,
            "latency_ms_avg": 0.0,
            "failure_samples": [{"row_index": 0, "error": "boom"}],
        }

        with patch.object(
            self.engine,
            "_evaluate_model_on_rows",
            side_effect=[base_eval, candidate_eval],
        ):
            result = self.engine._benchmark_finetune_candidate(
                rows,
                candidate_model_ref="candidate-model-ref",
                train_count=4,
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["base_score"], 0.75)
        self.assertEqual(result["candidate_score"], 0.0)
        self.assertEqual(result["improvement"], 0.0)
        self.assertEqual(result["failure"]["reason"], "model_evaluation_failed")
        self.assertEqual(result["failure"]["details"][0]["model"], "candidate")
        self.assertEqual(result["failure"]["details"][0]["error"], "all_inference_failed")

    def test_select_gliner_training_strategy_uses_lora_when_examples_below_full_threshold(self):
        self.engine._state["gliner_benchmark_history"] = []
        total_examples = max(1, int(config.GLINER_FULL_FINETUNE_MIN_EXAMPLES) - 1)
        decision = self.engine._select_gliner_training_strategy(total_examples)
        self.assertEqual(decision["mode"], "lora")
        self.assertEqual(decision["reason"], "insufficient_examples_for_full_finetune")

    def test_select_gliner_training_strategy_switches_to_full_when_lora_plateaus(self):
        window = max(1, int(config.GLINER_LORA_PLATEAU_WINDOW))
        epsilon = float(config.GLINER_LORA_PLATEAU_EPSILON)
        self.engine._state["gliner_benchmark_history"] = [
            {
                "strategy": "lora",
                "benchmark_ok": True,
                "improvement": max(0.0, epsilon * 0.5),
            }
            for _ in range(window)
        ]
        total_examples = int(config.GLINER_FULL_FINETUNE_MIN_EXAMPLES)
        decision = self.engine._select_gliner_training_strategy(total_examples)
        self.assertEqual(decision["mode"], "full")
        self.assertEqual(decision["reason"], "lora_plateau_detected")

    def test_select_gliner_training_strategy_keeps_lora_when_recent_improvements_are_strong(self):
        window = max(1, int(config.GLINER_LORA_PLATEAU_WINDOW))
        epsilon = float(config.GLINER_LORA_PLATEAU_EPSILON)
        self.engine._state["gliner_benchmark_history"] = [
            {
                "strategy": "lora",
                "benchmark_ok": True,
                "improvement": epsilon + 0.02,
            }
            for _ in range(window)
        ]
        total_examples = int(config.GLINER_FULL_FINETUNE_MIN_EXAMPLES) + 100
        decision = self.engine._select_gliner_training_strategy(total_examples)
        self.assertEqual(decision["mode"], "lora")
        self.assertEqual(decision["reason"], "lora_still_improving")


if __name__ == "__main__":
    unittest.main()
