"""Qwen3 LoRA fine-tune pipeline for triage + email classification (Phase 5C.3).

Fine-tunes Qwen3-4B (the local triage/classification fallback model) on
labeled data.  This model serves dual roles:
  1. **Triage classification** (orchestrator fallback): direct/simple/complex
  2. **Email triage**: urgent/relevant/background/noise

The orchestrator chain is:  Kimi K2.5 → Gemini Flash-Lite → **Qwen3-4B LoRA** → hardcoded.

Different from GLiNER LoRA (``evolution/gliner_training.py``) which fine-tunes
entity extraction.  Qwen3 LoRA fine-tunes message **classification**.

Pipeline:
  1. Load labeled examples from ``triage_labels.jsonl`` (minimum 500)
  2. Split: 80% train / 20% held-out eval
  3. LoRA fine-tune Qwen3-4B on Mac Mini M4 (on-device, no cloud)
  4. Benchmark against held-out set
  5. Deploy only if accuracy improves over baseline
  6. A/B eval: 100 held-out examples through base vs LoRA, compare accuracy + latency

Mirrors ``evolution/gliner_training.py`` structure.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import config
import db_pool

log = logging.getLogger(__name__)

# Constants
QWEN_LORA_MIN_EXAMPLES = 500
QWEN_LORA_COOLDOWN_DAYS = 7
QWEN_LORA_EVAL_RATIO = 0.2
QWEN_LORA_SEED = 42
QWEN_LORA_AB_EVAL_SIZE = 100
QWEN_MODEL_NAME = "Qwen/Qwen3-4B"


class QwenTrainingService:
    """Qwen3 LoRA training pipeline for triage + email classification.

    Receives explicit ``InfraService`` and ``OwnerCommsService`` dependencies
    (same pattern as ``GLiNERTrainingService``).
    """

    def __init__(self, ctx, infra, comms):
        from evolution.context import EngineContext
        from evolution.infra import InfraService
        from evolution.owner_comms import OwnerCommsService
        self.ctx: EngineContext = ctx
        self.infra: InfraService = infra
        self.comms: OwnerCommsService = comms

    # -- directory helpers --------------------------------------------------

    def training_dir(self) -> Path:
        return config.WORKSPACE / "memory" / "qwen_training"

    def labels_path(self) -> Path:
        return self.training_dir() / "triage_labels.jsonl"

    def models_dir(self) -> Path:
        return config.WORKSPACE / "models" / "qwen_lora"

    def candidate_dir(self) -> Path:
        return self.models_dir() / "candidate"

    def active_dir(self) -> Path:
        return self.models_dir() / "active"

    # -- should_run check ---------------------------------------------------

    def should_run(self) -> bool:
        """Check if training should run (enough data + cooldown passed)."""
        if not config.QWEN_LORA_ENABLED:
            return False

        total = self.count_examples()
        if total < config.QWEN_LORA_MIN_EXAMPLES:
            log.debug(
                "Qwen LoRA: insufficient examples (%d/%d)",
                total,
                config.QWEN_LORA_MIN_EXAMPLES,
            )
            return False

        last_run = self._last_training_time()
        if last_run:
            elapsed = datetime.now(timezone.utc) - last_run
            if elapsed < timedelta(days=QWEN_LORA_COOLDOWN_DAYS):
                log.debug("Qwen LoRA: cooldown active (%s remaining)", timedelta(days=QWEN_LORA_COOLDOWN_DAYS) - elapsed)
                return False

        return True

    # -- accumulation -------------------------------------------------------

    def accumulate_data(self, limit: int = 2000) -> dict[str, Any]:
        """Scan orchestrator DB for labeled triage results and accumulate.

        Each row needs: message_text, classification, profile assignments.
        Sources:
          - orchestrator.db: triage results with model_used != "disabled"
          - email triage results from email monitoring
        """
        training_dir = self.training_dir()
        training_dir.mkdir(parents=True, exist_ok=True)

        existing_count = self.count_examples()
        new_examples: list[dict[str, Any]] = []

        # Source 1: Orchestrator triage results
        orch_db = config.WORKSPACE / "store" / "orchestrator.db"
        if orch_db.exists():
            try:
                orch_examples = self._scan_orchestrator_db(orch_db, limit)
                new_examples.extend(orch_examples)
            except Exception:
                log.warning("Failed to scan orchestrator DB for Qwen training", exc_info=True)

        # Source 2: Email triage from mollygraph
        try:
            email_examples = self._scan_email_triage(limit)
            new_examples.extend(email_examples)
        except Exception:
            log.warning("Failed to scan email triage for Qwen training", exc_info=True)

        # Deduplicate by text hash
        seen_hashes: set[str] = self._load_seen_hashes()
        unique_examples = []
        for ex in new_examples:
            text_hash = self._hash_text(ex.get("text", ""))
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_examples.append(ex)

        # Write new examples
        written_path: str | None = None
        if unique_examples:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            batch_path = training_dir / f"batch-{ts}.jsonl"
            with open(batch_path, "w", encoding="utf-8") as f:
                for ex in unique_examples:
                    f.write(json.dumps(ex, ensure_ascii=True) + "\n")
            written_path = str(batch_path)
            log.info("Qwen training: wrote %d examples to %s", len(unique_examples), batch_path)

        total = self.count_examples()
        return {
            "new_examples": len(unique_examples),
            "total_examples": total,
            "batch_path": written_path,
            "required": config.QWEN_LORA_MIN_EXAMPLES,
        }

    def _scan_orchestrator_db(self, db_path: Path, limit: int) -> list[dict[str, Any]]:
        """Extract labeled triage results from orchestrator DB.

        Uses the ``orchestrator_log`` table (created by orchestrator.py)
        which stores: message_preview, classification, model, raw_response.
        """
        examples = []
        try:
            conn = db_pool.sqlite_connect(str(db_path))
            try:
                rows = conn.execute(
                    """SELECT message_preview, classification, model,
                              raw_response, timestamp
                       FROM orchestrator_log
                       WHERE model NOT IN ('disabled', 'hardcoded')
                         AND classification IS NOT NULL
                       ORDER BY timestamp DESC
                       LIMIT ?""",
                    (limit,),
                ).fetchall()
                for row in rows:
                    text = str(row[0] or "").strip()
                    classification = str(row[1] or "").strip().lower()
                    if not text or not classification:
                        continue

                    # Parse subtasks from raw_response JSON if available
                    subtasks = []
                    try:
                        raw_resp = str(row[3] or "")
                        if raw_resp:
                            import re
                            # Try to extract JSON from raw response
                            cleaned = re.sub(r"```(?:json)?\s*", "", raw_resp)
                            cleaned = re.sub(r"```\s*$", "", cleaned.strip())
                            data = json.loads(cleaned) if cleaned else {}
                            subtasks_raw = data.get("subtasks", [])
                            if isinstance(subtasks_raw, list):
                                subtasks = [
                                    {"profile": s.get("profile", "general"), "task": s.get("description", "")}
                                    for s in subtasks_raw
                                    if isinstance(s, dict)
                                ]
                    except Exception:
                        pass

                    examples.append({
                        "text": text,
                        "classification": classification,
                        "subtasks": subtasks,
                        "source": "orchestrator",
                        "model_used": str(row[2] or ""),
                        "created_at": str(row[4] or ""),
                    })
            finally:
                conn.close()
        except Exception:
            log.debug("Orchestrator DB scan failed", exc_info=True)
        return examples

    def _scan_email_triage(self, limit: int) -> list[dict[str, Any]]:
        """Extract email triage results for classification training."""
        examples = []
        try:
            conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))
            try:
                # Check if email_triage table exists
                tables = [
                    r[0] for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='email_triage'"
                    ).fetchall()
                ]
                if "email_triage" not in tables:
                    return examples

                rows = conn.execute(
                    """SELECT subject, sender, classification, score, created_at
                       FROM email_triage
                       WHERE classification IS NOT NULL
                       ORDER BY created_at DESC
                       LIMIT ?""",
                    (limit,),
                ).fetchall()
                for row in rows:
                    subject = str(row[0] or "").strip()
                    sender = str(row[1] or "").strip()
                    classification = str(row[2] or "").strip().lower()
                    if not subject or not classification:
                        continue
                    text = f"Email from {sender}: {subject}" if sender else f"Email: {subject}"
                    examples.append({
                        "text": text,
                        "classification": classification,
                        "subtasks": [],
                        "source": "email",
                        "created_at": str(row[4] or ""),
                    })
            finally:
                conn.close()
        except Exception:
            log.debug("Email triage scan failed", exc_info=True)
        return examples

    # -- training -----------------------------------------------------------

    async def run_training(self) -> dict[str, Any]:
        """Run the full Qwen3 LoRA training pipeline."""
        # Step 1: Load and split data
        rows = self.load_all_examples()
        total = len(rows)
        required = config.QWEN_LORA_MIN_EXAMPLES

        if total < required:
            return {
                "status": "insufficient_examples",
                "count": total,
                "required": required,
            }

        train_rows, eval_rows = self.split_data(rows)
        if not train_rows or not eval_rows:
            return {"status": "split_failed", "count": total}

        # Step 2: Fine-tune with LoRA
        log.info("Starting Qwen3 LoRA fine-tune: %d train, %d eval", len(train_rows), len(eval_rows))
        fine_tune = await asyncio.to_thread(self._fine_tune_lora, train_rows)

        if not fine_tune.get("ok"):
            self._update_state("finetune_failed", fine_tune.get("error", ""))
            return {"status": "finetune_failed", "error": fine_tune.get("error"), "count": total}

        # Step 3: Benchmark candidate vs baseline
        benchmark = await asyncio.to_thread(self._benchmark, eval_rows)

        if not benchmark.get("ok"):
            self._update_state("benchmark_failed", "")
            return {"status": "benchmark_failed", "benchmark": benchmark, "count": total}

        improvement = benchmark.get("improvement", 0.0)
        if improvement <= 0.0:
            self._update_state(
                "below_threshold",
                f"No improvement ({improvement:+.2%})",
            )
            return {
                "status": "below_threshold",
                "improvement": improvement,
                "benchmark": benchmark,
                "count": total,
            }

        # Step 4: Deploy if improved
        deployed = await asyncio.to_thread(self._deploy_candidate)
        if deployed.get("ok"):
            self._update_state(
                "deployed",
                f"LoRA +{improvement:.2%} accuracy, deployed",
            )
        else:
            self._update_state("deploy_failed", deployed.get("error", ""))

        return {
            "status": "deployed" if deployed.get("ok") else "deploy_failed",
            "improvement": improvement,
            "benchmark": benchmark,
            "fine_tune": fine_tune,
            "count": total,
        }

    def _fine_tune_lora(self, train_rows: list[dict]) -> dict[str, Any]:
        """Fine-tune Qwen3-4B with LoRA on triage classification data."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
            )
            from datasets import Dataset
        except ImportError as e:
            return {"ok": False, "error": f"Missing dependency: {e}"}

        models_dir = self.models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        candidate = self.candidate_dir()
        candidate.mkdir(parents=True, exist_ok=True)

        # Format training data as instruction prompts
        formatted = []
        for row in train_rows:
            text = row.get("text", "")
            classification = row.get("classification", "direct")
            subtasks = row.get("subtasks", [])

            prompt = self._format_triage_prompt(text)
            response = self._format_triage_response(classification, subtasks)
            formatted.append({"prompt": prompt, "response": response})

        try:
            # Load model and tokenizer
            model_path = str(config.TRIAGE_MODEL_PATH)
            tokenizer = AutoTokenizer.from_pretrained(
                QWEN_MODEL_NAME, trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_NAME,
                trust_remote_code=True,
                device_map="auto",
            )

            # LoRA config
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
            )
            model = get_peft_model(model, lora_config)

            # Tokenize dataset
            def tokenize_fn(example):
                full_text = f"{example['prompt']}\n{example['response']}"
                return tokenizer(
                    full_text,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                )

            dataset = Dataset.from_list(formatted)
            tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

            # Training args
            training_args = TrainingArguments(
                output_dir=str(candidate),
                num_train_epochs=2,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                warmup_steps=10,
                logging_steps=10,
                save_strategy="epoch",
                fp16=False,
                bf16=False,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized,
            )
            trainer.train()
            trainer.save_model(str(candidate))
            tokenizer.save_pretrained(str(candidate))

            return {"ok": True, "candidate_dir": str(candidate), "train_count": len(formatted)}

        except Exception as exc:
            log.error("Qwen3 LoRA fine-tune failed", exc_info=True)
            return {"ok": False, "error": str(exc)}

    def _benchmark(self, eval_rows: list[dict]) -> dict[str, Any]:
        """Benchmark candidate vs base model on held-out data."""
        # Select subset for A/B evaluation
        eval_subset = eval_rows[:QWEN_LORA_AB_EVAL_SIZE]

        base_correct = 0
        candidate_correct = 0
        base_latency_ms = 0.0
        candidate_latency_ms = 0.0

        for row in eval_subset:
            text = row.get("text", "")
            expected = row.get("classification", "")
            prompt = self._format_triage_prompt(text)

            # Base model evaluation
            t0 = time.monotonic()
            base_result = self._classify_with_base(prompt)
            base_latency_ms += (time.monotonic() - t0) * 1000

            if base_result == expected:
                base_correct += 1

            # Candidate model evaluation
            t0 = time.monotonic()
            candidate_result = self._classify_with_candidate(prompt)
            candidate_latency_ms += (time.monotonic() - t0) * 1000

            if candidate_result == expected:
                candidate_correct += 1

        total = len(eval_subset)
        if total == 0:
            return {"ok": False, "error": "empty_eval_set"}

        base_accuracy = base_correct / total
        candidate_accuracy = candidate_correct / total
        improvement = candidate_accuracy - base_accuracy

        return {
            "ok": True,
            "base_accuracy": round(base_accuracy, 4),
            "candidate_accuracy": round(candidate_accuracy, 4),
            "improvement": round(improvement, 4),
            "eval_count": total,
            "base_latency_ms_avg": round(base_latency_ms / total, 2),
            "candidate_latency_ms_avg": round(candidate_latency_ms / total, 2),
        }

    def _classify_with_base(self, prompt: str) -> str:
        """Run classification with the base (GGUF) model."""
        try:
            from memory.triage import classify_local_sync
            result = classify_local_sync(prompt)
            return self._parse_classification(result)
        except Exception:
            return "direct"

    def _classify_with_candidate(self, prompt: str) -> str:
        """Run classification with the LoRA candidate model."""
        candidate = self.candidate_dir()
        if not candidate.exists():
            return "direct"

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(candidate), trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(candidate), trust_remote_code=True, device_map="auto",
            )

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract classification from generated text
            return self._parse_classification(result)
        except Exception:
            return "direct"

    def _deploy_candidate(self) -> dict[str, Any]:
        """Deploy candidate model as active LoRA adapter."""
        import shutil

        candidate = self.candidate_dir()
        active = self.active_dir()

        if not candidate.exists():
            return {"ok": False, "error": "candidate_not_found"}

        try:
            # Backup existing active model
            if active.exists():
                backup_dir = self.models_dir() / "backups"
                backup_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                backup_path = backup_dir / ts
                shutil.copytree(active, backup_path)
                shutil.rmtree(active)

            # Move candidate to active
            shutil.copytree(candidate, active)
            shutil.rmtree(candidate)

            return {"ok": True, "active_dir": str(active)}
        except Exception as exc:
            log.error("Qwen LoRA deployment failed", exc_info=True)
            return {"ok": False, "error": str(exc)}

    # -- evaluation helpers -------------------------------------------------

    def evaluate(self) -> dict[str, Any]:
        """Return current training statistics."""
        total = self.count_examples()
        return {
            "total_examples": total,
            "required": config.QWEN_LORA_MIN_EXAMPLES,
            "ready": total >= config.QWEN_LORA_MIN_EXAMPLES,
            "last_result": str(self.ctx.state.get("qwen_last_result", "")),
            "last_status": str(self.ctx.state.get("qwen_last_cycle_status", "")),
            "last_training_at": str(self.ctx.state.get("qwen_last_training_at", "")),
            "active_lora": self.active_dir().exists(),
        }

    # -- data helpers -------------------------------------------------------

    def count_examples(self) -> int:
        """Count total accumulated training examples."""
        training_dir = self.training_dir()
        if not training_dir.exists():
            return 0
        total = 0
        for path in training_dir.glob("*.jsonl"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    total += sum(1 for line in f if line.strip())
            except Exception:
                pass
        return total

    def load_all_examples(self) -> list[dict[str, Any]]:
        """Load all accumulated training examples."""
        rows: list[dict[str, Any]] = []
        training_dir = self.training_dir()
        if not training_dir.exists():
            return rows
        for path in sorted(training_dir.glob("*.jsonl")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass
        return rows

    def split_data(
        self,
        rows: list[dict],
        eval_ratio: float = QWEN_LORA_EVAL_RATIO,
        seed: int = QWEN_LORA_SEED,
    ) -> tuple[list[dict], list[dict]]:
        """Split data into train and eval sets."""
        if not rows:
            return [], []
        indices = list(range(len(rows)))
        random.Random(seed).shuffle(indices)
        eval_count = max(1, int(len(rows) * eval_ratio))
        eval_count = min(eval_count, len(rows) - 1)
        eval_idx = set(indices[:eval_count])
        train = [rows[i] for i in range(len(rows)) if i not in eval_idx]
        eval_set = [rows[i] for i in range(len(rows)) if i in eval_idx]
        return train, eval_set

    def _load_seen_hashes(self) -> set[str]:
        """Load hashes of already-accumulated examples to avoid duplicates."""
        import hashlib
        seen: set[str] = set()
        training_dir = self.training_dir()
        if not training_dir.exists():
            return seen
        for path in training_dir.glob("*.jsonl"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            text = str(data.get("text", "")).strip()
                            if text:
                                seen.add(self._hash_text(text))
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass
        return seen

    @staticmethod
    def _hash_text(text: str) -> str:
        """Hash text for dedup."""
        import hashlib
        return hashlib.md5(text.strip().lower().encode()).hexdigest()

    @staticmethod
    def _format_triage_prompt(text: str) -> str:
        """Format a message for triage classification."""
        return (
            "Classify this message as one of: direct, simple, complex.\n"
            "Also identify which worker profiles are needed.\n\n"
            f"Message: {text}\n\n"
            "Classification:"
        )

    @staticmethod
    def _format_triage_response(classification: str, subtasks: list[dict]) -> str:
        """Format expected triage response."""
        profiles = [s.get("profile", "general") for s in subtasks] if subtasks else ["general"]
        return f" {classification}\nProfiles: {', '.join(profiles)}"

    @staticmethod
    def _parse_classification(result: str) -> str:
        """Parse classification from model output."""
        result_lower = result.strip().lower()
        for cls in ("complex", "simple", "direct"):
            if cls in result_lower:
                return cls
        # Email triage classifications
        for cls in ("urgent", "relevant", "background", "noise"):
            if cls in result_lower:
                return cls
        return "direct"

    # -- state management ---------------------------------------------------

    def _update_state(self, status: str, result: str):
        """Update persistent state."""
        self.ctx.state["qwen_last_cycle_status"] = status
        self.ctx.state["qwen_last_result"] = result
        if status in ("deployed", "finetune_failed", "benchmark_failed"):
            self.ctx.state["qwen_last_training_at"] = datetime.now(timezone.utc).isoformat()
        self.ctx.save_state()

    def _last_training_time(self) -> datetime | None:
        """Get the last training timestamp."""
        raw = str(self.ctx.state.get("qwen_last_training_at", "")).strip()
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None

    # -- weekly summary line ------------------------------------------------

    def weekly_summary_line(self) -> str:
        """Summary for weekly assessment."""
        total = self.count_examples()
        last_result = str(self.ctx.state.get("qwen_last_result", "no training yet"))
        return (
            f"Qwen3 LoRA training data: {total} examples. "
            f"Active adapter: {'yes' if self.active_dir().exists() else 'no'}. "
            f"Last: {last_result}"
        )


# ---------------------------------------------------------------------------
# Facade functions — package-level entry points
# ---------------------------------------------------------------------------

async def run_qwen_training_pipeline(
    engine=None,
    molly=None,
) -> dict[str, Any]:
    """Run the full Qwen3 LoRA training pipeline.

    Parameters
    ----------
    engine : SelfImprovementEngine | None
    molly : object | None

    Returns
    -------
    dict
        Pipeline result.
    """
    if engine is None:
        from evolution.skills import SelfImprovementEngine
        engine = SelfImprovementEngine(molly=molly)
        await engine.initialize()

    if not hasattr(engine, "qwen"):
        return {"status": "qwen_service_unavailable"}

    return await engine.qwen.run_training()


def get_qwen_stats(engine=None) -> dict[str, Any]:
    """Return current Qwen training statistics."""
    if engine is None or not hasattr(engine, "qwen"):
        return {"status": "no_engine", "total_examples": 0}
    return engine.qwen.evaluate()
