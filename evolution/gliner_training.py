"""GLiNER LoRA fine-tune pipeline — service + thin facade.

The ``GLiNERTrainingService`` class contains all GLiNER-related methods
extracted from ``SelfImprovementEngine`` (evolution/skills.py).  The engine
constructs this service so the methods are available via ``self.gliner.*``.

The facade functions at the bottom (``run_gliner_finetune_pipeline``,
``run_gliner_accumulation``, ``get_gliner_stats``) provide a package-level
entry point so maintenance jobs can import directly::

    from evolution.gliner_training import run_gliner_finetune_pipeline

Label *generation* stays in ``monitoring/jobs/entity_audit.py``.  This
module is the training *consumer* only.

Constants re-exported for discoverability:
    GLINER_BASE_MODEL, GLINER_BENCHMARK_SEED, GLINER_BENCHMARK_EVAL_RATIO,
    GLINER_BENCHMARK_THRESHOLD, GLINER_FINETUNE_COOLDOWN_DAYS
"""
from __future__ import annotations

from utils import atomic_write

import asyncio
import json
import logging
import random
import re
import shutil
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import config
import db_pool
from evolution.infra import (
    _parse_datetime,
    _GIT_AUTHOR_ENV,
    _GLINER_BASE_MODEL,
    _GLINER_BENCHMARK_SEED,
    _GLINER_BENCHMARK_EVAL_RATIO,
    _GLINER_BENCHMARK_THRESHOLD,
    _GLINER_FINETUNE_COOLDOWN_DAYS,
    _GLINER_TRAINING_SCAN_LIMIT,
)

log = logging.getLogger(__name__)

# Re-export constants used by the pipeline (originally module-level in
# self_improve.py, now in evolution/skills.py).
GLINER_BASE_MODEL = "fastino/gliner2-large-v1"
GLINER_BENCHMARK_SEED = 1337
GLINER_BENCHMARK_EVAL_RATIO = 0.2
GLINER_BENCHMARK_THRESHOLD = 0.4
GLINER_FINETUNE_COOLDOWN_DAYS = 7
GLINER_TRAINING_SCAN_LIMIT = 4000



class GLiNERTrainingService:
    """GLiNER training pipeline service.

    Receives explicit ``InfraService`` and ``OwnerCommsService`` dependencies.
    """

    _GLINER_MAX_RUNS = 3
    _GLINER_MAX_BACKUPS = 2

    def __init__(self, ctx, infra, comms):
        from evolution.context import EngineContext
        from evolution.infra import InfraService
        from evolution.owner_comms import OwnerCommsService
        self.ctx: EngineContext = ctx
        self.infra: InfraService = infra
        self.comms: OwnerCommsService = comms

    # -- directory helpers --------------------------------------------------

    def gliner_training_dir(self) -> Path:
        return config.WORKSPACE / "memory" / "gliner_training"

    def gliner_models_dir(self) -> Path:
        return config.WORKSPACE / "models"

    def gliner_candidate_model_dir(self) -> Path:
        return self.gliner_models_dir() / "gliner_candidate"

    def gliner_active_model_dir(self) -> Path:
        return self.gliner_models_dir() / "gliner_active"

    def gliner_training_config_path(self) -> Path:
        return self.ctx.project_root / "memory" / "gliner_finetune_config.json"

    # -- weekly summary -----------------------------------------------------

    def gliner_weekly_summary_line(self) -> str:
        total = self.count_accumulated_gliner_examples()
        last_finetune = _parse_datetime(str(self.ctx.state.get("gliner_last_finetune_at", "")))
        last_finetune_str = last_finetune.date().isoformat() if last_finetune else "never"
        strategy = str(self.ctx.state.get("gliner_last_training_strategy", "lora")).strip().lower() or "lora"
        last_result = str(self.ctx.state.get("gliner_last_result", "no fine-tune runs yet")).strip()
        if not last_result:
            last_result = "no fine-tune runs yet"
        return (
            f"GLiNER training data: {total} examples. "
            f"Last fine-tune: {last_finetune_str} ({strategy}). Result: {last_result}"
        )

    # -- nightly cycle ------------------------------------------------------

    async def run_gliner_nightly_cycle(self) -> dict[str, Any]:
        accumulation = await asyncio.to_thread(
            self.accumulate_gliner_training_data,
            _GLINER_TRAINING_SCAN_LIMIT,
        )
        total_examples = int(accumulation.get("total_examples", 0))
        required = int(config.GLINER_FINETUNE_MIN_EXAMPLES)
        progress_line = f"GLiNER training data: {total_examples}/{required} examples accumulated"
        log.info(progress_line)

        self.ctx.state["gliner_training_examples"] = total_examples
        self.ctx.state["gliner_last_result"] = progress_line
        self.ctx.state["gliner_last_cycle_status"] = "accumulated"
        self.ctx.save_state()

        if total_examples < required:
            self.comms.log_improvement_event(
                event_type="model",
                category="gliner",
                title="GLiNER training data accumulation",
                payload=json.dumps(accumulation, ensure_ascii=True),
                status="insufficient_examples",
            )
            return {
                "status": "insufficient_examples",
                "count": total_examples,
                "required": required,
                "accumulation": accumulation,
                "message": progress_line,
            }

        now_utc = datetime.now(timezone.utc)
        last_run = _parse_datetime(str(self.ctx.state.get("gliner_last_finetune_at", "")))
        if last_run and (now_utc - last_run) < timedelta(days=_GLINER_FINETUNE_COOLDOWN_DAYS):
            elapsed = now_utc - last_run
            remaining = timedelta(days=_GLINER_FINETUNE_COOLDOWN_DAYS) - elapsed
            hours_remaining = max(0, int(remaining.total_seconds() // 3600))
            cooldown_line = (
                f"GLiNER fine-tune skipped: last run {last_run.date().isoformat()} "
                f"({hours_remaining}h cooldown remaining)."
            )
            log.info(cooldown_line)
            self.ctx.state["gliner_last_result"] = cooldown_line
            self.ctx.state["gliner_last_cycle_status"] = "cooldown_active"
            self.ctx.save_state()
            self.comms.log_improvement_event(
                event_type="model",
                category="gliner",
                title="GLiNER fine-tune trigger",
                payload=json.dumps(
                    {"count": total_examples, "required": required, "last_run": last_run.isoformat()},
                    ensure_ascii=True,
                ),
                status="cooldown_active",
            )
            return {
                "status": "cooldown_active",
                "count": total_examples,
                "required": required,
                "last_run": last_run.isoformat(),
                "accumulation": accumulation,
                "message": cooldown_line,
            }

        pipeline = await self.run_gliner_finetune_pipeline()
        return {
            "status": "finetune_triggered",
            "accumulation": accumulation,
            "pipeline": pipeline,
        }

    async def run_gliner_finetune_pipeline(self) -> dict[str, Any]:
        rows = await asyncio.to_thread(self.load_accumulated_gliner_examples)
        total_rows = len(rows)
        required = int(config.GLINER_FINETUNE_MIN_EXAMPLES)
        if total_rows < required:
            msg = f"GLiNER training data: {total_rows}/{required} examples accumulated"
            self.ctx.state["gliner_training_examples"] = total_rows
            self.ctx.state["gliner_last_result"] = msg
            self.ctx.state["gliner_last_cycle_status"] = "insufficient_examples"
            self.ctx.save_state()
            return {
                "status": "insufficient_examples",
                "count": total_rows,
                "required": required,
                "message": msg,
            }

        train_rows, eval_rows = self.split_holdout_rows(rows, eval_ratio=0.2, seed=_GLINER_BENCHMARK_SEED)
        if not train_rows or not eval_rows:
            failure = {
                "status": "split_failed",
                "count": total_rows,
                "train_count": len(train_rows),
                "eval_count": len(eval_rows),
            }
            self.ctx.state["gliner_last_cycle_status"] = "split_failed"
            self.ctx.state["gliner_last_result"] = "GLiNER fine-tune skipped: invalid train/eval split."
            self.ctx.save_state()
            self.comms.log_improvement_event(
                event_type="model",
                category="gliner",
                title="GLiNER2 fine-tune split",
                payload=json.dumps(failure, ensure_ascii=True),
                status="split_failed",
            )
            return failure

        training_strategy = await asyncio.to_thread(self.select_gliner_training_strategy, total_rows)
        strategy_mode = str(training_strategy.get("mode") or "lora").strip().lower()
        if strategy_mode not in {"lora", "full"}:
            strategy_mode = "lora"

        self.ctx.state["gliner_last_finetune_at"] = datetime.now(timezone.utc).isoformat()
        self.ctx.state["gliner_training_examples"] = total_rows
        self.ctx.state["gliner_last_cycle_status"] = "finetune_started"
        self.ctx.state["gliner_last_training_strategy"] = strategy_mode
        self.ctx.save_state()

        fine_tune = await asyncio.to_thread(self.fine_tune_gliner_candidate, train_rows, strategy_mode)
        if not fine_tune.get("ok", False):
            payload = {
                "count": total_rows,
                "train_count": len(train_rows),
                "eval_count": len(eval_rows),
                "training_strategy": training_strategy,
                "fine_tune": fine_tune,
            }
            self.ctx.state["gliner_last_cycle_status"] = "finetune_failed"
            self.ctx.state["gliner_last_result"] = (
                f"GLiNER {strategy_mode} fine-tune failed before benchmarking."
            )
            self.ctx.save_state()
            self.comms.log_improvement_event(
                event_type="model",
                category="gliner",
                title="GLiNER2 fine-tune run",
                payload=json.dumps(payload, ensure_ascii=True),
                status="finetune_failed",
            )
            return {
                "status": "finetune_failed",
                "count": total_rows,
                "train_count": len(train_rows),
                "eval_count": len(eval_rows),
                "training_strategy": training_strategy,
                "fine_tune": fine_tune,
            }

        candidate_model_ref = str(fine_tune.get("candidate_model") or "").strip()
        benchmark = await asyncio.to_thread(
            self.benchmark_finetune_candidate,
            eval_rows,
            candidate_model_ref,
            len(train_rows),
        )
        payload = {
            "count": total_rows,
            "train_count": len(train_rows),
            "eval_count": len(eval_rows),
            "candidate_model": candidate_model_ref,
            "training_strategy": training_strategy,
            "fine_tune": fine_tune,
            "benchmark": benchmark,
        }

        status = "benchmark_failed"
        if benchmark.get("ok", False):
            if benchmark["improvement"] >= config.GLINER_FINETUNE_BENCHMARK_THRESHOLD:
                status = "proposal_ready"
            else:
                status = "below_threshold"

        if status != "proposal_ready":
            await asyncio.to_thread(
                self.discard_gliner_candidate_model,
                Path(candidate_model_ref) if candidate_model_ref else None,
            )
            if status == "below_threshold":
                self.ctx.state["gliner_last_result"] = (
                    f"GLiNER {strategy_mode} fine-tune benchmark below threshold "
                    f"({benchmark.get('improvement', 0.0):+.2%} F1)."
                )
            else:
                self.ctx.state["gliner_last_result"] = f"GLiNER {strategy_mode} fine-tune benchmark failed."
            self.ctx.state["gliner_last_cycle_status"] = status
            self.ctx.save_state()
            await asyncio.to_thread(
                self.record_gliner_benchmark,
                strategy_mode,
                benchmark,
                status,
                total_rows,
            )
            self.comms.log_improvement_event(
                event_type="model",
                category="gliner",
                title="GLiNER2 fine-tune benchmark",
                payload=json.dumps(payload, ensure_ascii=True),
                status=status,
            )
            return {
                "status": status,
                "count": total_rows,
                "train_count": len(train_rows),
                "eval_count": len(eval_rows),
                "candidate_model": candidate_model_ref or None,
                "training_strategy": training_strategy,
                "benchmark": benchmark,
                "fine_tune": fine_tune,
            }

        proposal = self.format_gliner_swap_proposal(benchmark, strategy_mode)
        await self.comms.notify_owner(proposal)
        decision = await self.comms.request_owner_decision(
            category="self-improve-gliner-model-swap",
            description=proposal,
            required_keyword="YES",
            allow_edit=False,
        )
        if decision is True:
            deploy = await asyncio.to_thread(
                self.deploy_gliner_candidate_model,
                Path(candidate_model_ref),
                benchmark,
                fine_tune,
            )
            status = "deployed" if deploy.get("ok", False) else "deploy_failed"
            payload["deploy"] = deploy
            if deploy.get("ok", False):
                self.ctx.state["gliner_last_cycle_status"] = "deployed"
                self.ctx.state["gliner_last_result"] = (
                    f"{strategy_mode} +{benchmark.get('improvement', 0.0):.2%} F1 "
                    f"(P {benchmark.get('base', {}).get('metrics', {}).get('precision', 0.0):.2f} "
                    f"-> {benchmark.get('candidate', {}).get('metrics', {}).get('precision', 0.0):.2f}), "
                    "approved and deployed."
                )
            else:
                self.ctx.state["gliner_last_cycle_status"] = "deploy_failed"
                self.ctx.state["gliner_last_result"] = (
                    f"GLiNER {strategy_mode} swap approved but deployment failed."
                )
            self.ctx.save_state()
            await asyncio.to_thread(
                self.record_gliner_benchmark,
                strategy_mode,
                benchmark,
                status,
                total_rows,
            )
            self.comms.log_improvement_event(
                event_type="model",
                category="gliner",
                title="GLiNER2 fine-tune benchmark",
                payload=json.dumps(payload, ensure_ascii=True),
                status=status,
            )
            return {
                "status": status,
                "count": total_rows,
                "train_count": len(train_rows),
                "eval_count": len(eval_rows),
                "training_strategy": training_strategy,
                "benchmark": benchmark,
                "fine_tune": fine_tune,
                "deploy": deploy,
            }

        await asyncio.to_thread(
            self.discard_gliner_candidate_model,
            Path(candidate_model_ref) if candidate_model_ref else None,
        )
        self.ctx.state["gliner_last_cycle_status"] = "rejected"
        self.ctx.state["gliner_last_result"] = (
            f"GLiNER {strategy_mode} candidate rejected; keeping active model."
        )
        self.ctx.save_state()
        await asyncio.to_thread(
            self.record_gliner_benchmark,
            strategy_mode,
            benchmark,
            "rejected",
            total_rows,
        )
        self.comms.log_improvement_event(
            event_type="model",
            category="gliner",
            title="GLiNER2 fine-tune benchmark",
            payload=json.dumps(payload, ensure_ascii=True),
            status="rejected",
        )
        return {
            "status": "rejected",
            "count": total_rows,
            "train_count": len(train_rows),
            "eval_count": len(eval_rows),
            "training_strategy": training_strategy,
            "benchmark": benchmark,
            "fine_tune": fine_tune,
        }

    # -- accumulation -------------------------------------------------------

    def accumulate_gliner_training_data(self, limit: int = _GLINER_TRAINING_SCAN_LIMIT) -> dict[str, Any]:
        from memory.graph import get_driver

        training_dir = self.gliner_training_dir()
        training_dir.mkdir(parents=True, exist_ok=True)
        seen_episode_ids = self.load_existing_gliner_episode_ids(training_dir)
        opus_analysis_text = self.latest_maintenance_analysis_text()
        correction_texts = self.load_recent_correction_texts()
        cursor = str(self.ctx.state.get("gliner_training_cursor", "")).strip()

        where_cursor = "AND ep.created_at > $cursor" if cursor else ""
        params: dict[str, Any] = {"limit": int(limit)}
        if cursor:
            params["cursor"] = cursor

        driver = get_driver()
        with driver.session() as session:
            rows = [
                dict(r)
                for r in session.run(
                    f"""
                    MATCH (ep:Episode)
                    WHERE ep.content_preview IS NOT NULL
                      AND trim(ep.content_preview) <> ''
                      AND ep.entities_extracted IS NOT NULL
                      AND size(ep.entities_extracted) > 0
                      {where_cursor}
                    RETURN ep.id AS episode_id,
                           ep.created_at AS created_at,
                           ep.content_preview AS source_text,
                           ep.entities_extracted AS entity_names
                    ORDER BY ep.created_at ASC
                    LIMIT $limit
                    """,
                    **params,
                )
            ]

            new_examples: list[dict[str, Any]] = []
            latest_seen_created_at = cursor
            for row in rows:
                episode_id = str(row.get("episode_id") or "").strip()
                if not episode_id or episode_id in seen_episode_ids:
                    if row.get("created_at"):
                        latest_seen_created_at = str(row["created_at"])
                    continue

                example = self.build_training_example_from_episode(
                    session=session,
                    episode=row,
                    opus_analysis_text=opus_analysis_text,
                    correction_texts=correction_texts,
                )
                if example:
                    new_examples.append(example)
                    seen_episode_ids.add(episode_id)

                if row.get("created_at"):
                    latest_seen_created_at = str(row["created_at"])

        written_path: str | None = None
        if new_examples:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            batch_path = training_dir / f"examples-{ts}.jsonl"
            with open(batch_path, "w", encoding="utf-8") as f:
                for example in new_examples:
                    f.write(json.dumps(example, ensure_ascii=True) + "\n")
            written_path = str(batch_path)
            log.info("GLiNER accumulation: wrote %d examples to %s", len(new_examples), batch_path)
        else:
            log.info("GLiNER accumulation: no new high-confidence examples this run")

        if latest_seen_created_at:
            self.ctx.state["gliner_training_cursor"] = latest_seen_created_at
        total_examples = self.count_accumulated_gliner_examples(training_dir)
        self.ctx.state["gliner_training_examples"] = total_examples
        self.ctx.save_state()

        return {
            "new_examples": len(new_examples),
            "total_examples": total_examples,
            "batch_path": written_path,
            "cursor": self.ctx.state.get("gliner_training_cursor", ""),
        }

    def load_existing_gliner_episode_ids(self, training_dir: Path) -> set[str]:
        seen: set[str] = set()
        for path in sorted(training_dir.glob("*.jsonl")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        episode_id = str(data.get("episode_id") or "").strip()
                        if episode_id:
                            seen.add(episode_id)
            except Exception:
                log.debug("Failed scanning GLiNER training file %s", path, exc_info=True)
        return seen

    def latest_maintenance_analysis_text(self) -> str:
        maintenance_dir = config.WORKSPACE / "memory" / "maintenance"
        if not maintenance_dir.exists():
            return ""
        candidates = sorted(maintenance_dir.glob("*.md"))
        if not candidates:
            return ""
        latest_path = candidates[-1]
        try:
            content = latest_path.read_text(encoding="utf-8")
        except Exception:
            return ""
        marker = "\n## Analysis\n"
        if marker in content:
            content = content.split(marker, 1)[1]
        return self.normalize_entity_text(content)

    def load_recent_correction_texts(self, limit: int = 400) -> list[str]:
        rows = self.infra.rows(
            config.MOLLYGRAPH_PATH,
            """
            SELECT context, molly_output, user_correction
            FROM corrections
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        texts = []
        for row in rows:
            merged = " ".join(
                str(row.get(key, "") or "").strip()
                for key in ("context", "molly_output", "user_correction")
            )
            normalized = self.normalize_entity_text(merged)
            if normalized:
                texts.append(normalized)
        return texts

    def build_training_example_from_episode(
        self,
        session: Any,
        episode: dict[str, Any],
        opus_analysis_text: str,
        correction_texts: list[str],
    ) -> dict[str, Any] | None:
        source_text = str(episode.get("source_text") or "").strip()
        if not source_text:
            return None
        entity_names_raw = episode.get("entity_names") or []
        if not isinstance(entity_names_raw, list):
            return None
        entity_names = sorted(
            {
                str(name).strip()
                for name in entity_names_raw
                if str(name).strip()
            }
        )
        if not entity_names:
            return None

        entity_rows = [
            dict(r)
            for r in session.run(
                """
                MATCH (e:Entity)
                WHERE e.name IN $names
                RETURN e.name AS name,
                       e.entity_type AS entity_type,
                       coalesce(e.mention_count, 0) AS mention_count,
                       EXISTS { MATCH (e)--(:Entity) } AS has_relationship
                """,
                names=entity_names,
            )
        ]
        if not entity_rows:
            return None

        selected_entities: list[dict[str, Any]] = []
        signal_counts = {
            "opus_confirmed": 0,
            "multi_mentions": 0,
            "relationship_backed": 0,
        }
        for entity in entity_rows:
            name = str(entity.get("name") or "").strip()
            label = str(entity.get("entity_type") or "Concept").strip() or "Concept"
            mention_count = int(entity.get("mention_count") or 0)
            has_relationship = bool(entity.get("has_relationship"))
            normalized = self.normalize_entity_text(name)
            if not normalized:
                continue

            corrected = any(normalized in text for text in correction_texts)
            opus_confirmed = bool(opus_analysis_text) and (normalized in opus_analysis_text) and not corrected
            multi_mentions = mention_count >= 2
            relationship_backed = has_relationship

            if not (opus_confirmed or multi_mentions or relationship_backed):
                continue

            if opus_confirmed:
                signal_counts["opus_confirmed"] += 1
            if multi_mentions:
                signal_counts["multi_mentions"] += 1
            if relationship_backed:
                signal_counts["relationship_backed"] += 1

            selected_entities.append(
                {
                    "text": name,
                    "label": label,
                }
            )

        if not selected_entities:
            return None

        selected_names = sorted({str(entity["text"]).strip() for entity in selected_entities})
        relation_rows = [
            dict(r)
            for r in session.run(
                """
                MATCH (h:Entity)-[r]->(t:Entity)
                WHERE h.name IN $names AND t.name IN $names
                  AND (r.audit_status IS NULL OR r.audit_status <> 'quarantined')
                RETURN h.name AS head,
                       t.name AS tail,
                       type(r) AS label,
                       coalesce(r.mention_count, 0) AS mention_count
                """,
                names=selected_names,
            )
        ]
        relations: list[dict[str, Any]] = []
        seen_rel_keys: set[tuple[str, str, str]] = set()
        for rel in relation_rows:
            head = str(rel.get("head") or "").strip()
            tail = str(rel.get("tail") or "").strip()
            label = str(rel.get("label") or "").strip()
            if not head or not tail or not label:
                continue
            key = (head, label, tail)
            if key in seen_rel_keys:
                continue
            seen_rel_keys.add(key)
            relations.append(
                {
                    "head": head,
                    "tail": tail,
                    "label": label,
                }
            )

        return {
            "episode_id": str(episode.get("episode_id") or ""),
            "created_at": str(episode.get("created_at") or ""),
            "source_text": source_text,
            "extracted_entities": selected_entities,
            "extracted_relations": relations,
            "quality_signals": signal_counts,
        }

    def count_accumulated_gliner_examples(self, training_dir: Path | None = None) -> int:
        target_dir = training_dir or self.gliner_training_dir()
        if not target_dir.exists():
            return 0
        total = 0
        for path in target_dir.glob("*.jsonl"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    total += sum(1 for line in f if line.strip())
            except Exception:
                log.debug("Failed counting GLiNER examples in %s", path, exc_info=True)
        return total

    def load_accumulated_gliner_examples(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        training_dir = self.gliner_training_dir()
        if not training_dir.exists():
            return rows
        for path in sorted(training_dir.glob("*.jsonl")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        row = self.to_benchmark_row(payload)
                        if row:
                            rows.append(row)
            except Exception:
                log.debug("Failed loading GLiNER examples from %s", path, exc_info=True)
        return rows

    @staticmethod
    def to_benchmark_row(payload: dict[str, Any]) -> dict[str, Any] | None:
        text = str(payload.get("source_text") or payload.get("text") or "").strip()
        entities = payload.get("extracted_entities", payload.get("entities", []))
        relations = payload.get("extracted_relations", payload.get("relations", []))
        if not text or not isinstance(entities, list) or not entities:
            return None
        return {
            "episode_id": str(payload.get("episode_id") or ""),
            "created_at": str(payload.get("created_at") or ""),
            "text": text,
            "entities": entities,
            "relations": relations if isinstance(relations, list) else [],
        }

    def to_gliner_training_record(self, row: dict[str, Any]) -> dict[str, Any] | None:
        text = str(row.get("text") or "").strip()
        if not text:
            return None
        text_norm = self.normalize_entity_text(text)

        entities_map: dict[str, list[str]] = {}
        entity_names: set[str] = set()
        for ent in row.get("entities") or []:
            if isinstance(ent, str):
                name = ent.strip()
                label = "Concept"
            elif isinstance(ent, dict):
                name = str(ent.get("text") or ent.get("name") or "").strip()
                label = str(ent.get("label") or ent.get("type") or "Concept").strip() or "Concept"
            else:
                continue
            if not name:
                continue
            if self.normalize_entity_text(name) not in text_norm:
                continue
            entities_map.setdefault(label, [])
            if name not in entities_map[label]:
                entities_map[label].append(name)
                entity_names.add(name)

        relations_out: list[dict[str, Any]] = []
        for rel in row.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            head = str(rel.get("head") or "").strip()
            tail = str(rel.get("tail") or "").strip()
            label = str(rel.get("label") or "").strip().lower().replace("_", " ")
            if not head or not tail or not label:
                continue
            if head not in entity_names or tail not in entity_names:
                continue
            if self.normalize_entity_text(head) not in text_norm:
                continue
            if self.normalize_entity_text(tail) not in text_norm:
                continue
            relations_out.append({label: {"head": head, "tail": tail}})

        output: dict[str, Any] = {}
        if entities_map:
            output["entities"] = entities_map
        if relations_out:
            output["relations"] = relations_out
        if not output:
            return None
        return {"input": text, "output": output}

    # -- model refs ---------------------------------------------------------

    def active_gliner_model_ref(self) -> str:
        active_dir = self.gliner_active_model_dir()
        if active_dir.exists():
            return str(active_dir)
        active_state_ref = str(self.ctx.state.get("gliner_active_model_ref", "")).strip()
        if active_state_ref and Path(active_state_ref).exists():
            return active_state_ref
        return _GLINER_BASE_MODEL

    # -- training strategy --------------------------------------------------

    def select_gliner_training_strategy(self, total_examples: int) -> dict[str, Any]:
        full_min_examples = max(1, int(config.GLINER_FULL_FINETUNE_MIN_EXAMPLES))
        plateau_window = max(1, int(config.GLINER_LORA_PLATEAU_WINDOW))
        plateau_epsilon = max(0.0, float(config.GLINER_LORA_PLATEAU_EPSILON))
        if total_examples < full_min_examples:
            return {
                "mode": "lora",
                "reason": "insufficient_examples_for_full_finetune",
                "full_min_examples": full_min_examples,
                "total_examples": total_examples,
            }

        history = self.ctx.state.get("gliner_benchmark_history")
        if not isinstance(history, list):
            history = []
        recent_lora = [
            row for row in history
            if isinstance(row, dict)
            and str(row.get("strategy", "")).lower() == "lora"
            and bool(row.get("benchmark_ok"))
        ]
        recent_lora = recent_lora[-plateau_window:]
        if len(recent_lora) < plateau_window:
            return {
                "mode": "lora",
                "reason": "not_enough_lora_history",
                "required_runs": plateau_window,
                "available_runs": len(recent_lora),
                "total_examples": total_examples,
            }

        improvements = [
            float(row.get("improvement", 0.0) or 0.0)
            for row in recent_lora
        ]
        max_gain = max(improvements) if improvements else 0.0
        plateaued = max_gain <= plateau_epsilon
        if plateaued:
            return {
                "mode": "full",
                "reason": "lora_plateau_detected",
                "plateau_window": plateau_window,
                "plateau_epsilon": plateau_epsilon,
                "recent_improvements": improvements,
                "total_examples": total_examples,
            }
        return {
            "mode": "lora",
            "reason": "lora_still_improving",
            "plateau_window": plateau_window,
            "plateau_epsilon": plateau_epsilon,
            "recent_improvements": improvements,
            "total_examples": total_examples,
        }

    def record_gliner_benchmark(
        self,
        strategy: str,
        benchmark: dict[str, Any],
        status: str,
        total_examples: int,
    ):
        history = self.ctx.state.get("gliner_benchmark_history")
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "strategy": strategy,
                "status": status,
                "total_examples": int(total_examples),
                "benchmark_ok": bool(benchmark.get("ok", False)),
                "improvement": float(benchmark.get("improvement", 0.0) or 0.0),
                "base_score": float(benchmark.get("base_score", 0.0) or 0.0),
                "candidate_score": float(benchmark.get("candidate_score", 0.0) or 0.0),
                "eval_count": int(benchmark.get("split", {}).get("eval_count", 0) or 0),
            }
        )
        self.ctx.state["gliner_benchmark_history"] = history[-20:]
        self.ctx.save_state()

    # -- fine-tuning --------------------------------------------------------

    def fine_tune_gliner_candidate(self, train_rows: list[dict[str, Any]], mode: str = "lora") -> dict[str, Any]:
        train_records = [r for r in (self.to_gliner_training_record(row) for row in train_rows) if r]
        if not train_records:
            return {"ok": False, "error": "no_valid_train_records"}

        from gliner2.training.trainer import train_gliner2

        models_dir = self.gliner_models_dir()
        candidate_dir = self.gliner_candidate_model_dir()
        runs_dir = models_dir / "gliner_runs"
        splits_dir = self.gliner_training_dir() / "splits"
        models_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        train_split_path = splits_dir / f"train-{ts}.jsonl"
        with open(train_split_path, "w", encoding="utf-8") as f:
            for record in train_records:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

        output_dir = runs_dir / ts
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_size = min(4, max(1, len(train_records)))
        normalized_mode = "full" if str(mode).strip().lower() == "full" else "lora"
        use_lora = normalized_mode == "lora"
        num_epochs = 1 if use_lora else 2

        try:
            result = train_gliner2(
                model_path=self.active_gliner_model_ref(),
                train_data=train_records,
                output_dir=str(output_dir),
                num_epochs=num_epochs,
                batch_size=batch_size,
                eval_strategy="no",
                fp16=False,
                bf16=False,
                num_workers=0,
                logging_steps=max(1, min(25, len(train_records))),
                use_lora=use_lora,
                save_adapter_only=False,
                seed=_GLINER_BENCHMARK_SEED,
            )
        except Exception as exc:
            log.error("GLiNER fine-tuning failed", exc_info=True)
            return {"ok": False, "error": str(exc)}

        final_dir = output_dir / "final"
        if not final_dir.exists():
            best_dir = output_dir / "best"
            if best_dir.exists():
                final_dir = best_dir
            else:
                return {"ok": False, "error": "trained_model_not_found", "output_dir": str(output_dir)}

        self.discard_gliner_candidate_model(candidate_dir)
        shutil.copytree(final_dir, candidate_dir, dirs_exist_ok=False)
        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "base_model": self.active_gliner_model_ref(),
            "train_examples": len(train_records),
            "batch_size": batch_size,
            "mode": normalized_mode,
            "num_epochs": num_epochs,
            "output_dir": str(output_dir),
            "result": result,
        }
        metadata_path = candidate_dir / "fine_tune_metadata.json"
        atomic_write(metadata_path, json.dumps(metadata, indent=2, ensure_ascii=True))

        self.prune_gliner_dirs()

        return {
            "ok": True,
            "candidate_model": str(candidate_dir),
            "train_split_path": str(train_split_path),
            "output_dir": str(output_dir),
            "metadata_path": str(metadata_path),
            "mode": normalized_mode,
            "result": result,
        }

    def prune_gliner_dirs(self) -> dict[str, Any]:
        pruned: dict[str, list[str]] = {"runs": [], "backups": []}
        for subdir, limit in [
            (self.gliner_models_dir() / "gliner_runs", self._GLINER_MAX_RUNS),
            (self.gliner_models_dir() / "gliner_backups", self._GLINER_MAX_BACKUPS),
        ]:
            if not subdir.exists():
                continue
            dirs = sorted(
                [d for d in subdir.iterdir() if d.is_dir()],
                key=lambda p: p.name,
                reverse=True,
            )
            for old_dir in dirs[limit:]:
                try:
                    shutil.rmtree(old_dir, ignore_errors=True)
                    pruned[subdir.name].append(old_dir.name)
                    log.info("Pruned old GLiNER dir: %s", old_dir)
                except Exception:
                    log.warning("Failed to prune %s", old_dir, exc_info=True)
        return pruned

    @staticmethod
    def discard_gliner_candidate_model(candidate_path: Path | None):
        if not candidate_path:
            return
        if candidate_path.exists():
            shutil.rmtree(candidate_path, ignore_errors=True)

    def format_gliner_swap_proposal(self, benchmark: dict[str, Any], mode: str = "lora") -> str:
        base_metrics = benchmark.get("base", {}).get("metrics", {})
        candidate_metrics = benchmark.get("candidate", {}).get("metrics", {})
        base_precision = float(base_metrics.get("precision", 0.0) or 0.0)
        cand_precision = float(candidate_metrics.get("precision", 0.0) or 0.0)
        eval_count = int(benchmark.get("split", {}).get("eval_count", 0) or 0)
        strategy_label = "full fine-tune" if str(mode).strip().lower() == "full" else "LoRA fine-tune"
        return (
            f"GLiNER {strategy_label} ready: precision improved "
            f"from {base_precision:.2f} to {cand_precision:.2f} "
            f"on {eval_count} held-out examples. Approve model swap?\n\n"
            "Reply YES to deploy or NO to keep the current model."
        )

    def deploy_gliner_candidate_model(
        self,
        candidate_path: Path,
        benchmark: dict[str, Any],
        fine_tune: dict[str, Any],
    ) -> dict[str, Any]:
        if not candidate_path.exists():
            return {"ok": False, "error": "candidate_path_missing"}

        active_dir = self.gliner_active_model_dir()
        active_dir.parent.mkdir(parents=True, exist_ok=True)
        previous_active_ref = self.active_gliner_model_ref()

        backup_dir = None
        if active_dir.exists():
            backup_root = self.gliner_models_dir() / "gliner_backups"
            backup_root.mkdir(parents=True, exist_ok=True)
            backup_dir = backup_root / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            shutil.copytree(active_dir, backup_dir, dirs_exist_ok=False)

        if active_dir.exists():
            shutil.rmtree(active_dir, ignore_errors=True)
        shutil.copytree(candidate_path, active_dir, dirs_exist_ok=False)

        deployed_at = datetime.now(timezone.utc).isoformat()
        self.ctx.state["gliner_active_model_ref"] = str(active_dir)
        self.ctx.state["gliner_last_deployed_at"] = deployed_at
        self.ctx.save_state()

        config_payload = {
            "updated_at": deployed_at,
            "active_model_ref": str(active_dir),
            "previous_model_ref": previous_active_ref,
            "backup_model_ref": str(backup_dir) if backup_dir else None,
            "benchmark": benchmark,
            "fine_tune": {
                "mode": fine_tune.get("mode", "lora"),
                "train_split_path": fine_tune.get("train_split_path"),
                "output_dir": fine_tune.get("output_dir"),
                "metadata_path": fine_tune.get("metadata_path"),
            },
        }
        config_path = self.gliner_training_config_path()
        atomic_write(config_path, json.dumps(config_payload, indent=2, ensure_ascii=True))
        commit = self.commit_gliner_training_config(config_path)

        pruned = self.prune_gliner_dirs()

        return {
            "ok": True,
            "active_model": str(active_dir),
            "backup_model": str(backup_dir) if backup_dir else None,
            "training_config": str(config_path),
            "git": commit,
            "pruned": pruned,
        }

    def commit_gliner_training_config(self, config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            return {"ok": False, "error": "config_missing"}
        try:
            rel_path = config_path.relative_to(self.ctx.project_root)
        except ValueError:
            rel_path = config_path
        rel = str(rel_path)

        try:
            self.infra.git(["add", "--", rel])
            staged = self.infra.git(["diff", "--cached", "--name-only", "--", rel], check=False)
            if not staged.stdout.strip():
                return {"ok": True, "status": "no_changes"}
            commit_msg = f"[molly-self-improve] GLiNER fine-tune config update ({date.today().isoformat()})"
            commit = self.infra.git(
                ["commit", "-m", commit_msg, "--", rel],
                check=False,
                env_override=_GIT_AUTHOR_ENV,
            )
            if commit.returncode != 0:
                return {
                    "ok": False,
                    "status": "commit_failed",
                    "error": (commit.stderr or commit.stdout).strip()[:1000],
                }
            commit_hash = self.infra.git(["rev-parse", "HEAD"], check=False).stdout.strip()
            return {"ok": True, "status": "committed", "commit": commit_hash}
        except Exception as exc:
            return {"ok": False, "status": "commit_exception", "error": str(exc)}

    # -- holdout split & normalization --------------------------------------

    @staticmethod
    def split_holdout_rows(
        rows: list[dict[str, Any]],
        eval_ratio: float = _GLINER_BENCHMARK_EVAL_RATIO,
        seed: int = _GLINER_BENCHMARK_SEED,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not rows:
            return [], []
        indices = list(range(len(rows)))
        random.Random(seed).shuffle(indices)
        if len(rows) == 1:
            eval_count = 1
        else:
            eval_count = max(1, int(round(len(rows) * eval_ratio)))
            eval_count = min(eval_count, len(rows) - 1)
        eval_idx = set(indices[:eval_count])
        train_rows = [rows[i] for i in range(len(rows)) if i not in eval_idx]
        eval_rows = [rows[i] for i in range(len(rows)) if i in eval_idx]
        return train_rows, eval_rows

    @staticmethod
    def normalize_entity_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        return re.sub(r"\s+", " ", text)

    def extract_expected_entity_set(self, row: dict[str, Any]) -> set[str]:
        entities = row.get("entities") or []
        if not isinstance(entities, list):
            return set()
        normalized: set[str] = set()
        for ent in entities:
            if isinstance(ent, str):
                name = self.normalize_entity_text(ent)
            elif isinstance(ent, dict):
                name = self.normalize_entity_text(
                    ent.get("text") or ent.get("name") or ent.get("entity")
                )
            else:
                name = ""
            if name:
                normalized.add(name)
        return normalized

    @staticmethod
    def extract_predicted_entity_set(result: Any) -> set[str]:
        if not isinstance(result, dict):
            return set()
        entity_dict = result.get("entities", result)
        if not isinstance(entity_dict, dict):
            return set()
        predicted: set[str] = set()
        for items in entity_dict.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, str):
                    name = GLiNERTrainingService.normalize_entity_text(item)
                elif isinstance(item, dict):
                    name = GLiNERTrainingService.normalize_entity_text(item.get("text"))
                else:
                    name = ""
                if name:
                    predicted.add(name)
        return predicted

    @staticmethod
    def compute_prf_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
        precision = (tp / (tp + fp)) if (tp + fp) else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    def load_gliner_entity_model(self, model_ref: str) -> tuple[Any, Any]:
        from gliner2 import GLiNER2
        from memory.extractor import ENTITY_SCHEMA

        model = GLiNER2.from_pretrained(model_ref)
        schema = model.create_schema().entities(ENTITY_SCHEMA)
        return model, schema

    def evaluate_model_on_rows(
        self,
        model_ref: str,
        rows: list[dict[str, Any]],
        threshold: float = _GLINER_BENCHMARK_THRESHOLD,
    ) -> dict[str, Any]:
        if not rows:
            return {
                "ok": False,
                "error": "empty_eval_set",
                "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "counts": {"tp": 0, "fp": 0, "fn": 0},
                "rows_total": 0,
                "rows_evaluated": 0,
                "rows_failed": 0,
                "latency_ms_avg": 0.0,
            }
        try:
            model, schema = self.load_gliner_entity_model(model_ref)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"model_load_failed: {exc}",
                "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "counts": {"tp": 0, "fp": 0, "fn": 0},
                "rows_total": len(rows),
                "rows_evaluated": 0,
                "rows_failed": len(rows),
                "latency_ms_avg": 0.0,
            }

        tp = 0
        fp = 0
        fn = 0
        rows_evaluated = 0
        rows_failed = 0
        latency_sum_ms = 0.0
        failure_samples: list[dict[str, Any]] = []

        for idx, row in enumerate(rows):
            text = str(row.get("text") or "").strip()
            if not text:
                rows_failed += 1
                if len(failure_samples) < 5:
                    failure_samples.append({"row_index": idx, "error": "missing_text"})
                continue

            expected = self.extract_expected_entity_set(row)
            try:
                t0 = time.monotonic()
                result = model.extract(
                    text,
                    schema,
                    threshold=threshold,
                    include_confidence=True,
                )
                latency_sum_ms += (time.monotonic() - t0) * 1000.0
            except Exception as exc:
                rows_failed += 1
                if len(failure_samples) < 5:
                    failure_samples.append({"row_index": idx, "error": str(exc)[:300]})
                continue

            predicted = self.extract_predicted_entity_set(result)
            tp += len(predicted & expected)
            fp += len(predicted - expected)
            fn += len(expected - predicted)
            rows_evaluated += 1

        if rows_evaluated == 0:
            return {
                "ok": False,
                "error": "all_inference_failed",
                "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "counts": {"tp": tp, "fp": fp, "fn": fn},
                "rows_total": len(rows),
                "rows_evaluated": 0,
                "rows_failed": rows_failed,
                "latency_ms_avg": 0.0,
                "failure_samples": failure_samples,
            }

        metrics = self.compute_prf_metrics(tp=tp, fp=fp, fn=fn)
        return {
            "ok": rows_failed == 0,
            "error": "" if rows_failed == 0 else "partial_inference_failures",
            "metrics": metrics,
            "counts": {"tp": tp, "fp": fp, "fn": fn},
            "rows_total": len(rows),
            "rows_evaluated": rows_evaluated,
            "rows_failed": rows_failed,
            "latency_ms_avg": round(latency_sum_ms / rows_evaluated, 2),
            "failure_samples": failure_samples,
        }

    def benchmark_finetune_candidate(
        self,
        rows: list[dict[str, Any]],
        candidate_model_ref: str | None = None,
        train_count: int | None = None,
    ) -> dict[str, Any]:
        eval_rows = rows
        resolved_train_count = int(train_count or 0)
        if train_count is None:
            split_train, split_eval = self.split_holdout_rows(rows)
            eval_rows = split_eval
            resolved_train_count = len(split_train)

        if not eval_rows:
            return {
                "ok": False,
                "base_score": 0.0,
                "candidate_score": 0.0,
                "improvement": 0.0,
                "split": {"seed": _GLINER_BENCHMARK_SEED, "train_count": resolved_train_count, "eval_count": 0},
                "failure": {"reason": "no_eval_rows"},
            }

        base_model_ref = self.active_gliner_model_ref()
        candidate_model_ref = (candidate_model_ref or "").strip()
        if not candidate_model_ref:
            candidate_path = self.gliner_candidate_model_dir()
            if candidate_path.exists():
                candidate_model_ref = str(candidate_path)
        if not candidate_model_ref:
            return {
                "ok": False,
                "base_score": 0.0,
                "candidate_score": 0.0,
                "improvement": 0.0,
                "split": {
                    "seed": _GLINER_BENCHMARK_SEED,
                    "train_count": resolved_train_count,
                    "eval_count": len(eval_rows),
                },
                "failure": {"reason": "candidate_model_missing"},
            }

        base_eval = self.evaluate_model_on_rows(base_model_ref, eval_rows)
        candidate_eval = self.evaluate_model_on_rows(candidate_model_ref, eval_rows)

        base_score = float(base_eval.get("metrics", {}).get("f1", 0.0) or 0.0)
        candidate_score = float(candidate_eval.get("metrics", {}).get("f1", 0.0) or 0.0)
        benchmark_ok = bool(base_eval.get("ok")) and bool(candidate_eval.get("ok"))
        improvement = (candidate_score - base_score) if benchmark_ok else 0.0

        failure_details = []
        if not base_eval.get("ok"):
            failure_details.append(
                {"model": "base", "error": base_eval.get("error", "unknown")}
            )
        if not candidate_eval.get("ok"):
            failure_details.append(
                {"model": "candidate", "error": candidate_eval.get("error", "unknown")}
            )

        return {
            "ok": benchmark_ok,
            "split": {
                "seed": _GLINER_BENCHMARK_SEED,
                "train_count": resolved_train_count,
                "eval_count": len(eval_rows),
            },
            "base_model": base_model_ref,
            "candidate_model": candidate_model_ref or None,
            "base": base_eval,
            "candidate": candidate_eval,
            "base_score": round(base_score, 4),
            "candidate_score": round(candidate_score, 4),
            "improvement": round(improvement, 4),
            "failure": None if benchmark_ok else {"reason": "model_evaluation_failed", "details": failure_details},
        }


# ---------------------------------------------------------------------------
# Facade functions — package-level entry points for maintenance jobs
# ---------------------------------------------------------------------------

async def run_gliner_finetune_pipeline(
    engine=None,
    molly=None,
) -> dict[str, Any]:
    """Run the full GLiNER nightly cycle: accumulate, split, train, benchmark, deploy.

    Parameters
    ----------
    engine : SelfImprovementEngine | None
        Pre-initialised engine.  If *None*, one is created from *molly*.
    molly : object | None
        Molly agent instance (used to construct engine if needed).

    Returns
    -------
    dict
        Pipeline result with keys like ``status``, ``count``, ``benchmark``, etc.
    """
    if engine is None:
        from evolution.skills import SelfImprovementEngine
        engine = SelfImprovementEngine(molly=molly)
        await engine.initialize()

    return await engine.run_gliner_nightly_cycle()


async def run_gliner_accumulation(
    engine=None,
    molly=None,
    limit: int = GLINER_TRAINING_SCAN_LIMIT,
) -> dict[str, Any]:
    """Accumulate GLiNER training data without triggering fine-tune.

    Useful for nightly maintenance when you want accumulation only.
    """
    if engine is None:
        from evolution.skills import SelfImprovementEngine
        engine = SelfImprovementEngine(molly=molly)
        await engine.initialize()

    return await asyncio.to_thread(engine.gliner.accumulate_gliner_training_data, limit)


def get_gliner_stats(engine=None) -> dict[str, Any]:
    """Return current GLiNER training statistics from engine state."""
    if engine is None:
        return {
            "status": "no_engine",
            "training_examples": 0,
            "last_result": "",
            "last_cycle_status": "",
        }
    ctx = getattr(engine, "ctx", None)
    state = ctx.state if ctx is not None else {}
    return {
        "training_examples": int(state.get("gliner_training_examples", 0)),
        "last_result": str(state.get("gliner_last_result", "")),
        "last_cycle_status": str(state.get("gliner_last_cycle_status", "")),
        "last_finetune_at": str(state.get("gliner_last_finetune_at", "")),
        "last_training_strategy": str(state.get("gliner_last_training_strategy", "")),
    }
