"""Structured mutation operators + trajectory recombination (SE-Agent style).

Seven mutation operators with a structured vocabulary.  The bandit selects
which operator to apply, and the operator generates the payload.
"""
from __future__ import annotations

import logging
import random
from typing import Any

from evolution._base import Mutation

log = logging.getLogger(__name__)

# Operator registry: operator_name → (area, description_template)
OPERATORS: dict[str, tuple[str, str]] = {
    "change_system_prompt": ("prompts", "Tweak system prompt phrasing"),
    "add_few_shot": ("prompts", "Inject successful example into prompt"),
    "adjust_temperature": ("inference", "Raise/lower temperature ±0.1"),
    "inject_memory": ("context", "Add relevant past experience to context"),
    "inject_rag": ("context", "Add RAG-retrieved context"),
    "refactor_tool": ("code", "Restructure an existing tool function"),
    "recombine_trajectories": ("reasoning", "Combine two successful reasoning traces"),
}


def select_mutation(
    eligible_operators: list[str] | None = None,
    context: dict | None = None,
) -> Mutation:
    """Select a mutation operator and generate payload.

    If *eligible_operators* is provided, only those are considered.
    Otherwise all operators are eligible.
    """
    candidates = eligible_operators or list(OPERATORS.keys())
    candidates = [c for c in candidates if c in OPERATORS]
    if not candidates:
        candidates = list(OPERATORS.keys())

    op = random.choice(candidates)
    area, desc_template = OPERATORS[op]

    payload: dict[str, Any] = {"operator": op}

    # Operator-specific payload generation
    if op == "adjust_temperature":
        delta = random.choice([-0.1, 0.1])
        payload["temperature_delta"] = delta
        desc = f"Adjust temperature by {delta:+.1f}"
    elif op == "add_few_shot":
        payload["source"] = "trajectory"
        desc = "Inject high-reward example as few-shot"
    elif op == "change_system_prompt":
        payload["strategy"] = random.choice(["conciseness", "thoroughness", "safety"])
        desc = f"Tweak system prompt for {payload['strategy']}"
    elif op == "inject_memory":
        payload["k"] = 3
        desc = "Inject top-3 similar experiences into context"
    elif op == "inject_rag":
        payload["source"] = "knowledge_graph"
        desc = "Add RAG-retrieved context from knowledge graph"
    elif op == "refactor_tool":
        payload["target"] = context.get("weakest_tool", "unknown") if context else "unknown"
        desc = f"Refactor tool: {payload['target']}"
    elif op == "recombine_trajectories":
        payload["min_reward"] = 0.7
        desc = "Combine two high-reward reasoning traces"
    else:
        desc = desc_template

    mutation = Mutation(operator=op, area=area, description=desc, payload=payload)
    log.debug("Selected mutation: %s (%s)", op, area)
    return mutation


async def recombine_trajectories(
    trace_a: dict,
    trace_b: dict,
    judge_fn=None,
) -> dict:
    """SE-Agent style: combine two successful reasoning traces into a hybrid.

    *judge_fn* is an async callable that takes a recombination prompt and
    returns the hybrid strategy.  When None, returns a simple merge.
    """
    hybrid = {
        "source_a": trace_a.get("task_hash", ""),
        "source_b": trace_b.get("task_hash", ""),
        "combined_tools": list(set(
            (trace_a.get("tools", []) or []) + (trace_b.get("tools", []) or [])
        )),
        "strategy": "hybrid",
    }

    if judge_fn is not None:
        prompt = (
            f"Combine the efficiency of Trace A with the thoroughness of Trace B.\n"
            f"Trace A: {trace_a}\n"
            f"Trace B: {trace_b}\n"
            f"Produce a hybrid strategy."
        )
        try:
            result = await judge_fn(prompt)
            hybrid["judge_recommendation"] = result
        except Exception:
            log.warning("Judge recombination failed", exc_info=True)

    return hybrid
