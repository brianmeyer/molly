"""Pre/post execution hooks for the evolution engine.

Called by ``main.py`` around every message handling cycle:

- ``pre_execution_hook()`` — select bandit arm, retrieve memories, inject guidelines,
  query graph context
- ``post_execution_hook()`` — compute reward, update bandit, log trajectory,
  store experience with embeddings, track guideline activation

**SAFETY:** Both hooks are wrapped in try/except → no-op on any
evolution.db failure.  They NEVER block message handling.
"""
from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)


def _format_memory_context(
    memories: list,
    guidelines: list[str],
    graph_context: str = "",
    arm_id: str = "baseline",
) -> str:
    """Format memories, guidelines, and graph context for injection into agent context.

    Each bandit arm varies the context budget:
      baseline:        3 memories, 5 guidelines, 500 chars graph (default)
      concise_prompt:  0 memories, 2 guidelines, no graph
      thorough_prompt: 3 memories, 5 guidelines, 500 chars graph + instruction prefix
      memory_heavy:    5 memories, 7 guidelines, 500 chars graph
      graph_context:   3 memories, 5 guidelines, 1000 chars graph

    Returns a formatted string suitable for appending to the system prompt.
    Returns empty string if no relevant context available.
    """
    # Arm-specific budgets
    arm_config = {
        "baseline":        {"mem": 3, "guide": 5, "graph": 500,  "prefix": ""},
        "concise_prompt":  {"mem": 0, "guide": 2, "graph": 0,    "prefix": ""},
        "thorough_prompt": {"mem": 3, "guide": 5, "graph": 500,  "prefix": "Be thorough and detailed in your response.\n\n"},
        "memory_heavy":    {"mem": 5, "guide": 7, "graph": 500,  "prefix": ""},
        "graph_context":   {"mem": 3, "guide": 5, "graph": 1000, "prefix": ""},
    }
    cfg = arm_config.get(arm_id, arm_config["baseline"])

    parts: list[str] = []

    if cfg["prefix"]:
        parts.append(cfg["prefix"])

    if guidelines and cfg["guide"] > 0:
        parts.append("## Applicable Guidelines")
        for g in guidelines[:cfg["guide"]]:
            parts.append(f"- {g}")

    if memories and cfg["mem"] > 0:
        parts.append("\n## Relevant Past Experiences")
        for mem in memories[:cfg["mem"]]:
            tc = getattr(mem, "task_class", "") or ""
            reward = getattr(mem, "reward", 0.0) or 0.0
            content = getattr(mem, "content", {}) or {}
            summary = content.get("outcome", {}).get("summary", "") if isinstance(content, dict) else ""
            line = f"- {tc} (reward={reward:.2f})"
            if summary:
                line += f": {summary[:120]}"
            parts.append(line)

    if graph_context and cfg["graph"] > 0:
        parts.append(f"\n## Entity Context\n{graph_context[:cfg['graph']]}")

    return "\n".join(parts) if parts else ""


def pre_execution_hook(
    task_hash: str = "",
    task_class: str = "",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select bandit arm and inject relevant context before execution.

    Returns a context dict with keys:
    - ``arm_id``: selected bandit arm (empty string if unavailable)
    - ``memories``: list of relevant past experiences
    - ``guidelines``: list of applicable IF-THEN rules
    - ``graph_context``: entity/relationship context from Neo4j
    - ``context_injection``: formatted text for system prompt injection
    - ``start_time``: high-resolution timer for latency tracking
    - ``task_hash``: echo back for post-hook

    NEVER raises — returns empty context on any error.
    """
    result: dict[str, Any] = {
        "arm_id": "",
        "memories": [],
        "guidelines": [],
        "graph_context": "",
        "context_injection": "",
        "start_time": time.time(),
        "task_hash": task_hash,
    }

    try:
        from evolution.bandit import BANDIT_ARMS, ThompsonBandit, register_default_arms
        from evolution.db import ensure_schema

        ensure_schema()
        register_default_arms()
        bandit = ThompsonBandit()
        arm = bandit.select_arm(arm_ids=BANDIT_ARMS)
        result["arm_id"] = arm
    except Exception:
        log.debug("pre_execution_hook: bandit selection failed", exc_info=True)

    # Retrieve similar past experiences from episodic memory
    try:
        from evolution.memory import retrieve_similar

        memories = retrieve_similar(task_class=task_class)
        result["memories"] = memories
    except Exception:
        log.debug("pre_execution_hook: memory retrieval failed", exc_info=True)

    # Retrieve applicable IF-THEN guidelines
    guidelines: list[str] = []
    try:
        from evolution.memory import get_relevant_guidelines

        if task_class:
            guidelines = get_relevant_guidelines(task_class)
            result["guidelines"] = guidelines
    except Exception:
        log.debug("pre_execution_hook: guideline retrieval failed", exc_info=True)

    # Query Neo4j graph for entity context relevant to this task
    graph_context = ""
    try:
        ctx = context or {}
        msg_text = ctx.get("message", "") or ctx.get("chat_jid", "")
        if msg_text:
            graph_context = _query_graph_context(msg_text[:500])
            result["graph_context"] = graph_context
    except Exception:
        log.debug("pre_execution_hook: graph context query failed", exc_info=True)

    # Build formatted context injection string
    result["context_injection"] = _format_memory_context(
        result["memories"], guidelines, graph_context,
        arm_id=result["arm_id"],
    )

    return result


def _query_graph_context(text: str) -> str:
    """Query Neo4j for entities mentioned in the text and return formatted context."""
    try:
        from memory.extractor import extract
        extraction = extract(text, entities_only=True)
        entity_names = [e.get("text", "") for e in extraction.get("entities", []) if e.get("text")]
        if not entity_names:
            return ""

        from memory.graph import get_driver
        driver = get_driver()
        parts: list[str] = []
        with driver.session() as session:
            for name in entity_names[:5]:
                rows = session.run(
                    """
                    MATCH (e:Entity {name: $name})-[r]->(t:Entity)
                    RETURN e.name AS source, type(r) AS rel, t.name AS target
                    LIMIT 5
                    """,
                    name=name,
                ).data()
                for row in rows:
                    parts.append(f"{row['source']} → {row['rel']} → {row['target']}")
        return "; ".join(parts) if parts else ""
    except Exception:
        return ""


async def post_execution_hook(
    task_hash: str = "",
    task_class: str = "",
    arm_id: str = "",
    start_time: float = 0.0,
    outcome: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute reward, update bandit, log trajectory, store experience.

    *outcome* should contain:
    - ``success``: bool
    - ``tool_calls``: int
    - ``error``: optional str
    - ``safety_flags``: optional list[str]
    - ``tokens_used``: optional int

    NEVER raises — returns empty result on any error.
    """
    result: dict[str, Any] = {
        "reward": 0.0,
        "bandit_updated": False,
        "trajectory_logged": False,
        "experience_stored": False,
    }

    if outcome is None:
        outcome = {}

    elapsed = time.time() - start_time if start_time else 0.0

    # Compute reward
    reward_detail: dict = {}
    try:
        from evolution.rewards import compute_reward

        reward_detail = compute_reward(
            outcome=1.0 if outcome.get("success", False) else 0.0,
            latency_s=elapsed,
            tokens=outcome.get("tokens_used", 0),
            tools=outcome.get("tool_calls", 0),
            safety=0.0 if outcome.get("safety_flags") else 1.0,
            arm_id=arm_id,
        )
        result["reward"] = reward_detail.get("reward", 0.0)
    except Exception:
        log.debug("post_execution_hook: reward computation failed", exc_info=True)

    # Update bandit
    try:
        if arm_id:
            from evolution.bandit import ThompsonBandit

            bandit = ThompsonBandit()
            bandit.update_arm(arm_id, reward=result["reward"])
            result["bandit_updated"] = True
    except Exception:
        log.debug("post_execution_hook: bandit update failed", exc_info=True)

    # Log trajectory with full reward breakdown
    try:
        from evolution.trajectory import log_trajectory

        log_trajectory(
            arm_id=arm_id,
            task_hash=task_hash,
            action=task_class or "message",
            reward=result["reward"],
            outcome_score=reward_detail.get("outcome_score", 0.0),
            process_score=reward_detail.get("process_score", 0.0),
            safety_score=reward_detail.get("safety_score", 0.0),
            cost_penalty=reward_detail.get("cost_penalty", 0.0),
            diversity_bonus=reward_detail.get("diversity_bonus", 0.0),
            latency_seconds=elapsed,
            tokens_used=outcome.get("tokens_used", 0),
            tools_used=outcome.get("tool_calls", 0),
        )
        result["trajectory_logged"] = True
    except Exception:
        log.debug("post_execution_hook: trajectory logging failed", exc_info=True)

    # Store experience WITH embedding for semantic search
    try:
        from evolution.memory import store_experience

        # Generate embedding from task content for semantic retrieval
        embedding = b""
        try:
            task_summary = f"{task_class}: {outcome.get('summary', '')}"
            embedding = _get_embedding(task_summary)
        except Exception:
            pass

        exp_id = store_experience(
            task_hash=task_hash,
            task_class=task_class,
            reward=result["reward"],
            confidence=1.0 if outcome.get("success") else 0.3,
            content={
                "arm_id": arm_id,
                "outcome": outcome,
                "latency_seconds": elapsed,
            },
            embedding=embedding,
        )
        result["experience_stored"] = bool(exp_id)
    except Exception:
        log.debug("post_execution_hook: experience storage failed", exc_info=True)

    # Track guideline activation counts for feedback loop
    try:
        ctx = context or {}
        used_guidelines = ctx.get("guidelines", [])
        if used_guidelines:
            _increment_guideline_activations(used_guidelines)
    except Exception:
        log.debug("post_execution_hook: guideline activation tracking failed", exc_info=True)

    return result


def _get_embedding(text: str) -> bytes:
    """Generate embedding bytes for semantic search.

    Uses the vectorstore's embedding function if available,
    otherwise returns empty bytes.
    """
    if not text or not text.strip():
        return b""
    try:
        from memory.retriever import get_vectorstore
        vs = get_vectorstore()
        if hasattr(vs, "embed_text"):
            return vs.embed_text(text)
    except Exception:
        pass
    return b""


def _increment_guideline_activations(guidelines: list) -> None:
    """Increment activation_count for used guidelines.

    This closes the feedback loop: guidelines that get activated more often
    are surfaced more prominently by ``get_relevant_guidelines()``.
    """
    try:
        from evolution.db import get_connection
        conn = get_connection()
        try:
            for g in guidelines:
                if isinstance(g, str) and g:
                    conn.execute(
                        "UPDATE guidelines SET activation_count = activation_count + 1 "
                        "WHERE rule_text = ?",
                        (g,),
                    )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        log.debug("Guideline activation increment failed", exc_info=True)
