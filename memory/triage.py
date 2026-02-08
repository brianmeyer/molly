"""Local triage model for intelligent message filtering.

Runs Qwen3-4B GGUF in-process via llama-cpp-python to classify group messages
into: urgent, relevant, background, or noise.
"""

import json
import logging
import os
import re
from contextlib import redirect_stderr
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

import config

log = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover - import failure handled at runtime
    Llama = None

# Dedicated executor for triage — isolates model calls from the default
# executor (which Neo4j and other async tasks may share). Single worker ensures
# sequential model inference.
_TRIAGE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="triage")

_TRIAGE_MODEL = None
_MODEL_LOCK = Lock()
_MODEL_LOAD_FAILED = False

# Patterns that suggest event-like content worth deeper analysis
EVENT_PATTERNS = re.compile(
    r"""
    \b(?:
        \d{1,2}[:/]\d{2}           # times: 3:00, 15:30
        | \d{1,2}\s*(?:am|pm)\b    # 3pm, 10 AM
        | (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d  # month + day
        | (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)  # day names
        | (?:tomorrow|tonight|next\s+week|this\s+weekend)
        | (?:rsvp|deadline|due\s+date|meeting|event|invite|reservation)
        | (?:location|address|venue|zoom|teams\s+link)
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

SYSTEM_PROMPT = (
    "You classify group chat messages for Brian's relevance. "
    "Respond with JSON: "
    '{"classification":"urgent|relevant|background|noise",'
    '"score":0.0-1.0,"reason":"brief explanation"}\n\n'
    "Definitions:\n"
    "- urgent: Directly mentions Brian, requires his immediate attention, or is time-sensitive for him\n"
    "- relevant: Mentions tracked entities, topics Brian follows, or people he knows\n"
    "- background: General conversation that might be useful context but needs no action\n"
    "- noise: Completely irrelevant to Brian (random chat, memes, off-topic)"
)


@dataclass
class TriageResult:
    classification: str  # urgent, relevant, background, noise
    score: float  # 0.0-1.0 relevance
    reason: str  # brief explanation


def _load_model() -> object | None:
    """Load the local GGUF model once and cache it."""
    global _TRIAGE_MODEL, _MODEL_LOAD_FAILED

    if _TRIAGE_MODEL is not None:
        return _TRIAGE_MODEL
    if _MODEL_LOAD_FAILED:
        return None

    with _MODEL_LOCK:
        if _TRIAGE_MODEL is not None:
            return _TRIAGE_MODEL
        if _MODEL_LOAD_FAILED:
            return None

        model_path = config.TRIAGE_MODEL_PATH.expanduser()

        if Llama is None:
            log.warning("llama-cpp-python is not installed — triage unavailable")
            _MODEL_LOAD_FAILED = True
            return None

        if not model_path.exists():
            log.warning("Triage model file not found: %s", model_path)
            return None

        try:
            # Keep llama.cpp Metal kernel fallback logs quiet on Apple Silicon.
            os.environ.setdefault("GGML_METAL_LOG_LEVEL", "0")
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with redirect_stderr(devnull):
                    _TRIAGE_MODEL = Llama(
                        model_path=str(model_path),
                        n_ctx=config.TRIAGE_N_CTX,
                        n_threads=config.TRIAGE_N_THREADS,
                        n_gpu_layers=config.TRIAGE_GPU_LAYERS,
                        verbose=False,
                    )
            log.info("Loaded triage model: %s", model_path)
            return _TRIAGE_MODEL
        except Exception:
            log.error("Failed to load triage model from %s", model_path, exc_info=True)
            _MODEL_LOAD_FAILED = True
            return None


def preload_model() -> bool:
    """Preload the triage model at startup."""
    return _load_model() is not None


def _build_context() -> str:
    """Build context blob for the triage prompt: who Brian is + tracked entities."""
    parts = []

    # Load USER.md summary
    user_file = config.WORKSPACE / "USER.md"
    if user_file.exists():
        content = user_file.read_text().strip()
        if content:
            parts.append(f"About Brian:\n{content}")

    # Top entities from knowledge graph
    try:
        from memory.graph import get_top_entities
        entities = get_top_entities(config.TRIAGE_CONTEXT_ENTITIES)
        if entities:
            entity_lines = []
            for e in entities:
                entity_lines.append(
                    f"- {e['name']} ({e.get('type', '?')}, "
                    f"strength: {e.get('strength', 0):.1f})"
                )
            parts.append("Entities Brian tracks:\n" + "\n".join(entity_lines))
    except Exception:
        log.debug("Could not load graph entities for triage context", exc_info=True)

    return "\n\n".join(parts) if parts else "No additional context available."


def _should_use_think(message: str) -> bool:
    """Check if a message looks event-like and warrants deeper analysis."""
    return bool(EVENT_PATTERNS.search(message))


def _build_user_prompt(
    message: str, sender_name: str, group_name: str, context: str,
) -> str:
    """Build the user message for the triage chat."""
    return (
        f"{context}\n\n"
        f"What to look for (relevant/urgent indicators):\n"
        f"- Direct mentions of Brian or people he knows\n"
        f"- Events, meetings, dates, times, locations, RSVPs\n"
        f"- Career opportunities, professional networking\n"
        f"- Topics Brian actively follows (see tracked entities)\n"
        f"- Action items or decisions that affect Brian\n\n"
        f'Message from {sender_name} in {group_name}:\n"{message}"'
    )


async def triage_message(
    message: str,
    sender_name: str = "Unknown",
    group_name: str = "Unknown Group",
) -> TriageResult | None:
    """Run a message through the local triage model.

    Returns TriageResult or None if triage is unavailable.
    """
    if not message.strip():
        return TriageResult(classification="noise", score=0.0, reason="Empty message")

    # Very short messages are almost always noise
    if len(message.strip()) < 5:
        return TriageResult(classification="noise", score=0.0, reason="Too short")

    use_think = _should_use_think(message)

    # Run ALL blocking work (Neo4j context query + model inference) in a
    # dedicated executor thread. This prevents two issues:
    # 1. Neo4j sync I/O blocking the event loop
    # 2. Default executor contention causing subsequent triage calls to hang
    import asyncio
    loop = asyncio.get_running_loop()

    try:
        result = await loop.run_in_executor(
            _TRIAGE_EXECUTOR,
            _sync_triage, message, sender_name, group_name, use_think,
        )
        return result
    except Exception:
        log.error("Triage failed", exc_info=True)
        return None


def _sync_triage(
    message: str, sender_name: str, group_name: str, use_think: bool,
) -> TriageResult | None:
    """Synchronous triage: build context + run local model + parse response."""
    model = _load_model()
    if model is None:
        return None

    context = _build_context()
    user_prompt = _build_user_prompt(message, sender_name, group_name, context)

    max_tokens = 2048 if use_think else 512
    raw = _run_model(model, user_prompt, max_tokens=max_tokens)
    if not raw:
        return TriageResult(
            classification="background", score=0.5,
            reason="Empty triage response — defaulting to background",
        )
    return _parse_response(raw)


def _run_model(model: object, user_prompt: str, max_tokens: int) -> str:
    """Run the local model and return raw text output."""
    kwargs = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }

    # First try strict JSON output via chat completion.
    try:
        resp = model.create_chat_completion(
            **kwargs,
            response_format={"type": "json_object"},
        )
        return resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    except TypeError:
        # Older llama-cpp builds may not support response_format.
        pass
    except Exception:
        log.debug("Triage chat completion (json mode) failed", exc_info=True)

    # Fallback: regular chat completion.
    try:
        resp = model.create_chat_completion(**kwargs)
        return resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        log.debug("Triage chat completion failed", exc_info=True)

    # Last fallback: plain completion prompt.
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{user_prompt}\n\n"
        "Return JSON only."
    )
    try:
        resp = model.create_completion(
            prompt=prompt,
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return resp.get("choices", [{}])[0].get("text", "")
    except Exception:
        log.debug("Triage completion fallback failed", exc_info=True)
        return ""


def _parse_response(raw: str) -> TriageResult:
    """Parse the JSON response from Qwen3, with fallback for malformed output."""
    if not raw or not raw.strip():
        return TriageResult(
            classification="background", score=0.5,
            reason="Empty triage response — defaulting to background",
        )

    try:
        data = json.loads(raw)
        classification = data.get("classification", "background").lower()
        if classification not in ("urgent", "relevant", "background", "noise"):
            classification = "background"
        score = float(data.get("score", 0.5))
        score = max(0.0, min(1.0, score))
        reason = data.get("reason", "")
        return TriageResult(
            classification=classification,
            score=score,
            reason=reason,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: try to extract JSON from mixed output
    json_match = re.search(r"\{[^}]+\}", raw)
    if json_match:
        try:
            data = json.loads(json_match.group())
            classification = data.get("classification", "background").lower()
            if classification not in ("urgent", "relevant", "background", "noise"):
                classification = "background"
            return TriageResult(
                classification=classification,
                score=float(data.get("score", 0.5)),
                reason=data.get("reason", ""),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    log.debug("Could not parse triage response: %s", raw[:200])
    return TriageResult(
        classification="background", score=0.5,
        reason="Unparseable triage response — defaulting to background",
    )
