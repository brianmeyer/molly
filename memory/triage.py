"""Local triage model for intelligent message filtering.

Calls Qwen3-4B via Ollama's chat API to classify group messages
into: urgent, relevant, background, or noise.

Uses Ollama's format=json for reliable structured output.
Qwen3 thinks internally (separate token stream) then outputs clean JSON.
"""

import json
import logging
import re
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import config

log = logging.getLogger(__name__)

# Dedicated executor for triage — isolates Ollama HTTP calls from the default
# executor (which Neo4j and other async tasks may share). Single worker ensures
# sequential model inference (Qwen3 can only run one request at a time anyway).
_TRIAGE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="triage")

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

    # Run ALL blocking work (Neo4j context query + Ollama HTTP call) in a
    # dedicated executor thread. This prevents two issues:
    # 1. Neo4j sync I/O blocking the event loop
    # 2. Default executor contention causing subsequent Ollama calls to hang
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
    """Synchronous triage: build context + call Ollama + parse response.

    Runs entirely in the dedicated triage executor thread.
    """
    context = _build_context()
    user_prompt = _build_user_prompt(message, sender_name, group_name, context)

    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    payload = json.dumps({
        "model": config.TRIAGE_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "think": use_think,
        "options": {
            "temperature": 0.1,
            "num_predict": 2048 if use_think else 512,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Content-Type": "application/json",
            "Connection": "close",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=config.TRIAGE_TIMEOUT) as resp:
            data = json.loads(resp.read())
            raw = data.get("message", {}).get("content", "")
    except urllib.error.URLError:
        log.debug("Ollama not reachable — triage unavailable")
        return None
    except Exception:
        log.debug("Ollama call failed", exc_info=True)
        return None

    return _parse_response(raw)


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
