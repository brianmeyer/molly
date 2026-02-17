"""Orchestrator — triage + decomposition via Kimi K2.5.

Routes incoming messages through a single Kimi K2.5 call that classifies
the message AND decomposes complex tasks into subtasks.  Workers then
execute subtasks in parallel.

Fallback chain: Kimi K2.5 → Gemini 2.5 Flash-Lite → Qwen3-4B local → hardcoded "general"

Key design decisions:
  - Uses httpx directly (matches existing Kimi patterns in tools/kimi.py)
  - NO response_format parameter (Moonshot doesn't support it — uses prompt-based JSON)
  - Thinking mode disabled for triage (speed over depth — extra_body not needed, just lower temp)
  - Every call logged to SQLite for observability
  - Feature-flagged: ORCHESTRATOR_ENABLED = True by default
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config

log = logging.getLogger(__name__)

# Thread-safety locks for global state
import threading
_DB_PATH_LOCK = threading.Lock()
_TABLES_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Subtask:
    """A single decomposed subtask for a worker."""
    id: str
    profile: str  # Worker profile name (calendar, email, research, etc.)
    description: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class TriageResult:
    """Result of orchestrator triage + decomposition."""
    classification: str  # "direct" | "simple" | "complex"
    confidence: float  # 0.0–1.0
    subtasks: list[Subtask] = field(default_factory=list)
    model_used: str = ""
    latency_ms: float = 0.0
    fallback_reason: str = ""


# ---------------------------------------------------------------------------
# Orchestrator prompt
# ---------------------------------------------------------------------------

ORCHESTRATOR_PROMPT = """\
You are a message router for an AI assistant named Molly. Classify the user \
message and, if complex, decompose it into subtasks.

Respond ONLY with a JSON object (no markdown, no explanation). Schema:

{
  "type": "direct" | "simple" | "complex",
  "confidence": 0.0 to 1.0,
  "subtasks": [
    {
      "id": "t1",
      "profile": "<worker_profile>",
      "description": "<what to do>",
      "depends_on": []
    }
  ]
}

Classification rules:
- "direct": Greetings, chitchat, simple Q&A, acknowledgments. No tools needed. \
subtasks = [].
- "simple": Single-domain tasks (check calendar, send email, create task). \
1 subtask.
- "complex": Multi-step or cross-domain tasks (schedule meeting + email attendees, \
research + summarize + create task). 2+ subtasks.

Available worker profiles:
  calendar  — Google Calendar operations (create/read/update/delete events)
  email     — Gmail operations (send/draft/reply/search)
  contacts  — Google Contacts lookup
  tasks     — Google Tasks operations
  research  — Web search, deep research, knowledge queries
  writer    — Draft text, compose messages, format documents
  files     — Read/search local files
  imessage  — Send/read iMessages
  browser   — Web page interaction (navigate, click, extract)
  general   — Catch-all for anything that doesn't fit above

Rules for subtasks:
- Use depends_on to express ordering (e.g. "t2" depends_on ["t1"] means t2 runs after t1)
- Independent subtasks run in parallel
- Each subtask description should be self-contained (worker doesn't see other subtasks)
- Maximum 5 subtasks per message
- For "direct" and "simple", confidence should be >= 0.8
"""

# ---------------------------------------------------------------------------
# SQLite observability
# ---------------------------------------------------------------------------

_DB_PATH: Path | None = None


def _get_db_path() -> Path:
    """Thread-safe lazy init of orchestrator DB path."""
    global _DB_PATH
    if _DB_PATH is not None:
        return _DB_PATH
    with _DB_PATH_LOCK:
        if _DB_PATH is None:
            _DB_PATH = getattr(config, "WORKSPACE", Path.home() / ".molly") / "store" / "orchestrator.db"
            _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _DB_PATH


def _ensure_tables() -> None:
    """Create the orchestrator log table if it doesn't exist."""
    path = _get_db_path()
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orchestrator_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                model TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                classification TEXT NOT NULL,
                subtask_count INTEGER NOT NULL,
                confidence REAL NOT NULL,
                fallback_reason TEXT DEFAULT '',
                message_preview TEXT DEFAULT '',
                raw_response TEXT DEFAULT ''
            )
        """)
        conn.commit()
    finally:
        conn.close()


_tables_ensured = False


def _log_triage(result: TriageResult, message_preview: str, raw_response: str = "") -> None:
    """Log a triage result to SQLite (thread-safe)."""
    global _tables_ensured
    if not _tables_ensured:
        with _TABLES_LOCK:
            if not _tables_ensured:
                _ensure_tables()
                _tables_ensured = True
    conn = None
    try:
        conn = sqlite3.connect(str(_get_db_path()))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """INSERT INTO orchestrator_log
               (timestamp, model, latency_ms, classification, subtask_count,
                confidence, fallback_reason, message_preview, raw_response)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                result.model_used,
                result.latency_ms,
                result.classification,
                len(result.subtasks),
                result.confidence,
                result.fallback_reason,
                message_preview[:200],
                raw_response[:2000],
            ),
        )
        conn.commit()
    except Exception:
        log.debug("Failed to log triage result", exc_info=True)
    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------
# JSON extraction from response text
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from response text.

    Handles common LLM quirks: markdown code blocks, leading text, etc.
    """
    if not text:
        return None

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip())

    # Try to parse the whole thing first
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find a JSON object in the text
    brace_depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}" and brace_depth > 0:
            brace_depth -= 1
            if brace_depth == 0 and start >= 0:
                try:
                    data = json.loads(cleaned[start:i + 1])
                    if isinstance(data, dict):
                        return data
                except (json.JSONDecodeError, ValueError):
                    start = -1

    return None


def _parse_triage_response(text: str) -> TriageResult | None:
    """Parse a triage JSON response into a TriageResult."""
    data = _extract_json(text)
    if not data:
        return None

    classification = str(data.get("type", "general")).lower()
    if classification not in ("direct", "simple", "complex"):
        classification = "simple"

    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))

    subtasks = []
    for st in data.get("subtasks", []):
        if not isinstance(st, dict):
            continue
        subtasks.append(Subtask(
            id=str(st.get("id", f"t{len(subtasks) + 1}")),
            profile=str(st.get("profile", "general")),
            description=str(st.get("description", "")),
            depends_on=(
                st["depends_on"] if isinstance(st.get("depends_on"), list)
                else [st["depends_on"]] if isinstance(st.get("depends_on"), str)
                else list(st.get("depends_on", []))
            ),
        ))

    # Cap subtasks at 5
    subtasks = subtasks[:5]

    return TriageResult(
        classification=classification,
        confidence=confidence,
        subtasks=subtasks,
    )


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------

async def _call_kimi(message: str, timeout: float = 15.0) -> tuple[str, str]:
    """Call Kimi K2.5 for triage. Returns (response_text, model_string).

    Uses the established httpx pattern from tools/kimi.py.
    NO response_format parameter — Moonshot doesn't support it.
    """
    import httpx

    api_key = config.MOONSHOT_API_KEY
    if not api_key:
        raise RuntimeError("MOONSHOT_API_KEY not configured")

    model = getattr(config, "KIMI_TRIAGE_MODEL", "kimi-k2.5")
    base_url = config.MOONSHOT_BASE_URL

    # Kimi K2.5 enforces fixed parameter values:
    #   thinking enabled: temperature=1.0, top_p=0.95
    #   thinking disabled: temperature=0.6, top_p=0.95
    # Any other values will ERROR. We disable thinking for triage (speed)
    # which locks temperature at 0.6.
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user", "content": message},
        ],
        "thinking": {"type": "disabled"},  # Instant mode for fast triage
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_exc: Exception | None = None
    for attempt in range(2):  # 2 attempts for triage (speed matters)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(content, list):
                content = "\n".join(str(chunk) for chunk in content)
            return str(content).strip(), model

        except (httpx.TimeoutException, httpx.ConnectTimeout) as exc:
            last_exc = exc
            if attempt == 0:
                await asyncio.sleep(2)
                continue
            raise
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (429, 502, 503) and attempt == 0:
                last_exc = exc
                await asyncio.sleep(3)
                continue
            raise

    raise last_exc or RuntimeError("Kimi triage failed")


async def _call_gemini(message: str, timeout: float = 12.0) -> tuple[str, str]:
    """Fallback: call Gemini 2.5 Flash-Lite for triage.

    Includes 2-attempt retry (matching Kimi's retry logic).
    """
    import httpx

    api_key = config.GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not configured")

    model = getattr(config, "GEMINI_TRIAGE_FALLBACK", "gemini-2.5-flash-lite")
    url = f"{config.GEMINI_BASE_URL}/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {"parts": [{"text": f"{ORCHESTRATOR_PROMPT}\n\n{message}"}]},
        ],
        "generationConfig": {
            "temperature": 0.2,
        },
    }

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = parts[0].get("text", "") if parts else ""
            return str(text).strip(), model

        except (httpx.TimeoutException, httpx.ConnectTimeout) as exc:
            last_exc = exc
            if attempt == 0:
                await asyncio.sleep(2)
                continue
            raise
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (429, 502, 503) and attempt == 0:
                last_exc = exc
                await asyncio.sleep(3)
                continue
            raise

    raise last_exc or RuntimeError("Gemini triage failed")


async def _call_qwen_local(message: str) -> tuple[str, str]:
    """Fallback: call local Qwen3-4B for triage classification.

    Uses the existing triage model loaded via llama-cpp-python.
    Falls back to hardcoded if model not loaded.
    """
    try:
        from memory.triage import _load_model
        model = _load_model()
        if model is None:
            raise RuntimeError("Qwen3 not loaded")

        prompt = f"""<|im_start|>system
{ORCHESTRATOR_PROMPT}<|im_end|>
<|im_start|>user
{message}<|im_end|>
<|im_start|>assistant
"""
        loop = asyncio.get_running_loop()
        output = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: model(prompt, max_tokens=512, temperature=0.2, stop=["<|im_end|>"]),
            ),
            timeout=15.0,
        )
        text = output.get("choices", [{}])[0].get("text", "")
        return str(text).strip(), "qwen3-4b-local"

    except Exception:
        raise


def _hardcoded_classification(message: str) -> TriageResult:
    """Last-resort: simple regex-based classification."""
    msg_lower = message.lower().strip()

    # Greetings / chitchat
    greetings = {"hi", "hello", "hey", "good morning", "good afternoon",
                 "good evening", "thanks", "thank you", "ok", "okay",
                 "bye", "goodbye", "gm", "gn"}
    if msg_lower in greetings or len(msg_lower) < 10:
        return TriageResult(
            classification="direct",
            confidence=0.6,
            subtasks=[],
            model_used="hardcoded",
            fallback_reason="all models unavailable",
        )

    # Simple: contains a single obvious domain keyword
    domain_map = {
        "calendar": ["calendar", "schedule", "meeting", "event", "appointment"],
        "email": ["email", "mail", "send email", "draft", "inbox"],
        "tasks": ["task", "todo", "to-do", "reminder"],
        "research": ["search", "look up", "find out", "research", "what is"],
        "imessage": ["imessage", "text message", "sms"],
    }
    matched_profiles = []
    for profile, keywords in domain_map.items():
        if any(kw in msg_lower for kw in keywords):
            matched_profiles.append(profile)

    if len(matched_profiles) == 1:
        return TriageResult(
            classification="simple",
            confidence=0.5,
            subtasks=[Subtask(
                id="t1",
                profile=matched_profiles[0],
                description=message,
            )],
            model_used="hardcoded",
            fallback_reason="all models unavailable",
        )

    if len(matched_profiles) > 1:
        subtasks = [
            Subtask(id=f"t{i+1}", profile=p, description=message)
            for i, p in enumerate(matched_profiles[:5])
        ]
        return TriageResult(
            classification="complex",
            confidence=0.4,
            subtasks=subtasks,
            model_used="hardcoded",
            fallback_reason="all models unavailable",
        )

    # Default: single general worker
    return TriageResult(
        classification="simple",
        confidence=0.4,
        subtasks=[Subtask(id="t1", profile="general", description=message)],
        model_used="hardcoded",
        fallback_reason="all models unavailable",
    )


# ---------------------------------------------------------------------------
# Main classify entry point
# ---------------------------------------------------------------------------

async def classify_message(message: str) -> TriageResult:
    """Classify and decompose a message through the orchestrator fallback chain.

    Chain: Kimi K2.5 → Gemini 2.5 Flash-Lite → Qwen3-4B local → hardcoded

    Returns a TriageResult with classification, confidence, and subtasks.
    Guaranteed to return a result — never raises.
    """
    if not getattr(config, "ORCHESTRATOR_ENABLED", False):
        return TriageResult(
            classification="simple",
            confidence=1.0,
            subtasks=[Subtask(id="t1", profile="general", description=message)],
            model_used="disabled",
            fallback_reason="orchestrator disabled",
        )

    start = time.monotonic()
    message_preview = message[:200]

    # Attempt 1: Kimi K2.5
    try:
        timeout = float(getattr(config, "ORCHESTRATOR_TIMEOUT", 15))
        text, model = await _call_kimi(message, timeout=timeout)
        result = _parse_triage_response(text)
        if result:
            elapsed = (time.monotonic() - start) * 1000
            result.model_used = model
            result.latency_ms = elapsed
            _log_triage(result, message_preview, raw_response=text)
            log.info(
                "Orchestrator: %s (%.0fms, %s, confidence=%.2f, subtasks=%d)",
                result.classification, elapsed, model,
                result.confidence, len(result.subtasks),
            )
            return result
        log.warning("Kimi returned unparseable triage response: %s", text[:200])
    except Exception as exc:
        log.warning("Kimi triage failed: %s", exc)

    # Attempt 2: Gemini 2.5 Flash-Lite
    try:
        text, model = await _call_gemini(message)
        result = _parse_triage_response(text)
        if result:
            elapsed = (time.monotonic() - start) * 1000
            result.model_used = model
            result.latency_ms = elapsed
            result.fallback_reason = "kimi unavailable"
            _log_triage(result, message_preview, raw_response=text)
            log.info(
                "Orchestrator fallback (Gemini): %s (%.0fms, subtasks=%d)",
                result.classification, elapsed, len(result.subtasks),
            )
            return result
        log.warning("Gemini returned unparseable triage response: %s", text[:200])
    except Exception as exc:
        log.warning("Gemini triage fallback failed: %s", exc)

    # Attempt 3: Qwen3-4B local
    try:
        text, model = await _call_qwen_local(message)
        result = _parse_triage_response(text)
        if result:
            elapsed = (time.monotonic() - start) * 1000
            result.model_used = model
            result.latency_ms = elapsed
            result.fallback_reason = "kimi+gemini unavailable"
            _log_triage(result, message_preview, raw_response=text)
            log.info(
                "Orchestrator fallback (Qwen local): %s (%.0fms, subtasks=%d)",
                result.classification, elapsed, len(result.subtasks),
            )
            return result
        log.warning("Qwen local returned unparseable triage response: %s", text[:200])
    except Exception as exc:
        log.warning("Qwen local triage fallback failed: %s", exc)

    # Attempt 4: Hardcoded regex
    result = _hardcoded_classification(message)
    elapsed = (time.monotonic() - start) * 1000
    result.latency_ms = elapsed
    _log_triage(result, message_preview)
    log.info(
        "Orchestrator fallback (hardcoded): %s (%.0fms, subtasks=%d)",
        result.classification, elapsed, len(result.subtasks),
    )
    return result


# ---------------------------------------------------------------------------
# Stats / introspection
# ---------------------------------------------------------------------------

def get_orchestrator_stats(hours: int = 24) -> dict:
    """Return orchestrator usage statistics."""
    try:
        _ensure_tables()
        conn = sqlite3.connect(str(_get_db_path()))
        conn.row_factory = sqlite3.Row
        try:
            cutoff = time.time() - (hours * 3600)

            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM orchestrator_log WHERE timestamp > ?",
                (cutoff,),
            ).fetchone()

            by_model = conn.execute(
                "SELECT model, COUNT(*) as cnt FROM orchestrator_log "
                "WHERE timestamp > ? GROUP BY model ORDER BY cnt DESC",
                (cutoff,),
            ).fetchall()

            by_class = conn.execute(
                "SELECT classification, COUNT(*) as cnt FROM orchestrator_log "
                "WHERE timestamp > ? GROUP BY classification ORDER BY cnt DESC",
                (cutoff,),
            ).fetchall()

            avg_latency = conn.execute(
                "SELECT AVG(latency_ms) as avg_ms FROM orchestrator_log "
                "WHERE timestamp > ?",
                (cutoff,),
            ).fetchone()

            fallback_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM orchestrator_log "
                "WHERE timestamp > ? AND fallback_reason != ''",
                (cutoff,),
            ).fetchone()

            return {
                "total_calls": total["cnt"] if total else 0,
                "by_model": {r["model"]: r["cnt"] for r in by_model},
                "by_classification": {r["classification"]: r["cnt"] for r in by_class},
                "avg_latency_ms": round(avg_latency["avg_ms"] or 0, 1) if avg_latency else 0,
                "fallback_count": fallback_count["cnt"] if fallback_count else 0,
                "enabled": getattr(config, "ORCHESTRATOR_ENABLED", False),
                "hours": hours,
            }
        finally:
            conn.close()
    except Exception:
        return {"error": "failed to read stats", "enabled": getattr(config, "ORCHESTRATOR_ENABLED", False)}
