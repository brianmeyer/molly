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

# ---------- Automated sender / noise subject patterns ----------

AUTOMATED_SENDER_PATTERNS = re.compile(
    r"""(?i)
    (?:^|\b)(?:
        no[\-_.]?reply
        | noreply
        | donotreply
        | do[\-_.]?not[\-_.]?reply
        | notifications?
        | mailer[\-_.]?daemon
        | postmaster
        | receipt[s]?
        | marketing
        | newsletter[s]?
        | promo(?:tions?)?
        | info@
        | support@
        | updates?@
        | alerts?@
        | news@
        | billing@
        | service@
        | hello@
        | rewards@
        | deals@
        | offers@
        | feedback@
        | survey@
        | confirm(?:ation)?@
        | tracking@
        | shipment@
        | delivery@
        | orders?@
        | account@
        | security@
        | team@
        | digest@
        | notify@
    )
    """,
    re.VERBOSE,
)

NOISE_SUBJECT_PATTERNS = re.compile(
    r"""(?i)
    (?:
        order\s+confirm(?:ation|ed)
        | shipping\s+(?:confirm|notif|update)
        | delivery\s+(?:confirm|notif|update)
        | (?:your\s+)?(?:package|shipment)\s+(?:has\s+)?(?:shipped|arrived|delivered|out\s+for)
        | track(?:ing)?\s+(?:number|update|your)
        | password\s+reset
        | verify\s+your\s+(?:email|account)
        | verification\s+code
        | one[\-\s]?time\s+(?:code|password|pin)
        | sign[\-\s]?in\s+(?:code|attempt)
        | login\s+(?:code|attempt|alert)
        | your\s+receipt
        | payment\s+(?:received|confirm|processed|success)
        | subscription\s+(?:confirm|renew)
        | unsubscribe
        | weekly\s+(?:digest|recap|summary|report|update)
        | daily\s+(?:summary|digest|recap|report)
        | monthly\s+(?:statement|summary|digest|recap|report|update)
        | (?:reservation|booking)\s+confirm(?:ation|ed)
        | check[\-\s]?in\s+(?:confirm|remind|now\s+available|ready|online|details)
        | hotel\s+(?:confirm|reserv|book|receipt)
        | your\s+(?:stay|trip|travel|itinerary|flight|booking)
        | (?:flight|travel)\s+(?:confirm|itinerary|update|receipt)
        | your\s+(?:statement|balance|account\s+(?:summary|update|activity|statement))
        | (?:account|portfolio)\s+(?:statement|summary|update|activity|alert)
        | (?:investment|dividend|market)\s+(?:update|summary|report|alert)
        | (?:auto[\-\s]?pay|payment)\s+(?:scheduled|processed|received|reminder)
        | your\s+(?:\w+\s+)?bill\s+is\s+(?:ready|available|due)
        | (?:smart\s+home|thermostat|sensor|device|camera)\s+(?:alert|update|report|notification|summary)
        | (?:energy|usage)\s+(?:report|summary|update|alert)
        | (?:you\s+(?:earned|saved)|points?\s+(?:earned|balance|summary))
        | (?:rewards?|loyalty|cashback)\s+(?:update|summary|earned|statement)
        | (?:sale|deal|offer|discount|coupon|save)\s+(?:alert|ends|today|now|inside|up\s+to)
        | (?:limited[\-\s]?time|flash\s+sale|clearance|exclusive\s+offer)
        | don.?t\s+miss|act\s+now|last\s+chance|hurry
        | (?:new\s+arrivals?|back\s+in\s+stock|trending|best\s+sellers?)
        | (?:ups|fedex|usps|dhl)\s+(?:delivery|shipping|tracking|your\s+package)
        | (?:your\s+)?(?:home|energy|monthly)\s+report
    )
    """,
    re.VERBOSE,
)

# ---------- Channel-specific system prompts ----------

_BASE_CLASSIFICATION_SCHEMA = (
    'Respond with JSON: '
    '{"classification":"urgent|relevant|background|noise",'
    '"score":0.0-1.0,"reason":"brief explanation"}\n\n'
    "Definitions:\n"
    "- urgent: ONLY for VIP senders or messages that explicitly require Brian to take action within 24 hours (e.g. a real person asking Brian a direct question, a deadline). Most emails are NOT urgent.\n"
    "- relevant: A real human who Brian knows personally wrote to him about something actionable. NOT automated notifications.\n"
    "- background: Informational but no action needed — very few emails qualify.\n"
    "- noise: THE DEFAULT. Automated emails, company notifications, transactional messages, marketing, newsletters, shipping updates, account alerts, financial statements, smart home alerts, travel confirmations, receipts, and anything from a company rather than a person.\n\n"
    "CRITICAL: When in doubt, classify as noise. 90%+ of emails are noise.\n"
    "If the sender is flagged as VIP or upgraded, bias toward urgent/relevant.\n"
    "If the sender is flagged as muted or frequently dismissed, classify as noise."
)

SYSTEM_PROMPT_EMAIL = (
    "You classify email messages for Brian's relevance. "
    + _BASE_CLASSIFICATION_SCHEMA + "\n\n"
    "Email-specific rules (follow strictly):\n"
    "- DEFAULT IS NOISE. Assume every email is noise unless you have a strong reason otherwise.\n"
    "- Company/brand emails = ALWAYS noise (hotels, airlines, banks, retailers, utilities, smart home, package tracking, insurance, SaaS products)\n"
    "- Newsletters, digests, marketing, promotions = ALWAYS noise\n"
    "- Transactional: receipts, confirmations, statements, alerts = ALWAYS noise\n"
    "- Travel: reservations, check-in, itineraries, booking updates = ALWAYS noise\n"
    "- Financial: statements, balance updates, investment reports, bill reminders = ALWAYS noise\n"
    "- Smart home / IoT: device alerts, energy reports, sensor notifications = ALWAYS noise\n"
    "- Shipping / delivery: tracking, delivery status, package updates = ALWAYS noise\n"
    "- Security: password resets, login alerts, verification codes = ALWAYS noise\n"
    "- Only classify as relevant/urgent if a REAL PERSON Brian knows is writing to him directly about something that requires his response or action\n"
    "- Meeting invites from a real person = relevant\n"
    "- VIP senders = urgent"
)

SYSTEM_PROMPT_IMESSAGE = (
    "You classify iMessage messages for Brian's relevance. "
    + _BASE_CLASSIFICATION_SCHEMA + "\n\n"
    "iMessage-specific guidance:\n"
    "- Direct questions to Brian = urgent\n"
    "- Group mentions of Brian or @Brian = urgent\n"
    "- Plans, events, logistics involving Brian = relevant\n"
    "- Reactions, emoji-only, tapbacks = noise\n"
    "- VIP senders = urgent"
)

SYSTEM_PROMPT_GROUP = (
    "You classify group chat messages for Brian's relevance. "
    + _BASE_CLASSIFICATION_SCHEMA + "\n\n"
    "Group chat guidance:\n"
    "- Direct mentions of Brian = urgent\n"
    "- Events, meetings, plans = relevant\n"
    "- General conversation = background\n"
    "- Memes, random chat, off-topic = noise"
)

# Backward-compat alias
SYSTEM_PROMPT = SYSTEM_PROMPT_GROUP


def _get_channel_prompt(group_name: str) -> str:
    """Select the right system prompt based on channel/group name."""
    name = (group_name or "").strip().lower()
    if name in {"email", "gmail", "mail"}:
        return SYSTEM_PROMPT_EMAIL
    if name in {"imessage", "imessages", "messages", "sms"}:
        return SYSTEM_PROMPT_IMESSAGE
    return SYSTEM_PROMPT_GROUP


def _check_prefilter(
    message: str, sender_name: str, group_name: str,
) -> "TriageResult | None":
    """Deterministic pre-filter: skip the model for obvious cases.

    Returns a TriageResult if a deterministic match is found, or None to
    fall through to the model.
    """
    sender_lower = (sender_name or "").strip().lower()

    # 1. VIP contacts from config
    for vip in config.VIP_CONTACTS:
        vip_name = (vip.get("name") or "").lower()
        vip_email = (vip.get("email") or "").lower()
        if vip_name and vip_name in sender_lower:
            log.debug("Triage pre-filter: VIP config match '%s'", sender_name)
            return TriageResult(
                classification="urgent", score=1.0,
                reason=f"VIP sender (config): {sender_name}",
            )
        if vip_email and vip_email in sender_lower:
            log.debug("Triage pre-filter: VIP email match '%s'", sender_name)
            return TriageResult(
                classification="urgent", score=1.0,
                reason=f"VIP sender (config): {sender_name}",
            )

    # 2. Sender tier from DB
    try:
        from memory.retriever import get_vectorstore
        vs = get_vectorstore()
        tier = vs.get_sender_tier(sender_lower)
        if tier == "vip":
            log.debug("Triage pre-filter: DB VIP tier '%s'", sender_name)
            return TriageResult(
                classification="urgent", score=1.0,
                reason=f"VIP sender (tier): {sender_name}",
            )
        if tier == "muted":
            log.debug("Triage pre-filter: muted tier '%s'", sender_name)
            return TriageResult(
                classification="noise", score=0.0,
                reason=f"Muted sender: {sender_name}",
            )
        if tier == "upgraded":
            log.debug("Triage pre-filter: upgraded tier '%s'", sender_name)
            return TriageResult(
                classification="relevant", score=0.8,
                reason=f"Upgraded sender: {sender_name}",
            )
        if tier == "downgraded":
            log.debug("Triage pre-filter: downgraded tier '%s'", sender_name)
            return TriageResult(
                classification="background", score=0.3,
                reason=f"Downgraded sender: {sender_name}",
            )
    except Exception:
        log.debug("Sender tier lookup failed", exc_info=True)

    # 3. Email only: automated sender regex (address patterns)
    is_email = (group_name or "").strip().lower() in {"email", "gmail", "mail"}
    if is_email and AUTOMATED_SENDER_PATTERNS.search(sender_lower):
        log.debug("Triage pre-filter: automated sender '%s'", sender_name)
        return TriageResult(
            classification="noise", score=0.1,
            reason=f"Automated sender pattern: {sender_name}",
        )

    # 4. Email only: noise subject patterns (check first 500 chars — covers From+Subject+snippet start)
    if is_email and NOISE_SUBJECT_PATTERNS.search(message[:500]):
        log.debug("Triage pre-filter: noise subject in '%s'", message[:60])
        return TriageResult(
            classification="noise", score=0.1,
            reason="Automated/transactional email subject",
        )

    return None


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


def _build_context(sender_name: str = "", group_name: str = "") -> str:
    """Build context blob for the triage prompt: who Brian is + tracked entities + signals."""
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

    # Sender tiers + dismissed sender signals
    try:
        from memory.retriever import get_vectorstore
        vs = get_vectorstore()
        signals = vs.get_triage_context_signals(limit=20)

        tier_lines = []
        for t in signals.get("sender_tiers", []):
            tier_lines.append(f"- {t['sender_pattern']}: {t['tier']}")
        if tier_lines:
            parts.append("Sender tier overrides:\n" + "\n".join(tier_lines))

        dismissed_lines = []
        for d in signals.get("dismissed_senders", []):
            dismissed_lines.append(
                f"- {d['sender_pattern']} (dismissed {d['dismissals']}x)"
            )
        if dismissed_lines:
            parts.append(
                "Frequently dismissed senders (bias toward noise):\n"
                + "\n".join(dismissed_lines)
            )
    except Exception:
        log.debug("Could not load triage context signals", exc_info=True)

    return "\n\n".join(parts) if parts else "No additional context available."


def _should_use_think(message: str) -> bool:
    """Check if a message looks event-like and warrants deeper analysis."""
    return bool(EVENT_PATTERNS.search(message))


def _build_user_prompt(
    message: str, sender_name: str, group_name: str, context: str,
) -> str:
    """Build the user message for the triage chat."""
    # Channel-aware source label
    channel = (group_name or "").strip().lower()
    if channel in {"email", "gmail", "mail"}:
        source_label = f"Email from {sender_name}"
    elif channel in {"imessage", "imessages", "messages", "sms"}:
        source_label = f"iMessage from {sender_name}"
    else:
        source_label = f"Message from {sender_name} in {group_name}"

    return (
        f"{context}\n\n"
        f"What to look for (relevant/urgent indicators):\n"
        f"- Direct mentions of Brian or people he knows\n"
        f"- Events, meetings, dates, times, locations, RSVPs\n"
        f"- Career opportunities, professional networking\n"
        f"- Topics Brian actively follows (see tracked entities)\n"
        f"- Action items or decisions that affect Brian\n\n"
        f'{source_label}:\n"{message}"'
    )


def classify_local(prompt: str, text: str) -> str:
    """Run a cheap local classification task and return the model label text."""
    model = _load_model()
    if model is None:
        return ""

    base_prompt = (prompt or "").strip()
    if not base_prompt:
        return ""

    payload = text or ""
    user_prompt = (
        base_prompt.replace("{reply}", payload).replace("{text}", payload)
    )
    if user_prompt == base_prompt and payload:
        user_prompt = f"{base_prompt}\n\n{payload}"

    messages = [
        {
            "role": "system",
            "content": "You are a local classifier. Follow instructions and respond tersely.",
        },
        {"role": "user", "content": user_prompt},
    ]

    raw = ""
    try:
        resp = model.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=16,
        )
        raw = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        log.debug("Local classify chat completion failed", exc_info=True)

    if not raw:
        try:
            resp = model.create_completion(
                prompt=f"{messages[0]['content']}\n\n{messages[1]['content']}",
                temperature=0.0,
                max_tokens=16,
            )
            raw = resp.get("choices", [{}])[0].get("text", "")
        except Exception:
            log.debug("Local classify completion fallback failed", exc_info=True)

    cleaned = (raw or "").strip()
    yn = re.search(r"\b(YES|NO)\b", cleaned, flags=re.IGNORECASE)
    if yn:
        return yn.group(1).upper()
    if not cleaned:
        return ""
    return cleaned.splitlines()[0].strip()[:64]


async def classify_local_async(prompt: str, text: str) -> str:
    """Async wrapper for local classification using the triage executor."""
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _TRIAGE_EXECUTOR,
        classify_local,
        prompt,
        text,
    )


# ---------------------------------------------------------------------------
# Local LLM time-expression parsing
# ---------------------------------------------------------------------------

def _parse_time_offset(raw: str, now_local) -> "datetime | None":
    """Parse a structured time-offset string into a UTC datetime.

    Accepted formats (case-insensitive):
        +Nm           — relative minutes from *now_local* (e.g. ``+30m``)
        +Nh           — relative hours from *now_local*   (e.g. ``+2h``)
        +NdHH:MM      — N days ahead, target local time   (e.g. ``+0d17:00``)
        +DAY HH:MM    — next occurrence of weekday        (e.g. ``+SAT10:00``)
        NONE | empty   — no time expression recognised → returns ``None``
    """
    from datetime import datetime, time, timedelta, timezone

    cleaned = (raw or "").strip()
    # Strip Qwen3 <think>...</think> reasoning blocks if present.
    cleaned = re.sub(r"(?si)<think>.*?</think>", "", cleaned).strip()
    cleaned = cleaned.upper()
    if not cleaned or cleaned == "NONE":
        return None

    # +Nm  — relative minutes
    # Handles: +30M, +15MIN, +30MINS, +15MINUTES
    # Negative lookahead (?![A-Z]) prevents matching day-name prefixes
    # like "+1MON09:00" as 1 minute.
    m = re.search(r"\+\s*(\d+)M(?:IN(?:UTES?|S)?)?(?![A-Z])", cleaned)
    if m:
        minutes = min(int(m.group(1)), 1440)  # cap at 24 hours
        # Add in UTC to avoid DST wall-clock surprises (e.g., spring-forward
        # or fall-back shifting the result by ±1 hour).
        now_utc = now_local.astimezone(timezone.utc)
        return now_utc + timedelta(minutes=minutes)

    # +Nh  — relative hours
    # Handles: +2H, +1HR, +3HRS, +1HOUR, +3HOURS
    # Negative lookahead prevents matching inside hybrid formats like "+17H:00"
    m = re.search(r"\+\s*(\d+)H(?:(?:OU)?RS?)?(?![A-Z\d:])", cleaned)
    if m:
        hours = min(int(m.group(1)), 8760)  # cap at 1 year
        # Add in UTC to avoid DST wall-clock surprises.
        now_utc = now_local.astimezone(timezone.utc)
        return now_utc + timedelta(hours=hours)

    # +NdHH:MM  — relative days + target local time
    m = re.search(r"\+\s*(\d+)D(\d{1,2}):(\d{2})", cleaned)
    if m:
        days = min(int(m.group(1)), 365)  # cap at 1 year
        hour = min(int(m.group(2)), 23)
        minute = min(int(m.group(3)), 59)
        target_date = now_local.date() + timedelta(days=days)
        tz = now_local.tzinfo
        result_local = datetime.combine(target_date, time(hour=hour, minute=minute), tzinfo=tz)
        return result_local.astimezone(timezone.utc)

    # +DAY HH:MM or DAY HH:MM  — next occurrence of weekday
    # Explicit suffix group handles all full day names:
    # MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY
    # Negative lookahead (?![A-Z]) prevents matching non-day words
    # like MONITOR or SUNLIGHT while allowing digits to follow immediately.
    m = re.search(r"\+?(MON(?:DAY)?|TUE(?:S(?:DAY)?)?|WED(?:S|NESDAY)?|THU(?:R(?:S(?:DAY)?)?)?|FRI(?:DAY)?|SAT(?:URDAY)?|SUN(?:DAY)?)(?![A-Z])\s*(\d{1,2}):(\d{2})", cleaned)
    if m:
        day_abbr = m.group(1)[:3]
        hour = min(int(m.group(2)), 23)
        minute = min(int(m.group(3)), 59)
        day_map = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
        target_day = day_map.get(day_abbr)
        if target_day is not None:
            days_ahead = (target_day - now_local.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7  # always advance to *next* occurrence
            target_date = now_local.date() + timedelta(days=days_ahead)
            tz = now_local.tzinfo
            result_local = datetime.combine(target_date, time(hour=hour, minute=minute), tzinfo=tz)
            return result_local.astimezone(timezone.utc)

    log.debug("Could not parse time offset from LLM: %s", raw[:100])
    return None


def parse_time_local(text: str, now_local) -> "datetime | None":
    """Use the local Qwen3-4B model to interpret a vague time expression.

    Returns a UTC ``datetime`` if a time expression is found, otherwise ``None``.
    This is the sync version; prefer :func:`parse_time_local_async` from async code.
    """
    model = _load_model()
    if model is None:
        return None

    day_name = now_local.strftime("%A")
    time_str = now_local.strftime("%I:%M %p")
    date_str = now_local.strftime("%Y-%m-%d")

    system_prompt = (
        "You interpret vague time expressions and return a structured offset. "
        "Respond with ONLY the offset, nothing else."
    )

    user_prompt = (
        "Given the current time, interpret when the person means.\n\n"
        "Return ONLY one of these formats:\n"
        "- +Nm (minutes from now, e.g. +30m)\n"
        "- +Nh (hours from now, e.g. +2h)\n"
        "- +NdHH:MM (days ahead + local time, e.g. +0d17:00, +1d09:00)\n"
        "- +DAYNAME HH:MM (next weekday, e.g. +SAT10:00, +MON09:00)\n"
        "- NONE (no time expression found)\n\n"
        "Examples:\n"
        "- 'in 30 minutes' -> +30m\n"
        "- 'in 15 min' -> +15m\n"
        "- 'end of day' -> +0d17:00\n"
        "- 'after lunch' -> +0d13:00\n"
        "- 'this weekend' -> +SAT10:00\n"
        "- 'in a couple hours' -> +2h\n"
        "- 'sometime next week' -> +MON09:00\n"
        "- 'before dinner' -> +0d17:00\n"
        "- 'in an hour' -> +1h\n"
        "- 'in 3 days' -> +3d09:00\n"
        "- 'day after tomorrow' -> +2d09:00\n"
        "- 'grab groceries on the way home' -> NONE\n\n"
        f"Current: {day_name} {date_str} {time_str}\n"
        f'Message: "{text[:300]}"'
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw = ""
    try:
        resp = model.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=64,
        )
        raw = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        log.debug("parse_time_local chat completion failed", exc_info=True)

    if not raw:
        try:
            resp = model.create_completion(
                prompt=f"{system_prompt}\n\n{user_prompt}",
                temperature=0.0,
                max_tokens=64,
            )
            raw = resp.get("choices", [{}])[0].get("text", "")
        except Exception:
            log.debug("parse_time_local completion fallback failed", exc_info=True)

    return _parse_time_offset(raw, now_local)


async def parse_time_local_async(text: str, now_local) -> "datetime | None":
    """Async wrapper — runs :func:`parse_time_local` on the triage executor."""
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _TRIAGE_EXECUTOR,
        parse_time_local,
        text,
        now_local,
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
    """Synchronous triage: pre-filter → build context → run local model → parse."""
    # Deterministic pre-filter — skip model for obvious cases
    prefilter = _check_prefilter(message, sender_name, group_name)
    if prefilter is not None:
        return prefilter

    model = _load_model()
    if model is None:
        return None

    context = _build_context(sender_name=sender_name, group_name=group_name)
    user_prompt = _build_user_prompt(message, sender_name, group_name, context)
    system_prompt = _get_channel_prompt(group_name)

    max_tokens = 2048 if use_think else 512
    raw = _run_model(model, user_prompt, max_tokens=max_tokens, system_prompt=system_prompt)
    if not raw:
        return TriageResult(
            classification="background", score=0.5,
            reason="Empty triage response — defaulting to background",
        )
    return _parse_response(raw)


def _run_model(
    model: object, user_prompt: str, max_tokens: int,
    system_prompt: str = "",
) -> str:
    """Run the local model and return raw text output."""
    prompt_text = system_prompt or SYSTEM_PROMPT
    kwargs = {
        "messages": [
            {"role": "system", "content": prompt_text},
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
        f"{prompt_text}\n\n"
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
