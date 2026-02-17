"""Local triage model for intelligent message filtering.

Runs Qwen3-4B GGUF in-process via llama-cpp-python to classify group messages
into: urgent, relevant, background, or noise.
"""

import hashlib
import json
import logging
import os
import re
from contextlib import redirect_stderr
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from threading import Lock
from zoneinfo import ZoneInfo

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

_SUBJECT_LINE_RE = re.compile(r"(?im)^subject:\s*(.+)$")
_HEADER_LINE_RE = re.compile(r"(?im)^(?:from|to|date|cc|bcc):")
_LOCATION_LINE_RE = re.compile(r"(?im)^(?:location|where|venue|address)\s*:\s*(.+)$")
_LOCATION_INLINE_RE = re.compile(r"(?i)\b(?:at|@)\s+([a-z0-9][^\n]{2,80})")
_ONLY_TIME_RE = re.compile(r"^\d{1,2}(?::\d{2})?\s*(?:am|pm)?$", re.IGNORECASE)

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
    '"score":0.0-1.0,"reason":"brief explanation",'
    '"calendar_event":null}\n\n'
    "Definitions:\n"
    "- urgent: ONLY for VIP senders or messages that explicitly require Brian to take action within 24 hours (e.g. a real person asking Brian a direct question, a deadline). Most emails are NOT urgent.\n"
    "- relevant: A real human who Brian knows personally wrote to him about something actionable. NOT automated notifications.\n"
    "- background: Informational but no action needed — very few emails qualify.\n"
    "- noise: THE DEFAULT. Automated emails, company notifications, transactional messages, marketing, newsletters, shipping updates, account alerts, financial statements, smart home alerts, travel confirmations, receipts, and anything from a company rather than a person.\n\n"
    "CRITICAL: When in doubt, classify as noise. 90%+ of emails are noise.\n"
    "If the sender is flagged as VIP or upgraded, bias toward urgent/relevant.\n"
    "If the sender is flagged as muted or frequently dismissed, classify as noise.\n\n"
    "CALENDAR EVENT DETECTION (applies to ALL classifications including noise):\n"
    "If the email contains a schedulable event (flight confirmation, event invite/RSVP, "
    "restaurant reservation, doctor/dentist appointment, meeting with specific date+time, "
    "concert/show tickets, hotel check-in), set calendar_event to:\n"
    '{"title":"short event title","start":"ISO datetime","end":"ISO datetime or null",'
    '"location":"venue/address or null","notes":"key details (confirmation codes, seat numbers, etc.)"}\n'
    "If no calendar event is detected, set calendar_event to null.\n"
    "An email can be noise AND still have a calendar_event — these are independent."
)

SYSTEM_PROMPT_EMAIL = (
    "You classify email messages for Brian's relevance. "
    + _BASE_CLASSIFICATION_SCHEMA + "\n\n"
    "Email-specific rules (follow strictly):\n"
    "- DEFAULT IS NOISE. Assume every email is noise unless you have a strong reason otherwise.\n"
    "- Company/brand emails = ALWAYS noise (hotels, airlines, banks, retailers, utilities, smart home, package tracking, insurance, SaaS products)\n"
    "- Newsletters, digests, marketing, promotions = ALWAYS noise\n"
    "- Transactional: receipts, confirmations, statements, alerts = ALWAYS noise\n"
    "- Travel: reservations, check-in, itineraries, booking updates = noise for classification, BUT still extract calendar_event if it has a flight/hotel with date+time\n"
    "- Financial: statements, balance updates, investment reports, bill reminders = ALWAYS noise\n"
    "- Smart home / IoT: device alerts, energy reports, sensor notifications = ALWAYS noise\n"
    "- Shipping / delivery: tracking, delivery status, package updates = ALWAYS noise\n"
    "- Security: password resets, login alerts, verification codes = ALWAYS noise\n"
    "- If an email from a known school (Guidepost, Montessori), childcare provider, doctor's office, or personal service contains dates, events, deadlines, or RSVPs, classify as relevant even if it looks like a newsletter. Brian's child's school events are personally relevant.\n"
    "- If EVENT_PATTERNS-like signals are present in the body and the sender is not muted, bias toward relevant (not noise).\n"
    "- Only classify as relevant/urgent if a REAL PERSON Brian knows is writing to him directly about something that requires his response or action\n"
    "- Meeting invites from a real person = relevant\n"
    "- VIP senders = urgent\n"
    "- Event invites (Evite, Eventbrite, Paperless Post, etc.) = noise for classification, BUT always extract calendar_event with full details\n"
    "- Flight confirmations (AA, United, Delta, Southwest, etc.) = noise for classification, BUT always extract calendar_event with flight number, airports, times"
)

SYSTEM_PROMPT_IMESSAGE = (
    "You classify iMessage messages for Brian's relevance. "
    + _BASE_CLASSIFICATION_SCHEMA + "\n\n"
    "iMessage-specific guidance:\n"
    "- Direct questions to Brian = urgent\n"
    "- Group mentions of Brian or @Brian = urgent\n"
    "- Plans, events, logistics involving Brian = relevant\n"
    "- If the message contains plans, events, dates, or meetups relevant to Brian, classify as relevant and mention calendar_event in the reason.\n"
    "- Reactions, emoji-only, tapbacks = noise\n"
    "- VIP senders = urgent\n"
    "- If a message contains event details (date + time + activity/location), extract calendar_event\n"
    "  Examples: 'Birthday party Saturday at 2pm at Sky Zone', 'Dinner Friday 7pm at Doya',\n"
    "  'Can you pick her up at 3:30?', 'Flight lands at 9am Sunday'\n"
    "- Include the sender's name in the event title when relevant (e.g. 'Dinner with Danielle')"
)

SYSTEM_PROMPT_GROUP = (
    "You classify group chat messages for Brian's relevance. "
    + _BASE_CLASSIFICATION_SCHEMA + "\n\n"
    "Group chat guidance:\n"
    "- Direct mentions of Brian = urgent\n"
    "- Events, meetings, plans = relevant\n"
    "- If the message contains plans, events, dates, or meetups relevant to Brian, classify as relevant and mention calendar_event in the reason.\n"
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

    # 1. Muted/downgraded sender tier check FIRST (overrides VIP wildcards)
    #    e.g. muting "mkt.databricks.com" takes priority over VIP "@databricks.com"
    #    Uses both exact match AND substring scan so domain-level mutes work.
    tier = None
    try:
        from memory.retriever import get_vectorstore
        vs = get_vectorstore()
        # Exact match first
        tier = vs.get_sender_tier(sender_lower)
        if tier in ("muted", "downgraded"):
            log.debug("Triage pre-filter: %s tier (exact) '%s'", tier, sender_name)
            if tier == "muted":
                return TriageResult(
                    classification="noise", score=0.0,
                    reason=f"Muted sender: {sender_name}",
                )
            return TriageResult(
                classification="background", score=0.3,
                reason=f"Downgraded sender: {sender_name}",
            )
        # Substring scan: check all muted/downgraded patterns against sender
        all_tiers = vs.get_sender_tiers()
        for t in all_tiers:
            pat = t["sender_pattern"]
            t_tier = t["tier"]
            if t_tier not in ("muted", "downgraded"):
                continue
            if pat in sender_lower:
                log.debug("Triage pre-filter: %s tier (substring '%s') '%s'",
                          t_tier, pat, sender_name)
                if t_tier == "muted":
                    return TriageResult(
                        classification="noise", score=0.0,
                        reason=f"Muted sender (pattern {pat}): {sender_name}",
                    )
                return TriageResult(
                    classification="background", score=0.3,
                    reason=f"Downgraded sender (pattern {pat}): {sender_name}",
                )
    except Exception:
        log.debug("Sender tier lookup (muted/downgraded) failed", exc_info=True)
        tier = None

    # 2. VIP contacts from config
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

    # 3. Remaining sender tiers from DB (vip, upgraded)
    try:
        if tier is None:
            from memory.retriever import get_vectorstore
            vs = get_vectorstore()
            tier = vs.get_sender_tier(sender_lower)
        if tier == "vip":
            log.debug("Triage pre-filter: DB VIP tier '%s'", sender_name)
            return TriageResult(
                classification="urgent", score=1.0,
                reason=f"VIP sender (tier): {sender_name}",
            )
        if tier == "upgraded":
            log.debug("Triage pre-filter: upgraded tier '%s'", sender_name)
            return TriageResult(
                classification="relevant", score=0.8,
                reason=f"Upgraded sender: {sender_name}",
            )
    except Exception:
        log.debug("Sender tier lookup failed", exc_info=True)

    # 4. Email only: automated sender regex (address patterns)
    is_email = (group_name or "").strip().lower() in {"email", "gmail", "mail"}
    if is_email and AUTOMATED_SENDER_PATTERNS.search(sender_lower):
        log.debug("Triage pre-filter: automated sender '%s'", sender_name)
        return TriageResult(
            classification="noise", score=0.1,
            reason=f"Automated sender pattern: {sender_name}",
        )

    # 5. Email only: noise subject patterns (check first 500 chars — covers From+Subject+snippet start)
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
    calendar_event: dict | None = None  # extracted event: {title, start, end, location, notes}


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


def _build_context(
    sender_name: str = "", group_name: str = "", chat_jid: str = "",
) -> str:
    """Build context blob for the triage prompt: who Brian is + tracked entities + signals.

    Enhanced (PLAN-16): also injects recent thread history, sender→Brian relationship
    from Neo4j, and sender message frequency for richer classification context.
    """
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

    # --- PLAN-16: Sender relationship to Brian from knowledge graph ---
    if sender_name and sender_name not in ("Unknown", ""):
        try:
            from memory.graph import query_entity
            entity_info = query_entity(sender_name)
            if entity_info and entity_info.get("relationships"):
                rel_lines = []
                for rel in entity_info["relationships"]:
                    rel_type = rel.get("type", "RELATED_TO")
                    if rel.get("direction") == "outgoing":
                        rel_lines.append(
                            f"- {sender_name} {rel_type} {rel.get('target', '?')}"
                        )
                    else:
                        rel_lines.append(
                            f"- {rel.get('source', '?')} {rel_type} {sender_name}"
                        )
                if rel_lines:
                    parts.append(
                        f"Brian's relationship with {sender_name}:\n"
                        + "\n".join(rel_lines[:5])  # cap at 5 to save tokens
                    )
        except Exception:
            log.debug("Could not load sender relationship for triage", exc_info=True)

    # --- PLAN-16: Recent thread messages for conversation context ---
    if chat_jid:
        try:
            import sqlite3 as _sql
            db_path = config.DATABASE_PATH
            with _sql.connect(str(db_path), timeout=5) as conn:
                conn.row_factory = _sql.Row
                rows = conn.execute(
                    """SELECT sender_name, content
                       FROM messages
                       WHERE chat_jid = ?
                       ORDER BY timestamp DESC
                       LIMIT 5""",
                    (chat_jid,),
                ).fetchall()
            if rows:
                thread_lines = []
                for r in reversed(rows):
                    name = r["sender_name"] or "Unknown"
                    snippet = (r["content"] or "")[:120]
                    thread_lines.append(f"  {name}: {snippet}")
                parts.append(
                    "Recent messages in this thread:\n" + "\n".join(thread_lines)
                )
        except Exception:
            log.debug("Could not load thread history for triage", exc_info=True)

    # --- PLAN-16: Sender message frequency (last 7 days) ---
    if sender_name and sender_name not in ("Unknown", "") and chat_jid:
        try:
            import sqlite3 as _sql
            from datetime import datetime, timedelta, timezone
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            db_path = config.DATABASE_PATH
            with _sql.connect(str(db_path)) as conn:
                row = conn.execute(
                    """SELECT COUNT(*) FROM messages
                       WHERE sender_name = ? AND timestamp > ?""",
                    (sender_name, cutoff),
                ).fetchone()
            msg_count = row[0] if row else 0
            if msg_count > 0:
                freq_label = (
                    "very active" if msg_count > 50
                    else "active" if msg_count > 20
                    else "moderate" if msg_count > 5
                    else "infrequent"
                )
                parts.append(
                    f"Sender frequency (last 7 days): {sender_name} sent "
                    f"{msg_count} messages ({freq_label})"
                )
        except Exception:
            log.debug("Could not load sender frequency for triage", exc_info=True)

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
    return await asyncio.wait_for(
        loop.run_in_executor(
            _TRIAGE_EXECUTOR,
            classify_local,
            prompt,
            text,
        ),
        timeout=20.0,
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


async def parse_time_local_async(text: str, now_local=None) -> "datetime | None":
    """Async wrapper — runs :func:`parse_time_local` on the triage executor."""
    import asyncio

    if now_local is None:
        now_local = datetime.now(ZoneInfo(config.TIMEZONE))

    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(
            _TRIAGE_EXECUTOR,
            parse_time_local,
            text,
            now_local,
        ),
        timeout=20.0,
    )


def _clean_event_text(value: str, limit: int = 140) -> str:
    text = " ".join(str(value or "").split())
    text = text.strip(" -,:;")
    if len(text) > limit:
        text = text[:limit].rstrip(" ,.;:-")
    return text


def _extract_event_title_heuristic(
    message_text: str,
    sender_name: str = "",
    channel: str = "",
) -> str:
    subject_match = _SUBJECT_LINE_RE.search(message_text or "")
    if subject_match:
        title = _clean_event_text(subject_match.group(1))
        if title:
            return title

    for raw_line in (message_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _HEADER_LINE_RE.match(line):
            continue
        line = re.sub(r"^[>\-*]+\s*", "", line)
        title = _clean_event_text(line)
        if len(title) >= 4:
            return title

    sender = _clean_event_text(sender_name, limit=60)
    if sender and sender.lower() != "unknown":
        return f"Event with {sender}"

    channel_name = _clean_event_text(channel, limit=40)
    if channel_name:
        return f"{channel_name} event"
    return "Calendar event"


def _extract_event_location_heuristic(message_text: str) -> str | None:
    content = message_text or ""

    for match in _LOCATION_LINE_RE.finditer(content):
        candidate = _clean_event_text(match.group(1), limit=120)
        if candidate and not _ONLY_TIME_RE.fullmatch(candidate):
            return candidate

    inline = _LOCATION_INLINE_RE.search(content)
    if inline:
        candidate = re.split(r"[\n,.;]", inline.group(1), maxsplit=1)[0]
        candidate = _clean_event_text(candidate, limit=120)
        if candidate and not _ONLY_TIME_RE.fullmatch(candidate):
            return candidate

    return None


def _normalize_event_title(title: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", (title or "").lower())
    return " ".join(normalized.split())


def calendar_event_dedup_key(title: str, when_utc: datetime) -> tuple[str, str]:
    normalized = _normalize_event_title(title)
    title_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    if when_utc.tzinfo is None:
        when_utc = when_utc.replace(tzinfo=timezone.utc)
    else:
        when_utc = when_utc.astimezone(timezone.utc)
    local_day = when_utc.astimezone(ZoneInfo(config.TIMEZONE)).date().isoformat()
    return (title_hash, local_day)


def mark_calendar_event_seen(
    event: dict,
    cycle_seen: set[tuple[str, str]] | None,
) -> tuple[str, str] | None:
    if cycle_seen is None:
        return None
    title = _clean_event_text(str(event.get("title") or ""))
    event_dt = event.get("datetime")
    if not title or not isinstance(event_dt, datetime):
        return None
    key = calendar_event_dedup_key(title, event_dt)
    cycle_seen.add(key)
    return key


def _decode_tool_payload(tool_response) -> object:
    if not isinstance(tool_response, dict):
        return None

    content = tool_response.get("content")
    if not isinstance(content, list):
        return None

    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        raw_text = str(part.get("text", "")).strip()
        if not raw_text:
            continue
        try:
            return json.loads(raw_text)
        except Exception:
            return raw_text
    return None


def _calendar_search_events(search_result) -> list[dict]:
    payload = _decode_tool_payload(search_result)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _parse_calendar_time_utc(raw_when) -> "datetime | None":
    if isinstance(raw_when, dict):
        raw_when = raw_when.get("dateTime") or raw_when.get("date")
    if not raw_when:
        return None
    when_text = str(raw_when).strip()
    if not when_text:
        return None

    try:
        parsed = datetime.fromisoformat(when_text.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(config.TIMEZONE))
    return parsed.astimezone(timezone.utc)


def _titles_similar(left: str, right: str) -> bool:
    left_norm = _normalize_event_title(left)
    right_norm = _normalize_event_title(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if left_norm in right_norm or right_norm in left_norm:
        return True

    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    if left_tokens and right_tokens:
        overlap = left_tokens.intersection(right_tokens)
        min_size = min(len(left_tokens), len(right_tokens))
        if min_size > 0 and (len(overlap) / min_size) >= 0.6:
            return True

    return SequenceMatcher(None, left_norm, right_norm).ratio() >= 0.78


async def extract_calendar_event_async(
    message_text: str,
    sender_name: str = "",
    channel: str = "",
) -> dict | None:
    """Extract a calendar event from event-like message text.

    Caller should invoke this only after triage marks the message relevant/urgent.
    Returns {"title": str, "datetime": datetime, "location": str | None} or None.
    """
    text = (message_text or "").strip()
    if not text or not EVENT_PATTERNS.search(text):
        return None

    now_local = datetime.now(ZoneInfo(config.TIMEZONE))
    event_dt = await parse_time_local_async(text, now_local)
    if event_dt is None:
        return None

    if event_dt.tzinfo is None:
        event_dt = event_dt.replace(tzinfo=ZoneInfo(config.TIMEZONE))
    event_dt = event_dt.astimezone(timezone.utc)

    title = _extract_event_title_heuristic(
        text, sender_name=sender_name, channel=channel,
    )
    if not title:
        return None

    return {
        "title": title,
        "datetime": event_dt,
        "location": _extract_event_location_heuristic(text),
    }


async def calendar_event_is_duplicate_async(
    event: dict,
    cycle_seen: set[tuple[str, str]] | None = None,
    window_minutes: int = 30,
) -> bool:
    """Check for duplicate calendar events via in-memory + calendar_search.

    Returns True when a duplicate is known or when dedup safety checks fail.
    """
    title = _clean_event_text(str(event.get("title") or ""))
    event_dt = event.get("datetime")
    if not title or not isinstance(event_dt, datetime):
        return False

    if event_dt.tzinfo is None:
        event_dt = event_dt.replace(tzinfo=timezone.utc)
    event_dt = event_dt.astimezone(timezone.utc)
    dedup_key = calendar_event_dedup_key(title, event_dt)

    if cycle_seen is not None and dedup_key in cycle_seen:
        return True

    try:
        from tools.calendar import calendar_search

        search_result = await calendar_search({"query": title})
    except Exception:
        log.warning("Calendar dedup search failed for '%s'", title, exc_info=True)
        return True

    if isinstance(search_result, dict) and search_result.get("is_error"):
        log.warning("Calendar dedup search returned an error for '%s'", title)
        return True

    window_seconds = max(1, int(window_minutes)) * 60
    for candidate in _calendar_search_events(search_result):
        existing_title = _clean_event_text(str(candidate.get("summary") or ""))
        if not existing_title or not _titles_similar(existing_title, title):
            continue

        existing_start = _parse_calendar_time_utc(candidate.get("start"))
        if existing_start is None:
            continue

        if abs((existing_start - event_dt).total_seconds()) <= window_seconds:
            if cycle_seen is not None:
                cycle_seen.add(dedup_key)
            return True

    return False


async def triage_message(
    message: str,
    sender_name: str = "Unknown",
    group_name: str = "Unknown Group",
    chat_jid: str = "",
) -> TriageResult | None:
    """Run a message through the local triage model.

    Returns TriageResult or None if triage is unavailable.
    chat_jid: Optional WhatsApp/iMessage chat ID for thread context (PLAN-16).
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
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _TRIAGE_EXECUTOR,
                _sync_triage, message, sender_name, group_name, use_think, chat_jid,
            ),
            timeout=45.0,
        )
        return result
    except Exception:
        log.error("Triage failed", exc_info=True)
        return None


def _sync_triage(
    message: str, sender_name: str, group_name: str, use_think: bool,
    chat_jid: str = "",
) -> TriageResult | None:
    """Synchronous triage: pre-filter → build context → run local model → parse."""
    # Deterministic pre-filter — skip model for obvious cases
    prefilter = _check_prefilter(message, sender_name, group_name)
    if prefilter is not None:
        return prefilter

    model = _load_model()
    if model is None:
        return None

    context = _build_context(
        sender_name=sender_name, group_name=group_name, chat_jid=chat_jid,
    )
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


def _get_logit_bias(model: object) -> dict[str, float] | None:
    """Build logit_bias dict constraining the classification token to one of 4 categories.

    Returns None if tokenization is unavailable; callers should fall back to
    unconstrained generation.
    """
    try:
        tokenize = getattr(model, "tokenize", None)
        if tokenize is None:
            return None

        bias: dict[str, float] = {}
        for label in ("urgent", "relevant", "background", "noise"):
            # Tokenize each label and boost the first token.
            # llama-cpp-python tokenize() returns list[int].
            tokens = tokenize(label.encode("utf-8"), add_bos=False)
            if tokens:
                bias[str(tokens[0])] = 10.0
        return bias if bias else None
    except Exception:
        log.debug("Failed to build logit_bias for triage", exc_info=True)
        return None


def _run_model(
    model: object, user_prompt: str, max_tokens: int,
    system_prompt: str = "",
) -> str:
    """Run the local model and return raw text output.

    Uses logit bias when available to constrain the classification token
    to one of the 4 valid categories (urgent/relevant/background/noise),
    improving speed and reliability.
    """
    prompt_text = system_prompt or SYSTEM_PROMPT
    kwargs: dict = {
        "messages": [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }

    # Build logit bias to constrain classification token
    logit_bias = _get_logit_bias(model)
    if logit_bias:
        kwargs["logit_bias"] = logit_bias

    # First try strict JSON output via chat completion.
    try:
        resp = model.create_chat_completion(
            **kwargs,
            response_format={"type": "json_object"},
        )
        return resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    except TypeError:
        # Older llama-cpp builds may not support response_format or logit_bias.
        # Retry without logit_bias.
        kwargs.pop("logit_bias", None)
        try:
            resp = model.create_chat_completion(
                **kwargs,
                response_format={"type": "json_object"},
            )
            return resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        except TypeError:
            pass
        except Exception:
            log.debug("Triage chat completion (json mode, no bias) failed", exc_info=True)
    except Exception:
        log.debug("Triage chat completion (json mode) failed", exc_info=True)

    # Fallback: regular chat completion (no logit_bias — already tried).
    kwargs.pop("logit_bias", None)
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
        cal_event = data.get("calendar_event")
        if cal_event is not None and not isinstance(cal_event, dict):
            cal_event = None
        return TriageResult(
            classification=classification,
            score=score,
            reason=reason,
            calendar_event=cal_event,
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
            cal_event = data.get("calendar_event")
            if cal_event is not None and not isinstance(cal_event, dict):
                cal_event = None
            return TriageResult(
                classification=classification,
                score=float(data.get("score", 0.5)),
                reason=data.get("reason", ""),
                calendar_event=cal_event,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    log.debug("Could not parse triage response: %s", raw[:200])
    return TriageResult(
        classification="background", score=0.5,
        reason="Unparseable triage response — defaulting to background",
    )


# ---------- Triage Training Data Accumulation (Task 4.18) ----------

_TRIAGE_LABELS_PATH = config.WORKSPACE / "training" / "triage_labels.jsonl"


def log_triage_label(
    *,
    sender: str,
    message_snippet: str = "",
    group_name: str = "",
    model_classification: str = "",
    model_score: float = 0.0,
    corrected_classification: str = "",
    feedback_source: str = "manual",
) -> bool:
    """Log a labeled triage example for future fine-tuning.

    Called when:
    - ``/upgrade``, ``/downgrade``, ``/mute`` commands override triage output
    - A user dismisses a proactive notification (implicit feedback)
    - Triage completes with high confidence (self-labeled)

    Returns True if the label was written successfully.
    """
    try:
        _TRIAGE_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sender": sender[:200],
            "message_snippet": message_snippet[:500],
            "group_name": group_name[:100],
            "model_classification": model_classification,
            "model_score": round(model_score, 3),
            "corrected_classification": corrected_classification,
            "feedback_source": feedback_source,
        }
        with open(_TRIAGE_LABELS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception:
        log.debug("Failed to log triage label", exc_info=True)
        return False


def get_triage_training_stats() -> dict:
    """Return basic statistics about accumulated triage training data."""
    if not _TRIAGE_LABELS_PATH.exists():
        return {"total": 0, "by_source": {}, "by_corrected": {}}
    try:
        by_source: dict[str, int] = {}
        by_corrected: dict[str, int] = {}
        total = 0
        with open(_TRIAGE_LABELS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    total += 1
                    src = rec.get("feedback_source", "unknown")
                    by_source[src] = by_source.get(src, 0) + 1
                    corr = rec.get("corrected_classification", "")
                    if corr:
                        by_corrected[corr] = by_corrected.get(corr, 0) + 1
                except json.JSONDecodeError:
                    continue
        return {"total": total, "by_source": by_source, "by_corrected": by_corrected}
    except Exception:
        return {"total": 0, "by_source": {}, "by_corrected": {}}
