"""Email digest queue — captures triaged emails for periodic digest delivery.

Heartbeat's _check_email() appends non-noise triage results here.
Automation slots (morning/noon/afternoon/evening) call build_digest()
to produce a WhatsApp-friendly summary and advance the high-water mark.

NOTE: Imported lazily by heartbeat.py and automations.py to avoid circular imports.
Do not add top-level imports from those modules.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
from utils import atomic_write_json, load_json

log = logging.getLogger(__name__)

QUEUE_DIR = config.EMAIL_DIGEST_QUEUE_DIR

_STATE_KEY = "email_digest_consumed_ts_ms"
_SLOTS = ("morning", "noon", "afternoon", "evening")
_SLOT_LABELS = {
    "morning": "Morning",
    "noon": "Noon",
    "afternoon": "Afternoon",
    "evening": "Evening",
}


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def append_digest_item(
    msg_id: str,
    sender: str,
    subject: str,
    snippet: str,
    classification: str,
    score: float,
    reason: str,
    internal_ts_ms: int,
) -> None:
    """Append one triaged email to today's JSONL queue file."""
    try:
        QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = QUEUE_DIR / f"{today}.jsonl"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "msg_id": msg_id,
            "sender": sender,
            "subject": subject,
            "snippet": (snippet or "")[:300],
            "classification": classification,
            "score": round(score, 3),
            "reason": (reason or "")[:200],
            "internal_ts_ms": internal_ts_ms,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")
    except OSError:
        log.debug("Failed to write email digest item", exc_info=True)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_queue_items(date_str: str | None = None) -> list[dict]:
    """Read items from a JSONL queue file for the given date (default: today)."""
    target_date = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = QUEUE_DIR / f"{target_date}.jsonl"

    if not path.exists():
        return []

    entries: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if isinstance(entry, dict):
                    entries.append(entry)
            except json.JSONDecodeError:
                continue
    except OSError:
        log.debug("Failed to read digest queue file: %s", path, exc_info=True)

    return entries


# ---------------------------------------------------------------------------
# Digest builder
# ---------------------------------------------------------------------------

def _get_consumed_ts(slot: str) -> int:
    """Read the last-consumed timestamp for a slot from state.json."""
    try:
        state = load_json(config.STATE_FILE, {})
        return int(state.get(_STATE_KEY, {}).get(slot, 0))
    except Exception:
        log.debug("Failed to read consumed ts for slot %s", slot, exc_info=True)
    return 0


def _set_consumed_ts(slot: str, ts_ms: int) -> None:
    """Update the high-water timestamp for a slot in state.json."""
    try:
        state = load_json(config.STATE_FILE, {})
        consumed = state.get(_STATE_KEY, {})
        if not isinstance(consumed, dict):
            consumed = {}
        consumed[slot] = ts_ms
        state[_STATE_KEY] = consumed
        atomic_write_json(config.STATE_FILE, state)
    except OSError:
        log.debug("Failed to update consumed ts for slot %s", slot, exc_info=True)


def _sender_display(sender: str) -> str:
    """Extract a display name from an email sender string."""
    # "Jane Doe <jane@example.com>" -> "Jane Doe"
    if "<" in sender:
        name = sender.split("<")[0].strip().strip('"').strip("'")
        if name:
            return name
    return sender


def build_digest(slot: str) -> str:
    """Build a WhatsApp-friendly email digest for the given slot.

    Reads today's + yesterday's JSONL, filters items newer than the
    last-consumed timestamp for this slot, formats the summary, and
    advances the high-water mark.

    Returns "NO_DIGEST_ITEMS" if there are no items to report.
    """
    if slot not in _SLOTS:
        log.warning("Invalid digest slot %r, falling back to 'morning'", slot)
        slot = "morning"

    consumed_ts = _get_consumed_ts(slot)

    # Read today + yesterday to handle midnight boundary
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    all_items = get_queue_items(yesterday) + get_queue_items(today)

    # Filter to items newer than last consumed
    new_items = [
        item for item in all_items
        if item.get("internal_ts_ms", 0) > consumed_ts
    ]

    if not new_items:
        return "NO_DIGEST_ITEMS"

    # Separate urgent (already notified) from new items
    already_notified = [i for i in new_items if i.get("classification") == "urgent"]
    new_emails = [i for i in new_items if i.get("classification") != "urgent"]

    label = _SLOT_LABELS.get(slot, slot.title())
    lines: list[str] = [f"Email Digest ({label})", ""]

    if already_notified:
        lines.append("ALREADY NOTIFIED:")
        for item in already_notified:
            name = _sender_display(item.get("sender", "Unknown"))
            subj = item.get("subject", "(no subject)")
            lines.append(f"- {name}: {subj} (urgent)")
        lines.append("")

    if new_emails:
        lines.append("NEW ITEMS:")
        for item in new_emails:
            name = _sender_display(item.get("sender", "Unknown"))
            subj = item.get("subject", "(no subject)")
            cls = item.get("classification", "")
            lines.append(f"- {name}: {subj} ({cls})")
        lines.append("")

    # Summary line
    parts = []
    if new_emails:
        parts.append(f"{len(new_emails)} new email{'s' if len(new_emails) != 1 else ''}")
    if already_notified:
        parts.append(f"{len(already_notified)} previously notified")
    lines.append(", ".join(parts))

    # Advance high-water mark
    max_ts = max(item.get("internal_ts_ms", 0) for item in new_items)
    _set_consumed_ts(slot, max_ts)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_old_files(keep_days: int = 3) -> int:
    """Remove JSONL queue files older than keep_days. Returns count deleted."""
    if not QUEUE_DIR.is_dir():
        return 0

    cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).strftime("%Y-%m-%d")
    deleted = 0
    for path in QUEUE_DIR.glob("????-??-??.jsonl"):
        if path.stem < cutoff:
            try:
                path.unlink()
                deleted += 1
            except OSError:
                log.debug("Failed to delete %s", path, exc_info=True)
    return deleted
