"""Graph suggestions system — captures relationship fallbacks and hotspots.

Real-time logging captures:
  1. Relationship type fallbacks (original type not in VALID_REL_TYPES → RELATED_TO)
  2. RELATED_TO hotspots (edges with 3+ mentions, candidates for reclassification)

Nightly maintenance builds a digest from today's JSONL + Neo4j hotspot query.

NOTE: Imported lazily by graph.py and maintenance.py to avoid circular imports.
Do not add top-level imports from those modules.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config

log = logging.getLogger(__name__)

SUGGESTIONS_DIR = config.WORKSPACE / "memory" / "graph_suggestions"


# ---------------------------------------------------------------------------
# Real-time JSONL logging
# ---------------------------------------------------------------------------

def _append_jsonl(entry: dict) -> None:
    """Best-effort append a JSON line to today's suggestion file."""
    try:
        SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
        ts = entry.get("timestamp", "")
        today = ts[:10] if len(ts) >= 10 else datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = SUGGESTIONS_DIR / f"{today}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")
    except OSError:
        log.debug("Failed to write graph suggestion", exc_info=True)


def log_relationship_fallback(
    head: str,
    tail: str,
    original_type: str,
    confidence: float,
    context: str,
) -> None:
    """Log when a relationship type falls back to RELATED_TO."""
    _append_jsonl({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "relationship_fallback",
        "head": head,
        "tail": tail,
        "original_type": original_type,
        "fell_back_to": "RELATED_TO",
        "confidence": round(confidence, 3),
        "context": (context or "")[:200],
        "suggestion": f"Consider adding '{original_type.strip().upper().replace(' ', '_')}' to VALID_REL_TYPES",
    })


def log_repeated_related_to(
    head: str,
    tail: str,
    mention_count: int,
) -> None:
    """Log when a RELATED_TO edge reaches 3+ mentions (reclassification candidate)."""
    _append_jsonl({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "related_to_hotspot",
        "head": head,
        "tail": tail,
        "mention_count": mention_count,
        "suggestion": f"{head} -> {tail}: RELATED_TO {mention_count}x — consider specific type",
    })


# ---------------------------------------------------------------------------
# Reading suggestions
# ---------------------------------------------------------------------------

def get_suggestions(date_str: str | None = None) -> list[dict]:
    """Read today's (or specified date's) suggestion JSONL file."""
    target_date = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = SUGGESTIONS_DIR / f"{target_date}.jsonl"

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
        log.debug("Failed to read suggestions file: %s", path, exc_info=True)

    return entries


def get_related_to_hotspots(min_mentions: int = 3) -> list[dict]:
    """Query Neo4j for RELATED_TO edges with high mention counts."""
    try:
        from memory.graph import get_driver

        driver = get_driver()
        with driver.session() as session:
            result = session.run(
                """MATCH (h:Entity)-[r:RELATED_TO]->(t:Entity)
                   WHERE r.mention_count >= $min_mentions
                   RETURN h.name AS head, t.name AS tail,
                          r.mention_count AS mentions,
                          r.context_snippets AS contexts
                   ORDER BY r.mention_count DESC
                   LIMIT 20""",
                min_mentions=min_mentions,
            )
            return [dict(record) for record in result]
    except Exception:
        log.debug("Failed to query RELATED_TO hotspots", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Nightly digest
# ---------------------------------------------------------------------------

def build_suggestion_digest() -> str:
    """Combine today's JSONL suggestions + Neo4j hotspots into a digest.

    Returns empty string if nothing to report.
    """
    suggestions = get_suggestions()
    hotspots = get_related_to_hotspots()

    if not suggestions and not hotspots:
        return ""

    # Deduplicate: count fallback types and hotspot edges
    fallback_counts: dict[str, int] = {}
    hotspot_from_jsonl: dict[str, int] = {}

    for entry in suggestions:
        entry_type = entry.get("type", "")
        if entry_type == "relationship_fallback":
            original = entry.get("original_type", "unknown")
            fallback_counts[original] = fallback_counts.get(original, 0) + 1
        elif entry_type == "related_to_hotspot":
            key = f"{entry.get('head', '?').strip().lower()} -> {entry.get('tail', '?').strip().lower()}"
            count = entry.get("mention_count", 0)
            hotspot_from_jsonl[key] = max(hotspot_from_jsonl.get(key, 0), count)

    # Separate Neo4j-only hotspots from those already in today's JSONL
    neo4j_only = [
        h for h in hotspots
        if f"{h.get('head', '?').strip().lower()} -> {h.get('tail', '?').strip().lower()}" not in hotspot_from_jsonl
    ]

    lines: list[str] = []
    total_items = len(fallback_counts) + len(hotspot_from_jsonl) + len(neo4j_only)
    if total_items == 0:
        return ""
    total_events = sum(fallback_counts.values()) + len(hotspot_from_jsonl) + len(neo4j_only)
    lines.append(f"{total_items} graph suggestion(s) today ({total_events} events):")

    # Fallback types
    for original_type, count in sorted(fallback_counts.items(), key=lambda x: -x[1]):
        normalized = original_type.strip().upper().replace(" ", "_")
        lines.append(f"- '{original_type}' fell back to RELATED_TO {count}x -> Add {normalized}?")

    # JSONL hotspots (real-time detections from today)
    for key, count in sorted(hotspot_from_jsonl.items(), key=lambda x: -x[1]):
        lines.append(f"- {key}: RELATED_TO {count}x -> specific type?")

    # Neo4j hotspots (persistent edges not already in today's JSONL)
    for hotspot in neo4j_only:
        head = hotspot.get("head", "?")
        tail = hotspot.get("tail", "?")
        mentions = hotspot.get("mentions", 0)
        lines.append(f"- {head} -> {tail}: RELATED_TO {mentions}x -> specific type?")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-adoption pipeline: adopt suggestions seen 3+ distinct nights
# ---------------------------------------------------------------------------

_ADOPTION_HISTORY_PATH = SUGGESTIONS_DIR / "adoption_history.jsonl"
_ADOPTION_THRESHOLD = 3  # distinct nights before auto-adopt


def _load_adoption_history() -> dict[str, dict]:
    """Load {suggestion_key: {nights: set[str], type: str, ...}} from history file."""
    history: dict[str, dict] = {}
    if not _ADOPTION_HISTORY_PATH.exists():
        return history
    for line in _ADOPTION_HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            key = entry.get("key", "")
            if key:
                if key not in history:
                    history[key] = {"nights": set(), "entry": entry}
                for n in entry.get("nights", []):
                    history[key]["nights"].add(n)
        except json.JSONDecodeError:
            continue
    return history


def _save_adoption_history(history: dict[str, dict]) -> None:
    """Persist adoption history (one JSONL line per unique suggestion)."""
    SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, data in history.items():
        entry = dict(data.get("entry", {}))
        entry["key"] = key
        entry["nights"] = sorted(data["nights"])
        lines.append(json.dumps(entry, ensure_ascii=True))
    _ADOPTION_HISTORY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_auto_adoption() -> str:
    """Check suggestion history and auto-adopt mature suggestions.

    Called from nightly maintenance. Returns a summary string.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_suggestions = get_suggestions()
    hotspots = get_related_to_hotspots()

    # Build today's suggestion keys
    todays_keys: dict[str, dict] = {}
    for entry in today_suggestions:
        if entry.get("type") == "relationship_fallback":
            original = entry.get("original_type", "").strip().upper().replace(" ", "_")
            if original:
                key = f"rel_type:{original}"
                todays_keys[key] = {"action": "add_rel_type", "rel_type": original, "entry": entry}
    for h in hotspots:
        head = h.get("head", "").strip()
        tail = h.get("tail", "").strip()
        if head and tail:
            key = f"reclassify:{head.lower()}->{tail.lower()}"
            todays_keys[key] = {
                "action": "reclassify", "head": head, "tail": tail,
                "mentions": h.get("mentions", 0), "entry": h,
            }

    if not todays_keys:
        return "no suggestions to track"

    # Update history with today's observations
    history = _load_adoption_history()
    for key, data in todays_keys.items():
        if key not in history:
            history[key] = {"nights": set(), "entry": data}
        history[key]["nights"].add(today)

    # Check for mature suggestions (3+ distinct nights)
    adopted = []
    for key, data in list(history.items()):
        if len(data["nights"]) >= _ADOPTION_THRESHOLD:
            entry = data.get("entry", {})
            action = entry.get("action", "")
            if action == "add_rel_type":
                rel_type = entry.get("rel_type", "")
                if rel_type and _adopt_rel_type(rel_type):
                    adopted.append(f"Added rel type: {rel_type}")
                    del history[key]
            elif action == "reclassify":
                # Don't auto-reclassify — just report for Kimi audit to handle
                adopted.append(f"Flagged for reclassify: {entry.get('head')} -> {entry.get('tail')}")

    _save_adoption_history(history)

    tracked = len(todays_keys)
    mature = len(adopted)
    parts = [f"tracked {tracked} suggestions"]
    if mature:
        parts.append(f"adopted {mature}: " + "; ".join(adopted))
    return ", ".join(parts)


def _adopt_rel_type(rel_type: str) -> bool:
    """Add a new relationship type to VALID_REL_TYPES and persist to schema."""
    try:
        from memory import graph
        normalized = rel_type.strip().upper().replace(" ", "_")
        if normalized in graph.VALID_REL_TYPES:
            return False
        graph.VALID_REL_TYPES.add(normalized)
        log.info("Auto-adopted new relationship type: %s", normalized)
        return True
    except Exception:
        log.error("Failed to adopt rel type: %s", rel_type, exc_info=True)
        return False
