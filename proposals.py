"""Automation proposal mining from operational log traces.

Extracted from automations_legacy.py during Phase 3 refactor.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import yaml

import config
from utils import normalize_timestamp

log = logging.getLogger(__name__)

_SEQUENCE_SPLIT_RE = re.compile(r"\s*->\s*")
_WORD_TOKEN_RE = re.compile(r"[a-z0-9]+")

_AUTOMATION_PROPOSAL_AUTO_ENABLE_THRESHOLD = 0.88
_PROPOSAL_MIN_OCCURRENCES = 3
_PROPOSAL_MAX_PATTERNS = 3

_OPERATIONAL_LOG_SCHEMA_DOC = {
    "tool_name": "Primary action/tool identifier for each event.",
    "created_at": "ISO-8601 event timestamp used for sequencing and recurrence detection.",
    "success": "Boolean or int success flag used to measure reliability of repeated workflows.",
    "latency_ms": "Optional duration signal used as evidence context.",
    "error_message": "Failure detail; non-empty values reduce confidence.",
    "parameters": "Optional JSON payload sampled to contextualize inferred workflows.",
}


def _safe_iso_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        normalized = normalize_timestamp(text)
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_dict(value: Any) -> dict:
    import json

    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    raw = value.strip()
    if not raw or not raw.startswith("{"):
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _tokenize(text: str) -> set[str]:
    return set(_WORD_TOKEN_RE.findall((text or "").lower()))


def _format_hhmm(minutes_since_midnight: int) -> str:
    minutes_since_midnight %= 24 * 60
    hour = minutes_since_midnight // 60
    minute = minutes_since_midnight % 60
    return f"{hour:02d}:{minute:02d}"


def _percentile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        return 0
    idx = int((len(sorted_values) - 1) * pct)
    return sorted_values[max(0, min(idx, len(sorted_values) - 1))]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _infer_schedule(occurrence_times: list[datetime]) -> dict[str, Any]:
    if not occurrence_times:
        return {
            "cron": "0 9 * * 1-5",
            "day_names": [],
            "window_start": "08:00",
            "window_end": "10:00",
        }

    tz = ZoneInfo(config.TIMEZONE)
    local_times = [ts.astimezone(tz) for ts in occurrence_times]

    minute_of_day = [lt.hour * 60 + lt.minute for lt in local_times]
    minute_counts = Counter(minute_of_day)
    center = minute_counts.most_common(1)[0][0]
    rounded_minute = int(round((center % 60) / 5.0) * 5)
    hour = center // 60
    if rounded_minute >= 60:
        rounded_minute = 0
        hour = (hour + 1) % 24

    weekday_counts = Counter(lt.weekday() for lt in local_times)
    total = len(local_times)
    active_weekdays = sorted(
        day for day, count in weekday_counts.items()
        if count >= 2 and (count / total) >= 0.2
    )

    cron_days = "*"
    if active_weekdays:
        if active_weekdays == [0, 1, 2, 3, 4]:
            cron_days = "1-5"
        else:
            to_cron = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "0"}
            cron_days = ",".join(to_cron[d] for d in active_weekdays)

    sorted_minutes = sorted(minute_of_day)
    p25 = _percentile(sorted_minutes, 0.25)
    p75 = _percentile(sorted_minutes, 0.75)
    half_window = max(30, int((p75 - p25) / 2) + 15)
    window_start = _format_hhmm(center - half_window)
    window_end = _format_hhmm(center + half_window)

    day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    day_labels = [day_names[d] for d in active_weekdays]

    return {
        "cron": f"{rounded_minute} {hour} * * {cron_days}",
        "day_names": day_labels,
        "window_start": window_start,
        "window_end": window_end,
    }


def _confidence_score(
    count: int,
    sequence_len: int,
    total_events: int,
    intervals_m: list[float],
    success_ratio: float,
) -> float:
    freq_score = min(1.0, count / 8.0)
    density_score = min(1.0, (count * sequence_len) / max(1, total_events))

    if len(intervals_m) >= 2:
        mean = sum(intervals_m) / len(intervals_m)
        if mean > 0:
            variance = sum((x - mean) ** 2 for x in intervals_m) / len(intervals_m)
            cv = (variance ** 0.5) / mean
            regularity_score = 1.0 / (1.0 + cv)
        else:
            regularity_score = 0.0
    elif len(intervals_m) == 1:
        regularity_score = 0.75
    else:
        regularity_score = 0.5

    score = (
        (0.38 * freq_score)
        + (0.28 * regularity_score)
        + (0.18 * max(0.0, min(1.0, success_ratio)))
        + (0.16 * density_score)
    )
    return round(max(0.05, min(0.99, score)), 3)


def _normalize_operational_events(operational_logs: list[dict]) -> tuple[list[dict], list[str]]:
    events: list[dict] = []
    schema_fields: set[str] = set()

    for idx, row in enumerate(operational_logs):
        if not isinstance(row, dict):
            continue
        schema_fields.update(str(k) for k in row.keys())

        tool_name = str(
            row.get("tool_name")
            or row.get("tool")
            or row.get("action")
            or row.get("step")
            or ""
        ).strip()
        if not tool_name or tool_name.startswith("approval:"):
            continue

        created_at = _safe_iso_datetime(
            row.get("created_at") or row.get("timestamp") or row.get("time")
        )
        if created_at is None:
            continue

        success_raw = row.get("success")
        if isinstance(success_raw, bool):
            success = success_raw
        elif isinstance(success_raw, (int, float)):
            success = bool(success_raw)
        elif isinstance(success_raw, str):
            success = success_raw.strip().lower() not in {"0", "false", "failed", "error", "no"}
        else:
            success = not bool(str(row.get("error_message", "")).strip())

        events.append(
            {
                "index": idx,
                "tool_name": tool_name,
                "created_at": created_at,
                "created_at_iso": created_at.isoformat(),
                "success": success,
                "latency_ms": _safe_int(row.get("latency_ms"), default=0),
                "error_message": str(row.get("error_message", "")).strip(),
                "parameters": _safe_dict(row.get("parameters")),
            }
        )

    events.sort(key=lambda r: r["created_at"])
    return events, sorted(schema_fields)


def _mine_repeated_sequences(events: list[dict], min_occurrences: int) -> list[dict]:
    if len(events) < 4:
        return []

    sequence_hits: dict[tuple[str, ...], list[int]] = defaultdict(list)
    max_len = min(4, len(events))
    for start in range(len(events)):
        for size in range(2, max_len + 1):
            end = start + size
            if end > len(events):
                break
            seq = tuple(events[i]["tool_name"] for i in range(start, end))
            if len(set(seq)) == 1:
                continue
            sequence_hits[seq].append(start)

    patterns: list[dict] = []
    for sequence, starts in sequence_hits.items():
        selected: list[int] = []
        last_end = -1
        for s in sorted(starts):
            if s <= last_end:
                continue
            selected.append(s)
            last_end = s + len(sequence) - 1

        if len(selected) < min_occurrences:
            continue

        start_times = [events[s]["created_at"] for s in selected]
        interval_minutes = [
            (start_times[i] - start_times[i - 1]).total_seconds() / 60.0
            for i in range(1, len(start_times))
        ]
        median_interval = _median(interval_minutes) if interval_minutes else 0.0

        step_successes = []
        for s in selected:
            for idx in range(s, s + len(sequence)):
                step_successes.append(1.0 if events[idx]["success"] else 0.0)
        success_ratio = (sum(step_successes) / len(step_successes)) if step_successes else 0.0

        inferred_schedule = _infer_schedule(start_times)
        confidence = _confidence_score(
            count=len(selected),
            sequence_len=len(sequence),
            total_events=len(events),
            intervals_m=interval_minutes,
            success_ratio=success_ratio,
        )

        sample_events = []
        for s in selected[:3]:
            event = events[s]
            sample_events.append(
                {
                    "tool_name": event["tool_name"],
                    "created_at": event["created_at_iso"],
                    "success": event["success"],
                    "latency_ms": event["latency_ms"],
                    "parameters": event["parameters"],
                }
            )

        patterns.append(
            {
                "sequence": list(sequence),
                "sequence_key": " -> ".join(sequence),
                "occurrence_count": len(selected),
                "sequence_len": len(sequence),
                "first_seen": start_times[0].isoformat(),
                "last_seen": start_times[-1].isoformat(),
                "median_interval_minutes": round(median_interval, 1) if median_interval else None,
                "success_ratio": round(success_ratio, 3),
                "confidence": confidence,
                "schedule": inferred_schedule,
                "sample_events": sample_events,
            }
        )

    return sorted(
        patterns,
        key=lambda p: (p["confidence"], p["occurrence_count"], p["sequence_len"]),
        reverse=True,
    )


def _coerce_pattern_summaries(operational_logs: list[dict]) -> list[dict]:
    patterns: list[dict] = []
    for row in operational_logs:
        if not isinstance(row, dict):
            continue
        steps = str(row.get("steps", "")).strip()
        if not steps:
            continue
        sequence = [s.strip() for s in _SEQUENCE_SPLIT_RE.split(steps) if s.strip()]
        if len(sequence) < 2:
            continue
        count = max(1, _safe_int(row.get("count"), default=1))
        confidence = float(row.get("confidence", 0.0) or 0.0)
        confidence = round(max(0.05, min(0.99, confidence)), 3)
        patterns.append(
            {
                "sequence": sequence,
                "sequence_key": " -> ".join(sequence),
                "occurrence_count": count,
                "sequence_len": len(sequence),
                "first_seen": "",
                "last_seen": "",
                "median_interval_minutes": None,
                "success_ratio": 0.0,
                "confidence": confidence,
                "schedule": {
                    "cron": "0 9 * * 1-5",
                    "day_names": ["mon", "tue", "wed", "thu", "fri"],
                    "window_start": "08:30",
                    "window_end": "09:30",
                },
                "sample_events": [],
            }
        )
    return sorted(
        patterns,
        key=lambda p: (p["confidence"], p["occurrence_count"], p["sequence_len"]),
        reverse=True,
    )


def _load_existing_automation_signatures() -> list[dict]:
    signatures: list[dict] = []
    root = config.AUTOMATIONS_DIR
    try:
        paths = sorted(root.glob("*.yaml"))
    except Exception:
        return signatures

    for path in paths:
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", path.stem))
        automation_id = str(raw.get("id", path.stem))
        trigger = raw.get("trigger") or {}
        cron = str(trigger.get("cron", "")).strip() if isinstance(trigger, dict) else ""
        pipeline = raw.get("pipeline") or []

        text_parts = [name, automation_id]
        if isinstance(pipeline, list):
            for step in pipeline:
                if not isinstance(step, dict):
                    continue
                text_parts.append(str(step.get("action", "")))
                text_parts.append(str(step.get("step", "")))
                text_parts.append(str(step.get("agent", "")))

        signatures.append(
            {
                "path": str(path),
                "name": name,
                "id": automation_id,
                "cron": cron,
                "tokens": _tokenize(" ".join(text_parts)),
            }
        )
    return signatures


def _pattern_overlaps_existing(pattern: dict, existing: list[dict]) -> tuple[bool, str]:
    sequence = [str(s) for s in pattern.get("sequence", [])]
    seq_tokens = _tokenize(" ".join(sequence))
    if not seq_tokens:
        return False, ""

    pattern_cron = str((pattern.get("schedule") or {}).get("cron", "")).strip()
    for sig in existing:
        existing_tokens = sig.get("tokens") or set()
        if not existing_tokens:
            continue
        overlap = len(seq_tokens & existing_tokens) / max(1, len(seq_tokens))
        cron_match = bool(pattern_cron and pattern_cron == sig.get("cron", ""))
        if overlap >= 0.6 or (cron_match and overlap >= 0.4):
            return True, f"overlap={overlap:.2f} with {sig.get('id', '')}"
    return False, ""


def _humanize_tool(tool_name: str) -> str:
    cleaned = tool_name.replace("routing:subagent_start:", "subagent ")
    cleaned = cleaned.replace(":", " ").replace("_", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.title() if cleaned else "Workflow"


def _sequence_slug(sequence: list[str]) -> str:
    raw = "-".join(sequence[:3]).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    return slug[:64] or "proposal"


def propose_automation(operational_logs: list[dict]) -> str | None:
    """Generate a log-driven automation proposal YAML from operational traces."""
    if not operational_logs:
        return None

    min_occurrences = max(
        _PROPOSAL_MIN_OCCURRENCES,
        int(getattr(config, "AUTOMATION_MIN_PATTERN_COUNT", _PROPOSAL_MIN_OCCURRENCES)),
    )
    detected_events, detected_schema_fields = _normalize_operational_events(operational_logs)
    mined_patterns = _mine_repeated_sequences(detected_events, min_occurrences=min_occurrences)
    source_mode = "operational_logs"

    if not mined_patterns:
        mined_patterns = _coerce_pattern_summaries(operational_logs)
        source_mode = "pattern_summaries"

    if not mined_patterns:
        return None

    existing = _load_existing_automation_signatures()
    selected_patterns: list[dict] = []
    dedupe_notes: list[str] = []
    for pattern in mined_patterns:
        overlaps, reason = _pattern_overlaps_existing(pattern, existing)
        if overlaps:
            dedupe_notes.append(f"Skipped {pattern.get('sequence_key', '')}: {reason}")
            continue
        selected_patterns.append(pattern)
        if len(selected_patterns) >= _PROPOSAL_MAX_PATTERNS:
            break

    if not selected_patterns:
        return None

    primary = selected_patterns[0]
    primary_schedule = primary.get("schedule") or {}
    primary_sequence = [str(t) for t in primary.get("sequence", [])]
    if len(primary_sequence) < 2:
        return None

    confidence = float(primary.get("confidence", 0.0) or 0.0)
    auto_enable_threshold = _AUTOMATION_PROPOSAL_AUTO_ENABLE_THRESHOLD
    enabled = confidence >= auto_enable_threshold

    median_interval_minutes = float(primary.get("median_interval_minutes") or 0.0)
    if median_interval_minutes > 0:
        min_interval_s = int(max(600, median_interval_minutes * 60 * 0.6))
    else:
        min_interval_s = 3600

    day_names = [str(d) for d in primary_schedule.get("day_names", []) if str(d).strip()]
    conditions: list[dict] = [{"type": "not_quiet_hours"}]
    if day_names and len(day_names) < 7:
        conditions.append({"type": "day_of_week", "days": day_names})
    conditions.append(
        {
            "type": "not_duplicate",
            "key": f"{_sequence_slug(primary_sequence)}:{{date}}",
            "cooldown": f"{max(30, int(min_interval_s / 60))}m",
        }
    )

    top_pattern_lines = []
    for idx, pattern in enumerate(selected_patterns, start=1):
        top_pattern_lines.append(
            (
                f"{idx}. {pattern.get('sequence_key', '')} "
                f"(occurrences={pattern.get('occurrence_count', 0)}, "
                f"confidence={pattern.get('confidence', 0.0)})"
            )
        )

    primary_human_start = _humanize_tool(primary_sequence[0])
    primary_human_end = _humanize_tool(primary_sequence[-1])
    proposal_name = f"{primary_human_start} to {primary_human_end} Routine"

    rationale = (
        "Proposed because operational logs show a repeated workflow sequence with stable timing "
        "and high recurrence count."
    )
    pipeline_prompt = (
        "Execute the primary recurring workflow observed in operational logs.\n"
        f"Primary sequence: {primary.get('sequence_key', '')}\n"
        f"Schedule window (local {config.TIMEZONE}): "
        f"{primary_schedule.get('window_start', '')}-{primary_schedule.get('window_end', '')}\n"
        "Supporting patterns:\n"
        + "\n".join(top_pattern_lines)
        + "\n"
        "If exact inputs are missing, produce a concise operator-ready checklist and next actions."
    )

    deliver_message = (
        "Automation run for pattern "
        f"'{primary.get('sequence_key', '')}' "
        "(confidence "
        f"{confidence:.2f}"
        ")\n"
        "{execute_pattern.output}"
    )

    evidence = {
        "top_pattern": {
            "sequence": primary_sequence,
            "sequence_key": primary.get("sequence_key", ""),
            "occurrence_count": primary.get("occurrence_count", 0),
            "first_seen": primary.get("first_seen", ""),
            "last_seen": primary.get("last_seen", ""),
            "median_interval_minutes": primary.get("median_interval_minutes"),
            "success_ratio": primary.get("success_ratio"),
            "window_start": primary_schedule.get("window_start", ""),
            "window_end": primary_schedule.get("window_end", ""),
        },
        "supporting_patterns": [
            {
                "sequence_key": p.get("sequence_key", ""),
                "occurrence_count": p.get("occurrence_count", 0),
                "confidence": p.get("confidence", 0.0),
            }
            for p in selected_patterns[1:]
        ],
        "sample_events": primary.get("sample_events", []),
    }

    skeleton = {
        "id": f"proposal-{_sequence_slug(primary_sequence)}",
        "name": proposal_name,
        "enabled": enabled,
        "version": 1,
        "trigger": {
            "type": "schedule",
            "cron": str(primary_schedule.get("cron", "0 9 * * 1-5")),
            "timezone": config.TIMEZONE,
        },
        "min_interval": f"{max(5, int(min_interval_s / 60))}m",
        "conditions": conditions,
        "pipeline": [
            {
                "step": "execute_pattern",
                "agent": "analyst",
                "prompt": pipeline_prompt,
            },
            {
                "step": "deliver",
                "channel": "whatsapp",
                "to": "owner",
                "message": deliver_message,
            },
        ],
        "meta": {
            "source_mode": source_mode,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_from_log_count": len(operational_logs),
            "parsed_event_count": len(detected_events),
            "schema_fields_detected": detected_schema_fields,
            "operational_log_schema_reference": _OPERATIONAL_LOG_SCHEMA_DOC,
            "fields_used_for_pattern_detection": [
                "tool_name",
                "created_at",
                "success",
                "latency_ms",
                "error_message",
                "parameters",
            ],
            "proposal_confidence": confidence,
            "auto_enable_threshold": auto_enable_threshold,
            "why_proposed": rationale,
            "dedupe_notes": dedupe_notes,
            "evidence": evidence,
        },
    }
    return yaml.safe_dump(skeleton, sort_keys=False)
