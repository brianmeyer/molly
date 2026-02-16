from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import config

log = logging.getLogger(__name__)

FOUNDRY_OBSERVATIONS_DIR = config.WORKSPACE / "foundry" / "observations"
_SUCCESS_OUTCOMES = {"success", "ok", "completed"}


@dataclass(frozen=True)
class FoundrySequenceSignal:
    steps: tuple[str, str, str]
    count: int
    successes: int
    latest_at: str

    @property
    def success_rate(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.successes / self.count


def load_foundry_sequence_signals(
    *,
    days: int = 30,
    observations_dir: Path = FOUNDRY_OBSERVATIONS_DIR,
    is_low_value_tool: Callable[[str], bool] | None = None,
    now_utc: datetime | None = None,
) -> dict[str, FoundrySequenceSignal]:
    now = now_utc.astimezone(timezone.utc) if now_utc else datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(1, int(days)))
    aggregate: dict[str, dict[str, object]] = {}

    if not observations_dir.exists() or not observations_dir.is_dir():
        return {}

    for path in sorted(observations_dir.glob("*.jsonl")):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            log.debug("Foundry observations unreadable: %s", path, exc_info=True)
            continue

        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                log.debug("Skipping malformed Foundry observation: %s:%s", path, line_number)
                continue
            if not isinstance(payload, dict):
                continue

            observed_at = _parse_timestamp(payload.get("timestamp"))
            if observed_at is None or observed_at < cutoff:
                continue

            raw_sequence = payload.get("tool_sequence")
            if not isinstance(raw_sequence, list):
                continue
            steps = []
            for raw_step in raw_sequence:
                step = str(raw_step or "").strip()
                if not step:
                    continue
                if callable(is_low_value_tool) and is_low_value_tool(step):
                    continue
                steps.append(step)
            if len(steps) < 3:
                continue

            outcome = str(payload.get("outcome", "")).strip().lower()
            is_success = outcome in _SUCCESS_OUTCOMES
            for index in range(0, len(steps) - 2):
                sequence = tuple(steps[index:index + 3])
                if len(set(sequence)) == 1:
                    continue
                key = " -> ".join(sequence)
                entry = aggregate.get(key)
                if entry is None:
                    aggregate[key] = {
                        "steps": sequence,
                        "count": 1,
                        "successes": 1 if is_success else 0,
                        "latest_at": observed_at,
                    }
                    continue

                entry["count"] = int(entry.get("count", 0) or 0) + 1
                if is_success:
                    entry["successes"] = int(entry.get("successes", 0) or 0) + 1

                latest = entry.get("latest_at")
                if not isinstance(latest, datetime) or observed_at > latest:
                    entry["latest_at"] = observed_at

    signals: dict[str, FoundrySequenceSignal] = {}
    for key, entry in aggregate.items():
        raw_steps = entry.get("steps")
        if not isinstance(raw_steps, tuple) or len(raw_steps) != 3:
            continue
        latest_at = entry.get("latest_at")
        latest_at_text = latest_at.isoformat() if isinstance(latest_at, datetime) else ""
        steps = tuple(str(step) for step in raw_steps)
        signals[key] = FoundrySequenceSignal(
            steps=(steps[0], steps[1], steps[2]),
            count=int(entry.get("count", 0) or 0),
            successes=int(entry.get("successes", 0) or 0),
            latest_at=latest_at_text,
        )
    return signals


def write_observation(
    tool_sequence: list[str],
    outcome: str,
    context: str = "",
    observations_dir: Path = FOUNDRY_OBSERVATIONS_DIR,
) -> None:
    """Append a tool-sequence observation to today's JSONL file.

    Skips sequences shorter than 3 steps (not enough signal for pattern detection).

    See also: memory/graph_suggestions._append_jsonl (similar JSONL pattern).
    """
    if len(tool_sequence) < 3:
        return
    try:
        observations_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        path = observations_dir / f"{today}.jsonl"
        entry = {
            "timestamp": now.isoformat(),
            "tool_sequence": list(tool_sequence),
            "workflow": " -> ".join(str(step) for step in tool_sequence),
            "workflow_steps": len(tool_sequence),
            "outcome": outcome,
            "context": (context or "")[:200],
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")
    except OSError:
        log.debug("Failed to write foundry observation", exc_info=True)


def _parse_timestamp(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
