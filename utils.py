import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_NUMERIC_TS_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def _coerce_unix_epoch(value: float) -> float:
    """Convert unix timestamps in ns/us/ms/sec to seconds."""
    abs_value = abs(value)
    if abs_value >= 1e17:
        return value / 1_000_000_000  # nanoseconds
    if abs_value >= 1e14:
        return value / 1_000_000  # microseconds
    if abs_value >= 1e11:
        return value / 1_000  # milliseconds
    return value  # seconds


def normalize_timestamp(value: Any) -> str:
    """Normalize timestamps to timezone-aware UTC ISO-8601 strings."""
    if isinstance(value, datetime):
        dt = value
    else:
        raw = str(value or "").strip()
        if not raw:
            return datetime.now(timezone.utc).isoformat()

        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            if not _NUMERIC_TS_RE.fullmatch(raw):
                return raw
            try:
                dt = datetime.fromtimestamp(
                    _coerce_unix_epoch(float(raw)),
                    tz=timezone.utc,
                )
            except (ValueError, OverflowError, OSError):
                return raw

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def atomic_write(path: str | Path, payload: str | bytes) -> None:
    """Atomically write text/bytes by writing a sibling temp file then rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp_name = f".{target.name}.{os.getpid()}.tmp"
    tmp_path = target.with_name(tmp_name)

    if isinstance(payload, bytes):
        tmp_path.write_bytes(payload)
    else:
        tmp_path.write_text(payload, encoding="utf-8")

    os.replace(tmp_path, target)


def atomic_write_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    atomic_write(path, json.dumps(data, indent=indent, default=str))


def load_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return {} if default is None else default
    try:
        return json.loads(target.read_text())
    except (json.JSONDecodeError, OSError):
        return {} if default is None else default
