import asyncio
import contextlib
import functools
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Latency tracking utilities
# ---------------------------------------------------------------------------

class LatencyTracker:
    """Context manager that measures wall-clock latency for a block of code.

    Usage (sync)::

        with LatencyTracker("neo4j", "resolve_entity") as lt:
            result = session.run(query, params)
        # lt.elapsed_ms is now set, log entry emitted at DEBUG level

    Usage (async)::

        async with LatencyTracker("google_calendar", "list_events") as lt:
            events = await service.events().list(...).execute()
    """

    __slots__ = ("service", "operation", "elapsed_ms", "_start", "_logger")

    def __init__(self, service: str, operation: str, *, logger: logging.Logger | None = None):
        self.service = service
        self.operation = operation
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0
        self._logger = logger or logging.getLogger(f"latency.{service}")

    def _finish(self) -> None:
        self.elapsed_ms = (time.monotonic() - self._start) * 1000
        self._logger.debug(
            "%s.%s latency=%.1fms",
            self.service, self.operation, self.elapsed_ms,
        )

    def __enter__(self) -> "LatencyTracker":
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._finish()

    async def __aenter__(self) -> "LatencyTracker":
        self._start = time.monotonic()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        self._finish()


def track_latency(service: str, operation: str | None = None):
    """Decorator that logs wall-clock latency for sync or async functions.

    Usage::

        @track_latency("google_calendar")
        def calendar_list(args):
            ...

        @track_latency("neo4j", "query")
        async def run_cypher(query, params):
            ...

    The *operation* defaults to the function name if not given.
    Latency is logged at DEBUG to ``latency.<service>`` logger.
    """

    def decorator(fn: Any) -> Any:
        op = operation or fn.__name__
        _logger = logging.getLogger(f"latency.{service}")

        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.monotonic()
                try:
                    return await fn(*args, **kwargs)
                finally:
                    ms = (time.monotonic() - start) * 1000
                    _logger.debug("%s.%s latency=%.1fms", service, op, ms)
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.monotonic()
                try:
                    return fn(*args, **kwargs)
                finally:
                    ms = (time.monotonic() - start) * 1000
                    _logger.debug("%s.%s latency=%.1fms", service, op, ms)
            return sync_wrapper

    return decorator
