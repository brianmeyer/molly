import asyncio
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

try:
    import aiosqlite  # type: ignore
except Exception:  # pragma: no cover - optional dependency in some test envs
    aiosqlite = None  # type: ignore

log = logging.getLogger(__name__)

_ASYNC_CONNECTIONS: dict[str, Any] = {}
_ASYNC_POOL_LOCK: asyncio.Lock | None = None
_ASYNC_POOL_LOOP_ID: int | None = None


def _normalize_db_key(database: str | Path) -> str:
    raw = str(database)
    if raw == ":memory:" or raw.startswith("file:"):
        return raw
    return str(Path(raw).expanduser())


def _get_async_pool_lock() -> asyncio.Lock:
    global _ASYNC_POOL_LOCK, _ASYNC_POOL_LOOP_ID
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    if _ASYNC_POOL_LOCK is None or _ASYNC_POOL_LOOP_ID != loop_id:
        _ASYNC_POOL_LOCK = asyncio.Lock()
        _ASYNC_POOL_LOOP_ID = loop_id
    return _ASYNC_POOL_LOCK


async def get_async_connection(
    database: str | Path,
    *,
    row_factory: Any | None = None,
) -> Any:
    """Return a singleton aiosqlite connection for a database file."""
    if aiosqlite is None:
        raise RuntimeError("aiosqlite is not installed")

    key = _normalize_db_key(database)
    lock = _get_async_pool_lock()
    async with lock:
        conn = _ASYNC_CONNECTIONS.get(key)
        if conn is None:
            conn = await aiosqlite.connect(key)
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=5000")
            if row_factory is not None:
                conn.row_factory = row_factory
            _ASYNC_CONNECTIONS[key] = conn
            log.info("Initialized async sqlite pool connection: %s", key)
            return conn

        if row_factory is not None:
            conn.row_factory = row_factory
        return conn


async def close_async_connections() -> None:
    lock = _get_async_pool_lock()
    async with lock:
        for conn in list(_ASYNC_CONNECTIONS.values()):
            try:
                await conn.close()
            except Exception:
                log.debug("Failed closing async sqlite pool connection", exc_info=True)
        _ASYNC_CONNECTIONS.clear()


_SYNC_LOCK = threading.Lock()
_SYNC_CONNECTIONS: dict[tuple[Any, ...], sqlite3.Connection] = {}


def get_connection(
    database: str | Path,
    *args,
    pooled: bool = False,
    **kwargs,
) -> sqlite3.Connection:
    """Return a sqlite3 connection via one accessor.

    `pooled=False` returns a new connection (legacy-compatible).
    `pooled=True` returns a singleton connection for the exact connect signature.
    """
    db_key = _normalize_db_key(database)
    if not pooled or db_key == ":memory:":
        return sqlite3.connect(str(database), *args, **kwargs)

    cache_key = (
        db_key,
        args,
        tuple(sorted(kwargs.items())),
    )
    with _SYNC_LOCK:
        conn = _SYNC_CONNECTIONS.get(cache_key)
        if conn is None:
            conn = sqlite3.connect(str(database), *args, **kwargs)
            _SYNC_CONNECTIONS[cache_key] = conn
        return conn


def close_sync_connections() -> None:
    with _SYNC_LOCK:
        for conn in list(_SYNC_CONNECTIONS.values()):
            try:
                conn.close()
            except Exception:
                log.debug("Failed closing sync sqlite pooled connection", exc_info=True)
        _SYNC_CONNECTIONS.clear()


def sqlite_connect(database: str | Path, *args, **kwargs) -> sqlite3.Connection:
    """Drop-in replacement helper used during sqlite3.connect migration."""
    return get_connection(database, *args, **kwargs)
