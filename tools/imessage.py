"""iMessage MCP tools for Molly (read-only).

Read-only access to the macOS Messages SQLite database (chat.db).
Sender names fall back to raw handles when contacts are unavailable.

Tools:
  imessage_search  (AUTO) — Search messages by contact, keyword, or date range
  imessage_recent  (AUTO) — Get recent messages from last N hours
  imessage_thread  (AUTO) — Get conversation thread with a specific contact
  imessage_unread  (AUTO) — Messages since last check (high-water timestamp)

Date handling: iMessage uses Apple Cocoa Core Data timestamps (seconds since
2001-01-01). If value > 1e12, it's nanoseconds — divide by 1e9 first.

Requires Full Disk Access for the Python process.
"""

import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone

from claude_agent_sdk import create_sdk_mcp_server, tool

import config

log = logging.getLogger(__name__)

# Seconds between Unix epoch (1970-01-01) and Apple epoch (2001-01-01)
APPLE_EPOCH_OFFSET = 978307200
_NON_DIGITS = re.compile(r"\D+")


def _normalize_phone_for_match(phone: str) -> str:
    digits = _NON_DIGITS.sub("", str(phone or ""))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits[-10:] if len(digits) >= 10 else digits


def _connect() -> sqlite3.Connection:
    """Open a read-only connection to the iMessage database."""
    uri = f"file:{config.IMESSAGE_DB}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _text_result(data) -> dict:
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def _error_result(msg: str) -> dict:
    return {"content": [{"type": "text", "text": msg}], "is_error": True}


def _apple_to_unix(apple_ts) -> float:
    """Convert Apple Cocoa timestamp to Unix timestamp."""
    if apple_ts is None or apple_ts == 0:
        return 0.0
    ts = float(apple_ts)
    if ts > 1e12:
        ts = ts / 1e9
    return ts + APPLE_EPOCH_OFFSET


_DB_NANOSECONDS: bool | None = None


def _detect_db_format():
    """Detect whether the iMessage DB uses nanoseconds (macOS 10.13+) or seconds."""
    global _DB_NANOSECONDS
    try:
        conn = _connect()
        try:
            row = conn.execute("SELECT date FROM message ORDER BY ROWID DESC LIMIT 1").fetchone()
            _DB_NANOSECONDS = bool(row and row["date"] and float(row["date"]) > 1e12)
        finally:
            conn.close()
    except Exception:
        _DB_NANOSECONDS = True  # Default to nanoseconds on modern macOS


def _unix_to_apple(unix_ts: float) -> float:
    """Convert Unix timestamp to Apple Cocoa timestamp.

    Detects whether the DB uses seconds or nanoseconds (macOS 10.13+)
    and returns the matching format.
    """
    global _DB_NANOSECONDS
    if _DB_NANOSECONDS is None:
        _detect_db_format()
    apple_seconds = unix_ts - APPLE_EPOCH_OFFSET
    if _DB_NANOSECONDS:
        return apple_seconds * 1e9
    return apple_seconds


def _format_ts(apple_ts) -> str:
    """Convert Apple timestamp to ISO 8601 string."""
    unix = _apple_to_unix(apple_ts)
    if unix <= 0:
        return ""
    return datetime.fromtimestamp(unix, tz=timezone.utc).isoformat()


def _resolve_handle(handle_id: str) -> str:
    """Resolve an iMessage handle to a display value."""
    if not handle_id:
        return "Unknown"
    # For phone-like handles (not emails), try Google Contacts resolver
    if "@" not in handle_id:
        digits = _NON_DIGITS.sub("", handle_id)
        if len(digits) >= 10:
            try:
                from contacts import get_resolver
                name = get_resolver().resolve_phone(digits)
                if name:
                    return name
            except Exception:
                log.debug("Contact resolution failed for handle %s", handle_id, exc_info=True)
    return handle_id


def _format_message(row: sqlite3.Row, handle_map: dict[int, str] | None = None) -> dict:
    """Format a message row into a clean dict."""
    handle_id_str = ""
    if handle_map and row["handle_id"] in handle_map:
        handle_id_str = handle_map[row["handle_id"]]
    sender = _resolve_handle(handle_id_str) if handle_id_str else "Me"

    return {
        "id": row["ROWID"],
        "text": row["text"] or "",
        "date": _format_ts(row["date"]),
        "is_from_me": bool(row["is_from_me"]),
        "sender": sender if not row["is_from_me"] else "Me",
        "handle": handle_id_str,
    }


def _build_handle_map(conn: sqlite3.Connection) -> dict[int, str]:
    """Build a mapping of handle ROWID -> handle ID (phone/email)."""
    rows = conn.execute("SELECT ROWID, id FROM handle").fetchall()
    return {r["ROWID"]: r["id"] for r in rows}


def _prewarm_contacts(rows, handle_map: dict[int, str]):
    """Legacy no-op: contact prewarming moved to apple-mcp."""
    return


def _get_high_water() -> float:
    """Get the last-checked Unix timestamp from state.json."""
    try:
        state_data = json.loads(config.STATE_FILE.read_text()) if config.STATE_FILE.exists() else {}
        return float(state_data.get("imessage_high_water", 0))
    except (json.JSONDecodeError, ValueError):
        return 0.0


def _set_high_water(unix_ts: float):
    """Update the last-checked Unix timestamp in state.json."""
    try:
        state_data = json.loads(config.STATE_FILE.read_text()) if config.STATE_FILE.exists() else {}
        state_data["imessage_high_water"] = unix_ts
        config.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.STATE_FILE.write_text(json.dumps(state_data, indent=2))
    except Exception:
        log.debug("Failed to update iMessage high-water mark", exc_info=True)


def _find_handles_for_contact(
    conn: sqlite3.Connection,
    contact: str,
    handle_map: dict[int, str],
) -> list[int]:
    """Find handle ROWIDs matching a contact name or identifier."""
    # First try direct handle match (phone/email)
    direct = conn.execute(
        "SELECT ROWID FROM handle WHERE id LIKE ?",
        (f"%{contact}%",),
    ).fetchall()
    if direct:
        return [r["ROWID"] for r in direct]

    # Try resolving contact name to phone numbers via Apple Contacts
    try:
        contacts_conn = sqlite3.connect(
            f"file:{config.CONTACTS_DB}?mode=ro", uri=True,
        )
        contacts_conn.row_factory = sqlite3.Row
        try:
            q = f"%{contact}%"
            phone_rows = contacts_conn.execute(
                """SELECT p.ZFULLNUMBER FROM ZABCDRECORD r
                   JOIN ZABCDPHONENUMBER p ON r.Z_PK = p.ZOWNER
                   WHERE r.ZFIRSTNAME LIKE ? COLLATE NOCASE
                      OR r.ZLASTNAME LIKE ? COLLATE NOCASE
                      OR (r.ZFIRSTNAME || ' ' || r.ZLASTNAME) LIKE ? COLLATE NOCASE""",
                (q, q, q),
            ).fetchall()

            email_rows = contacts_conn.execute(
                """SELECT e.ZADDRESS FROM ZABCDRECORD r
                   JOIN ZABCDEMAILADDRESS e ON r.Z_PK = e.ZOWNER
                   WHERE r.ZFIRSTNAME LIKE ? COLLATE NOCASE
                      OR r.ZLASTNAME LIKE ? COLLATE NOCASE
                      OR (r.ZFIRSTNAME || ' ' || r.ZLASTNAME) LIKE ? COLLATE NOCASE""",
                (q, q, q),
            ).fetchall()

            identifiers = set()
            for r in phone_rows:
                if r["ZFULLNUMBER"]:
                    identifiers.add(r["ZFULLNUMBER"])
            for r in email_rows:
                if r["ZADDRESS"]:
                    identifiers.add(r["ZADDRESS"])

            if identifiers:
                matching = []
                for rowid, handle_id in handle_map.items():
                    handle_norm = _normalize_phone_for_match(handle_id)
                    for ident in identifiers:
                        ident_norm = _normalize_phone_for_match(str(ident))
                        if handle_norm and ident_norm and handle_norm == ident_norm:
                            matching.append(rowid)
                            break
                        if str(handle_id).lower() == str(ident).lower():
                            matching.append(rowid)
                            break
                return matching
        finally:
            contacts_conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        log.debug("Could not cross-reference contacts for iMessage lookup", exc_info=True)

    return []


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@tool(
    "imessage_search",
    "Search iMessages by keyword and/or contact name/number. "
    "Returns matching messages from the last N days.",
    {"query": str, "contact": str, "days": int, "limit": int},
)
async def imessage_search(args: dict) -> dict:
    query = args.get("query", "")
    contact = args.get("contact", "")
    days = args.get("days", 7)
    limit = min(args.get("limit", 50), 200)

    if not query and not contact:
        return _error_result("Please provide a query and/or contact to search for.")

    apple_threshold = _unix_to_apple(time.time() - days * 86400)

    try:
        conn = _connect()
        try:
            handle_map = _build_handle_map(conn)

            handle_ids = None
            if contact:
                handle_ids = _find_handles_for_contact(conn, contact, handle_map)
                if not handle_ids:
                    return _text_result({"messages": [], "note": f"No handles found for '{contact}'"})

            sql = "SELECT m.ROWID, m.text, m.date, m.is_from_me, m.handle_id FROM message m WHERE m.date > ?"
            params: list = [apple_threshold]

            if query:
                sql += " AND m.text LIKE ?"
                params.append(f"%{query}%")

            if handle_ids is not None:
                placeholders = ",".join("?" * len(handle_ids))
                sql += f" AND m.handle_id IN ({placeholders})"
                params.extend(handle_ids)

            sql += " ORDER BY m.date DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(sql, params).fetchall()
            _prewarm_contacts(rows, handle_map)
            messages = [_format_message(r, handle_map) for r in rows]
            return _text_result(messages)
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        return _error_result(f"iMessage database not accessible: {e}")


@tool(
    "imessage_recent",
    "Get recent iMessages from the last N hours. Shows sender handles, "
    "message text, and timestamps.",
    {"hours": int, "limit": int},
)
async def imessage_recent(args: dict) -> dict:
    hours = args.get("hours", 24)
    limit = min(args.get("limit", 50), 200)

    apple_threshold = _unix_to_apple(time.time() - hours * 3600)

    try:
        conn = _connect()
        try:
            handle_map = _build_handle_map(conn)
            rows = conn.execute(
                """SELECT m.ROWID, m.text, m.date, m.is_from_me, m.handle_id
                   FROM message m
                   WHERE m.date > ? AND m.text IS NOT NULL AND m.text != ''
                   ORDER BY m.date DESC LIMIT ?""",
                (apple_threshold, limit),
            ).fetchall()
            _prewarm_contacts(rows, handle_map)
            messages = [_format_message(r, handle_map) for r in rows]
            if not messages:
                return _text_result(f"No messages in the last {hours} hours.")
            return _text_result(messages)
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        return _error_result(f"iMessage database not accessible: {e}")


@tool(
    "imessage_thread",
    "Get the conversation thread with a specific contact. "
    "Accepts a contact name, phone number, or email address.",
    {"contact": str, "limit": int},
)
async def imessage_thread(args: dict) -> dict:
    contact = args.get("contact", "")
    limit = min(args.get("limit", 50), 200)

    if not contact:
        return _error_result("Please provide a contact name or number.")

    try:
        conn = _connect()
        try:
            handle_map = _build_handle_map(conn)
            handle_ids = _find_handles_for_contact(conn, contact, handle_map)

            if not handle_ids:
                return _text_result({"messages": [], "note": f"No conversation found for '{contact}'"})

            placeholders = ",".join("?" * len(handle_ids))
            rows = conn.execute(
                f"""SELECT m.ROWID, m.text, m.date, m.is_from_me, m.handle_id
                    FROM message m
                    WHERE m.handle_id IN ({placeholders})
                      AND m.text IS NOT NULL AND m.text != ''
                    ORDER BY m.date DESC LIMIT ?""",
                [*handle_ids, limit],
            ).fetchall()

            _prewarm_contacts(rows, handle_map)
            messages = [_format_message(r, handle_map) for r in reversed(rows)]
            return _text_result(messages)
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        return _error_result(f"iMessage database not accessible: {e}")


@tool(
    "imessage_unread",
    "Get iMessages received since the last check. Uses a high-water "
    "timestamp to track what's been seen. Updates the marker after reading.",
    {},
)
async def imessage_unread(args: dict) -> dict:
    high_water = _get_high_water()
    apple_threshold = _unix_to_apple(high_water) if high_water > 0 else _unix_to_apple(time.time() - 3600)

    try:
        conn = _connect()
        try:
            handle_map = _build_handle_map(conn)
            rows = conn.execute(
                """SELECT m.ROWID, m.text, m.date, m.is_from_me, m.handle_id
                   FROM message m
                   WHERE m.date > ? AND m.text IS NOT NULL AND m.text != ''
                   ORDER BY m.date ASC""",
                (apple_threshold,),
            ).fetchall()

            _prewarm_contacts(rows, handle_map)
            messages = [_format_message(r, handle_map) for r in rows]

            # Update high-water mark
            if rows:
                latest_apple = max(r["date"] for r in rows)
                _set_high_water(_apple_to_unix(latest_apple))
            elif high_water == 0:
                _set_high_water(time.time())

            return _text_result({
                "messages": messages,
                "count": len(messages),
                "since": datetime.fromtimestamp(high_water, tz=timezone.utc).isoformat() if high_water > 0 else "first check",
            })
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        return _error_result(f"iMessage database not accessible: {e}")


# ---------------------------------------------------------------------------
# Heartbeat helper (called from heartbeat.py, not an MCP tool)
# ---------------------------------------------------------------------------

def get_new_messages_since(unix_ts: float) -> list[dict]:
    """Get new iMessages since a Unix timestamp.

    Returns list of message dicts with resolved contact names.
    Pre-warms the contacts cache with a single bulk lookup.
    """
    apple_threshold = _unix_to_apple(unix_ts)

    try:
        conn = _connect()
        try:
            handle_map = _build_handle_map(conn)
            rows = conn.execute(
                """SELECT m.ROWID, m.text, m.date, m.is_from_me, m.handle_id
                   FROM message m
                   WHERE m.date > ? AND m.is_from_me = 0
                     AND m.text IS NOT NULL AND m.text != ''
                   ORDER BY m.date ASC""",
                (apple_threshold,),
            ).fetchall()

            _prewarm_contacts(rows, handle_map)
            return [_format_message(r, handle_map) for r in rows]
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        log.debug("iMessage DB not accessible", exc_info=True)
        return []


def get_mention_messages_since(
    unix_ts: float,
    trigger_pattern: re.Pattern | None = None,
) -> list[dict]:
    """Get iMessages sent by *me* that contain the @molly trigger since *unix_ts*.

    Unlike ``get_new_messages_since`` (which reads received messages), this reads
    Brian's **own** sent messages (``is_from_me = 1``) to detect when he types
    ``@molly`` in an iMessage conversation.

    Each returned dict includes a ``chat_id`` key (the ROWID from the ``chat``
    table via ``chat_message_join``) so the caller can fetch thread context.
    """
    if trigger_pattern is None:
        trigger_pattern = config.TRIGGER_PATTERN

    apple_threshold = _unix_to_apple(unix_ts)

    try:
        conn = _connect()
        try:
            handle_map = _build_handle_map(conn)
            rows = conn.execute(
                """SELECT m.ROWID, m.text, m.date, m.is_from_me, m.handle_id
                   FROM message m
                   WHERE m.date > ? AND m.is_from_me = 1
                     AND m.text IS NOT NULL AND m.text != ''
                   ORDER BY m.date ASC""",
                (apple_threshold,),
            ).fetchall()

            results: list[dict] = []
            for r in rows:
                text = r["text"] or ""
                if not trigger_pattern.search(text):
                    continue

                msg = _format_message(r, handle_map)

                # Resolve the chat this message belongs to
                chat_row = conn.execute(
                    "SELECT cmj.chat_id FROM chat_message_join cmj "
                    "WHERE cmj.message_id = ? LIMIT 1",
                    (r["ROWID"],),
                ).fetchone()
                msg["chat_id"] = chat_row["chat_id"] if chat_row else None

                results.append(msg)

            return results
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        log.debug("iMessage DB not accessible for mention scan", exc_info=True)
        return []


def get_thread_context(
    chat_id: int,
    before_message_id: int,
    count: int = 8,
) -> list[dict]:
    """Fetch recent messages from a chat thread for surrounding context.

    Returns up to *count* messages from the chat identified by *chat_id* that
    appear **before** *before_message_id*, in chronological order.  Both sent
    and received messages are included so the caller sees the full conversation.
    """
    try:
        conn = _connect()
        try:
            handle_map = _build_handle_map(conn)
            rows = conn.execute(
                """SELECT m.ROWID, m.text, m.date, m.is_from_me, m.handle_id
                   FROM message m
                   JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                   WHERE cmj.chat_id = ? AND m.ROWID < ?
                     AND m.text IS NOT NULL AND m.text != ''
                   ORDER BY m.date DESC
                   LIMIT ?""",
                (chat_id, before_message_id, count),
            ).fetchall()

            # Reverse to chronological order (query was DESC for LIMIT)
            return [_format_message(r, handle_map) for r in reversed(rows)]
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        log.debug("iMessage DB not accessible for thread context", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

imessage_server = create_sdk_mcp_server(
    name="imessage",
    version="1.0.0",
    tools=[imessage_search, imessage_recent, imessage_thread, imessage_unread],
)
