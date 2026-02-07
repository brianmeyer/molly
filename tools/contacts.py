"""Apple Contacts MCP tools for Molly.

Read-only access to the macOS AddressBook SQLite database.
Also exposes resolve_phone_to_name() for WhatsApp pipeline name resolution.

Tools:
  contacts_search  (AUTO) — Search by name, phone, email, or company
  contacts_get     (AUTO) — Full details for a specific contact
  contacts_list    (AUTO) — List all contacts (paginated)
  contacts_recent  (AUTO) — Contacts modified in last N days

Requires Full Disk Access or Contacts permission for the Python process.
"""

import json
import logging
import re
import sqlite3

from claude_agent_sdk import create_sdk_mcp_server, tool

import config

log = logging.getLogger(__name__)

# Phone normalization: strip everything except digits and leading +
_PHONE_STRIP = re.compile(r"[^\d+]")


def _normalize_phone(phone: str) -> str:
    """Normalize a phone number to digits only (with optional leading +)."""
    return _PHONE_STRIP.sub("", phone)


def _connect() -> sqlite3.Connection:
    """Open a read-only connection to the Apple Contacts database."""
    uri = f"file:{config.CONTACTS_DB}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _text_result(data) -> dict:
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def _error_result(msg: str) -> dict:
    return {"content": [{"type": "text", "text": msg}], "is_error": True}


def _format_contact(row: sqlite3.Row, conn: sqlite3.Connection) -> dict:
    """Build a contact dict from a ZABCDRECORD row + joined sub-tables."""
    pk = row["Z_PK"]
    first = row["ZFIRSTNAME"] or ""
    last = row["ZLASTNAME"] or ""
    name = f"{first} {last}".strip() or row["ZORGANIZATION"] or "(unnamed)"

    contact = {
        "id": pk,
        "name": name,
        "first_name": first,
        "last_name": last,
        "organization": row["ZORGANIZATION"] or "",
        "job_title": row["ZJOBTITLE"] or "",
        "notes": row["ZNOTE"] or "",
    }

    # Phone numbers
    phones = conn.execute(
        "SELECT ZFULLNUMBER, ZLABEL FROM ZABCDPHONENUMBER WHERE ZOWNER = ?",
        (pk,),
    ).fetchall()
    contact["phones"] = [
        {"number": p["ZFULLNUMBER"], "label": _clean_label(p["ZLABEL"])}
        for p in phones if p["ZFULLNUMBER"]
    ]

    # Email addresses
    emails = conn.execute(
        "SELECT ZADDRESS, ZLABEL FROM ZABCDEMAILADDRESS WHERE ZOWNER = ?",
        (pk,),
    ).fetchall()
    contact["emails"] = [
        {"address": e["ZADDRESS"], "label": _clean_label(e["ZLABEL"])}
        for e in emails if e["ZADDRESS"]
    ]

    # Postal addresses
    try:
        addrs = conn.execute(
            """SELECT ZSTREET, ZCITY, ZSTATE, ZZIPCODE, ZCOUNTRYNAME, ZLABEL
               FROM ZABCDPOSTALADDRESS WHERE ZOWNER = ?""",
            (pk,),
        ).fetchall()
        contact["addresses"] = [
            {
                "street": a["ZSTREET"] or "",
                "city": a["ZCITY"] or "",
                "state": a["ZSTATE"] or "",
                "zip": a["ZZIPCODE"] or "",
                "country": a["ZCOUNTRYNAME"] or "",
                "label": _clean_label(a["ZLABEL"]),
            }
            for a in addrs
        ]
    except sqlite3.OperationalError:
        contact["addresses"] = []

    return contact


def _clean_label(label: str | None) -> str:
    """Clean Apple's internal label format (_$!<Home>!$_ → Home)."""
    if not label:
        return ""
    return label.replace("_$!<", "").replace(">!$_", "")


def _contact_summary(row: sqlite3.Row) -> dict:
    """Lightweight contact summary (no sub-table joins)."""
    first = row["ZFIRSTNAME"] or ""
    last = row["ZLASTNAME"] or ""
    name = f"{first} {last}".strip() or row["ZORGANIZATION"] or "(unnamed)"
    return {
        "id": row["Z_PK"],
        "name": name,
        "organization": row["ZORGANIZATION"] or "",
    }


# ---------------------------------------------------------------------------
# Phone resolution for WhatsApp pipeline
# ---------------------------------------------------------------------------

_phone_cache: dict[str, str | None] = {}


def resolve_phone_to_name(phone: str) -> str | None:
    """Look up a phone number in Apple Contacts. Returns full name or None.

    Used by the WhatsApp pipeline to resolve sender JIDs to real names.
    Results are cached in-memory for the session lifetime.
    """
    normalized = _normalize_phone(phone)
    if not normalized:
        return None

    if normalized in _phone_cache:
        return _phone_cache[normalized]

    try:
        conn = _connect()
        try:
            rows = conn.execute(
                """SELECT r.ZFIRSTNAME, r.ZLASTNAME, r.ZORGANIZATION, p.ZFULLNUMBER
                   FROM ZABCDRECORD r
                   JOIN ZABCDPHONENUMBER p ON r.Z_PK = p.ZOWNER
                   WHERE p.ZFULLNUMBER IS NOT NULL""",
            ).fetchall()

            for row in rows:
                db_phone = _normalize_phone(row["ZFULLNUMBER"])
                # Match on last 10 digits (handles country code differences)
                if db_phone[-10:] == normalized[-10:] and len(normalized) >= 10:
                    first = row["ZFIRSTNAME"] or ""
                    last = row["ZLASTNAME"] or ""
                    name = f"{first} {last}".strip()
                    if not name:
                        name = row["ZORGANIZATION"] or None
                    _phone_cache[normalized] = name
                    return name
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        log.debug("Contacts DB not accessible (need Full Disk Access?)", exc_info=True)

    _phone_cache[normalized] = None
    return None


_email_cache: dict[str, str | None] = {}


def resolve_email_to_name(email: str) -> str | None:
    """Look up an email address in Apple Contacts. Returns full name or None.

    Used by the iMessage pipeline to resolve email-based handles.
    Results are cached in-memory for the session lifetime.
    """
    email_lower = email.lower().strip()
    if not email_lower or "@" not in email_lower:
        return None

    if email_lower in _email_cache:
        return _email_cache[email_lower]

    try:
        conn = _connect()
        try:
            row = conn.execute(
                """SELECT r.ZFIRSTNAME, r.ZLASTNAME, r.ZORGANIZATION
                   FROM ZABCDRECORD r
                   JOIN ZABCDEMAILADDRESS e ON r.Z_PK = e.ZOWNER
                   WHERE e.ZADDRESS = ? COLLATE NOCASE""",
                (email_lower,),
            ).fetchone()

            if row:
                first = row["ZFIRSTNAME"] or ""
                last = row["ZLASTNAME"] or ""
                name = f"{first} {last}".strip()
                if not name:
                    name = row["ZORGANIZATION"] or None
                _email_cache[email_lower] = name
                return name
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        log.debug("Contacts DB not accessible for email lookup", exc_info=True)

    _email_cache[email_lower] = None
    return None


def resolve_handle_to_name(handle: str) -> str | None:
    """Resolve a phone number or email address to a contact name.

    Tries phone lookup first, then email. Used by iMessage tools
    to resolve handles regardless of type.
    """
    if not handle:
        return None

    # Try phone first
    name = resolve_phone_to_name(handle)
    if name:
        return name

    # Try email
    if "@" in handle:
        return resolve_email_to_name(handle)

    return None


def clear_phone_cache():
    """Clear the phone resolution cache (call after contacts change)."""
    _phone_cache.clear()
    _email_cache.clear()


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@tool(
    "contacts_search",
    "Search Apple Contacts by name, phone number, email, or company. "
    "Returns matching contacts with full details (phones, emails, addresses).",
    {"query": str},
)
async def contacts_search(args: dict) -> dict:
    query = args.get("query", "")
    if not query:
        return _error_result("Please provide a search query.")

    try:
        conn = _connect()
        try:
            q = f"%{query}%"
            norm_phone = _normalize_phone(query)

            # Search by name, org
            rows = conn.execute(
                """SELECT DISTINCT r.* FROM ZABCDRECORD r
                   WHERE r.ZFIRSTNAME LIKE ? COLLATE NOCASE
                      OR r.ZLASTNAME LIKE ? COLLATE NOCASE
                      OR r.ZORGANIZATION LIKE ? COLLATE NOCASE
                      OR (r.ZFIRSTNAME || ' ' || r.ZLASTNAME) LIKE ? COLLATE NOCASE
                   LIMIT 20""",
                (q, q, q, q),
            ).fetchall()

            pks = {r["Z_PK"] for r in rows}

            # Search by phone number
            if norm_phone and len(norm_phone) >= 3:
                phone_rows = conn.execute(
                    """SELECT DISTINCT r.* FROM ZABCDRECORD r
                       JOIN ZABCDPHONENUMBER p ON r.Z_PK = p.ZOWNER
                       WHERE p.ZFULLNUMBER LIKE ?
                       LIMIT 10""",
                    (f"%{norm_phone[-10:]}%",),
                ).fetchall()
                for r in phone_rows:
                    if r["Z_PK"] not in pks:
                        rows.append(r)
                        pks.add(r["Z_PK"])

            # Search by email
            if "@" in query:
                email_rows = conn.execute(
                    """SELECT DISTINCT r.* FROM ZABCDRECORD r
                       JOIN ZABCDEMAILADDRESS e ON r.Z_PK = e.ZOWNER
                       WHERE e.ZADDRESS LIKE ? COLLATE NOCASE
                       LIMIT 10""",
                    (q,),
                ).fetchall()
                for r in email_rows:
                    if r["Z_PK"] not in pks:
                        rows.append(r)
                        pks.add(r["Z_PK"])

            contacts = [_format_contact(r, conn) for r in rows]
            if not contacts:
                return _text_result(f"No contacts found for '{query}'.")
            return _text_result(contacts)
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        return _error_result(f"Contacts database not accessible: {e}")


@tool(
    "contacts_get",
    "Get full details for a specific contact by their ID (from a search result).",
    {"contact_id": int},
)
async def contacts_get(args: dict) -> dict:
    contact_id = args.get("contact_id")
    if contact_id is None:
        return _error_result("Please provide a contact_id.")

    try:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM ZABCDRECORD WHERE Z_PK = ?",
                (contact_id,),
            ).fetchone()
            if not row:
                return _text_result(f"No contact found with ID {contact_id}.")
            return _text_result(_format_contact(row, conn))
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        return _error_result(f"Contacts database not accessible: {e}")


@tool(
    "contacts_list",
    "List all contacts with pagination. Returns name and organization for each.",
    {"offset": int, "limit": int},
)
async def contacts_list(args: dict) -> dict:
    offset = args.get("offset", 0)
    limit = min(args.get("limit", 50), 200)

    try:
        conn = _connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM ZABCDRECORD").fetchone()[0]
            rows = conn.execute(
                """SELECT * FROM ZABCDRECORD
                   ORDER BY ZLASTNAME, ZFIRSTNAME
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
            contacts = [_contact_summary(r) for r in rows]
            return _text_result({
                "total": total,
                "offset": offset,
                "count": len(contacts),
                "contacts": contacts,
            })
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        return _error_result(f"Contacts database not accessible: {e}")


@tool(
    "contacts_recent",
    "Get contacts modified or added in the last N days.",
    {"days": int},
)
async def contacts_recent(args: dict) -> dict:
    days = args.get("days", 7)
    import time
    apple_epoch_offset = 978307200
    threshold = time.time() - apple_epoch_offset - (days * 86400)

    try:
        conn = _connect()
        try:
            rows = conn.execute(
                """SELECT * FROM ZABCDRECORD
                   WHERE ZMODIFICATIONDATE > ? OR ZCREATIONDATE > ?
                   ORDER BY ZMODIFICATIONDATE DESC
                   LIMIT 50""",
                (threshold, threshold),
            ).fetchall()
            contacts = [_format_contact(r, conn) for r in rows]
            if not contacts:
                return _text_result(f"No contacts modified in the last {days} days.")
            return _text_result(contacts)
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        return _error_result(f"Contacts database not accessible: {e}")


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

contacts_server = create_sdk_mcp_server(
    name="apple-contacts",
    version="1.0.0",
    tools=[contacts_search, contacts_get, contacts_list, contacts_recent],
)
