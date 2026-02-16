"""WhatsApp message search MCP tool for Molly.

Read-only access to the stored WhatsApp message history in messages.db.
Lets Molly search past group conversations, DMs, and her own sent messages.

Tools:
  whatsapp_search (AUTO) — Search stored WhatsApp messages by text, chat, sender
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone

from claude_agent_sdk import create_sdk_mcp_server, tool

import config
import db_pool

log = logging.getLogger(__name__)


def _connect() -> sqlite3.Connection:
    """Open a read-only connection to the WhatsApp message database."""
    uri = f"file:{config.DATABASE_PATH}?mode=ro"
    conn = db_pool.sqlite_connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _text_result(data) -> dict:
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def _error_result(msg: str) -> dict:
    return {"content": [{"type": "text", "text": msg}], "is_error": True}


def _format_message(row: sqlite3.Row) -> dict:
    """Format a message row for display."""
    return {
        "sender": row["sender_name"] or row["sender"].split("@")[0],
        "chat": row["chat_jid"].split("@")[0],
        "timestamp": row["timestamp"],
        "text": row["content"],
        "is_from_me": bool(row["is_from_me"]),
    }


@tool(
    "whatsapp_search",
    "Search Molly's stored WhatsApp message history. "
    "Find messages by keyword, chat, or sender. "
    "Use this to recall group conversations, past discussions, or what someone said.",
    {"query": str, "chat_id": str, "sender": str, "limit": int, "hours_back": int},
)
async def whatsapp_search(args: dict) -> dict:
    query_text = args.get("query", "")
    chat_id = args.get("chat_id", "")
    sender = args.get("sender", "")
    limit = min(args.get("limit", 20), 100)
    hours_back = min(args.get("hours_back", 24), 720)  # cap at 30 days

    if not query_text and not chat_id and not sender:
        return _error_result(
            "Provide at least one filter: query (text search), chat_id, or sender."
        )

    # Compute time threshold as ISO timestamp
    threshold = datetime.fromtimestamp(
        time.time() - hours_back * 3600, tz=timezone.utc
    ).isoformat()

    try:
        conn = _connect()
        try:
            sql = (
                "SELECT chat_jid, sender, sender_name, content, timestamp, is_from_me "
                "FROM messages WHERE timestamp > ?"
            )
            params: list = [threshold]

            if query_text:
                sql += " AND content LIKE ?"
                params.append(f"%{query_text}%")

            if chat_id:
                sql += " AND chat_jid LIKE ?"
                params.append(f"%{chat_id}%")

            if sender:
                sql += " AND (sender_name LIKE ? OR sender LIKE ?)"
                params.append(f"%{sender}%")
                params.append(f"%{sender}%")

            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(sql, params).fetchall()

            if not rows:
                filters = []
                if query_text:
                    filters.append(f"query='{query_text}'")
                if chat_id:
                    filters.append(f"chat='{chat_id}'")
                if sender:
                    filters.append(f"sender='{sender}'")
                return _text_result(
                    f"No messages found in the last {hours_back}h "
                    f"matching {', '.join(filters)}."
                )

            messages = [_format_message(r) for r in rows]
            return _text_result(messages)
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        return _error_result(f"Message database not accessible: {e}")


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

whatsapp_server = create_sdk_mcp_server(
    name="whatsapp-history",
    version="1.0.0",
    tools=[whatsapp_search],
)
