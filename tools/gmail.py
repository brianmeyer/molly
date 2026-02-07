"""Gmail MCP tools for Molly.

Tools:
  gmail_search (AUTO)    — Search inbox with Gmail query syntax
  gmail_read   (AUTO)    — Read a specific email or thread
  gmail_draft  (CONFIRM) — Create a draft email
  gmail_send   (CONFIRM) — Send an email
  gmail_reply  (CONFIRM) — Reply to a thread
"""

import base64
import json
import logging
from email.mime.text import MIMEText

from claude_agent_sdk import create_sdk_mcp_server, tool

from tools.google_auth import get_gmail_service

log = logging.getLogger(__name__)


def _format_message(msg: dict, full: bool = False) -> dict:
    """Extract key fields from a Gmail API message."""
    headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}

    result = {
        "id": msg.get("id"),
        "thread_id": msg.get("threadId"),
        "from": headers.get("from"),
        "to": headers.get("to"),
        "subject": headers.get("subject"),
        "date": headers.get("date"),
        "snippet": msg.get("snippet"),
        "labels": msg.get("labelIds", []),
    }

    if full:
        result["body"] = _extract_body(msg.get("payload", {}))

    return result


def _extract_body(payload: dict) -> str:
    """Recursively extract plain text body from a Gmail message payload."""
    # Direct body
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")

    # Multipart — look for text/plain
    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")

    # Fallback: try HTML
    if payload.get("mimeType") == "text/html" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/html" and part.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")

    # Nested multipart
    for part in payload.get("parts", []):
        body = _extract_body(part)
        if body:
            return body

    return ""


def _text_result(data) -> dict:
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def _error_result(msg: str) -> dict:
    return {"content": [{"type": "text", "text": msg}], "is_error": True}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool(
    "gmail_search",
    "Search Gmail using Gmail query syntax (e.g. 'from:dave subject:deployment', "
    "'is:unread', 'newer_than:7d'). Returns message summaries.",
    {"query": str, "max_results": int},
)
async def gmail_search(args: dict) -> dict:
    try:
        query = args["query"]
        max_results = args.get("max_results", 10)
        service = get_gmail_service()

        result = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )

        messages = result.get("messages", [])
        if not messages:
            return _text_result(f"No emails matching '{query}'.")

        # Fetch details for each message
        details = []
        for msg_ref in messages:
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=msg_ref["id"], format="metadata",
                     metadataHeaders=["From", "To", "Subject", "Date"])
                .execute()
            )
            details.append(_format_message(msg))

        return _text_result(details)
    except Exception as e:
        log.error("gmail_search failed", exc_info=True)
        return _error_result(f"Gmail search failed: {e}")


@tool(
    "gmail_read",
    "Read the full content of a specific email by message ID. "
    "Returns headers, body text, and metadata.",
    {"message_id": str},
)
async def gmail_read(args: dict) -> dict:
    try:
        message_id = args["message_id"]
        service = get_gmail_service()

        msg = (
            service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        return _text_result(_format_message(msg, full=True))
    except Exception as e:
        log.error("gmail_read failed", exc_info=True)
        return _error_result(f"Gmail read failed: {e}")


@tool(
    "gmail_draft",
    "Create a draft email. The draft is saved but not sent. "
    "Provide recipient, subject, and body text.",
    {"to": str, "subject": str, "body": str},
)
async def gmail_draft(args: dict) -> dict:
    try:
        service = get_gmail_service()

        message = MIMEText(args["body"])
        message["to"] = args["to"]
        message["subject"] = args["subject"]

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        draft = (
            service.users()
            .drafts()
            .create(userId="me", body={"message": {"raw": raw}})
            .execute()
        )

        return _text_result({
            "status": "draft_created",
            "draft_id": draft.get("id"),
            "to": args["to"],
            "subject": args["subject"],
        })
    except Exception as e:
        log.error("gmail_draft failed", exc_info=True)
        return _error_result(f"Gmail draft failed: {e}")


@tool(
    "gmail_send",
    "Send an email immediately. Provide recipient email, subject, and body text. "
    "This sends the email right away (requires approval).",
    {"to": str, "subject": str, "body": str},
)
async def gmail_send(args: dict) -> dict:
    try:
        service = get_gmail_service()

        message = MIMEText(args["body"])
        message["to"] = args["to"]
        message["subject"] = args["subject"]

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        sent = (
            service.users()
            .messages()
            .send(userId="me", body={"raw": raw})
            .execute()
        )

        return _text_result({
            "status": "sent",
            "message_id": sent.get("id"),
            "to": args["to"],
            "subject": args["subject"],
        })
    except Exception as e:
        log.error("gmail_send failed", exc_info=True)
        return _error_result(f"Gmail send failed: {e}")


@tool(
    "gmail_reply",
    "Reply to an existing email thread. Provide the original message_id, "
    "thread_id, and your reply body text.",
    {"message_id": str, "thread_id": str, "body": str},
)
async def gmail_reply(args: dict) -> dict:
    try:
        service = get_gmail_service()

        # Fetch the original message for headers
        original = (
            service.users()
            .messages()
            .get(userId="me", id=args["message_id"], format="metadata",
                 metadataHeaders=["From", "Subject", "Message-ID"])
            .execute()
        )
        headers = {h["name"].lower(): h["value"] for h in original.get("payload", {}).get("headers", [])}

        reply_to = headers.get("from", "")
        subject = headers.get("subject", "")
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        message = MIMEText(args["body"])
        message["to"] = reply_to
        message["subject"] = subject
        message["In-Reply-To"] = headers.get("message-id", "")
        message["References"] = headers.get("message-id", "")

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        sent = (
            service.users()
            .messages()
            .send(userId="me", body={"raw": raw, "threadId": args["thread_id"]})
            .execute()
        )

        return _text_result({
            "status": "replied",
            "message_id": sent.get("id"),
            "thread_id": args["thread_id"],
            "to": reply_to,
            "subject": subject,
        })
    except Exception as e:
        log.error("gmail_reply failed", exc_info=True)
        return _error_result(f"Gmail reply failed: {e}")


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

gmail_server = create_sdk_mcp_server(
    name="gmail",
    version="1.0.0",
    tools=[
        gmail_search,
        gmail_read,
        gmail_draft,
        gmail_send,
        gmail_reply,
    ],
)
