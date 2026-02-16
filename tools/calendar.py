"""Google Calendar MCP tools for Molly.

Tools:
  calendar_list   (AUTO)    — List upcoming events for N days
  calendar_get    (AUTO)    — Get details of a specific event
  calendar_search (AUTO)    — Search events by query string
  calendar_create (AUTO)    — Create a new event
  calendar_update (CONFIRM) — Modify an existing event
  calendar_delete (CONFIRM) — Delete an event
"""

import json
import logging
from datetime import datetime, timedelta, timezone

from claude_agent_sdk import create_sdk_mcp_server, tool

import config
from tools.google_auth import get_calendar_service

log = logging.getLogger(__name__)


def _format_event(event: dict) -> dict:
    """Extract key fields from a Calendar API event into a clean dict."""
    start = event.get("start", {})
    end = event.get("end", {})
    return {
        "id": event.get("id"),
        "summary": event.get("summary", "(no title)"),
        "start": start.get("dateTime", start.get("date", "?")),
        "end": end.get("dateTime", end.get("date", "?")),
        "location": event.get("location"),
        "description": event.get("description"),
        "attendees": [
            {"email": a.get("email"), "status": a.get("responseStatus")}
            for a in event.get("attendees", [])
        ],
        "status": event.get("status"),
        "html_link": event.get("htmlLink"),
    }


def _text_result(data) -> dict:
    """Wrap data as a text content MCP response."""
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def _error_result(msg: str) -> dict:
    return {"content": [{"type": "text", "text": msg}], "is_error": True}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool(
    "calendar_list",
    "List upcoming calendar events. Returns events for the next N days (default 7). "
    "Shows title, time, location, and attendees.",
    {"days": int},
)
async def calendar_list(args: dict) -> dict:
    try:
        days = args.get("days", 7)
        service = get_calendar_service()

        now = datetime.now(timezone.utc)
        time_max = now + timedelta(days=days)

        result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now.isoformat(),
                timeMax=time_max.isoformat(),
                maxResults=50,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )

        events = [_format_event(e) for e in result.get("items", [])]
        if not events:
            return _text_result(f"No events in the next {days} days.")
        return _text_result(events)
    except Exception as e:
        log.error("calendar_list failed", exc_info=True)
        return _error_result(f"Calendar list failed: {e}")


@tool(
    "calendar_get",
    "Get full details of a specific calendar event by its event ID.",
    {"event_id": str},
)
async def calendar_get(args: dict) -> dict:
    try:
        event_id = args["event_id"]
        service = get_calendar_service()

        event = service.events().get(calendarId="primary", eventId=event_id).execute()
        return _text_result(_format_event(event))
    except Exception as e:
        log.error("calendar_get failed", exc_info=True)
        return _error_result(f"Calendar get failed: {e}")


@tool(
    "calendar_search",
    "Search calendar events by a text query. Searches event titles, descriptions, "
    "and locations. Returns matching events from the past 30 days to 90 days ahead.",
    {"query": str},
)
async def calendar_search(args: dict) -> dict:
    try:
        query = args["query"]
        service = get_calendar_service()

        now = datetime.now(timezone.utc)
        time_min = now - timedelta(days=30)
        time_max = now + timedelta(days=90)

        result = (
            service.events()
            .list(
                calendarId="primary",
                q=query,
                timeMin=time_min.isoformat(),
                timeMax=time_max.isoformat(),
                maxResults=20,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )

        events = [_format_event(e) for e in result.get("items", [])]
        if not events:
            return _text_result(f"No events matching '{query}'.")
        return _text_result(events)
    except Exception as e:
        log.error("calendar_search failed", exc_info=True)
        return _error_result(f"Calendar search failed: {e}")


@tool(
    "calendar_create",
    "Create a new calendar event. Requires title and start time. "
    "Optionally include end time, location, description, and attendee emails.",
    {
        "summary": str,
        "start": str,
        "end": str,
        "location": str,
        "description": str,
        "attendees": str,
    },
)
async def calendar_create(args: dict) -> dict:
    try:
        service = get_calendar_service()

        event_body: dict = {
            "summary": args["summary"],
            "start": _parse_datetime(args["start"]),
            "end": _parse_datetime(args.get("end", "")),
        }

        # If no end time, default to 1 hour after start
        if not args.get("end"):
            start_dt = _parse_to_datetime(args["start"])
            end_dt = start_dt + timedelta(hours=1)
            event_body["end"] = {"dateTime": end_dt.isoformat(), "timeZone": config.TIMEZONE}

        if args.get("location"):
            event_body["location"] = args["location"]
        if args.get("description"):
            event_body["description"] = args["description"]
        if args.get("attendees"):
            emails = [e.strip() for e in args["attendees"].split(",")]
            event_body["attendees"] = [{"email": e} for e in emails if e]

        created = service.events().insert(calendarId="primary", body=event_body).execute()
        return _text_result({
            "status": "created",
            "event": _format_event(created),
        })
    except Exception as e:
        log.error("calendar_create failed", exc_info=True)
        return _error_result(f"Calendar create failed: {e}")


@tool(
    "calendar_update",
    "Update an existing calendar event. Provide the event_id and any fields to change: "
    "summary, start, end, location, description, attendees.",
    {
        "event_id": str,
        "summary": str,
        "start": str,
        "end": str,
        "location": str,
        "description": str,
        "attendees": str,
    },
)
async def calendar_update(args: dict) -> dict:
    try:
        event_id = args["event_id"]
        service = get_calendar_service()

        # Fetch current event
        event = service.events().get(calendarId="primary", eventId=event_id).execute()

        # Apply updates
        if args.get("summary"):
            event["summary"] = args["summary"]
        if args.get("start"):
            event["start"] = _parse_datetime(args["start"])
        if args.get("end"):
            event["end"] = _parse_datetime(args["end"])
        if args.get("location"):
            event["location"] = args["location"]
        if args.get("description"):
            event["description"] = args["description"]
        if args.get("attendees"):
            emails = [e.strip() for e in args["attendees"].split(",")]
            event["attendees"] = [{"email": e} for e in emails if e]

        updated = (
            service.events()
            .update(calendarId="primary", eventId=event_id, body=event)
            .execute()
        )
        return _text_result({
            "status": "updated",
            "event": _format_event(updated),
        })
    except Exception as e:
        log.error("calendar_update failed", exc_info=True)
        return _error_result(f"Calendar update failed: {e}")


@tool(
    "calendar_delete",
    "Delete a calendar event by its event ID.",
    {"event_id": str},
)
async def calendar_delete(args: dict) -> dict:
    try:
        event_id = args["event_id"]
        service = get_calendar_service()

        service.events().delete(calendarId="primary", eventId=event_id).execute()
        return _text_result({"status": "deleted", "event_id": event_id})
    except Exception as e:
        log.error("calendar_delete failed", exc_info=True)
        return _error_result(f"Calendar delete failed: {e}")


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------

def _parse_datetime(dt_str: str) -> dict:
    """Parse a datetime string into a Calendar API datetime object."""
    if not dt_str:
        return {}

    # If it's just a date (YYYY-MM-DD), use date format
    if len(dt_str) == 10 and dt_str.count("-") == 2:
        return {"date": dt_str}

    # Otherwise parse as datetime and return with timezone
    dt = _parse_to_datetime(dt_str)
    return {"dateTime": dt.isoformat(), "timeZone": config.TIMEZONE}


def _parse_to_datetime(dt_str: str) -> datetime:
    """Parse various datetime string formats into a datetime object."""
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse datetime: {dt_str}")


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

calendar_server = create_sdk_mcp_server(
    name="google-calendar",
    version="1.0.0",
    tools=[
        calendar_list,
        calendar_get,
        calendar_search,
        calendar_create,
        calendar_update,
        calendar_delete,
    ],
)
