"""Apple Reminders MCP tools for Molly.

Tools:
  reminders_lists  (AUTO)    — List reminder lists in Apple Reminders
  list_reminders   (AUTO)    — List reminders in a list (active by default)
  search_reminders (AUTO)    — Search reminder titles in a list
  create_reminder  (CONFIRM) — Create a new reminder

Implementation uses osascript against the Reminders app so created reminders
sync through iCloud to iPhone/iPad when Apple's account sync is enabled.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

log = logging.getLogger(__name__)

DEFAULT_LIST_NAME = "Molly"
_RECORD_SEP = chr(30)
_FIELD_SEP = chr(31)


def _text_result(data: Any) -> dict:
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def _error_result(msg: str) -> dict:
    return {"content": [{"type": "text", "text": msg}], "is_error": True}


def _run_osascript(script_lines: list[str], args: list[str] | None = None) -> str:
    cmd: list[str] = ["osascript"]
    for line in script_lines:
        cmd.extend(["-e", line])
    if args:
        cmd.extend(args)

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or "osascript execution failed"
        raise RuntimeError(err)
    return (proc.stdout or "").strip()


def _epoch_to_iso(raw_epoch: str) -> str:
    text = (raw_epoch or "").strip()
    if not text:
        return ""
    try:
        return datetime.fromtimestamp(float(text), tz=timezone.utc).isoformat()
    except Exception:
        return ""


def _list_lists_script() -> list[str]:
    return [
        "on run argv",
        "    set recordSep to character id 30",
        '    tell application "Reminders"',
        "        set listNames to name of every list",
        "        if (count of listNames) = 0 then return \"\"",
        "        set AppleScript's text item delimiters to recordSep",
        "        return listNames as text",
        "    end tell",
        "end run",
    ]


def _list_reminders_script() -> list[str]:
    return [
        "on run argv",
        "    set listName to item 1 of argv",
        "    set includeCompleted to false",
        "    if (count of argv) > 1 then",
        "        set includeCompleted to ((item 2 of argv) is \"1\")",
        "    end if",
        "    set fieldSep to character id 31",
        "    set recordSep to character id 30",
        '    set baseEpoch to date "Thursday, January 1, 1970 at 12:00:00 AM"',
        '    tell application "Reminders"',
        "        if not (exists list listName) then return \"\"",
        "        set targetList to list listName",
        "        set linesOut to {}",
        "        repeat with r in reminders of targetList",
        "            set isDone to completed of r",
        "            if includeCompleted or (isDone is false) then",
        "                set rid to id of r as text",
        "                set titleText to my sanitize(name of r, fieldSep, recordSep)",
        "                set notesText to \"\"",
        "                if (body of r) is not missing value then",
        "                    set notesText to my sanitize(body of r, fieldSep, recordSep)",
        "                end if",
        "                set dueEpochText to \"\"",
        "                if (due date of r) is not missing value then",
        "                    set dueEpochText to ((due date of r) - baseEpoch) as integer",
        "                end if",
        "                set completionEpochText to \"\"",
        "                if (completion date of r) is not missing value then",
        "                    set completionEpochText to ((completion date of r) - baseEpoch) as integer",
        "                end if",
        "                set doneText to \"0\"",
        "                if isDone then set doneText to \"1\"",
        "                set lineText to rid & fieldSep & titleText & fieldSep & doneText & fieldSep & (dueEpochText as text) & fieldSep & (completionEpochText as text) & fieldSep & notesText",
        "                set end of linesOut to lineText",
        "            end if",
        "        end repeat",
        "    end tell",
        "    if (count of linesOut) = 0 then return \"\"",
        "    set AppleScript's text item delimiters to recordSep",
        "    return linesOut as text",
        "end run",
        "on sanitize(valueText, fieldSep, recordSep)",
        "    set txt to valueText as text",
        "    set txt to my replace_text(txt, fieldSep, \" \")",
        "    set txt to my replace_text(txt, recordSep, \" \")",
        "    set txt to my replace_text(txt, return, \" \")",
        "    set txt to my replace_text(txt, linefeed, \" \")",
        "    return txt",
        "end sanitize",
        "on replace_text(subjectText, findText, replaceText)",
        "    set AppleScript's text item delimiters to findText",
        "    set textItems to text items of subjectText",
        "    set AppleScript's text item delimiters to replaceText",
        "    set joinedText to textItems as text",
        "    set AppleScript's text item delimiters to \"\"",
        "    return joinedText",
        "end replace_text",
    ]


def _create_reminder_script() -> list[str]:
    return [
        "on run argv",
        "    set listName to item 1 of argv",
        "    set titleText to item 2 of argv",
        "    set notesText to \"\"",
        "    if (count of argv) > 2 then set notesText to item 3 of argv",
        "    set dueEpochText to \"\"",
        "    if (count of argv) > 3 then set dueEpochText to item 4 of argv",
        '    set baseEpoch to date "Thursday, January 1, 1970 at 12:00:00 AM"',
        '    tell application "Reminders"',
        "        if not (exists list listName) then",
        "            make new list with properties {name:listName}",
        "        end if",
        "        set targetList to list listName",
        "        set reminderProps to {name:titleText, body:notesText}",
        "        set newReminder to make new reminder at end of reminders of targetList with properties reminderProps",
        "        if dueEpochText is not \"\" then",
        "            set due date of newReminder to (baseEpoch + (dueEpochText as integer))",
        "        end if",
        "        return id of newReminder as text",
        "    end tell",
        "end run",
    ]


def list_lists() -> list[str]:
    raw = _run_osascript(_list_lists_script())
    if not raw:
        return []
    return [name.strip() for name in raw.split(_RECORD_SEP) if name.strip()]


def ensure_list_exists(list_name: str) -> bool:
    list_name = (list_name or "").strip()
    if not list_name:
        raise ValueError("list_name is required")
    script = [
        "on run argv",
        "    set listName to item 1 of argv",
        '    tell application "Reminders"',
        "        if not (exists list listName) then",
        "            make new list with properties {name:listName}",
        "        end if",
        "    end tell",
        "    return \"ok\"",
        "end run",
    ]
    _run_osascript(script, [list_name])
    return True


def list_reminders(list_name: str = DEFAULT_LIST_NAME, include_completed: bool = False) -> list[dict]:
    list_name = (list_name or "").strip() or DEFAULT_LIST_NAME
    raw = _run_osascript(_list_reminders_script(), [list_name, "1" if include_completed else "0"])
    if not raw:
        return []

    rows: list[dict] = []
    for line in raw.split(_RECORD_SEP):
        if not line:
            continue
        parts = line.split(_FIELD_SEP)
        if len(parts) < 6:
            continue

        reminder_id, title, done_raw, due_epoch, completion_epoch, notes = parts[:6]
        completed = done_raw.strip() == "1"
        rows.append(
            {
                "id": reminder_id.strip(),
                "title": title.strip(),
                "completed": completed,
                "due_at": _epoch_to_iso(due_epoch),
                "completed_at": _epoch_to_iso(completion_epoch),
                "notes": notes.strip(),
                "list_name": list_name,
            }
        )
    return rows


def search_reminders(
    query: str,
    list_name: str = DEFAULT_LIST_NAME,
    include_completed: bool = False,
) -> list[dict]:
    needle = (query or "").strip().lower()
    if not needle:
        return []
    reminders = list_reminders(list_name=list_name, include_completed=include_completed)
    return [r for r in reminders if needle in str(r.get("title", "")).lower()]


def create_reminder(
    title: str,
    notes: str = "",
    due_at: datetime | None = None,
    list_name: str = DEFAULT_LIST_NAME,
) -> dict:
    title = (title or "").strip()
    if not title:
        raise ValueError("title is required")
    list_name = (list_name or "").strip() or DEFAULT_LIST_NAME
    notes = (notes or "").strip()

    due_epoch = ""
    if due_at is not None:
        due_dt = due_at
        if due_dt.tzinfo is None:
            due_dt = due_dt.replace(tzinfo=timezone.utc)
        due_epoch = str(int(due_dt.timestamp()))

    reminder_id = _run_osascript(
        _create_reminder_script(),
        [list_name, title, notes, due_epoch],
    ).strip()
    return {
        "id": reminder_id,
        "title": title,
        "notes": notes,
        "due_at": due_at.isoformat() if due_at else "",
        "list_name": list_name,
    }


@tool(
    "reminders_lists",
    "List Apple Reminders lists available on this Mac/iCloud account.",
    {"type": "object", "properties": {}},
)
async def reminders_lists(args: dict) -> dict:
    del args
    try:
        return _text_result(list_lists())
    except Exception as e:
        log.error("reminders_lists failed", exc_info=True)
        return _error_result(f"Reminders list lookup failed: {e}")


@tool(
    "list_reminders",
    "List reminders in an Apple Reminders list (active by default).",
    {
        "type": "object",
        "properties": {
            "list_name": {"type": "string"},
            "include_completed": {"type": "boolean"},
        },
    },
)
async def reminders_list(args: dict) -> dict:
    try:
        list_name = str(args.get("list_name", DEFAULT_LIST_NAME) or DEFAULT_LIST_NAME)
        include_completed = bool(args.get("include_completed", False))
        rows = list_reminders(list_name=list_name, include_completed=include_completed)
        return _text_result(rows)
    except Exception as e:
        log.error("reminders_list failed", exc_info=True)
        return _error_result(f"Reminders fetch failed: {e}")


@tool(
    "search_reminders",
    "Search reminder titles in a list (defaults to list 'Molly').",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "list_name": {"type": "string"},
            "include_completed": {"type": "boolean"},
        },
        "required": ["query"],
    },
)
async def reminders_search(args: dict) -> dict:
    try:
        query = str(args.get("query", "")).strip()
        list_name = str(args.get("list_name", DEFAULT_LIST_NAME) or DEFAULT_LIST_NAME)
        include_completed = bool(args.get("include_completed", False))
        rows = search_reminders(
            query=query,
            list_name=list_name,
            include_completed=include_completed,
        )
        return _text_result(rows)
    except Exception as e:
        log.error("reminders_search failed", exc_info=True)
        return _error_result(f"Reminder search failed: {e}")


@tool(
    "create_reminder",
    "Create an Apple Reminder in a target list (defaults to list 'Molly').",
    {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "notes": {"type": "string"},
            "due_at": {"type": "string"},
            "list_name": {"type": "string"},
        },
        "required": ["title"],
    },
)
async def reminders_create(args: dict) -> dict:
    try:
        title = str(args.get("title", "")).strip()
        notes = str(args.get("notes", "")).strip()
        list_name = str(args.get("list_name", DEFAULT_LIST_NAME) or DEFAULT_LIST_NAME)
        due_at_raw = str(args.get("due_at", "")).strip()

        due_at: datetime | None = None
        if due_at_raw:
            due_at = datetime.fromisoformat(due_at_raw.replace("Z", "+00:00"))

        reminder = create_reminder(
            title=title,
            notes=notes,
            due_at=due_at,
            list_name=list_name,
        )
        return _text_result({"status": "created", "reminder": reminder})
    except Exception as e:
        log.error("reminders_create failed", exc_info=True)
        return _error_result(f"Reminder create failed: {e}")


reminders_server = create_sdk_mcp_server(
    name="apple-mcp",
    version="1.0.0",
    tools=[reminders_lists, reminders_list, reminders_search, reminders_create],
)
