"""Google Tasks API MCP tools for Molly.

Tools:
  tasks_list       (AUTO)    — List task lists
  tasks_list_tasks (AUTO)    — List tasks in a task list
  tasks_create     (AUTO)    — Create a new task
  tasks_complete   (CONFIRM) — Mark task as completed
  tasks_delete     (CONFIRM) — Delete a task
"""

import json
import logging

from claude_agent_sdk import create_sdk_mcp_server, tool

from tools.google_auth import get_tasks_service

log = logging.getLogger(__name__)


def _format_task(task: dict) -> dict:
    """Extract key fields from a Tasks API task resource."""
    return {
        "id": task.get("id"),
        "title": task.get("title"),
        "notes": task.get("notes"),
        "status": task.get("status"),
        "due": task.get("due"),
        "completed": task.get("completed"),
        "updated": task.get("updated"),
        "parent": task.get("parent"),
        "position": task.get("position"),
    }


def _text_result(data) -> dict:
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def _error_result(msg: str) -> dict:
    return {"content": [{"type": "text", "text": msg}], "is_error": True}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool(
    "tasks_list",
    "List all Google Task lists. Returns list IDs and titles.",
    {},
)
async def tasks_list(args: dict) -> dict:
    try:
        service = get_tasks_service()
        result = service.tasklists().list(maxResults=100).execute()

        lists = [
            {"id": tl.get("id"), "title": tl.get("title"), "updated": tl.get("updated")}
            for tl in result.get("items", [])
        ]
        if not lists:
            return _text_result("No task lists found.")
        return _text_result({"task_lists": lists})
    except Exception as e:
        log.error("tasks_list failed", exc_info=True)
        return _error_result(f"Tasks list failed: {e}")


@tool(
    "tasks_list_tasks",
    "List tasks in a specific task list. Use tasks_list to find list IDs. "
    "Defaults to the primary task list if tasklist_id is omitted.",
    {"tasklist_id": str},
)
async def tasks_list_tasks(args: dict) -> dict:
    try:
        tasklist_id = args.get("tasklist_id", "@default")
        show_completed = args.get("show_completed", True)
        service = get_tasks_service()

        kwargs = {"tasklist": tasklist_id, "maxResults": 100}
        if not show_completed:
            kwargs["showCompleted"] = False

        result = service.tasks().list(**kwargs).execute()

        tasks = [_format_task(t) for t in result.get("items", [])]
        if not tasks:
            return _text_result("No tasks found in this list.")
        data = {"tasks": tasks}
        if result.get("nextPageToken"):
            data["next_page_token"] = result["nextPageToken"]
        return _text_result(data)
    except Exception as e:
        log.error("tasks_list_tasks failed", exc_info=True)
        return _error_result(f"Tasks list tasks failed: {e}")


@tool(
    "tasks_create",
    "Create a new Google Task. Requires a title. Optionally provide notes, "
    "due date (RFC 3339), and tasklist_id (defaults to primary list).",
    {"title": str},
)
async def tasks_create(args: dict) -> dict:
    try:
        title = args["title"]
        tasklist_id = args.get("tasklist_id", "@default")
        service = get_tasks_service()

        body = {"title": title}
        if args.get("notes"):
            body["notes"] = args["notes"]
        if args.get("due"):
            body["due"] = args["due"]

        task = service.tasks().insert(tasklist=tasklist_id, body=body).execute()
        return _text_result({"created": _format_task(task)})
    except Exception as e:
        log.error("tasks_create failed", exc_info=True)
        return _error_result(f"Tasks create failed: {e}")


@tool(
    "tasks_complete",
    "Mark a Google Task as completed. Provide the task ID and optionally "
    "the tasklist_id (defaults to primary list).",
    {"task_id": str},
)
async def tasks_complete(args: dict) -> dict:
    try:
        task_id = args["task_id"]
        tasklist_id = args.get("tasklist_id", "@default")
        service = get_tasks_service()

        task = service.tasks().get(tasklist=tasklist_id, task=task_id).execute()
        task["status"] = "completed"
        updated = service.tasks().update(tasklist=tasklist_id, task=task_id, body=task).execute()
        return _text_result({"completed": _format_task(updated)})
    except Exception as e:
        log.error("tasks_complete failed", exc_info=True)
        return _error_result(f"Tasks complete failed: {e}")


@tool(
    "tasks_delete",
    "Delete a Google Task. Provide the task ID and optionally "
    "the tasklist_id (defaults to primary list).",
    {"task_id": str},
)
async def tasks_delete(args: dict) -> dict:
    try:
        task_id = args["task_id"]
        tasklist_id = args.get("tasklist_id", "@default")
        service = get_tasks_service()

        service.tasks().delete(tasklist=tasklist_id, task=task_id).execute()
        return _text_result(f"Task {task_id} deleted.")
    except Exception as e:
        log.error("tasks_delete failed", exc_info=True)
        return _error_result(f"Tasks delete failed: {e}")


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

tasks_server = create_sdk_mcp_server(
    name="google-tasks",
    version="1.0.0",
    tools=[tasks_list, tasks_list_tasks, tasks_create, tasks_complete, tasks_delete],
)
