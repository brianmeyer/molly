"""Google Drive API MCP tools for Molly.

Tools:
  drive_search (AUTO) — Search files by name or query
  drive_get    (AUTO) — Get file metadata
  drive_read   (AUTO) — Read/export file content
"""

import json
import logging

from claude_agent_sdk import create_sdk_mcp_server, tool

from tools.google_auth import get_drive_service
from utils import track_latency

log = logging.getLogger(__name__)

_FILE_FIELDS = "id,name,mimeType,modifiedTime,size,owners,webViewLink,parents"

# Google Workspace MIME types that need export (not direct download)
_EXPORT_MIME_TYPES = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
    "application/vnd.google-apps.drawing": "application/pdf",
}

_MAX_CONTENT_BYTES = 100_000  # 100KB content cap
_MAX_DOWNLOAD_BYTES = 10_000_000  # 10MB — refuse to download larger files


def _format_file(f: dict) -> dict:
    """Extract key fields from a Drive API file resource."""
    owners = f.get("owners", [])
    return {
        "id": f.get("id"),
        "name": f.get("name"),
        "mime_type": f.get("mimeType"),
        "modified": f.get("modifiedTime"),
        "size": f.get("size"),
        "owner": owners[0].get("displayName") if owners else None,
        "web_link": f.get("webViewLink"),
    }


def _text_result(data) -> dict:
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def _error_result(msg: str) -> dict:
    return {"content": [{"type": "text", "text": msg}], "is_error": True}


def _escape_drive_query(user_input: str) -> str:
    """Escape a user string for use in a Drive API name-contains query."""
    return user_input.replace("\\", "\\\\").replace("'", "\\'")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool(
    "drive_search",
    "Search Google Drive files by name or query. Pass a simple string to search "
    "by file name, or use Drive query syntax (e.g. \"mimeType='application/pdf'\").",
    {"query": str},
)
@track_latency("google_drive")
async def drive_search(args: dict) -> dict:
    try:
        query = args["query"]
        max_results = args.get("max_results", 20)
        service = get_drive_service()

        # If the query doesn't look like Drive query syntax, wrap as name search
        if "=" not in query and " contains " not in query:
            escaped = _escape_drive_query(query)
            q = f"name contains '{escaped}' and trashed = false"
        else:
            # Append trashed=false if not already present
            q = query
            if "trashed" not in query.lower():
                q = f"({query}) and trashed = false"

        result = (
            service.files()
            .list(
                q=q,
                fields=f"files({_FILE_FIELDS}),nextPageToken",
                pageSize=min(max_results, 100),
            )
            .execute()
        )

        files = [_format_file(f) for f in result.get("files", [])]
        if not files:
            return _text_result(f"No files matching '{query}'.")
        data = {"files": files}
        if result.get("nextPageToken"):
            data["next_page_token"] = result["nextPageToken"]
        return _text_result(data)
    except Exception as e:
        log.error("drive_search failed", exc_info=True)
        return _error_result(f"Drive search failed: {e}")


@tool(
    "drive_get",
    "Get metadata for a specific Google Drive file by file ID.",
    {"file_id": str},
)
@track_latency("google_drive")
async def drive_get(args: dict) -> dict:
    try:
        file_id = args["file_id"]
        service = get_drive_service()

        f = service.files().get(fileId=file_id, fields=_FILE_FIELDS).execute()
        return _text_result(_format_file(f))
    except Exception as e:
        log.error("drive_get failed", exc_info=True)
        return _error_result(f"Drive get failed: {e}")


@tool(
    "drive_read",
    "Read the content of a Google Drive file. Google Docs are exported as plain text, "
    "Sheets as CSV. Other text files are returned as-is. "
    "Binary files return metadata only. Max 100KB of content.",
    {"file_id": str},
)
@track_latency("google_drive")
async def drive_read(args: dict) -> dict:
    try:
        file_id = args["file_id"]
        service = get_drive_service()

        # Get full file metadata
        meta = service.files().get(fileId=file_id, fields=_FILE_FIELDS).execute()
        mime_type = meta.get("mimeType", "")
        name = meta.get("name", "unknown")

        # Check size before downloading (Google Workspace files have no size field)
        file_size = int(meta.get("size") or 0)
        if file_size > _MAX_DOWNLOAD_BYTES:
            return _text_result({
                "name": name,
                "mime_type": mime_type,
                "note": f"File too large ({file_size:,} bytes). Use web_link to view.",
                "metadata": _format_file(meta),
            })

        # Google Workspace files need export
        if mime_type in _EXPORT_MIME_TYPES:
            export_mime = _EXPORT_MIME_TYPES[mime_type]
            content = (
                service.files()
                .export_media(fileId=file_id, mimeType=export_mime)
                .execute()
            )
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            if len(content) > _MAX_CONTENT_BYTES:
                content = content[:_MAX_CONTENT_BYTES] + "\n\n... (truncated at 100KB)"
            return _text_result({"name": name, "mime_type": mime_type, "content": content})

        # Regular files — try to download as text
        if mime_type.startswith("text/") or mime_type in (
            "application/json",
            "application/xml",
            "application/javascript",
        ):
            content = service.files().get_media(fileId=file_id).execute()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            if len(content) > _MAX_CONTENT_BYTES:
                content = content[:_MAX_CONTENT_BYTES] + "\n\n... (truncated at 100KB)"
            return _text_result({"name": name, "mime_type": mime_type, "content": content})

        # Binary files — return metadata only
        return _text_result({
            "name": name,
            "mime_type": mime_type,
            "note": "Binary file — content not readable as text. Use web_link to view.",
            "metadata": _format_file(meta),
        })
    except Exception as e:
        log.error("drive_read failed", exc_info=True)
        return _error_result(f"Drive read failed: {e}")


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

drive_server = create_sdk_mcp_server(
    name="google-drive",
    version="1.0.0",
    tools=[drive_search, drive_get, drive_read],
)
