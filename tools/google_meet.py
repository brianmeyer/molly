"""Google Meet API MCP tools for Molly.

Tools:
  meet_list        (AUTO) — List recent conference records
  meet_get         (AUTO) — Get meeting details
  meet_transcripts (AUTO) — Get transcript entries for a meeting
  meet_recordings  (AUTO) — List recordings for a meeting
"""

import json
import logging

from claude_agent_sdk import create_sdk_mcp_server, tool

from tools.google_auth import get_meet_service

log = logging.getLogger(__name__)

_MAX_TRANSCRIPT_ENTRIES = 500  # cap to avoid unbounded responses


def _format_conference(record: dict) -> dict:
    """Extract key fields from a Meet API conference record."""
    return {
        "name": record.get("name"),
        "start_time": record.get("startTime"),
        "end_time": record.get("endTime"),
        "space": record.get("space"),
        "expire_time": record.get("expireTime"),
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
    "meet_list",
    "List recent Google Meet conference records. Returns meeting names, "
    "start/end times, and space info. All parameters are optional.",
    {"page_size": int},
)
async def meet_list(args: dict) -> dict:
    try:
        service = get_meet_service()
        page_size = args.get("page_size", 25)

        kwargs = {"pageSize": min(page_size, 100)}
        if args.get("page_token"):
            kwargs["pageToken"] = args["page_token"]

        result = service.conferenceRecords().list(**kwargs).execute()

        records = [_format_conference(r) for r in result.get("conferenceRecords", [])]
        data = {"meetings": records}
        if result.get("nextPageToken"):
            data["next_page_token"] = result["nextPageToken"]

        if not records:
            return _text_result("No recent meetings found.")
        return _text_result(data)
    except Exception as e:
        log.error("meet_list failed", exc_info=True)
        return _error_result(f"Meet list failed: {e}")


@tool(
    "meet_get",
    "Get details of a specific Google Meet conference record by name "
    "(e.g. 'conferenceRecords/abc123').",
    {"name": str},
)
async def meet_get(args: dict) -> dict:
    try:
        name = args["name"]
        service = get_meet_service()

        record = service.conferenceRecords().get(name=name).execute()
        return _text_result(_format_conference(record))
    except Exception as e:
        log.error("meet_get failed", exc_info=True)
        return _error_result(f"Meet get failed: {e}")


@tool(
    "meet_transcripts",
    "Get transcript entries for a Google Meet conference. Provide the conference "
    "record name (e.g. 'conferenceRecords/abc123'). Returns transcript text entries "
    "(capped at 500 entries).",
    {"conference_name": str},
)
async def meet_transcripts(args: dict) -> dict:
    try:
        conference_name = args["conference_name"]
        service = get_meet_service()

        # List transcripts for this conference
        transcripts_result = (
            service.conferenceRecords()
            .transcripts()
            .list(parent=conference_name)
            .execute()
        )

        transcripts = transcripts_result.get("transcripts", [])
        if not transcripts:
            return _text_result("No transcripts available for this meeting.")

        # Get entries for each transcript, paginating through all pages
        all_entries = []
        for transcript in transcripts:
            transcript_name = transcript.get("name", "")
            page_token = None
            while len(all_entries) < _MAX_TRANSCRIPT_ENTRIES:
                kwargs = {"parent": transcript_name}
                if page_token:
                    kwargs["pageToken"] = page_token
                entries_result = (
                    service.conferenceRecords()
                    .transcripts()
                    .entries()
                    .list(**kwargs)
                    .execute()
                )
                for entry in entries_result.get("transcriptEntries", []):
                    all_entries.append({
                        "participant": entry.get("participant"),
                        "text": entry.get("text"),
                        "start_time": entry.get("startTime"),
                        "end_time": entry.get("endTime"),
                        "language_code": entry.get("languageCode"),
                    })
                    if len(all_entries) >= _MAX_TRANSCRIPT_ENTRIES:
                        break
                page_token = entries_result.get("nextPageToken")
                if not page_token:
                    break

        if not all_entries:
            return _text_result("Transcripts exist but no entries found.")

        data = {"entries": all_entries}
        if len(all_entries) >= _MAX_TRANSCRIPT_ENTRIES:
            data["note"] = f"Truncated at {_MAX_TRANSCRIPT_ENTRIES} entries."
        return _text_result(data)
    except Exception as e:
        log.error("meet_transcripts failed", exc_info=True)
        return _error_result(f"Meet transcripts failed: {e}")


@tool(
    "meet_recordings",
    "List recordings for a Google Meet conference. Provide the conference "
    "record name (e.g. 'conferenceRecords/abc123').",
    {"conference_name": str},
)
async def meet_recordings(args: dict) -> dict:
    try:
        conference_name = args["conference_name"]
        service = get_meet_service()

        result = (
            service.conferenceRecords()
            .recordings()
            .list(parent=conference_name)
            .execute()
        )

        recordings = []
        for rec in result.get("recordings", []):
            recordings.append({
                "name": rec.get("name"),
                "state": rec.get("state"),
                "start_time": rec.get("startTime"),
                "end_time": rec.get("endTime"),
                "drive_destination": rec.get("driveDestination"),
            })

        if not recordings:
            return _text_result("No recordings available for this meeting.")
        return _text_result(recordings)
    except Exception as e:
        log.error("meet_recordings failed", exc_info=True)
        return _error_result(f"Meet recordings failed: {e}")


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

meet_server = create_sdk_mcp_server(
    name="google-meet",
    version="1.0.0",
    tools=[meet_list, meet_get, meet_transcripts, meet_recordings],
)
