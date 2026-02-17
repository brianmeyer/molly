"""Google People API MCP tools for Molly.

Tools:
  people_search (AUTO) — Search contacts by name or email
  people_get    (AUTO) — Get full contact details by resource name
  people_list   (AUTO) — List connections (paginated)
"""

import json
import logging

from claude_agent_sdk import create_sdk_mcp_server, tool

from tools.google_auth import get_people_service
from utils import track_latency

log = logging.getLogger(__name__)

_PERSON_FIELDS = "names,emailAddresses,phoneNumbers,organizations,birthdays,addresses"


def _format_person(person: dict) -> dict:
    """Extract key fields from a People API person resource."""
    names = person.get("names", [])
    emails = person.get("emailAddresses", [])
    phones = person.get("phoneNumbers", [])
    orgs = person.get("organizations", [])
    birthdays = person.get("birthdays", [])
    addresses = person.get("addresses", [])
    return {
        "resource_name": person.get("resourceName"),
        "name": names[0].get("displayName") if names else None,
        "emails": [e.get("value") for e in emails],
        "phones": [p.get("value") for p in phones],
        "organization": orgs[0].get("name") if orgs else None,
        "title": orgs[0].get("title") if orgs else None,
        "birthday": birthdays[0].get("text") if birthdays else None,
        "address": addresses[0].get("formattedValue") if addresses else None,
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
    "people_search",
    "Search Google Contacts by name or email. Returns matching contacts "
    "with names, emails, phones, and organizations.",
    {"query": str},
)
@track_latency("google_people")
async def people_search(args: dict) -> dict:
    try:
        query = args["query"]
        page_size = args.get("page_size", 10)
        service = get_people_service()

        result = (
            service.people()
            .searchContacts(query=query, readMask=_PERSON_FIELDS, pageSize=min(page_size, 30))
            .execute()
        )

        contacts = [_format_person(r.get("person", {})) for r in result.get("results", [])]
        if not contacts:
            return _text_result(f"No contacts matching '{query}'.")
        return _text_result({"contacts": contacts})
    except Exception as e:
        log.error("people_search failed", exc_info=True)
        return _error_result(f"People search failed: {e}")


@tool(
    "people_get",
    "Get full details of a Google Contact by resource name "
    "(e.g. 'people/c1234567890').",
    {"resource_name": str},
)
@track_latency("google_people")
async def people_get(args: dict) -> dict:
    try:
        resource_name = args["resource_name"]
        service = get_people_service()

        person = (
            service.people()
            .get(resourceName=resource_name, personFields=_PERSON_FIELDS)
            .execute()
        )
        return _text_result(_format_person(person))
    except Exception as e:
        log.error("people_get failed", exc_info=True)
        return _error_result(f"People get failed: {e}")


@tool(
    "people_list",
    "List Google Contacts (connections). Returns paginated results. "
    "All parameters are optional.",
    {"page_size": int},
)
@track_latency("google_people")
async def people_list(args: dict) -> dict:
    try:
        page_size = args.get("page_size", 20)
        service = get_people_service()

        kwargs = {
            "resourceName": "people/me",
            "personFields": _PERSON_FIELDS,
            "pageSize": min(page_size, 100),
        }
        if args.get("page_token"):
            kwargs["pageToken"] = args["page_token"]

        result = service.people().connections().list(**kwargs).execute()

        contacts = [_format_person(c) for c in result.get("connections", [])]
        data = {
            "contacts": contacts,
            "total": result.get("totalItems", len(contacts)),
        }
        if result.get("nextPageToken"):
            data["next_page_token"] = result["nextPageToken"]

        if not contacts:
            return _text_result("No contacts found.")
        return _text_result(data)
    except Exception as e:
        log.error("people_list failed", exc_info=True)
        return _error_result(f"People list failed: {e}")


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

people_server = create_sdk_mcp_server(
    name="google-people",
    version="1.0.0",
    tools=[people_search, people_get, people_list],
)
