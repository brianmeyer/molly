"""Grok reasoning MCP tool for Molly.

Sends queries to xAI's Grok model. Two modes:

1. **General reasoning** (search_x=false): OpenAI-compatible API via httpx.
   For second opinions, creative tasks, general reasoning.

2. **Live X search** (search_x=true): xai-sdk with x_search server-side tool.
   Grok searches live X/Twitter posts and synthesizes results.
   For social sentiment, trending topics, "what people are saying about..."

Tools:
  grok_reason (AUTO) — Query Grok for social intelligence and reasoning
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import httpx
from claude_agent_sdk import create_sdk_mcp_server, tool

import config
from utils import track_latency

log = logging.getLogger(__name__)


def _text_result(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _error_result(msg: str) -> dict:
    return {"content": [{"type": "text", "text": msg}], "is_error": True}


def _run_x_search(query_text: str, system_context: str, days_back: int) -> dict:
    """Run Grok with x_search in a synchronous context (called via executor).

    Uses the xai-sdk Client which is synchronous. We run this in a thread
    executor from the async handler.
    """
    from xai_sdk import Client
    from xai_sdk.chat import system, user
    from xai_sdk.tools import x_search

    now = datetime.now(tz=timezone.utc)
    from_date = now - timedelta(days=days_back)

    search_tool = x_search(
        from_date=from_date,
        to_date=now,
    )

    client = Client(api_key=config.XAI_API_KEY)
    chat = client.chat.create(
        # xAI server-side tools like x_search require an agentic reasoning model.
        model="grok-4-1-fast-reasoning",
        messages=[
            system(system_context),
            user(query_text),
        ],
        tools=[search_tool],
        include=["x_search_call_output", "inline_citations"],
    )

    response = chat.sample()

    content = response.content or ""
    if not content:
        return _error_result("Grok returned an empty response from X search.")

    # Build metadata footer
    parts = ["Grok + x_search"]
    if response.server_side_tool_usage:
        parts.append(f"tools: {dict(response.server_side_tool_usage)}")
    if response.citations:
        parts.append(f"citations: {len(response.citations)}")
    meta = f"\n\n[{' | '.join(parts)}]"

    # Append inline citation URLs if present
    if response.citations:
        cite_lines = "\n".join(f"- {url}" for url in response.citations[:10])
        content += f"\n\nSources:\n{cite_lines}"

    return _text_result(content + meta)


@tool(
    "grok_reason",
    "Query Grok (xAI) for social intelligence, sentiment analysis, "
    "trending topics, second opinions, and reasoning tasks. "
    "Set search_x=true to search live X/Twitter posts (for sentiment, "
    "trending topics, 'what people are saying about...'). "
    "Set search_x=false for general reasoning without live search.",
    {"query": str, "system_context": str, "search_x": bool, "days_back": int},
)
@track_latency("grok")
async def grok_reason(args: dict) -> dict:
    query_text = args.get("query", "").strip()
    if not query_text:
        return _error_result("A 'query' parameter is required.")

    if not config.XAI_API_KEY:
        return _error_result(
            "XAI_API_KEY not configured. "
            "Cannot reach Grok — ask Brian to add the key."
        )

    system_context = args.get(
        "system_context",
        "You are a helpful assistant with strong social awareness and reasoning skills.",
    )
    search_x = args.get("search_x", False)
    days_back = min(args.get("days_back", 21), 90)

    # --- Live X search mode (xai-sdk) ---
    if search_x:
        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None, _run_x_search, query_text, system_context, days_back,
                ),
                timeout=60.0,
            )
            return result
        except Exception as e:
            log.error("Grok x_search failed", exc_info=True)
            return _error_result(f"Grok X search failed: {e}")

    # --- General reasoning mode (httpx) ---
    messages = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": query_text},
    ]

    body = {
        "model": "grok-4-1-fast-reasoning",
        "messages": messages,
    }

    headers = {
        "Authorization": f"Bearer {config.XAI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{config.XAI_BASE_URL}/chat/completions",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        if not content:
            return _error_result("Grok returned an empty response.")

        usage = data.get("usage", {})
        meta = f"\n\n[Grok | tokens: {usage.get('total_tokens', '?')}]"
        return _text_result(content + meta)

    except httpx.TimeoutException:
        return _error_result("Grok request timed out after 90s.")
    except httpx.HTTPStatusError as e:
        return _error_result(f"Grok API error: {e.response.status_code} {e.response.text[:200]}")
    except Exception as e:
        log.error("Grok reasoning failed", exc_info=True)
        return _error_result(f"Grok request failed: {e}")


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

grok_server = create_sdk_mcp_server(
    name="grok",
    version="1.0.0",
    tools=[grok_reason],
)
