"""Kimi K2.5 research MCP tool for Molly.

Sends research queries to Moonshot's Kimi K2.5 model via their
OpenAI-compatible API. Use for parallel research, deep reasoning,
and knowledge-intensive tasks.

Tools:
  kimi_research (AUTO) — Send a research query to Kimi K2.5
"""

import logging

import httpx
from claude_agent_sdk import create_sdk_mcp_server, tool

import config

log = logging.getLogger(__name__)


def _text_result(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _error_result(msg: str) -> dict:
    return {"content": [{"type": "text", "text": msg}], "is_error": True}


@tool(
    "kimi_research",
    "Send a research query to Kimi K2.5 (Moonshot AI). "
    "Use for deep research, knowledge-intensive questions, parallel reasoning, "
    "or getting a second opinion on complex topics. "
    "Supports chain-of-thought thinking mode for harder problems.",
    {"query": str, "system_context": str, "thinking": bool},
)
async def kimi_research(args: dict) -> dict:
    query_text = args.get("query", "").strip()
    if not query_text:
        return _error_result("A 'query' parameter is required.")

    if not config.MOONSHOT_API_KEY:
        return _error_result(
            "MOONSHOT_API_KEY not configured. "
            "Cannot reach Kimi K2.5 — ask Brian to add the key."
        )

    system_context = args.get("system_context", "You are a helpful research assistant.")
    thinking = args.get("thinking", False)

    model = "kimi-k2-0711-preview"
    messages = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": query_text},
    ]

    body = {
        "model": model,
        "messages": messages,
    }
    if thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": 8192}

    headers = {
        "Authorization": f"Bearer {config.MOONSHOT_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{config.MOONSHOT_BASE_URL}/chat/completions",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        if not content:
            return _error_result("Kimi returned an empty response.")

        usage = data.get("usage", {})
        meta = f"\n\n[Kimi K2.5 | tokens: {usage.get('total_tokens', '?')}]"
        return _text_result(content + meta)

    except httpx.TimeoutException:
        return _error_result("Kimi request timed out after 120s.")
    except httpx.HTTPStatusError as e:
        return _error_result(f"Kimi API error: {e.response.status_code} {e.response.text[:200]}")
    except Exception as e:
        log.error("Kimi research failed", exc_info=True)
        return _error_result(f"Kimi request failed: {e}")


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

kimi_server = create_sdk_mcp_server(
    name="kimi",
    version="1.0.0",
    tools=[kimi_research],
)
