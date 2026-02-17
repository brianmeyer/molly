"""Groq reasoning MCP tool for Molly.

Tools:
  groq_reason (AUTO) - Query Groq-hosted reasoning models via OpenAI-compatible API.
"""

import logging

import httpx
from claude_agent_sdk import create_sdk_mcp_server, tool

import config
from utils import track_latency

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "deepseek-r1-distill-llama-70b"


def _text_result(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _error_result(msg: str) -> dict:
    return {"content": [{"type": "text", "text": msg}], "is_error": True}


@tool(
    "groq_reason",
    "Query Groq for fast reasoning, coding help, summarization, and fallback orchestration tasks.",
    {"query": str, "system_context": str, "model": str, "temperature": float},
)
@track_latency("groq")
async def groq_reason(args: dict) -> dict:
    query_text = str(args.get("query", "")).strip()
    if not query_text:
        return _error_result("A 'query' parameter is required.")

    if not config.GROQ_API_KEY:
        return _error_result(
            "GROQ_API_KEY not configured. Cannot reach Groq - ask Brian to add the key."
        )

    model = str(args.get("model", _DEFAULT_MODEL) or _DEFAULT_MODEL).strip()
    if not model:
        model = _DEFAULT_MODEL

    system_context = str(
        args.get(
            "system_context",
            "You are a concise, accurate assistant helping a personal chief-of-staff AI.",
        )
    ).strip()
    if not system_context:
        system_context = "You are a concise, accurate assistant."

    try:
        temperature = float(args.get("temperature", 0.2))
    except (TypeError, ValueError):
        temperature = 0.2
    temperature = max(0.0, min(1.5, temperature))

    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_context},
            {"role": "user", "content": query_text},
        ],
        "temperature": temperature,
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{config.GROQ_BASE_URL}/chat/completions",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            payload = resp.json()

        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        if not content:
            return _error_result("Groq returned an empty response.")

        usage = payload.get("usage", {})
        meta = (
            f"\n\n[Groq | model: {model} | "
            f"tokens: {usage.get('total_tokens', '?')}]"
        )
        return _text_result(str(content).strip() + meta)

    except httpx.TimeoutException:
        return _error_result("Groq request timed out after 90s.")
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        detail = exc.response.text[:240]
        return _error_result(f"Groq API error: {status} {detail}")
    except Exception as exc:
        log.error("Groq reasoning failed", exc_info=True)
        return _error_result(f"Groq request failed: {exc}")


groq_server = create_sdk_mcp_server(
    name="groq",
    version="1.0.0",
    tools=[groq_reason],
)
