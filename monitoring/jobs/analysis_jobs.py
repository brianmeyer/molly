"""Analysis Jobs — Opus analysis, operational insights, graph suggestions."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import config
import db_pool

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Opus analysis prompt
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """\
You are Molly's maintenance analyst. Review the maintenance report and graph data below.
Produce TWO sections separated by the exact marker `---WHATSAPP---`:

SECTION 1 — MEMORY.md update:
A dated section to append to MEMORY.md. Format: `## {date}` followed by bullet points.
Only include genuinely important, durable facts — skip noise and ephemeral details.

---WHATSAPP---

SECTION 2 — WhatsApp maintenance summary:
A comprehensive but concise summary of tonight's maintenance for Brian.
Use WhatsApp formatting (*bold*, bullets with -).
Include: graph health (entity/relationship counts and deltas), dedup merges,
relationship audit results (Tier 1 flags, Tier 2 Kimi outcomes),
GLiNER training progress, any failures or anomalies, and key insights.
Keep it under 500 words. Make it scannable — Brian reads this on his phone.
Do NOT include raw task results or status codes. Synthesize.

Output both sections with the ---WHATSAPP--- marker between them.
"""


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def run_opus_analysis(report: str, graph_summary: str, today: str) -> str:
    """Run a text-only Claude query for maintenance analysis — no tools, no permissions."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    prompt_text = (
        f"Today is {today}.\n\n"
        f"## Maintenance Report\n{report}\n\n"
        f"## Graph Summary\n{graph_summary}\n\n"
        "Based on this data, produce the MEMORY.md update section."
    )

    options = ClaudeAgentOptions(
        system_prompt=ANALYSIS_SYSTEM_PROMPT.format(date=today),
        model="sonnet",
        allowed_tools=[],
        cwd=str(config.WORKSPACE),
    )

    async def _prompt():
        yield {
            "type": "user",
            "session_id": "",
            "message": {"role": "user", "content": prompt_text},
            "parent_tool_use_id": None,
        }

    response_text = ""
    async for message in query(prompt=_prompt(), options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_text += block.text
        elif isinstance(message, ResultMessage):
            if message.is_error:
                log.error("Maintenance analysis error: %s", message.result)

    return response_text


def compute_operational_insights() -> dict:
    """Compute 24h tool success rates, flag failing tools, find unused skills."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    stale_cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    conn = None
    try:
        conn = db_pool.sqlite_connect(str(config.MOLLYGRAPH_PATH))

        # Tool success rates (last 24h)
        cursor = conn.execute(
            """SELECT tool_name,
                      COUNT(*) AS total,
                      SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successes
               FROM tool_calls
               WHERE created_at > ?
                 AND tool_name NOT LIKE 'routing:%'
                 AND tool_name NOT LIKE 'approval:%'
               GROUP BY tool_name
               ORDER BY total DESC""",
            (cutoff,),
        )
        failing_tools: list[str] = []
        tool_count = 0
        for row in cursor.fetchall():
            tool_count += 1
            total = row[1]
            successes = row[2]
            rate = successes / total if total > 0 else 0
            if total >= 3 and rate < 0.9:
                failing_tools.append(f"{row[0]} ({successes}/{total}={rate:.0%})")

        # Unused skills (7+ days since last execution)
        cursor = conn.execute(
            "SELECT DISTINCT skill_name FROM skill_executions WHERE created_at > ?",
            (stale_cutoff,),
        )
        recent_skills = {row[0] for row in cursor.fetchall()}
        cursor = conn.execute("SELECT DISTINCT skill_name FROM skill_executions")
        all_skills = {row[0] for row in cursor.fetchall()}
        unused = sorted(all_skills - recent_skills)

        return {
            "tool_count_24h": tool_count,
            "failing_tools": failing_tools,
            "unused_skills": unused,
        }
    except Exception as exc:
        log.error("Operational insights failed: %s", exc, exc_info=True)
        return {
            "tool_count_24h": 0,
            "failing_tools": [],
            "unused_skills": [],
        }
    finally:
        if conn is not None:
            conn.close()


def run_graph_suggestions_digest() -> str:
    """Build and return the daily graph suggestions digest."""
    from memory.graph_suggestions import build_suggestion_digest

    digest = build_suggestion_digest()
    if digest:
        return digest[:500]
    return "no suggestions today"
