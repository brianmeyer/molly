# Molly

Personal AI chief of staff. Mac Mini M4. Local-first. Three-layer memory. Multi-channel. Proactive automation.

## Architecture

```
                    ┌─ WhatsApp (Neonize)
                    │
Channels ──────────┼─ Web UI (FastAPI + WebSocket)    → async queue → Claude Agent SDK (Opus)
                    │                                                       ↑
                    ├─ Terminal REPL (CLI)                     Identity stack + memory context
                    │                                                       ↓
                    └─ Email monitoring (Gmail poll)              Response → channel
                                                                            ↓
                                                          Async post-processing:
                                                          ├── Embed → sqlite-vec (Layer 2)
                                                          └── GLiNER2 → Neo4j (Layer 3)
```

## Stack

| Component | Role |
|-----------|------|
| Neonize | WhatsApp client (Go whatsmeow via Python) |
| Claude Agent SDK | LLM brain with sub-agent routing (Claude Max) |
| FastAPI + uvicorn | Web UI with WebSocket chat |
| EmbeddingGemma-300M | 768-dim embeddings for semantic search |
| sqlite-vec | Layer 2: vector search + operational logs |
| GLiNER2 | Entity + relationship extraction (DeBERTa-large) |
| Neo4j | Layer 3: knowledge graph |
| Qwen3-4B GGUF (`llama-cpp-python`) | Local triage model for message classification |
| Kimi K2.5 (Moonshot) | External research model via MCP tool |
| Grok (xAI) | External reasoning model via MCP tool |
| Health Doctor (`health.py`) | Daily/preflight health reports with regression checks |
| Contract Audit (`contract_audit.py`) | Nightly/weekly deterministic contracts + optional LLM audit layer |
| Track F Pre-Prod Audit (`scripts/run_preprod_readiness_audit.py`) | Report-first rollout checks before production promotion |
| Self-Improvement Engine (`self_improve.py`) | Guarded self-edit loop (branch, tests, approval, rollback/restart) |
| Skill Analytics (`skill_analytics.py`) | Skill gap telemetry, underperformance detection, gap clustering |

## Project Structure

```
molly/
├── main.py              # Entry point, async lifecycle, preflight checks
├── config.py            # All settings, paths, constants
├── whatsapp.py          # Neonize client wrapper
├── formatting.py        # WhatsApp-safe markdown rendering + chunking
├── health.py            # Health Doctor checks + report generation
├── contract_audit.py    # Deterministic + model-backed maintenance/weekly audits
├── foundry_adapter.py   # Foundry observation signal adapter for self-improvement
├── health_remediation.py # Health signal routing policy (auto-fix/propose/escalate)
├── self_improve.py      # Self-improvement engine (guarded self-edit loop)
├── skill_analytics.py   # Skill gap telemetry, underperformance detection
├── web.py               # FastAPI web UI backend (WebSocket chat)
├── terminal.py          # Standalone CLI REPL for debugging
├── database.py          # SQLite message store
├── agent.py             # Claude Agent SDK wrapper, sub-agents, identity loading
├── approval.py          # Action approval flow (WhatsApp yes/no)
├── commands.py          # /help, /clear, /memory, /graph, /forget, /status
├── heartbeat.py         # Proactive check-in + iMessage/email monitoring + skill hot-reload
├── automations.py       # YAML-based proactive automation engine
├── automation_triggers.py # Trigger types (schedule, event, email, message, etc.)
├── maintenance.py       # Nightly maintenance (direct Python, no SDK tools)
├── skills.py            # Dynamic skill trigger matching, hot-reload, lifecycle
├── scripts/
│   ├── run_molly.sh     # Supervisor loop with restart-on-exit-code support
│   └── run_preprod_readiness_audit.py # Track F pre-prod gate/report generator
├── web/
│   └── index.html       # Chat UI (single-page, no framework)
├── tools/
│   ├── google_auth.py   # Google OAuth token management
│   ├── calendar.py      # Google Calendar MCP tools
│   ├── gmail.py         # Gmail MCP tools
│   ├── imessage.py      # iMessage MCP tools
│   ├── reminders.py     # Apple Reminders MCP tools
│   ├── whatsapp.py      # WhatsApp message search MCP tool
│   ├── kimi.py          # Kimi K2.5 research MCP tool (Moonshot API)
│   └── grok.py          # Grok reasoning MCP tool (xAI API)
└── memory/
    ├── embeddings.py    # EmbeddingGemma-300M wrapper
    ├── vectorstore.py   # sqlite-vec backed vector store + preference/self-improve logs
    ├── retriever.py     # Pre-query: semantic search + graph lookup
    ├── processor.py     # Post-response: embed, extract, store
    ├── extractor.py     # GLiNER2 entity/relation/classification
    ├── triage.py        # Local Qwen3 message triage
    ├── dedup.py         # Shared dedup engine used by maintenance + health
    ├── issue_registry.py # Persistent issue/event registry + notification cooldown
    └── graph.py         # Neo4j client + Cypher queries
```

## Channels

All channels share the same memory pipeline and call `handle_message()`.

| Channel | Interface | Session prefix | Notes |
|---------|-----------|---------------|-------|
| WhatsApp | Neonize (Go bridge) | JID | Primary channel, approval flow |
| Web UI | FastAPI + WebSocket | `web:` | `http://localhost:8080?token=XXX` |
| Terminal | CLI REPL | `terminal` | `python terminal.py` for debugging |
| Email | Gmail API poll | `email` | Surfaces urgent emails via WhatsApp |

## Memory System

**Layer 1 — Identity files** loaded every turn: SOUL.md, USER.md, AGENTS.md, MEMORY.md, daily logs.

**Layer 2 — Semantic search** via sqlite-vec. Every conversation turn is embedded and stored. Pre-query retrieval injects the top 5 most similar past conversations into the system prompt.

**Layer 3 — Knowledge graph** via Neo4j. GLiNER2 extracts entities (Person, Technology, Organization, Project, Place, Concept) and relationships from every conversation. Graph context is retrieved alongside semantic search results.

## Approval System

Sensitive actions require explicit approval via WhatsApp before execution. Three tiers:

- **AUTO** — Read-only tools execute immediately
- **CONFIRM** — Shell access, file writes, external sends require Brian's yes/no
- **BLOCKED** — Destructive actions are denied outright

## Sub-Agents

Opus orchestrates and delegates to sub-agents via the SDK's Task tool:

| Agent | Model | Use for |
|-------|-------|---------|
| `quick` | Haiku | Fast lookups, formatting, trivial subtasks |
| `worker` | Sonnet | Email drafts, research synthesis, multi-step tools |
| `analyst` | Opus | Deep analysis, strategic thinking, complex reasoning |

External models (Kimi K2.5 for research, Grok for social intelligence) are available as MCP tools, not sub-agents.

## Automations

YAML-based proactive automation engine. Automations live in `~/.molly/workspace/automations/` and run on triggers without requiring a message from the user.

| Automation | Trigger | What it does |
|-----------|---------|-------------|
| Morning Briefing | Cron (7 AM weekdays) | Calendar, email, commitments summary |
| Meeting Prep | 30 min before calendar event | Context gathering for upcoming meetings |
| Email Triage | New unread emails (polled) | Categorize and surface urgent emails |
| Commitment Tracker | Owner makes a commitment | Logs and tracks follow-through |
| End-of-Day Wrap | Cron (6 PM weekdays) | Daily summary if 3+ messages exchanged |
| Weekend Review | Cron (9 AM Saturday) | Weekly reflection and planning |

**Trigger types:** schedule (cron), event (calendar), email (Gmail poll), message (pattern match), commitment (extracted from conversation), condition (custom expression), webhook.

**Guards:** Quiet hours (10 PM–7 AM ET, VIP/urgent bypass), deduplication via state.json (payload hash + min interval), condition expressions.

**Pipeline steps** route through `handle_message()` → Claude Agent SDK, inheriting the full sub-agent and approval system.

## Recent Updates (February 2026)

- Added skill gap telemetry (`skill_analytics.py`): per-turn tool call logging, gap clustering, underperformance detection, and automatic proposal pipeline into self-improvement.
- Added skill hot-reload: heartbeat checks for skill file changes each cycle, swaps cache atomically, rolls back malformed files.
- Added skill lifecycle with pending approval flow: self-improvement can propose new skills or edits, owner approves/rejects via WhatsApp, rejections enter 30-day cooldown.
- Added YAML-first trigger contract parsing alongside legacy `## Trigger` format, with mixed-mode precedence and dedup.
- Expanded self-improvement engine with skill gap proposals, failure diagnostic tool generation, workflow pattern detection, and Foundry signal integration.
- Added guarded self-improvement workflows in `self_improve.py`: proposal drafting, branch/test gates, owner approval, deploy restart, and post-deploy rollback checks.
- Added Health Doctor in `health.py` with startup preflight snapshots, daily reports, and green→red regression detection.
- Added health remediation routing + issue registry persistence (`health_remediation.py`, `memory/issue_registry.py`) to separate observe/auto-fix/escalate behavior and avoid repeat alert spam.
- Added WhatsApp output hardening via `formatting.py` plus message chunking and JID guards in `whatsapp.py`.
- Added stable timestamp normalization in `database.py`/`whatsapp.py` for mixed epoch units (sec/ms/us/ns) and ISO values.
- Expanded commitment automation to sync with Apple Reminders and exposed commitment/reminder status via `/followups`.
- Added automation proposal mining from repeated tool traces and improved email-trigger high-water deduplication.
- Improved approval runtime behavior with request-scoped coalescing and owner-routed approvals for non-WhatsApp channels.
- Added Foundry signal ingestion (`foundry_adapter.py`) so self-improvement can use real observed workflow sequences as additive evidence.
- Added contract audits (`contract_audit.py`) with deterministic checks first and optional model audits (Opus/Kimi/Gemini, fallback-capable).
- Added Track F pre-prod readiness audit (`scripts/run_preprod_readiness_audit.py`) and report-first enforcement toggles in `config.py`.
- Added promotion contract/drift gates in workspace `promote-tool.py` (`validate`, `promote --dry-run`, force override with reason).
- Added `scripts/run_molly.sh` supervisor for restart-safe runtime operation.

## Preference Signals

Passive feedback logging for learning loops. When the owner dismisses a surfaced notification (e.g., "not important", "who cares", "stop sending"), the dismissal is logged with the source, summary, and sender pattern. Accumulated signals feed into nightly maintenance assessments and skill gap detection.

## Skills

Markdown skill files in `~/.molly/workspace/skills/` with trigger patterns parsed dynamically from:
- YAML front matter `triggers:` lists
- Legacy `## Trigger` quoted phrases/commands

Matched skills inject their instructions into the system prompt for that turn. This keeps runtime matching compatible with both old and new skill authoring formats.

**Hot-reload:** The heartbeat checks for skill file changes every cycle. Modified or new skills are picked up without restart; malformed files roll back to the previous state.

**Skill lifecycle:** The self-improvement engine can propose new skills or edits to existing skills. Proposals go through owner approval via WhatsApp before activation. Rejected proposals enter a 30-day cooldown.

**Skill gap analytics:** Tool call patterns are logged per-turn. `skill_analytics.py` clusters repeated tool sequences that lack a matching skill, surfaces underperforming skills, and feeds gap candidates into the self-improvement proposal pipeline.

## Pre-Prod Readiness Audit

Before promoting major behavior changes, run:

```bash
python scripts/run_preprod_readiness_audit.py --output-dir /tmp/molly-audits
```

Strict mode fails on red checks:

```bash
python scripts/run_preprod_readiness_audit.py --strict
```

Track F checks cover:
- parser compatibility
- skill telemetry presence
- Foundry ingestion health
- promotion drift status

Default posture is report-first (`MOLLY_TRACK_F_REPORT_ONLY=true`) so rollout safety can be observed before hard enforcement.

## Commands

```
/help                 Show available commands
/clear                Reset conversation session
/memory               Show MEMORY.md contents
/graph                Show graph summary (entity count, top connected)
/graph <entity>       Look up entity in knowledge graph
/forget <topic>       Remove entity from graph
/status               Uptime, model, connection info
/skills               List loaded skills
/skill <name>         Show skill details
/digest               Trigger daily digest manually
/register             Register current chat for responses
/register listen      Register current chat for listen-only monitoring
/unregister           Unregister current chat
/groups               List registered chats
/pending              Show pending approvals
/automations          List automations and their status
/followups            Show commitment tracker + Apple Reminders status
/commitments          Alias for /followups
/health               Show latest health report (or generate one)
/health history       Show 7-day health trend
```

## Running

```bash
cd ~/molly
./scripts/run_molly.sh
```

`scripts/run_molly.sh` supervises Molly and restarts automatically when she exits with
`MOLLY_RESTART_EXIT_CODE` (default `42`). This is used for self-edit deploy reloads and
rollback-restart recovery.

Preflight checks verify Docker, Neo4j, the local triage GGUF model, and Google OAuth are
ready before startup. WhatsApp reconnects automatically after initial QR pairing. Web UI
starts on port 8080.

For a single direct run (no supervisor loop):
```bash
source .venv/bin/activate
python main.py
```

For terminal-only debugging (no WhatsApp/web):
```bash
python terminal.py
```

## Web UI Access

The web UI binds to `127.0.0.1` by default (localhost only). To access from other devices on the local network, set `MOLLY_WEB_HOST=0.0.0.0`.

- **Local**: `http://localhost:8080?token=YOUR_TOKEN`
- **Phone/tablet**: `http://macmini.local:8080?token=YOUR_TOKEN` (requires `MOLLY_WEB_HOST=0.0.0.0`)

Set `MOLLY_WEB_TOKEN` env var for authentication. A warning is logged if unset.

## Requirements

- Python 3.12
- Docker (for Neo4j)
- Claude Max subscription (for Claude Agent SDK)
- HuggingFace account (for gated EmbeddingGemma model)
- `llama-cpp-python` (for local triage)
- Qwen3-4B GGUF file at `~/.molly/models/Qwen_Qwen3-4B-Q4_K_M.gguf`
- Google Cloud OAuth credentials (optional, for Calendar/Gmail)
- Moonshot API key (optional, for Kimi K2.5 research tool)
- xAI API key (optional, for Grok reasoning tool)

## Build Phases

- [x] Phase 0: Prerequisites
- [x] Phase 1: Core platform (WhatsApp + Claude Agent SDK)
- [x] Phase 2: Three-layer memory (semantic search + knowledge graph + nightly maintenance)
- [x] Phase 3: Tools + skills (Calendar, Gmail, iMessage, Skills)
- [x] Phase 4: Multi-channel (Web UI, Terminal REPL, Email monitoring)
- [x] Phase 5: Sub-agents, model routing, audit hardening
- [x] Phase 6: Proactive automation engine, preference signal logging, Grok x_search
- [x] Phase 7: Learning loops (preference signals, health doctor, automation proposal mining, entity dedup)
- [x] Phase 8: Self-improvement (guarded self-edit, skill lifecycle, skill gap analytics, hot-reload)
- [ ] Phase 9: Email triage overhaul (deterministic 3-tier classification, digest scheduling)
- [ ] Phase 10: Collaboration
