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
| Claude Agent SDK | LLM brain (Claude Max subscription) |
| FastAPI + uvicorn | Web UI with WebSocket chat |
| EmbeddingGemma-300M | 768-dim embeddings for semantic search |
| sqlite-vec | Layer 2: vector search + operational logs |
| GLiNER2 | Entity + relationship extraction (DeBERTa-large) |
| Neo4j | Layer 3: knowledge graph |
| Qwen3-4B (Ollama) | Local triage model for message classification |

## Project Structure

```
molly/
├── main.py              # Entry point, async lifecycle, preflight checks
├── config.py            # All settings, paths, constants
├── whatsapp.py          # Neonize client wrapper
├── web.py               # FastAPI web UI backend (WebSocket chat)
├── terminal.py          # Standalone CLI REPL for debugging
├── database.py          # SQLite message store
├── agent.py             # Claude Agent SDK wrapper + identity loading
├── approval.py          # Action approval flow (WhatsApp yes/no)
├── commands.py          # /help, /clear, /memory, /graph, /forget, /status
├── heartbeat.py         # Proactive check-in + iMessage/email monitoring
├── maintenance.py       # Nightly Opus maintenance + health check
├── skills.py            # Skill matching + loading
├── web/
│   └── index.html       # Chat UI (single-page, no framework)
├── tools/
│   ├── google_auth.py   # Google OAuth token management
│   ├── calendar.py      # Google Calendar MCP tools
│   ├── gmail.py         # Gmail MCP tools
│   ├── contacts.py      # Apple Contacts MCP tools
│   ├── imessage.py      # iMessage MCP tools
│   └── whatsapp.py      # WhatsApp message search MCP tool
└── memory/
    ├── embeddings.py    # EmbeddingGemma-300M wrapper
    ├── vectorstore.py   # sqlite-vec backed vector store
    ├── retriever.py     # Pre-query: semantic search + graph lookup
    ├── processor.py     # Post-response: embed, extract, store
    ├── extractor.py     # GLiNER2 entity/relation/classification
    ├── triage.py        # Local Qwen3 message triage
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

## Skills

Markdown skill files in `~/.molly/workspace/skills/` with trigger patterns. Matched skills inject their instructions into the system prompt for that turn. Built-in skills: daily digest, meeting prep.

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
/unregister           Unregister current chat
/groups               List registered chats
/pending              Show pending approvals
```

## Running

```bash
cd ~/molly
source .venv/bin/activate
python main.py
```

Preflight checks verify Docker, Neo4j, Ollama, and Google OAuth are ready before startup. WhatsApp reconnects automatically after initial QR pairing. Web UI starts on port 8080.

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
- Ollama (optional, for local triage)
- Google Cloud OAuth credentials (optional, for Calendar/Gmail)

## Build Phases

- [x] Phase 0: Prerequisites
- [x] Phase 1: Core platform (WhatsApp + Claude Agent SDK)
- [x] Phase 2: Three-layer memory (semantic search + knowledge graph + nightly maintenance)
- [x] Phase 3: Tools + skills (Calendar, Gmail, Contacts, iMessage, Skills)
- [x] Phase 4: Multi-channel (Web UI, Terminal REPL, Email monitoring)
- [ ] Phase 5: Learning loops
- [ ] Phase 6: Collaboration
- [ ] Phase 7: Self-improvement
