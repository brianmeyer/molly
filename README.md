# Molly

Personal AI chief of staff running on WhatsApp. Mac Mini M4. Local-first. Three-layer memory. Proactive automation.

## Architecture

```
WhatsApp (Neonize) → async queue → Claude Agent SDK (Opus)
                                        ↑
                          Identity stack + memory context
                                        ↓
                              Response → WhatsApp
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
| EmbeddingGemma-300M | 768-dim embeddings for semantic search |
| sqlite-vec | Layer 2: vector search + operational logs |
| GLiNER2 | Entity + relationship extraction (DeBERTa-large) |
| Neo4j | Layer 3: knowledge graph |

## Project Structure

```
molly/
├── main.py              # Entry point, async lifecycle, preflight checks
├── config.py            # All settings, paths, constants
├── whatsapp.py          # Neonize client wrapper
├── database.py          # SQLite message store
├── agent.py             # Claude Agent SDK wrapper + identity loading
├── approval.py          # Action approval flow (WhatsApp yes/no)
├── commands.py          # /help, /clear, /memory, /graph, /forget, /status
├── heartbeat.py         # Proactive check-in logic
├── maintenance.py       # Nightly Opus maintenance + health check
└── memory/
    ├── embeddings.py    # EmbeddingGemma-300M wrapper
    ├── vectorstore.py   # sqlite-vec backed vector store
    ├── retriever.py     # Pre-query: semantic search + graph lookup
    ├── processor.py     # Post-response: embed, extract, store
    ├── extractor.py     # GLiNER2 entity/relation/classification
    └── graph.py         # Neo4j client + Cypher queries
```

## Memory System

**Layer 1 — Identity files** loaded every turn: SOUL.md, USER.md, AGENTS.md, MEMORY.md, daily logs.

**Layer 2 — Semantic search** via sqlite-vec. Every conversation turn is embedded and stored. Pre-query retrieval injects the top 5 most similar past conversations into the system prompt.

**Layer 3 — Knowledge graph** via Neo4j. GLiNER2 extracts entities (Person, Technology, Organization, Project, Place, Concept) and relationships from every conversation. Graph context is retrieved alongside semantic search results.

## Approval System

Sensitive actions require explicit approval via WhatsApp before execution. Categories: send_email, api_write, bash_destructive, modify_identity, install_package, file_delete, calendar_modify, send_message_external.

Molly describes what she wants to do, Brian replies yes or no.

## Commands

```
/help                 Show available commands
/clear                Reset conversation session
/memory               Show MEMORY.md contents
/graph <entity>       Look up entity in knowledge graph
/forget <topic>       Remove entity from graph
/status               Uptime, model, connection info
```

## Running

```bash
cd ~/molly
source .venv/bin/activate
python main.py
```

Preflight checks verify Docker and Neo4j are running before startup. WhatsApp reconnects automatically after initial QR pairing.

## Requirements

- Python 3.12
- Docker (for Neo4j)
- Claude Max subscription (for Claude Agent SDK)
- HuggingFace account (for gated EmbeddingGemma model)

## Build Phases

- [x] Phase 0: Prerequisites
- [x] Phase 1: Core platform (WhatsApp + Claude Agent SDK)
- [x] Phase 2: Three-layer memory (semantic search + knowledge graph + nightly maintenance)
- [ ] Phase 3: Tools + skills (Calendar, Gmail, Contacts)
- [ ] Phase 4: Multi-channel
- [ ] Phase 5: Learning loops
- [ ] Phase 6: Collaboration
- [ ] Phase 7: Self-improvement
