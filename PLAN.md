# Molly Architecture Evolution: OpenClaw-Inspired Parallel Orchestration

## Problem Statement

Molly's current architecture has a **serial bottleneck**. Every user message goes through a single Claude SDK session that thinks → calls tool → waits → thinks → calls tool → waits, all sequentially. A request touching calendar + email + tasks takes 30-45s because each tool call is serial.

OpenClaw solves this differently — it's a persistent daemon with a hub-and-spoke gateway, multi-channel support, cron scheduling, and webhook triggers. It feels more autonomous because it's always on, always doing things proactively.

**Goal**: Take the best ideas from OpenClaw's architecture and combine them with Molly's unique strengths (knowledge graph, local ML, self-evolution) while:
1. Breaking the serial bottleneck using Kimi as a fast orchestrator dispatching parallel Claude SDK workers
2. Adding always-on proactive autonomy (cron, webhooks, gateway)
3. Adding a voice-to-voice channel so Brian can talk to Molly naturally
4. Making the orchestrator routing resilient with a local Qwen fallback

---

## What We Keep (Molly's Advantages)

These are genuinely ahead of OpenClaw and must not be degraded:

- **Knowledge graph** (Neo4j + GLiNER2) — structured entity/relationship extraction
- **Local ML** (Qwen3-4B triage, GLiNER2 extraction, EmbeddingGemma embeddings)
- **Self-evolution framework** (branch → test → approve → deploy → rollback)
- **Three-tier approval** (AUTO/CONFIRM/BLOCKED with per-tool classification)
- **Triage system** (deterministic pre-filter + local Qwen classifier)
- **Health doctor** (multi-layer monitoring, issue registry, remediation routing)

## What We Take From OpenClaw

1. **Always-on cron scheduler** — user-defined scheduled tasks with pre-approved tool sets
2. **Webhook ingestion** — HTTP endpoint for external event triggers
3. **Unified channel abstraction** — normalize all messaging behind one interface
4. **Context-aware auto-approval** — auto-approve within owner-initiated request scope
5. **Proactive task execution** — morning briefings, email triage, daily summaries
6. **Multi-channel architecture** — any platform connects through the same gateway

---

## The Core Change: Kimi Orchestrator + Parallel Claude Workers

### Current Flow (Serial)
```
Message → Claude SDK (single session, ~30-45s for multi-domain tasks)
            ↓
       think → tool → wait → think → tool → wait → ... → response
```

### New Flow (Parallel)
```
Message → Orchestrator classifies (~1-2s) →
  ├── "direct": Kimi/Qwen answers immediately (no Claude, ~1-2s total)
  ├── "single": One scoped Claude worker (~5-10s)
  └── "parallel": Multiple Claude workers via asyncio.gather (~5-10s)
                     ↓
              Kimi synthesizes results (~1-2s)
                     ↓
              Final response
```

### Why This Works

1. **Kimi K2.5 is fast and cheap** — classification ~1-2s, synthesis ~1-2s
2. **Multiple `query()` calls run truly in parallel** via `asyncio.gather()` — confirmed by Anthropic docs and SDK
3. **Each worker gets a scoped tool set** — calendar worker only loads calendar MCP, email worker only loads gmail MCP. Less context pollution, faster startup.
4. **Simple questions skip Claude entirely** — "what time is it?" or "thanks" get answered by Kimi in 1-2s instead of 10s+
5. **The Claude SDK stays** — workers are real SDK sessions with all the existing tool execution, MCP integration, context management
6. **Fallback is safe** — if Kimi API is down, local Qwen3-4B handles routing; if that fails too, the existing single-session `handle_message()` path runs unchanged

### Orchestrator Routing: Kimi Primary, Qwen Local Fallback

The orchestrator classification is the critical path — if it's slow or down, everything stalls. Solution: **two-tier routing**.

**Tier 1: Kimi K2.5 (primary)**
- Full JSON-mode classification with rich reasoning
- Handles complex decomposition (which subtasks, dependencies, model tiers)
- ~1-2s latency over API

**Tier 2: Qwen3-4B local (fallback)**
- Already loaded in-process for triage — zero additional startup cost
- Simpler classification: `direct | single:<profile> | parallel:<profile>,<profile>`
- Can't do rich dependency reasoning, but handles 90% of routing correctly
- ~0.5s latency, zero API dependency
- Activates automatically when Kimi API is unreachable or times out

**Tier 3: Hardcoded fallback**
- If both Kimi and Qwen fail: route to single "general" Claude worker
- Equivalent to today's behavior — zero regression

This means Molly can route messages even when completely offline from external APIs (Qwen runs locally), which is something OpenClaw cannot do.

---

## Phase 1: Orchestrator Layer (orchestrator.py)

**New file.** The brain that decides how to handle each message.

### Key Components

- `classify_message(user_message, memory_context, skill_context) → OrchestrationPlan`
  - Tries Kimi K2.5 first (JSON mode, ~1-2s)
  - Falls back to local Qwen3-4B if Kimi fails/times out
  - Falls back to hardcoded "single:general" if both fail
  - Returns: `{strategy: "direct", direct_answer: "..."}` or `{strategy: "single"|"parallel", subtasks: [...]}`

- `synthesize_results(user_message, worker_results, hint) → str`
  - Kimi combines multiple worker outputs into one coherent response
  - Fallback: concatenate results if Kimi synthesis fails

- `orchestrate(user_message, chat_id, ...) → (response_text, session_id)`
  - Same signature as `handle_message()` — drop-in replacement
  - Runs pre-processing (memory retrieval, skill matching) concurrently
  - Calls `classify_message()`, then dispatches accordingly

### Classification Prompt Design

Tells Kimi/Qwen about Molly's tool domains: calendar, email, contacts, tasks, research, writer, files, imessage, general. For each subtask, specifies: `worker_profile`, `prompt`, `model` (haiku/sonnet/opus), `depends_on` (dependency indices). Rules bias toward "direct" (fastest) and "single" (simplest).

### Worker Profiles (Scoped Tool Sets)

| Profile | MCP Servers | Default Model | Use Case |
|---------|-------------|---------------|----------|
| calendar | google-calendar, google-meet | haiku | Schedule lookups |
| email | gmail | sonnet | Email search/draft/send |
| contacts | google-people, imessage | haiku | Contact lookups |
| tasks | google-tasks, apple-mcp | haiku | Task/reminder management |
| research | kimi, grok | opus | Deep research |
| writer | (none) | sonnet | Drafting text |
| files | google-drive | sonnet | File/drive operations |
| imessage | imessage, apple-mcp | sonnet | iMessage operations |
| general | (none) | sonnet | Catch-all |

### Qwen Local Router Design

The local Qwen3-4B router uses a simpler prompt since it can't reliably produce complex JSON:

```
System: You are a message router. Classify the user's request.
Respond with ONLY one line in this format:
  direct: <answer>           — for simple questions you can answer
  single:<profile>            — needs one tool domain
  parallel:<profile>,<profile> — needs multiple domains simultaneously

Profiles: calendar, email, contacts, tasks, research, writer, files, imessage, general

User: <message>
```

Qwen's response is parsed with simple string splitting. If it says `direct:`, Molly uses the inline answer. If `single:calendar`, dispatch one calendar worker. If `parallel:calendar,email`, dispatch both. This is fast (~0.5s), reliable, and runs entirely locally.

### Fallback Chain

```
classify_message():
  try:
    return kimi_classify()           # ~1-2s, rich JSON
  except (timeout, api_error):
    try:
      return qwen_local_classify()   # ~0.5s, simple format
    except:
      return hardcoded_general()     # 0ms, safe default
```

---

## Phase 2: Worker Pool (workers.py)

**New file.** Parallel Claude SDK session management.

### Key Components

- `dispatch_workers(plan, chat_id, ...) → ([(description, response)], session_id)`
  - Groups subtasks by dependency level
  - Phase 0: independent subtasks run via `asyncio.gather()`
  - Phase 1+: dependent subtasks wait for prerequisites, inject their results

- `_run_worker(subtask, ...) → (response_text, session_id)`
  - Creates a scoped `ClaudeSDKClient` with only the tools this worker needs
  - Loads only the relevant MCP servers (from `_MCP_SERVER_SPECS` in agent.py)
  - Uses the per-worker model tier (haiku/sonnet/opus)
  - Applies the existing approval system for CONFIRM-tier tools

- `asyncio.Semaphore(3)` — caps concurrent Claude workers to avoid rate limits

### How Workers Reuse Existing Code

- **Identity stack**: loaded once by `dispatch_workers()`, shared to all workers
- **Approval system**: each worker gets its own `_make_worker_tool_checker()` that calls the same `get_action_tier()` and `approval_manager.request_tool_approval()` as today
- **MCP servers**: `_load_scoped_mcp_servers()` reuses `_MCP_SERVER_SPECS` from agent.py but only loads what the worker needs

---

## Phase 3: Gateway Scheduler (gateway.py)

**New file.** Always-on cron and webhook system, inspired by OpenClaw's gateway.

### GatewayScheduler Class

- Loads task definitions from `~/.molly/workspace/gateway/tasks/*.yaml`
- Loads webhook definitions from `~/.molly/workspace/gateway/webhooks/*.yaml`
- `tick()` — called from the main loop, checks for due cron tasks
- `handle_webhook(webhook_id, payload) → response`
- Tasks execute through the orchestrator (not directly through Claude)

### Scheduled Task YAML Format

```yaml
name: Morning Briefing
schedule: "0 8 * * *"       # cron expression
prompt: |
  Give Brian a morning briefing:
  1. Today's calendar events
  2. Important emails from last 12 hours
  3. Due tasks and reminders
worker_profile: general
model: sonnet
auto_approve: false          # whether to skip approval for this task
enabled: false               # user enables explicitly
channel: owner_dm            # where to send results
```

### Built-in Tasks (Created on First Run, Disabled by Default)

- `morning-briefing` — 8am daily calendar+email+tasks summary
- `email-triage` — every 30min during active hours, flag important emails
- `daily-summary` — 9pm end-of-day recap

### Webhook HTTP Format

```yaml
name: GitHub PR Webhook
prompt_template: |
  A GitHub webhook event was received:
  {payload}
  Summarize what happened and whether Brian needs to act.
worker_profile: general
model: haiku
secret: "webhook-secret-here"   # HMAC-SHA256 verification
channel: owner_dm
```

### HTTP Endpoints (Added to Existing FastAPI App)

- `POST /webhook/{webhook_id}` — receive external events
- `GET /gateway/status` — list tasks and their state
- `POST /gateway/tasks/{id}/enable` — enable a task
- `POST /gateway/tasks/{id}/disable` — disable a task
- `POST /gateway/tasks/{id}/run` — trigger a task manually

---

## Phase 4: Channel Abstraction (channels/)

**New package.** Unified messaging interface inspired by OpenClaw's channel dock.

### channels/base.py

- `InboundMessage` dataclass — normalized: `channel_type, chat_id, sender_id, sender_name, content, msg_id, timestamp, is_from_owner, is_group`
- `OutboundMessage` dataclass — normalized: `chat_id, content, channel_type`
- `Channel` abstract base class — `connect()`, `disconnect()`, `send()`, `send_typing()`, `stop_typing()`, `is_connected`
- `ChannelRegistry` — maps chat_ids to channels, routes outbound messages

### channels/whatsapp_channel.py

- Wraps existing `WhatsAppClient` behind the `Channel` interface
- Handles the thread-to-async bridge (same as today's `_on_whatsapp_message`)
- Normalizes neonize `msg_data` dict into `InboundMessage`

### Why This Matters

- Adding Telegram = implement `TelegramChannel`, register it
- Adding Slack = implement `SlackChannel`, register it
- Adding voice = implement `VoiceChannel`, register it (see Phase 6)
- The orchestrator and workers never know which platform the message came from

---

## Phase 5: Voice-to-Voice Channel (channels/voice_channel.py)

**New file.** Adds real-time voice conversation as a first-class Molly channel.

### Architecture Decision: Which Voice API?

Three viable options, each with different tradeoffs:

| | Gemini Live | OpenAI Realtime | xAI Grok Voice |
|---|---|---|---|
| **Protocol** | WebSocket (own format) | WebSocket (event-based) | WebSocket (OpenAI-compatible) |
| **Bidirectional audio** | Yes (native audio model) | Yes (native audio model) | Yes (native audio model) |
| **Text transcripts** | Both sides (input + output transcription) | Both sides (via gpt-4o-transcribe) | Both sides (OpenAI-compatible) |
| **Tool/function calling** | Yes (in-session) | Yes (in-session) | Yes (+ built-in web_search, x_search) |
| **System instructions** | Yes (setup message) | Yes (session.update, updatable mid-session) | Yes (session.update, OpenAI-compatible) |
| **Latency** | Sub-second (native audio) | Sub-second (native audio) | <1s (claims fastest) |
| **Cost per 5-min call** | ~$0.15-0.50 | ~$0.50 | $0.25 (flat $0.05/min) |
| **Max session** | 10 min default | 60 min | Not specified |
| **Molly already has** | Gemini API key (`GEMINI_API_KEY`) | No API key configured | xAI API key (`XAI_API_KEY`) |

**Recommended approach**: Support all three behind a common `VoiceProvider` interface, with provider selection via config. Grok Voice is the cheapest and uses the OpenAI-compatible protocol (shares implementation with OpenAI Realtime). Gemini Live is a different protocol but Molly already has a Gemini key.

### Voice Channel Design

The voice channel is fundamentally different from text channels because:
1. It's **continuous** — audio streams in/out constantly, not discrete messages
2. It needs its own **LLM brain** — the voice API's native model handles conversation
3. It needs to **feed Molly's memory** — transcripts from both sides must flow into the knowledge graph
4. It needs **Molly's context** — the voice model's system prompt must include identity, memory, and current state

### How It Plugs Into Molly

```
                  ┌─────────────────────┐
                  │   Voice Provider     │
                  │  (Gemini/OpenAI/Grok)│
                  │                      │
  Microphone ───► │  Audio In ──► Model  │
                  │             (native) │
                  │  Audio Out ◄── Model │ ───► Speaker
                  │                      │
                  │  Transcripts (both)  │
                  └──────────┬──────────┘
                             │
                    text transcripts
                             │
                             ▼
                  ┌─────────────────────┐
                  │   VoiceChannel       │
                  │                      │
                  │  1. Injects Molly's  │
                  │     identity + memory│
                  │     into system      │
                  │     instructions     │
                  │                      │
                  │  2. Captures both    │
                  │     sides' transcripts│
                  │     as InboundMessage│
                  │     + stores in      │
                  │     memory pipeline  │
                  │                      │
                  │  3. Registers tool   │
                  │     functions that   │
                  │     call back into   │
                  │     the orchestrator │
                  │     for multi-step   │
                  │     tasks            │
                  └─────────────────────┘
```

### Key Design Points

**1. The voice model IS the conversational brain during voice sessions.**

Unlike text channels where messages route through the orchestrator → Claude workers, voice sessions use the voice API's native model (Gemini/GPT-4o/Grok) for real-time conversation. This is necessary because voice-to-voice latency requires a single model handling the full loop — you can't route audio through Kimi → Claude → synthesis and keep sub-second response times.

**2. Molly's identity and memory inject into the voice model's system prompt.**

Before the WebSocket session starts (or via mid-session updates for OpenAI/Grok), we inject:
- `SOUL.md` — Molly's personality and behavioral guidelines
- `USER.md` — information about Brian
- Recent memory context from `retrieve_context()` — relevant past conversations
- Knowledge graph summary — key entities and relationships
- Current state — today's calendar, pending tasks, recent notifications

This makes the voice model "be Molly" for the duration of the call.

**3. Transcripts flow back into Molly's memory pipeline.**

Both sides of the conversation are captured as text transcripts (all three APIs support this). After each conversational turn:
- User transcript + model transcript → `process_conversation()` (embed + graph extract)
- This means voice conversations build knowledge graph entities, update memory, and are searchable later — same as text conversations

**4. Tool calls bridge to Molly's orchestrator for complex tasks.**

All three voice APIs support function/tool calling during live sessions. We register tools like:
- `check_calendar(date)` — calls orchestrator with calendar worker
- `search_email(query)` — calls orchestrator with email worker
- `create_reminder(title, due)` — calls orchestrator with tasks worker
- `do_research(query)` — calls orchestrator with research worker

When the voice model decides to call a tool, the VoiceChannel:
1. Receives the function call from the WebSocket
2. Dispatches it through the orchestrator (which may fan out to Claude workers)
3. Returns the result to the voice model via the WebSocket
4. The voice model speaks the result to the user

This means Brian can say "check my calendar for tomorrow and see if John emailed about the meeting" and the voice model will call both tools, get parallel results, and speak a synthesized answer.

**5. Voice sessions are triggered via a dedicated endpoint or command.**

Options for starting a voice session:
- `POST /voice/start` HTTP endpoint — returns WebSocket URL for audio streaming
- WhatsApp voice note detection — when Brian sends a voice note, Molly enters voice mode
- `/voice` command in any channel — starts a voice session
- Dedicated companion app (future) — native mic/speaker access

### VoiceProvider Interface

```python
class VoiceProvider(ABC):
    """Abstract interface for voice-to-voice providers."""

    async def connect(self, system_instructions: str, tools: list[dict]) -> None:
        """Open WebSocket connection with identity + tool definitions."""

    async def send_audio(self, audio_chunk: bytes) -> None:
        """Stream audio input to the model."""

    async def receive(self) -> VoiceEvent:
        """Receive next event: audio_chunk | transcript | tool_call | end."""

    async def send_tool_result(self, call_id: str, result: str) -> None:
        """Return a tool call result to the model."""

    async def update_instructions(self, instructions: str) -> None:
        """Update system instructions mid-session (OpenAI/Grok only)."""

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
```

### Provider Implementations

**`GrokVoiceProvider` / `OpenAIVoiceProvider`** — share 95% of code because Grok uses the OpenAI-compatible protocol. Key events:
- `session.update` → inject system instructions + tool definitions
- `input_audio_buffer.append` → stream mic audio
- `response.audio.delta` → receive speaker audio
- `conversation.item.input_audio_transcription.completed` → user transcript
- `response.audio_transcript.delta` → model transcript
- `response.function_call_arguments.done` → tool call request

**`GeminiVoiceProvider`** — different protocol:
- Setup message with `systemInstruction` + `tools.functionDeclarations`
- `realtimeInput.mediaChunks` → stream mic audio (PCM 16kHz)
- `serverContent.modelTurn.parts[].inlineData` → receive speaker audio (PCM 24kHz)
- `inputAudioTranscription` / `outputAudioTranscription` → transcripts
- `toolResponse` → return tool results

### Audio Transport

The VoiceChannel needs to get audio to/from the user's device. Options:

**Option A: WebSocket relay through FastAPI** (simplest)
- Client connects to `ws://molly:8080/voice`
- Client streams raw PCM audio in, receives PCM audio out
- VoiceChannel relays between client WebSocket ↔ provider WebSocket
- Works with any web client, companion app, or native app

**Option B: WebRTC (lower latency, more complex)**
- OpenAI Realtime natively supports WebRTC for browser clients
- Requires STUN/TURN infrastructure
- Better for mobile/browser use cases
- Can add later as an optimization

**Recommended**: Start with Option A (WebSocket relay). It works with all three providers, is simple to implement, and latency is acceptable over LAN. Add WebRTC later if needed.

### Voice Session Lifecycle

```
1. Client connects to ws://molly:8080/voice?provider=grok
2. VoiceChannel loads Molly's identity + memory context
3. VoiceChannel opens WebSocket to voice provider with system instructions + tools
4. Audio relay loop:
   - Client audio → provider (streaming)
   - Provider audio → client (streaming)
   - Provider transcript events → memory pipeline (async)
   - Provider tool calls → orchestrator → result → provider
5. Session ends when client disconnects or timeout
6. Full transcript written to daily log + embedded + graph-extracted
```

### Cost Controls

Voice sessions burn API credits continuously. Safeguards:
- `VOICE_MAX_SESSION_MINUTES` config (default: 10)
- `VOICE_DAILY_BUDGET_MINUTES` config (default: 60)
- Session timer with warning at 80% of max
- Daily minute tracking persisted in gateway state

---

## Phase 6: Integration (Wiring Changes)

### agent.py Changes

- `handle_message()` gets a new `use_orchestrator: bool = True` parameter
- When `use_orchestrator=True` and orchestrator is available, delegates to `orchestrator.orchestrate()`
- On any orchestrator failure, falls back to the existing serial path
- All existing code (identity stack, approval, MCP loading, skill gaps, foundry) stays intact as the fallback and as building blocks for workers

### main.py Changes

- Add `GatewayScheduler` to `Molly.__init__()`
- Add `await self.gateway.initialize()` in `run()`
- Add `gateway.tick()` in the main loop timeout handler (same pattern as automations/self-improvement)
- Register webhook routes and voice endpoint on the FastAPI app
- All existing behavior preserved — gateway and voice are purely additive

### config.py Changes

New settings:
```python
# Orchestrator
ORCHESTRATOR_ENABLED = _env_bool("MOLLY_ORCHESTRATOR_ENABLED", True)
MAX_CONCURRENT_WORKERS = int(os.getenv("MOLLY_MAX_WORKERS", "3"))
ORCHESTRATOR_KIMI_TIMEOUT = int(os.getenv("MOLLY_ORCHESTRATOR_KIMI_TIMEOUT", "5"))
ORCHESTRATOR_LOCAL_FALLBACK = _env_bool("MOLLY_ORCHESTRATOR_LOCAL_FALLBACK", True)

# Gateway
GATEWAY_TASK_DIR = WORKSPACE / "gateway" / "tasks"
GATEWAY_WEBHOOK_DIR = WORKSPACE / "gateway" / "webhooks"
GATEWAY_STATE_FILE = WORKSPACE / "gateway" / "state.json"

# Voice
VOICE_ENABLED = _env_bool("MOLLY_VOICE_ENABLED", False)
VOICE_PROVIDER = os.getenv("MOLLY_VOICE_PROVIDER", "grok")  # grok | openai | gemini
VOICE_MAX_SESSION_MINUTES = int(os.getenv("MOLLY_VOICE_MAX_SESSION_MINUTES", "10"))
VOICE_DAILY_BUDGET_MINUTES = int(os.getenv("MOLLY_VOICE_DAILY_BUDGET_MINUTES", "60"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
```

---

## Phase 7: Multimodal Embedding Layer (memory/embeddings.py)

**Evaluated**: Qwen3-VL-Embedding-2B vs current EmbeddingGemma-300M.

### Current State

Molly uses `google/embeddinggemma-300m` (768-dim, ~200MB, text-only) via `sentence-transformers`. All embeddings are stored in `sqlite-vec` with `EMBEDDING_DIM = 768`. Used for semantic search over conversation chunks in `retriever.py`.

### Why Consider Qwen3-VL-Embedding-2B

- **Multimodal**: Embeds text, images, and video frames into the same vector space
- **Shared semantic space**: "a photo of a sunset" (text) and an actual sunset.jpg produce nearby vectors — enables cross-modal retrieval
- **Matryoshka dimensionality**: Native 2048-dim, but can truncate to 768/512/256 with graceful degradation
- **32K token context**: Handles long documents without chunking (EmbeddingGemma: 8K)
- **GGUF available**: ~1.1GB quantized, runs on CPU via `llama-cpp-python` (already a dependency)
- **Apache 2.0 license**

### Head-to-Head Comparison

| Dimension | EmbeddingGemma-300M | Qwen3-VL-Embedding-2B |
|-----------|--------------------|-----------------------|
| Parameters | 300M | 2B |
| Native dims | 768 | 2048 (Matryoshka to 768/512/256) |
| Context window | 8K tokens | 32K tokens |
| Modalities | Text only | Text + Image + Video |
| VRAM/RAM | ~200MB | ~1.1GB (Q4 GGUF) / ~4GB (fp16) |
| Inference speed | ~5ms/embed | ~15-30ms/embed (text), ~50-100ms (image) |
| Text quality (MTEB) | Good for size | Slightly better at 768-dim, much better at 2048-dim |
| Framework | sentence-transformers | transformers or llama-cpp-python |
| Image understanding | None | Native (shared space with text) |

### What Qwen3-VL Unlocks for Molly

1. **Image memory** — Screenshots, photos, receipts shared via WhatsApp get embedded and become searchable via text queries ("find the receipt from last week")
2. **Visual document retrieval** — PDFs, diagrams, whiteboard photos become part of Molly's memory
3. **Graph enrichment** — Image entities can be extracted (GLiNER2 for text + CLIP/VL for visual) and linked in Neo4j
4. **Cross-modal recall** — "Show me what Brian sent about the house" finds both text messages and photo messages in one query
5. **Voice session context** — If voice sessions include screen sharing or camera, visual context can be embedded

### What Molly Would Lose (Single-Model Migration)

- **Speed regression**: ~15-30ms vs ~5ms per embed (3-6x slower, but still fast enough for real-time)
- **More RAM**: ~1.1GB vs ~200MB (manageable on M-series Mac with 32GB+)
- **Slight text quality regression at 768-dim**: Matryoshka truncation loses some nuance vs native 768-dim model
- **Framework change**: Would need `transformers` or `llama-cpp-python` instead of `sentence-transformers`

### Recommended Path: Hybrid (Keep Both)

**Keep EmbeddingGemma for text** (fast, proven, small) and **add Qwen3-VL for images** in a parallel table.

```
memory/vectorstore.py changes:
  - Existing `chunks_vec` table (768-dim, EmbeddingGemma) — unchanged
  - New `visual_vec` table (768-dim, Qwen3-VL at Matryoshka 768) — images only
  - `search()` queries both tables, merges results by cosine similarity

memory/embeddings.py changes:
  - Existing `embed()` / `embed_batch()` — unchanged (EmbeddingGemma)
  - New `embed_image(image_path) → np.ndarray` — Qwen3-VL
  - New `embed_multimodal(text, image_path) → np.ndarray` — Qwen3-VL with both

memory/retriever.py changes:
  - `retrieve_context()` adds visual search when query might reference images
  - Results from both text and visual search merged and ranked
```

**Why hybrid over single-model**: Zero regression on existing text search. Text embeddings stay fast. Images are a purely additive capability. Can migrate to single-model later if Qwen3-VL proves superior across the board.

### Alternative Path: Single-Model (Full Migration)

Replace EmbeddingGemma entirely with Qwen3-VL at 768-dim Matryoshka:
- One model for everything — simpler architecture
- Re-embed existing ~N conversation chunks (one-time batch job)
- Update `EMBEDDING_DIM = 768` (stays the same if using Matryoshka truncation)
- Accept the speed and RAM tradeoffs

### Image Storage Design

```
~/.molly/workspace/images/
  ├── {sha256_hash}.{ext}     # deduplicated by content hash
  └── metadata.json            # hash → {source_chat_id, timestamp, sender, caption}

vectorstore.py:
  visual_chunks table:
    id, image_hash, source_chat_id, sender_id, timestamp, caption

  visual_vec virtual table:
    rowid → visual_chunks.id, embedding (768-dim float32)
```

Images received via WhatsApp (or future channels) are saved, hashed, embedded, and optionally entity-extracted for the graph. The image file path is stored; only the embedding goes into sqlite-vec.

### Graph Integration

Neo4j nodes for images:
```cypher
(:Image {hash, path, timestamp, caption})
  -[:SENT_BY]->(:Person {name})
  -[:SENT_IN]->(:Chat {id})
  -[:CONTAINS]->(:Entity {name, type})  // from caption or visual extraction
  -[:SIMILAR_TO]->(:Image)              // cosine similarity > threshold
```

This means "who sent me photos of the house?" becomes a graph traversal + vector search hybrid query — same pattern Molly already uses for text.

---

## What Does NOT Change

- **Knowledge graph** — untouched, still fed by `process_conversation()` post-response (now also fed by voice transcripts)
- **Triage system** — still gates group messages before they reach the orchestrator
- **Memory retrieval** — still runs during pre-processing, results go to orchestrator and workers (and into voice system prompts)
- **Approval system** — same 3-tier classification, same WhatsApp approval flow; workers reuse the same `get_action_tier()` and `ApprovalManager`
- **Self-improvement** — untouched, still runs on its own tick cycle
- **Health doctor** — untouched
- **Skills** — still matched during pre-processing, injected into worker prompts
- **Automations** — existing automation engine untouched; gateway tasks are complementary
- **WhatsApp client** — still works via neonize; WhatsApp channel adapter wraps it

---

## Risk Mitigation

1. **Kimi API down** → Qwen local router takes over. If Qwen fails → single general worker (same as today).
2. **Worker fails** → Returns error text for that subtask. Other workers unaffected.
3. **Rate limits** → Semaphore caps concurrent workers at 3. Configurable.
4. **Wrong classification** → "single/general" is the safe default. Worst case = same as today.
5. **Circular dependencies in subtasks** → Detected and force-run remaining. Logged.
6. **Voice provider down** → Session fails to start. Text channels unaffected.
7. **Voice cost overrun** → Per-session and daily minute budgets enforced.
8. **Voice transcript quality** → All three APIs note transcripts may diverge slightly from what the model "heard". Memory pipeline handles this gracefully since it already handles imprecise input.

---

## File Inventory

| File | Status | Purpose |
|------|--------|---------|
| `orchestrator.py` | NEW | Kimi + Qwen local routing, synthesis, main orchestrate() entry point |
| `workers.py` | NEW | Parallel Claude SDK worker pool, scoped profiles, dispatch logic |
| `gateway.py` | NEW | Cron scheduler, webhook handler, built-in task definitions |
| `channels/__init__.py` | NEW | Package init |
| `channels/base.py` | NEW | Channel protocol, InboundMessage, OutboundMessage, ChannelRegistry |
| `channels/whatsapp_channel.py` | NEW | WhatsApp adapter wrapping existing neonize client |
| `channels/voice_channel.py` | NEW | Voice-to-voice session manager, provider relay, transcript capture |
| `channels/voice_providers.py` | NEW | VoiceProvider ABC + Grok/OpenAI/Gemini implementations |
| `agent.py` | MODIFIED | Add orchestrator delegation with fallback |
| `main.py` | MODIFIED | Add gateway scheduler init + tick + webhook routes + voice endpoint |
| `config.py` | MODIFIED | Add orchestrator, gateway, and voice settings |
| `memory/embeddings.py` | MODIFIED | Add Qwen3-VL image embedding functions alongside existing EmbeddingGemma |
| `memory/vectorstore.py` | MODIFIED | Add visual_chunks + visual_vec tables, merged search |
| `memory/retriever.py` | MODIFIED | Add visual search path to retrieve_context() |

---

## Implementation Order

Each phase is independently testable. The system works at every intermediate step because the fallback path is the existing code.

### Step 1: orchestrator.py
- Kimi classification + Qwen local fallback + hardcoded fallback
- Can test by calling `orchestrate()` directly from a script
- No changes to existing files yet

### Step 2: workers.py
- Parallel Claude SDK worker pool with scoped profiles
- Can test with manually constructed `Subtask` objects
- No changes to existing files yet

### Step 3: agent.py wiring
- Add `use_orchestrator` flag, delegate to orchestrator when available
- Flip the switch, test end-to-end with WhatsApp
- Full fallback to existing behavior if anything fails

### Step 4: gateway.py
- Cron scheduler + webhook handler
- Independent of orchestrator — can test in isolation
- Creates workspace/gateway/ directory with built-in task YAMLs

### Step 5: main.py wiring
- Integrate gateway tick, webhook routes, and gateway status endpoint
- All additive, no existing behavior changed

### Step 6: channels/base.py + channels/whatsapp_channel.py
- Unified channel abstraction
- WhatsApp adapter wraps existing client
- Prepares the interface for voice and future channels

### Step 7: channels/voice_providers.py
- VoiceProvider ABC + Grok/OpenAI implementation (shared protocol)
- Gemini implementation (different protocol)
- Can test each provider independently with a simple audio test script

### Step 8: channels/voice_channel.py
- Voice session manager: identity injection, tool bridging, transcript capture
- WebSocket endpoint on FastAPI for audio relay
- Integration with memory pipeline for transcript storage

### Step 9: config.py additions
- Small additions throughout steps 1-8
- Orchestrator, gateway, and voice settings

### Step 10: memory/embeddings.py — Qwen3-VL multimodal embeddings
- Load Qwen3-VL-Embedding-2B (GGUF or transformers)
- Add `embed_image()` and `embed_multimodal()` alongside existing `embed()`
- Add `visual_chunks` + `visual_vec` tables to vectorstore
- Update retriever to merge text + visual search results
- Add image storage pipeline (hash, save, embed, graph-link)
- Can test independently with sample images before wiring into channels
