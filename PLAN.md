# Molly Architecture Evolution: OpenClaw-Inspired Parallel Orchestration

## Problem Statement

Molly's current architecture has a **serial bottleneck**. Every user message goes through a single Claude SDK session that thinks → calls tool → waits → thinks → calls tool → waits, all sequentially. A request touching calendar + email + tasks takes 30-45s because each tool call is serial.

OpenClaw solves this differently — it's a persistent daemon with a hub-and-spoke gateway, multi-channel support, cron scheduling, and webhook triggers. It feels more autonomous because it's always on, always doing things proactively.

**Goal**: Take the best ideas from OpenClaw's architecture and combine them with Molly's unique strengths (knowledge graph, local ML, self-evolution) while:
1. Breaking the serial bottleneck using a two-phase Gemini orchestrator dispatching parallel Claude SDK workers
2. Adding always-on proactive autonomy (cron, webhooks, gateway)
3. Adding voice — both async voice notes and live voice-to-voice — unified through Gemini's multimodal API
4. Making the orchestrator routing resilient with a four-tier fallback chain
5. Adding a browser tool for interacting with websites lacking APIs
6. Staying in Python (evaluated Go, Rust, TypeScript — Python wins for this use case)

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

## The Core Change: Two-Phase Orchestrator + Parallel Claude Workers

### Current Flow (Serial)
```
Message → Claude SDK (single session, ~30-45s for multi-domain tasks)
            ↓
       think → tool → wait → think → tool → wait → ... → response
```

### New Flow (Parallel, Two-Phase)
```
Message (text or audio) →
  Phase 1: Gemini 2.5 Flash-Lite (~0.4s) →
    ├── "direct: <answer>"     → respond immediately (no Claude)
    ├── "simple: <profile>"    → dispatch one scoped Claude worker (~5-10s)
    └── "complex"              →
          Phase 2: Kimi K2.5 thinking mode (~0.7-1s) →
            Full decomposition: subtasks, dependencies, model tiers
            → Multiple Claude workers via asyncio.gather (~5-10s)
                  ↓
            Kimi synthesizes results (~0.5-1s)
                  ↓
            Final response
```

### Why Two Phases Instead of One

The orchestrator has two very different jobs:
1. **Simple routing** (~80% of messages): "check my calendar" → single:calendar. Any model can do this.
2. **Complex decomposition** (~20%): "check my calendar, find John's email about the meeting, and draft a reply incorporating both" → parallel subtasks with dependencies, model tier selection, prompt writing.

Using one model for both means either paying for reasoning you don't need (expensive fast model) or getting wrong decompositions (cheap model on complex requests). Two phases optimizes both.

### Why Gemini

**Model evaluation (February 2026)** — compared Kimi K2.5, DeepSeek V3.2, OpenAI, Grok, Gemini, and Claude for orchestration:

| Model | Input $/1M | Output $/1M | TTFT | Intelligence | Notes |
|---|---|---|---|---|---|
| GPT-4.1 Nano | **$0.02** | $0.15 | 0.37s | 13 | Cheapest, but too dumb for decomposition |
| Gemini 2.5 Flash-Lite | $0.10 | $0.40 | **0.29s** | ~20 | Fastest TTFT, free tier, multimodal (audio+image) |
| Gemini 3 Flash | $0.50 | $3.00 | 0.40s | **46** | Strong general reasoning, but not agent-specialized |
| Grok 4.1 Fast | $0.20 | $0.50 | 0.40s | 24 | Good value but no multimodal input |
| DeepSeek V3.2 | $0.28 | $0.42 | 1.2-7.5s | High | **Too slow** — 3.5-9s total latency |
| Kimi K2.5 | $0.60 | $2.50 | 0.62s | High | **Agent Swarm** — designed for multi-agent orchestration, thinking mode |
| Claude Haiku 4.5 | $1.00 | $5.00 | 0.50s | ~30 | 10x pricier than Flash-Lite on input |

**Decision**: Split by phase — each model where it's strongest:

**Phase 1 triage → Gemini 2.5 Flash-Lite** because:
1. **Multimodal** — accepts audio input natively, so voice notes get transcribed AND classified in one API call (no separate STT)
2. **Fastest TTFT** (0.29s) — sub-half-second for simple routing
3. **Free tier** — development and testing at zero cost
4. **Gemini Live API** — same key, same provider for live voice-to-voice sessions

**Phase 2 decomposition → Kimi K2.5 (thinking mode)** because:
1. **Agent Swarm architecture** — Kimi K2.5 was explicitly designed to coordinate up to 100 specialized agents. This is literally the decomposition task: figure out which agents to spawn, what each does, and how they depend on each other.
2. **Thinking mode** — chain-of-thought reasoning over task dependencies, model tier selection, and prompt writing for each subtask. Produces more reliable decompositions than a general-purpose model.
3. **1T MoE (32B active)** — deep reasoning capacity without the latency of a full dense model
4. **Molly already has `KIMI_API_KEY`** — no new credentials needed
5. **Cost-competitive** — $0.60/$2.50 vs Gemini 3 Flash $0.50/$3.00 — nearly identical per-call cost

**Provider diversity**: Gemini triages, Kimi decomposes, Claude works, Qwen falls back locally. Four providers, no single point of failure.

### Four-Tier Fallback Chain

```
classify_message():
  try:
    Phase 1: gemini_flash_lite_triage()     # ~0.4s, text or audio input
    if complex:
      Phase 2: kimi_k25_decompose()         # ~0.7-1s, thinking mode, rich JSON
  except (timeout, api_error):
    try:
      return qwen_local_classify()           # ~0.5s, simple format, offline
    except:
      try:
        return gpt41_nano_classify()         # ~0.4s, if OPENAI_API_KEY set
      except:
        return hardcoded_general()           # 0ms, safe default
```

Four providers, four levels of resilience: Gemini+Kimi → Qwen local → GPT-4.1 Nano → hardcoded. Molly can classify messages even with two cloud providers simultaneously down (Qwen runs locally).

### Language Decision: Stay in Python

**Evaluated**: Go, Rust, TypeScript, and hybrid approaches against current Python stack.

**Why Python wins for Molly:**
1. **Claude Agent SDK** — official first-class support in Python and TypeScript only. No Go or Rust SDK. The SDK is in Alpha (v0.1.36) with breaking changes — being on an official SDK is non-negotiable.
2. **ML ecosystem** — sentence-transformers, GLiNER2 fine-tuning, llama-cpp-python have no equivalent outside Python. The self-improvement engine's GLiNER fine-tuning is impossible in any other language.
3. **Bottleneck is API latency** — Claude API round-trips take 2-10s. Shaving 30ms off WebSocket frames by switching to Rust gains nothing.
4. **26K lines = 2-4 month rewrite** — zero feature work during that time, chasing Alpha SDK breaking changes.
5. **OpenClaw is TypeScript** (not Go/Rust as speculated) — proves TS works for this category, but OpenClaw has a team and doesn't run local ML in-process.

**Quick wins without rewriting:**
- `pip install neo4j-rust-ext` — 3-10x Neo4j driver speedup, zero code changes
- `pip install uvloop` — ~2x asyncio event loop throughput
- Profile with `py-spy` before optimizing anything

---

## Phase 1: Orchestrator Layer (orchestrator.py)

**New file.** The brain that decides how to handle each message.

### Key Components

- `classify_message(user_message, memory_context, skill_context, audio_bytes=None) → OrchestrationPlan`
  - **Phase 1 triage**: Gemini 2.5 Flash-Lite (~0.4s) — determines `direct`, `simple`, or `complex`
  - If `audio_bytes` provided (voice note), sends audio to Flash-Lite — gets transcript AND classification in one call
  - **Phase 2 decomposition** (only if `complex`): Kimi K2.5 thinking mode (~0.7-1s) — agent-specialized subtask decomposition with dependencies
  - Falls back through: Qwen3-4B local → GPT-4.1 Nano → hardcoded "single:general"
  - Returns: `{strategy: "direct", direct_answer: "...", transcript: "..."}` or `{strategy: "single"|"parallel", subtasks: [...]}`

- `synthesize_results(user_message, worker_results, hint) → str`
  - Kimi K2.5 combines multiple worker outputs into one coherent response (same model that decomposed — maintains context)
  - Fallback: concatenate results if Kimi synthesis fails

- `orchestrate(user_message, chat_id, ..., audio_bytes=None) → (response_text, session_id)`
  - Same signature as `handle_message()` — drop-in replacement
  - Runs pre-processing (memory retrieval, skill matching) concurrently
  - Calls `classify_message()`, then dispatches accordingly
  - `audio_bytes` parameter enables unified voice note handling

### Phase 1 Triage Prompt (Gemini 2.5 Flash-Lite)

Fast, cheap classification. The model receives the user message (text or audio) plus brief context and returns a one-line classification:

```
System: You are Molly's message router. Classify the user's request.
Respond with ONLY one of:
  direct: <brief answer>      — trivial questions you can answer (time, thanks, greetings)
  simple: <profile>            — needs one tool domain
  complex                      — needs multiple domains or multi-step reasoning

Profiles: calendar, email, contacts, tasks, research, writer, files, imessage, browser, general

If the input is audio, also include: transcript: <verbatim transcription>
```

Output is ~10-30 tokens. At $0.10/$0.40 per 1M tokens, cost is negligible (~$0.0001/call).

### Phase 2 Decomposition Prompt (Kimi K2.5 Thinking Mode)

Only triggered for `complex` messages (~20% of traffic). Kimi K2.5 uses its Agent Swarm reasoning to decompose into subtasks:

```json
{
  "subtasks": [
    {"profile": "calendar", "prompt": "Check Brian's calendar for tomorrow", "model": "haiku", "depends_on": []},
    {"profile": "email", "prompt": "Find John's email about the meeting", "model": "haiku", "depends_on": []},
    {"profile": "writer", "prompt": "Draft a reply incorporating calendar and email findings", "model": "sonnet", "depends_on": [0, 1]}
  ]
}
```

Kimi K2.5's Agent Swarm architecture (1T MoE, 32B active) was designed for exactly this: reasoning about which agents to spawn, what tools each needs, how they depend on each other, and what model tier each deserves. Thinking mode adds chain-of-thought reasoning over dependency graphs. ~0.7-1s latency, $0.60/$2.50 per 1M tokens. Cost per decomposition: ~$0.0012.

**Decomposition fallback**: If Kimi is down, fall through to Phase 1 fallbacks (Qwen local → GPT-4.1 Nano) which handle complex requests as simple `parallel:<profile>,<profile>` without dependency reasoning — less optimal but functional.

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
| browser | browser-tool | sonnet | Web interaction (no API available) |
| general | (none) | sonnet | Catch-all |

### Qwen Local Router Design (Fallback Tier 2)

The local Qwen3-4B router uses a simpler prompt since it can't reliably produce complex JSON:

```
System: You are a message router. Classify the user's request.
Respond with ONLY one line in this format:
  direct: <answer>           — for simple questions you can answer
  single:<profile>            — needs one tool domain
  parallel:<profile>,<profile> — needs multiple domains simultaneously

Profiles: calendar, email, contacts, tasks, research, writer, files, imessage, browser, general

User: <message>
```

Qwen's response is parsed with simple string splitting. If it says `direct:`, Molly uses the inline answer. If `single:calendar`, dispatch one calendar worker. If `parallel:calendar,email`, dispatch both. This is fast (~0.5s), reliable, and runs entirely locally.

Note: Qwen cannot handle audio input. If a voice note triggers the Qwen fallback, Molly must either use a separate local whisper model for transcription or respond with "I can't process voice right now, please type your message."

### GPT-4.1 Nano Fallback (Tier 3)

If `OPENAI_API_KEY` is configured, GPT-4.1 Nano ($0.02/$0.15) serves as a third-tier fallback. Same simple prompt as Qwen. ~0.4s latency, text-only.

### Fallback Chain

```
classify_message():
  try:
    phase1 = gemini_flash_lite_triage()       # ~0.4s, text or audio
  except (timeout, api_error):
    # Gemini triage down — fall through to simpler classifiers
    try:
      return qwen_local_classify()             # ~0.5s, simple format, offline
    except:
      if OPENAI_API_KEY:
        try:
          return gpt41_nano_classify()         # ~0.4s, text-only
        except:
          pass
      return hardcoded_general()               # 0ms, safe default

  if phase1.strategy == "complex":
    try:
      return kimi_k25_decompose(phase1)        # ~0.7-1s, thinking mode, rich JSON
    except (timeout, api_error):
      # Kimi down — degrade to simple parallel dispatch from Phase 1
      return phase1.as_simple_parallel()       # no dependency reasoning, but functional
  return phase1
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

## Phase 5: Unified Voice Pipeline (channels/voice_channel.py)

**New file.** Voice is handled entirely through Gemini, creating a unified pipeline with the orchestrator.

### Two Voice Modes, One Provider

| Mode | API | Model | Latency | When |
|---|---|---|---|---|
| **Async voice notes** | Gemini 2.5 Flash-Lite (standard API) | Flash-Lite | ~0.4s | WhatsApp voice notes, voice commands |
| **Live voice-to-voice** | Gemini Live API | Gemini 2.0 Flash (native audio) | Sub-second | Real-time conversation sessions |

Both use `GEMINI_API_KEY` — no additional credentials needed.

### Mode 1: Async Voice Notes (Integrated with Triage)

This is the key integration. When a voice note arrives (WhatsApp, future channels), it goes through the **same triage path as text messages** — but with audio input:

```
Voice note (audio bytes) →
  Gemini 2.5 Flash-Lite (audio input) →
    Returns BOTH:
      1. transcript: "Check my calendar for tomorrow"
      2. classification: "simple: calendar"
    → Same orchestrator dispatch as text messages
```

**No separate STT step.** The triage model does transcription AND classification in one API call. This means:
- Zero additional latency vs text messages (Flash-Lite processes audio natively)
- Zero additional cost (same call, audio is just another input modality)
- Voice notes and text messages flow through identical orchestrator logic
- Transcripts feed into memory pipeline (embed + graph extract) automatically

Implementation in `orchestrator.py`:
```python
async def classify_message(
    user_message: str,
    memory_context: str,
    skill_context: str,
    audio_bytes: bytes | None = None,  # voice note audio
) -> OrchestrationPlan:
    """Classify via Gemini Flash-Lite. Accepts text OR audio."""
    if audio_bytes:
        # Send audio as inline_data part alongside system prompt
        # Response includes both transcript and classification
        parts = [
            {"inline_data": {"mime_type": "audio/ogg", "data": base64.b64encode(audio_bytes).decode()}},
            {"text": f"Context: {memory_context}\n\nTranscribe the audio and classify the request."}
        ]
    else:
        parts = [{"text": f"Context: {memory_context}\n\nMessage: {user_message}"}]
    # ... same classification logic, returns OrchestrationPlan with optional transcript field
```

### Mode 2: Live Voice-to-Voice (Gemini Live API)

For real-time conversational sessions where Brian talks to Molly naturally.

```
                  ┌─────────────────────┐
                  │   Gemini Live API    │
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
                  │     → memory pipeline│
                  │                      │
                  │  3. Tool functions   │
                  │     call back into   │
                  │     the orchestrator │
                  └─────────────────────┘
```

### Key Design Points

**1. The Gemini Live model IS the conversational brain during voice sessions.**

Voice-to-voice latency requires a single model handling the full audio loop. You can't route audio through triage → Claude → synthesis and keep sub-second response times. The Gemini Live model handles real-time conversation natively.

**2. Molly's identity and memory inject into the session's system instructions.**

Before the WebSocket session starts, we inject:
- `SOUL.md` — Molly's personality and behavioral guidelines
- `USER.md` — information about Brian
- Recent memory context from `retrieve_context()` — relevant past conversations
- Knowledge graph summary — key entities and relationships
- Current state — today's calendar, pending tasks, recent notifications

This makes the Gemini Live model "be Molly" for the duration of the call.

**3. Transcripts flow back into Molly's memory pipeline.**

Both sides of the conversation are captured as text transcripts. After each conversational turn:
- User transcript + model transcript → `process_conversation()` (embed + graph extract)
- Voice conversations build knowledge graph entities, update memory, and are searchable later — same as text

**4. Tool calls bridge to Molly's orchestrator for complex tasks.**

Gemini Live supports function calling during live sessions. We register tools like:
- `check_calendar(date)` — calls orchestrator with calendar worker
- `search_email(query)` — calls orchestrator with email worker
- `create_reminder(title, due)` — calls orchestrator with tasks worker
- `do_research(query)` — calls orchestrator with research worker
- `browse_website(url, action)` — calls orchestrator with browser worker

When the Gemini Live model decides to call a tool, the VoiceChannel:
1. Receives the function call from the WebSocket
2. Dispatches it through the orchestrator (which may fan out to Claude workers)
3. Returns the result to the Gemini Live model via the WebSocket
4. The model speaks the result to the user

**5. Voice sessions are triggered via a dedicated endpoint or command.**

- `POST /voice/start` HTTP endpoint — returns WebSocket URL for audio streaming
- WhatsApp voice note detection — async mode (transcript + classify + respond via text)
- `/voice` command in any channel — starts a live voice-to-voice session
- Dedicated companion app (future) — native mic/speaker access

### Gemini Live Protocol

Single provider implementation — no abstraction layer needed:

```python
class GeminiLiveSession:
    """Manages a Gemini Live voice-to-voice session."""

    async def connect(self, system_instructions: str, tools: list[dict]) -> None:
        """Open WebSocket to Gemini Live API with identity + tool definitions."""
        # Setup message with systemInstruction + tools.functionDeclarations

    async def send_audio(self, audio_chunk: bytes) -> None:
        """Stream audio input via realtimeInput.mediaChunks (PCM 16kHz)."""

    async def receive(self) -> VoiceEvent:
        """Receive: audio chunk | transcript | tool_call | end.
        - serverContent.modelTurn.parts[].inlineData → speaker audio (PCM 24kHz)
        - inputAudioTranscription → user transcript
        - outputAudioTranscription → model transcript
        - toolCall → function call request
        """

    async def send_tool_result(self, call_id: str, result: str) -> None:
        """Return tool result via toolResponse message."""

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
```

### Audio Transport

**WebSocket relay through FastAPI** (simplest, works over LAN):
- Client connects to `ws://molly:8080/voice`
- Client streams raw PCM audio in, receives PCM audio out
- VoiceChannel relays between client WebSocket ↔ Gemini Live WebSocket
- Works with web client, companion app, or native app
- WebRTC can be added later as an optimization if latency becomes a measured problem

### Voice Session Lifecycle

```
1. Client connects to ws://molly:8080/voice
2. VoiceChannel loads Molly's identity + memory context
3. VoiceChannel opens WebSocket to Gemini Live API with system instructions + tools
4. Audio relay loop:
   - Client audio → Gemini Live (streaming)
   - Gemini Live audio → client (streaming)
   - Transcript events → memory pipeline (async)
   - Tool calls → orchestrator → Claude workers → result → Gemini Live
5. Session ends when client disconnects or timeout
6. Full transcript written to daily log + embedded + graph-extracted
```

### Cost Controls

Voice sessions burn API credits continuously. Safeguards:
- `VOICE_MAX_SESSION_MINUTES` config (default: 10)
- `VOICE_DAILY_BUDGET_MINUTES` config (default: 60)
- Session timer with warning at 80% of max
- Daily minute tracking persisted in gateway state
- Gemini Live cost: ~$0.15-0.50 per 5-min call

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
# Orchestrator (Phase 1: Gemini triage, Phase 2: Kimi decomposition)
ORCHESTRATOR_ENABLED = _env_bool("MOLLY_ORCHESTRATOR_ENABLED", True)
MAX_CONCURRENT_WORKERS = int(os.getenv("MOLLY_MAX_WORKERS", "3"))
ORCHESTRATOR_TRIAGE_TIMEOUT = int(os.getenv("MOLLY_ORCHESTRATOR_TRIAGE_TIMEOUT", "3"))   # Phase 1 (Gemini)
ORCHESTRATOR_DECOMPOSE_TIMEOUT = int(os.getenv("MOLLY_ORCHESTRATOR_DECOMPOSE_TIMEOUT", "5"))  # Phase 2 (Kimi)
ORCHESTRATOR_LOCAL_FALLBACK = _env_bool("MOLLY_ORCHESTRATOR_LOCAL_FALLBACK", True)
GEMINI_TRIAGE_MODEL = os.getenv("MOLLY_GEMINI_TRIAGE_MODEL", "gemini-2.5-flash-lite")
KIMI_DECOMPOSE_MODEL = os.getenv("MOLLY_KIMI_DECOMPOSE_MODEL", "kimi-k2.5")
KIMI_THINKING_ENABLED = _env_bool("MOLLY_KIMI_THINKING", True)  # enable thinking mode for decomposition

# Gateway
GATEWAY_TASK_DIR = WORKSPACE / "gateway" / "tasks"
GATEWAY_WEBHOOK_DIR = WORKSPACE / "gateway" / "webhooks"
GATEWAY_STATE_FILE = WORKSPACE / "gateway" / "state.json"

# Voice (Gemini unified)
VOICE_ENABLED = _env_bool("MOLLY_VOICE_ENABLED", False)
VOICE_MAX_SESSION_MINUTES = int(os.getenv("MOLLY_VOICE_MAX_SESSION_MINUTES", "10"))
VOICE_DAILY_BUDGET_MINUTES = int(os.getenv("MOLLY_VOICE_DAILY_BUDGET_MINUTES", "60"))

# Optional fallback orchestrator
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # GPT-4.1 Nano fallback (Tier 3)
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

- **Knowledge graph** — untouched, still fed by `process_conversation()` post-response (now also fed by voice transcripts from Gemini)
- **Triage system** — still gates group messages before they reach the orchestrator
- **Memory retrieval** — still runs during pre-processing, results go to orchestrator and workers (and into voice system prompts)
- **Approval system** — same 3-tier classification, same WhatsApp approval flow; workers reuse the same `get_action_tier()` and `ApprovalManager`
- **Self-improvement** — untouched, still runs on its own tick cycle
- **Health doctor** — untouched
- **Skills** — still matched during pre-processing, injected into worker prompts
- **Automations** — existing automation engine untouched; gateway tasks are complementary
- **WhatsApp client** — still works via neonize; WhatsApp channel adapter wraps it

---

## Technology Watch List

### Browser Tool (Active — New Worker Profile)

Molly is getting a browser tool for interacting with websites that lack APIs. This adds a `browser` worker profile to the orchestrator.

**Implementation approach**: Headless Playwright via MCP server. The browser runs as a headless Chromium instance controlled by an MCP tool server, giving Claude workers the ability to navigate, click, fill forms, extract content, and take screenshots.

**MCP server**: Use `@anthropic-ai/mcp-server-puppeteer` or `playwright-mcp-server` — both expose browser actions as MCP tools (`navigate`, `click`, `type`, `screenshot`, `extract_text`, `evaluate`). Runs headless on the Mac Mini.

**Worker profile**:
| Profile | MCP Servers | Default Model | Use Case |
|---------|-------------|---------------|----------|
| browser | browser-tool | sonnet | Web interaction, form filling, scraping |

**Use cases**:
- Restaurant reservations (OpenTable, Resy — no usable API)
- Form submissions (government sites, appointment booking)
- Price checking / comparison shopping
- Reading web pages that block API access
- Interacting with sites that require login (session cookies managed by Playwright)

**Why not WebMCP**: WebMCP (`navigator.modelContext`) is a browser-native standard for websites that opt in to exposing tools. It requires a visible browser and human-in-the-loop. Molly is a headless backend agent — she needs to control the browser, not wait for websites to expose tools. Playwright/Puppeteer is the right approach.

**Security**: Browser worker runs in a sandboxed Chromium profile. No access to Brian's main browser session/cookies. Login credentials for specific sites stored in Molly's encrypted config, injected per-session.

### Google Workspace Managed MCP Servers (Watch)

Google launched managed MCP servers for Cloud services (Dec 2025) and plans to cover all Google services. When Gmail/Calendar/Drive/Tasks get official managed MCP servers, they could replace Molly's hand-rolled tools with Google-maintained endpoints (less maintenance, enterprise auth for free). Revisit when announced.

### alibaba/zvec (Migration Candidate for Phase 7 — Maturity Gate)

**What it is**: Embedded vector database built on Alibaba's Proxima engine (battle-tested in Taobao/Alipay). HNSW ANN indexing, >8,000 QPS on 10M 768-dim vectors. Apache 2.0. Positions itself as "the SQLite of vector databases" — but it is NOT an SQLite extension. It's a separate embedded DB with its own storage format.

**Current state**: v0.2.0, released Feb 10 2026 (~5 days old), ~1,700 stars. No LangChain/LlamaIndex integrations yet. Documentation is sparse.

**Why it's interesting for Molly**:
- **Native multi-vector queries** — search across text and image embeddings simultaneously with built-in fusion/reranking. sqlite-vec can't do this; we'd need two separate searches + manual merging in Python.
- **HNSW indexing** — if conversation chunks grow past ~100K, sqlite-vec's brute-force KNN will slow down. zvec handles millions.
- **Scalar filter pushdown** — metadata filters execute inside the index path, not as post-filter JOINs.

**Why NOT to switch now**:
- **5 days old.** Unknown edge cases, incomplete docs, no ecosystem integrations.
- **Two-database architecture.** Molly's `VectorStore` class is tightly coupled to SQLite — operational tables (`tool_calls`, `skill_executions`, `corrections`, `preference_signals`, `sender_tiers`, `self_improvement_events`) live alongside vectors in one file. zvec would split this into SQLite (relational) + zvec (vectors), adding consistency concerns.
- **Current scale is fine.** Thousands of chunks, sub-millisecond brute-force. No bottleneck.

**Migration trigger**: When implementing Phase 7 (multimodal embeddings), evaluate zvec maturity at that time. If it has reached v1.0+, has published hybrid search examples, and has LangChain/LlamaIndex integrations, architect the new multi-vector search layer around zvec instead of adding more sqlite-vec virtual tables. The architecture would be:
- **SQLite** stays for all relational/operational tables
- **zvec** handles all vector storage (text + image embeddings, multi-vector queries with native fusion)
- `search()` becomes: zvec query → IDs + scores → SQLite metadata lookup

If zvec is still immature at that point, proceed with the sqlite-vec hybrid path (separate `visual_vec` table) as described in Phase 7, and revisit zvec later.

---

## Risk Mitigation

1. **Gemini API down** → Four-tier fallback: Qwen local → GPT-4.1 Nano → hardcoded general. Molly routes even when fully offline (Qwen is local).
2. **Worker fails** → Returns error text for that subtask. Other workers unaffected.
3. **Rate limits** → Semaphore caps concurrent workers at 3. Configurable.
4. **Wrong classification** → Two-phase design reduces this. Phase 1 only does simple triage (low error rate). Phase 2 uses a reasoning model (intelligence 46) for complex decomposition. Worst case = same as today.
5. **Circular dependencies in subtasks** → Detected and force-run remaining. Logged.
6. **Voice-to-voice session fails** → Gemini Live session fails to start → text channels unaffected. Voice notes still work (async through Flash-Lite).
7. **Voice cost overrun** → Per-session and daily minute budgets enforced.
8. **Voice transcript quality** → Gemini Live transcripts may diverge slightly from what the model "heard". Memory pipeline handles this gracefully since it already handles imprecise input.
9. **Browser tool security** → Sandboxed Chromium profile, no access to main browser cookies. Per-site credential isolation.
10. **Gemini single-provider dependency for routing** — Mitigated by the four-tier fallback chain. Text routing works without any cloud API (Qwen local). Only live voice-to-voice requires Gemini specifically.

---

## File Inventory

| File | Status | Purpose |
|------|--------|---------|
| `orchestrator.py` | NEW | Two-phase routing (Gemini Flash-Lite triage + Kimi K2.5 decompose), Qwen/GPT fallback, synthesis |
| `workers.py` | NEW | Parallel Claude SDK worker pool, scoped profiles, dispatch logic |
| `gateway.py` | NEW | Cron scheduler, webhook handler, built-in task definitions |
| `channels/__init__.py` | NEW | Package init |
| `channels/base.py` | NEW | Channel protocol, InboundMessage, OutboundMessage, ChannelRegistry |
| `channels/whatsapp_channel.py` | NEW | WhatsApp adapter wrapping existing neonize client |
| `channels/voice_channel.py` | NEW | Unified voice: async voice notes (via orchestrator) + live Gemini Live sessions |
| `tools/browser_mcp.py` | NEW | Browser tool MCP server config (Playwright headless) |
| `agent.py` | MODIFIED | Add orchestrator delegation with fallback, audio_bytes parameter |
| `main.py` | MODIFIED | Add gateway scheduler init + tick + webhook routes + voice endpoint |
| `config.py` | MODIFIED | Add orchestrator (Gemini models), gateway, voice, and browser settings |
| `memory/embeddings.py` | MODIFIED | Add Qwen3-VL image embedding functions alongside existing EmbeddingGemma |
| `memory/vectorstore.py` | MODIFIED | Add visual_chunks + visual_vec tables, merged search |
| `memory/retriever.py` | MODIFIED | Add visual search path to retrieve_context() |

---

## Implementation Order

Each phase is independently testable. The system works at every intermediate step because the fallback path is the existing code.

### Step 1: orchestrator.py
- Two-phase routing: Gemini Flash-Lite triage + Kimi K2.5 thinking decomposition
- Qwen local fallback + GPT-4.1 Nano fallback + hardcoded fallback
- `classify_message()` accepts text or audio_bytes
- Can test by calling `orchestrate()` directly from a script
- No changes to existing files yet

### Step 2: workers.py
- Parallel Claude SDK worker pool with scoped profiles (including browser)
- Can test with manually constructed `Subtask` objects
- No changes to existing files yet

### Step 3: agent.py wiring
- Add `use_orchestrator` flag and `audio_bytes` parameter
- Delegate to orchestrator when available
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
- Voice note detection → passes audio_bytes to orchestrator
- Prepares the interface for voice and future channels

### Step 7: channels/voice_channel.py
- Gemini Live session manager: identity injection, tool bridging, transcript capture
- WebSocket endpoint on FastAPI for audio relay (`ws://molly:8080/voice`)
- Integration with memory pipeline for transcript storage
- Can test with a simple audio test script against Gemini Live API

### Step 8: tools/browser_mcp.py
- Playwright headless browser MCP server configuration
- Sandboxed Chromium profile
- Register as `browser-tool` MCP server for the `browser` worker profile

### Step 9: config.py additions
- Small additions throughout steps 1-8
- Gemini + Kimi model names, orchestrator timeouts, voice budgets, browser config

### Step 10: memory/embeddings.py — Qwen3-VL multimodal embeddings
- Load Qwen3-VL-Embedding-2B (GGUF or transformers)
- Add `embed_image()` and `embed_multimodal()` alongside existing `embed()`
- Add `visual_chunks` + `visual_vec` tables to vectorstore
- Update retriever to merge text + visual search results
- Add image storage pipeline (hash, save, embed, graph-link)
- Can test independently with sample images before wiring into channels
