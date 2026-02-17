"""Porcupine wakeword → Gemini Live conversation loop (Phase 5C.1).

NOT a channel — Gemini Live handles its own STT, LLM reasoning, TTS,
and audio output in one bidirectional WebSocket stream.

State machine:
  [LISTENING]  ──wakeword──▶  [CONNECTING]  ──ready──▶  [CONVERSING]
       ▲                                                      │
       └──────── timeout / goodbye / error ───────────────────┘

Single mic stream (16kHz, 16-bit PCM, mono):
  LISTENING  → Porcupine.process(pcm) for wakeword
  CONVERSING → session.send_realtime_input(pcm) to Gemini Live

Cost: ~$0.023/min.  Max $1.38/day at 60-min budget.
VOICE_ENABLED=False by default.
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from enum import Enum
from pathlib import Path
from typing import Any

import config

log = logging.getLogger(__name__)


class VoiceState(str, Enum):
    LISTENING = "LISTENING"
    CONNECTING = "CONNECTING"
    CONVERSING = "CONVERSING"
    PAUSED = "PAUSED"


class VoiceLoop:
    """Porcupine wakeword → Gemini Live conversation loop."""

    # Approved voice tools (read-only, safe to call without confirmation)
    VOICE_APPROVED_TOOLS = {"check_calendar", "search_memory"}
    # Tools requiring owner confirmation (cannot auto-execute via voice)
    VOICE_CONFIRM_TOOLS = {"send_message", "create_task"}

    def __init__(self, molly: Any = None):
        self.molly = molly
        self.state = VoiceState.LISTENING
        self._state_lock = asyncio.Lock()
        self.porcupine: Any = None
        self.recorder: Any = None
        self._session: Any = None
        self._session_handle: str | None = None  # for session resumption
        self._client: Any = None
        self._running = True

        # Budget tracking (persisted to disk)
        self._daily_minutes_used = 0.0
        self._daily_reset_date: str = ""
        self._session_start_time: float = 0.0
        self._budget_state_file = (
            getattr(config, "WORKSPACE", Path.home() / ".molly") / "store" / "voice_budget.json"
        )
        self._load_budget_state()

        # Audio queues for Gemini Live conversation (bounded to prevent OOM)
        self._send_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._play_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self):
        """Main voice loop — runs as background task in Molly.run()."""
        if not config.PICOVOICE_ACCESS_KEY:
            log.warning("Voice loop disabled: PICOVOICE_ACCESS_KEY not set")
            return
        if not config.GEMINI_API_KEY:
            log.warning("Voice loop disabled: GEMINI_API_KEY not set")
            return

        try:
            self._init_porcupine()
        except Exception:
            log.error("Failed to initialize Porcupine wakeword engine", exc_info=True)
            return

        self._init_gemini_client()

        log.info(
            "Voice loop started: wakeword=%s, model=%s, budget=%dmin/day",
            config.PORCUPINE_MODEL_PATH,
            config.GEMINI_LIVE_MODEL,
            config.VOICE_DAILY_BUDGET_MINUTES,
        )

        try:
            while self._running:
                if self.state == VoiceState.LISTENING:
                    await self._listen_for_wakeword()
                elif self.state == VoiceState.CONNECTING:
                    await self._start_gemini_session()
                elif self.state == VoiceState.CONVERSING:
                    await self._run_conversation()
                elif self.state == VoiceState.PAUSED:
                    await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            log.info("Voice loop cancelled")
        finally:
            self._cleanup()

    def stop(self):
        """Signal the voice loop to stop."""
        self._running = False

    # ------------------------------------------------------------------
    # Porcupine wakeword detection
    # ------------------------------------------------------------------

    def _init_porcupine(self):
        """Initialize Porcupine wakeword engine and PvRecorder."""
        import pvporcupine
        from pvrecorder import PvRecorder

        model_path = config.PORCUPINE_MODEL_PATH
        self.porcupine = pvporcupine.create(
            access_key=config.PICOVOICE_ACCESS_KEY,
            keyword_paths=[model_path],
            sensitivities=[config.VOICE_SENSITIVITY],
        )

        device_index = config.VOICE_DEVICE_INDEX
        self.recorder = PvRecorder(
            frame_length=self.porcupine.frame_length,
            device_index=device_index if device_index >= 0 else -1,
        )
        self.recorder.start()
        log.info(
            "Porcupine initialized: frame_length=%d, sample_rate=%d",
            self.porcupine.frame_length,
            self.porcupine.sample_rate,
        )

    async def _listen_for_wakeword(self):
        """Block on mic → check for wakeword keyword."""
        try:
            pcm = await asyncio.wait_for(asyncio.to_thread(self.recorder.read), timeout=5.0)
            result = self.porcupine.process(pcm)
            if result >= 0:
                log.info("Wakeword detected! Transitioning to CONNECTING")
                async with self._state_lock:
                    self.state = VoiceState.CONNECTING
        except Exception:
            log.error("Wakeword detection error", exc_info=True)
            await asyncio.sleep(0.5)

    # ------------------------------------------------------------------
    # Gemini Live session management
    # ------------------------------------------------------------------

    def _init_gemini_client(self):
        """Initialize Google GenAI client for Gemini Live."""
        try:
            from google import genai
            self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        except Exception:
            log.error("Failed to initialize Gemini client", exc_info=True)

    async def _start_gemini_session(self):
        """Open a Gemini Live bidirectional stream."""
        if not self._check_budget():
            log.warning("Daily voice budget exhausted (%d min)", config.VOICE_DAILY_BUDGET_MINUTES)
            async with self._state_lock:
                self.state = VoiceState.LISTENING
            return

        if self._client is None:
            log.error("Gemini client not initialized")
            async with self._state_lock:
                self.state = VoiceState.LISTENING
            return

        self._session_start_time = time.monotonic()
        async with self._state_lock:
            self.state = VoiceState.CONVERSING
        log.info("Gemini Live session starting")

    def _build_live_config(self) -> dict:
        """Build the LiveConnectConfig for Gemini Live."""
        from google.genai import types

        # System instruction from identity files
        system_instruction = self._load_system_context()

        # Tool declarations so Gemini can invoke local tools
        tool_decls = get_voice_tool_declarations()

        live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            tools=[types.Tool(function_declarations=tool_decls)] if tool_decls else None,
            system_instruction=types.Content(
                parts=[types.Part(text=system_instruction)]
            ) if system_instruction else None,
        )

        # Enable session resumption if we have a previous handle
        if self._session_handle:
            live_config.session_resumption = types.SessionResumptionConfig(
                handle=self._session_handle,
            )

        return live_config

    def _load_system_context(self) -> str:
        """Load SOUL.md + USER.md for session preloading."""
        parts = []
        for identity_path in config.IDENTITY_FILES[:2]:  # SOUL.md + USER.md
            if identity_path.exists():
                try:
                    content = identity_path.read_text(encoding="utf-8").strip()
                    if content:
                        parts.append(content)
                except Exception:
                    pass
        return "\n\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Conversation (4-task pattern from Gemini Live cookbook)
    # ------------------------------------------------------------------

    async def _run_conversation(self):
        """Run a Gemini Live conversation session.

        4 concurrent async tasks (official cookbook pattern):
        1. listen_audio()  → mic → send queue
        2. send_audio()    → send queue → session.send_realtime_input()
        3. receive_audio() → session.receive() → play queue + tool calls
        4. play_audio()    → play queue → speaker (PyAudio, 24kHz output)
        """
        if self._client is None:
            async with self._state_lock:
                self.state = VoiceState.LISTENING
            return

        try:
            live_config = self._build_live_config()
            async with self._client.aio.live.connect(
                model=config.GEMINI_LIVE_MODEL,
                config=live_config,
            ) as session:
                self._session = session

                # Pre-warm with memory context if enabled
                if config.VOICE_PRELOAD_ENABLED:
                    await self._preload_context(session)

                # Run 4 concurrent tasks
                tasks = [
                    asyncio.create_task(self._listen_audio(), name="voice-listen"),
                    asyncio.create_task(self._send_audio(session), name="voice-send"),
                    asyncio.create_task(self._receive_audio(session), name="voice-receive"),
                    asyncio.create_task(self._play_audio(), name="voice-play"),
                ]

                # Also run a timeout watchdog
                max_seconds = config.VOICE_MAX_SESSION_MINUTES * 60
                tasks.append(
                    asyncio.create_task(
                        self._session_watchdog(max_seconds),
                        name="voice-watchdog",
                    )
                )

                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel remaining tasks
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

                # Check for errors in completed tasks
                for t in done:
                    if t.exception():
                        log.error(
                            "Voice task %s failed: %s",
                            t.get_name(),
                            t.exception(),
                        )

        except Exception:
            log.error("Gemini Live session error", exc_info=True)
        finally:
            self._session = None
            if self._session_start_time > 0:
                elapsed = time.monotonic() - self._session_start_time
                self._daily_minutes_used += elapsed / 60.0
                self._save_budget_state()
                log.info(
                    "Voice session ended: %.1fs, daily budget: %.1f/%.0f min",
                    elapsed,
                    self._daily_minutes_used,
                    config.VOICE_DAILY_BUDGET_MINUTES,
                )
            else:
                log.warning("Voice session ended without valid start time — budget not updated")
            self._session_start_time = 0.0

            # Clear queues (exception-based exit, not check-then-act)
            for q in (self._send_queue, self._play_queue):
                while True:
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            async with self._state_lock:
                self.state = VoiceState.LISTENING

    async def _preload_context(self, session):
        """Pre-warm the session with memory context."""
        try:
            from memory.retriever import retrieve_context
            context = await retrieve_context("voice conversation starting", top_k=3)
            if context:
                from google.genai import types
                await session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part(text=f"Context from memory:\n{context}")],
                    ),
                    turn_complete=True,
                )
                log.debug("Voice session preloaded with %d chars of context", len(context))
        except Exception:
            log.debug("Voice context preload failed", exc_info=True)

    async def _listen_audio(self):
        """Read mic frames and push to send queue (drop if full)."""
        while self._running and self.state == VoiceState.CONVERSING:
            try:
                pcm = await asyncio.wait_for(asyncio.to_thread(self.recorder.read), timeout=5.0)
                # Convert int16 PCM list to bytes for Gemini
                audio_bytes = struct.pack(f"{len(pcm)}h", *pcm)
                try:
                    self._send_queue.put_nowait(audio_bytes)
                except asyncio.QueueFull:
                    pass  # Audio frames are ephemeral — OK to drop
            except Exception:
                log.debug("Listen audio error", exc_info=True)
                break

    async def _send_audio(self, session):
        """Pull from send queue and stream to Gemini Live."""
        from google.genai import types

        while self._running and self.state == VoiceState.CONVERSING:
            try:
                audio_bytes = await asyncio.wait_for(
                    self._send_queue.get(), timeout=0.5
                )
                await session.send_realtime_input(
                    audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000"),
                )
            except asyncio.TimeoutError:
                continue
            except Exception:
                log.debug("Send audio error", exc_info=True)
                break

    async def _receive_audio(self, session):
        """Receive responses from Gemini Live (audio + tool calls)."""
        while self._running and self.state == VoiceState.CONVERSING:
            try:
                async for response in session.receive():
                    server_content = getattr(response, "server_content", None)
                    if server_content:
                        model_turn = getattr(server_content, "model_turn", None)
                        if model_turn and hasattr(model_turn, "parts"):
                            for part in model_turn.parts:
                                inline_data = getattr(part, "inline_data", None)
                                if inline_data and hasattr(inline_data, "data"):
                                    await self._play_queue.put(inline_data.data)

                        # Check for turn completion
                        turn_complete = getattr(server_content, "turn_complete", False)
                        if turn_complete:
                            # Signal end of audio with sentinel
                            await self._play_queue.put(None)

                        # Check for interruption
                        interrupted = getattr(server_content, "interrupted", False)
                        if interrupted:
                            # Clear play queue on interruption
                            while True:
                                try:
                                    self._play_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    break

                    # Handle tool calls
                    tool_call = getattr(response, "tool_call", None)
                    if tool_call:
                        await self._handle_tool_call(session, tool_call)

                    # Handle session resumption
                    session_resumption = getattr(
                        response, "session_resumption_update", None
                    )
                    if session_resumption:
                        new_handle = getattr(session_resumption, "new_handle", None)
                        if new_handle:
                            self._session_handle = new_handle
                            log.debug("Session handle updated for resumption")
                        else:
                            # Resumption update without a handle → clear stale handle
                            self._session_handle = None
                            log.debug("Session handle cleared (no new handle in update)")

                # Session ended naturally
                break

            except Exception:
                log.debug("Receive audio error", exc_info=True)
                break

    async def _play_audio(self):
        """Pull audio from play queue and output to speaker."""
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            # Gemini Live outputs at 24kHz
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True,
                frames_per_buffer=1024,
            )
        except Exception:
            log.warning("PyAudio unavailable — voice output disabled", exc_info=True)
            # Drain queue without playing
            while self._running and self.state == VoiceState.CONVERSING:
                try:
                    await asyncio.wait_for(self._play_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    log.debug("Voice stream read timed out, continuing")
                    continue
            return

        try:
            while self._running and self.state == VoiceState.CONVERSING:
                try:
                    audio_data = await asyncio.wait_for(
                        self._play_queue.get(), timeout=1.0,
                    )
                    if audio_data is None:
                        continue  # turn-complete sentinel
                    await asyncio.to_thread(stream.write, audio_data)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    log.debug("Play audio error", exc_info=True)
                    break
        finally:
            try:
                stream.stop_stream()
                stream.close()
                pa.terminate()
            except Exception:
                pass

    async def _handle_tool_call(self, session, tool_call):
        """Bridge Gemini Live tool calls to Molly's MCP tools."""
        function_calls = getattr(tool_call, "function_calls", [])
        if not function_calls:
            return

        from google.genai import types

        for fc in function_calls:
            fn_name = getattr(fc, "name", "")
            fn_args = getattr(fc, "args", {})
            fn_id = getattr(fc, "id", "")

            log.info("Voice tool call: %s(%s)", fn_name, fn_args)

            try:
                result = await self._execute_tool(fn_name, fn_args)
            except Exception as exc:
                log.error("Voice tool call failed: %s", exc, exc_info=True)
                result = {"error": str(exc)}

            # Send tool response back to Gemini
            try:
                await session.send_tool_response(
                    function_responses=[
                        types.FunctionResponse(
                            name=fn_name,
                            id=fn_id,
                            response=result if isinstance(result, dict) else {"result": str(result)},
                        )
                    ]
                )
            except Exception:
                log.error("Failed to send tool response to Gemini", exc_info=True)

    async def _execute_tool(self, tool_name: str, args: dict) -> dict:
        """Execute a tool by name, respecting approval tiers.

        Only VOICE_APPROVED_TOOLS (read-only) execute directly.
        VOICE_CONFIRM_TOOLS require owner confirmation (currently denied via voice).
        """
        # Tier check: only allow pre-approved read-only tools via voice
        if tool_name in self.VOICE_CONFIRM_TOOLS:
            log.warning(
                "Voice tool '%s' requires confirmation — denied (no approval channel in voice)",
                tool_name,
            )
            return {
                "error": f"Tool '{tool_name}' requires confirmation. "
                "Please use WhatsApp or web chat to execute this action."
            }
        if tool_name not in self.VOICE_APPROVED_TOOLS:
            log.warning("Voice tool '%s' not in approved list — denied", tool_name)
            return {"error": f"Tool '{tool_name}' is not available via voice."}

        tool_map = {
            "check_calendar": self._tool_check_calendar,
            "search_memory": self._tool_search_memory,
        }

        handler = tool_map.get(tool_name)
        if handler:
            return await handler(args)
        return {"error": f"Unknown tool: {tool_name}"}

    async def _tool_check_calendar(self, args: dict) -> dict:
        """Check calendar events."""
        try:
            from tools.calendar import calendar_search
            result = await calendar_search(args)
            return result if isinstance(result, dict) else {"result": str(result)}
        except Exception as exc:
            return {"error": str(exc)}

    async def _tool_send_message(self, args: dict) -> dict:
        """Send a WhatsApp message."""
        if self.molly and self.molly.wa:
            recipient = args.get("recipient", "")
            text = args.get("text", "")
            if recipient and text:
                owner_jid = self.molly._get_owner_dm_jid()
                if owner_jid:
                    msg_id = self.molly.wa.send_message(owner_jid, text)
                    self.molly._track_send(msg_id)
                    return {"status": "sent", "msg_id": str(msg_id)}
        return {"error": "messaging not available"}

    async def _tool_create_task(self, args: dict) -> dict:
        """Create a Google Task."""
        try:
            from tools.google_tasks import tasks_create
            result = await tasks_create(args)
            return result if isinstance(result, dict) else {"result": str(result)}
        except Exception as exc:
            return {"error": str(exc)}

    async def _tool_search_memory(self, args: dict) -> dict:
        """Search memory for relevant context."""
        try:
            from memory.retriever import retrieve_context
            query = args.get("query", "")
            context = await retrieve_context(query, top_k=3)
            return {"context": context or "No relevant memories found."}
        except Exception as exc:
            return {"error": str(exc)}

    async def _session_watchdog(self, max_seconds: int):
        """End session after max duration."""
        await asyncio.sleep(max_seconds)
        log.info("Voice session timeout after %ds", max_seconds)
        async with self._state_lock:
            self.state = VoiceState.LISTENING

    # ------------------------------------------------------------------
    # Budget tracking
    # ------------------------------------------------------------------

    def _check_budget(self) -> bool:
        """Check if daily budget allows another session."""
        today = time.strftime("%Y-%m-%d")
        if self._daily_reset_date != today:
            self._daily_reset_date = today
            self._daily_minutes_used = 0.0
            self._save_budget_state()
        return self._daily_minutes_used < config.VOICE_DAILY_BUDGET_MINUTES

    def _load_budget_state(self) -> None:
        """Load persisted budget state from disk."""
        try:
            if self._budget_state_file.exists():
                raw = self._budget_state_file.read_text(encoding="utf-8")
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError(f"Expected dict, got {type(data).__name__}")
                minutes = float(data.get("minutes_used", 0.0))
                if minutes < 0 or minutes > 1440:  # sanity: max 24h
                    log.warning("Voice budget state corrupt (minutes=%.1f), resetting", minutes)
                    minutes = 0.0
                self._daily_minutes_used = minutes
                self._daily_reset_date = str(data.get("reset_date", ""))
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("Voice budget state corrupt (%s), resetting to defaults", exc)
            self._daily_minutes_used = 0.0
            self._daily_reset_date = ""
        except Exception:
            log.warning("Failed to load voice budget state", exc_info=True)

    def _save_budget_state(self) -> None:
        """Persist budget state to disk atomically (survives crashes)."""
        try:
            self._budget_state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "minutes_used": round(self._daily_minutes_used, 2),
                "reset_date": self._daily_reset_date,
            }
            # Atomic write: temp file + os.replace to prevent corruption
            import os
            import tempfile as _tempfile
            fd, tmp_path = _tempfile.mkstemp(
                dir=str(self._budget_state_file.parent),
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.replace(tmp_path, str(self._budget_state_file))
            except BaseException:
                # Clean up temp file on any error
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception:
            log.debug("Failed to save voice budget state", exc_info=True)

    def get_stats(self) -> dict:
        """Return voice loop statistics."""
        return {
            "state": self.state.value,
            "daily_minutes_used": round(self._daily_minutes_used, 1),
            "daily_budget_minutes": config.VOICE_DAILY_BUDGET_MINUTES,
            "session_handle": self._session_handle is not None,
            "running": self._running,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self):
        """Release Porcupine and recorder resources."""
        if self.recorder is not None:
            try:
                self.recorder.stop()
                self.recorder.delete()
            except Exception:
                pass
            self.recorder = None

        if self.porcupine is not None:
            try:
                self.porcupine.delete()
            except Exception:
                pass
            self.porcupine = None

        log.info("Voice loop resources cleaned up")


# ---------------------------------------------------------------------------
# Gemini Live tool declarations (for session config)
# ---------------------------------------------------------------------------

def get_voice_tool_declarations() -> list[dict]:
    """Return function declarations for Gemini Live tool calling."""
    return [
        {
            "name": "check_calendar",
            "description": "Check Google Calendar for upcoming events or search for specific events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for calendar events",
                    },
                    "time_min": {
                        "type": "string",
                        "description": "Start time in ISO format",
                    },
                    "time_max": {
                        "type": "string",
                        "description": "End time in ISO format",
                    },
                },
            },
        },
        {
            "name": "send_message",
            "description": "Send a WhatsApp message to Brian.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Message text to send",
                    },
                    "recipient": {
                        "type": "string",
                        "description": "Recipient name or ID",
                    },
                },
                "required": ["text"],
            },
        },
        {
            "name": "create_task",
            "description": "Create a new task in Google Tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Task title",
                    },
                    "due": {
                        "type": "string",
                        "description": "Due date in ISO format",
                    },
                },
                "required": ["title"],
            },
        },
        {
            "name": "search_memory",
            "description": "Search Molly's memory for relevant past conversations and knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory",
                    },
                },
                "required": ["query"],
            },
        },
    ]


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if "--test" in sys.argv:
        print("Voice loop standalone test")
        print(f"  VOICE_ENABLED: {config.VOICE_ENABLED}")
        print(f"  PICOVOICE_ACCESS_KEY: {'set' if config.PICOVOICE_ACCESS_KEY else 'NOT SET'}")
        print(f"  PORCUPINE_MODEL_PATH: {config.PORCUPINE_MODEL_PATH}")
        print(f"  GEMINI_API_KEY: {'set' if config.GEMINI_API_KEY else 'NOT SET'}")
        print(f"  GEMINI_LIVE_MODEL: {config.GEMINI_LIVE_MODEL}")

        from pathlib import Path
        ppn_path = Path(config.PORCUPINE_MODEL_PATH)
        print(f"  .ppn exists: {ppn_path.exists()}")

        if config.PICOVOICE_ACCESS_KEY and ppn_path.exists():
            print("\n  Testing Porcupine initialization...")
            vl = VoiceLoop()
            try:
                vl._init_porcupine()
                print(f"  ✓ Porcupine ready (frame_length={vl.porcupine.frame_length})")
                vl._cleanup()
                print("  ✓ Cleanup OK")
            except Exception as e:
                print(f"  ✗ Porcupine init failed: {e}")
        else:
            print("  Skipping Porcupine test (missing key or model)")
    else:
        vl = VoiceLoop()
        asyncio.run(vl.run())
