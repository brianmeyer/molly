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
import importlib
import json
import logging
import struct
import time
from contextlib import suppress
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
        self._end_session = False  # set by end_conversation tool

        # Dynamic tool registry (loaded once, cached)
        self._tool_declarations: list[dict] | None = None
        self._tool_handlers: dict[str, Any] | None = None

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

        # Transcript collection for memory storage
        self._transcript: list[dict] = []  # [{"role": "user"|"model", "text": "..."}]

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

        # Initialize Porcupine wakeword engine (retry if audio device unavailable)
        initialized = False
        for attempt in range(5):
            try:
                self._init_porcupine()
                initialized = True
                break
            except Exception:
                if attempt == 0:
                    log.warning(
                        "Porcupine init failed (attempt %d/5) — "
                        "audio device may be unavailable, retrying in 10s",
                        attempt + 1, exc_info=True,
                    )
                else:
                    log.info("Porcupine init retry %d/5 failed", attempt + 1)
                await asyncio.sleep(10.0)

        if not initialized:
            log.error("Porcupine init failed after 5 attempts — voice loop entering wait mode")

        self._init_gemini_client()

        log.info(
            "Voice loop started: wakeword=%s, model=%s, budget=%dmin/day, audio=%s",
            config.PORCUPINE_MODEL_PATH,
            config.GEMINI_LIVE_MODEL,
            config.VOICE_DAILY_BUDGET_MINUTES,
            "ready" if initialized else "waiting for device",
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

    def _reinit_recorder(self) -> bool:
        """Reinitialize PvRecorder after audio device change (e.g. AirPods removed).

        Returns True if successful, False otherwise.
        """
        try:
            if self.recorder is not None:
                with suppress(Exception):
                    self.recorder.stop()
                with suppress(Exception):
                    self.recorder.delete()
                self.recorder = None

            from pvrecorder import PvRecorder
            device_index = config.VOICE_DEVICE_INDEX
            self.recorder = PvRecorder(
                frame_length=self.porcupine.frame_length,
                device_index=device_index if device_index >= 0 else -1,
            )
            self.recorder.start()
            log.info("PvRecorder reinitialized after audio device change")
            return True
        except Exception:
            log.warning("Failed to reinitialize PvRecorder", exc_info=True)
            self.recorder = None
            return False

    async def _listen_for_wakeword(self):
        """Block on mic → check for wakeword keyword."""
        if self.porcupine is None:
            # Porcupine not initialized — try full init periodically
            await asyncio.sleep(5.0)
            try:
                self._init_porcupine()
                log.info("Porcupine initialized (deferred)")
            except Exception:
                pass  # will retry next cycle
            return

        if self.recorder is None:
            # Audio device unavailable — try to reinitialize periodically
            await asyncio.sleep(3.0)
            self._reinit_recorder()
            return

        try:
            pcm = await asyncio.wait_for(asyncio.to_thread(self.recorder.read), timeout=5.0)
            result = self.porcupine.process(pcm)
            if result >= 0:
                log.info("Wakeword detected! Transitioning to CONNECTING")
                async with self._state_lock:
                    self.state = VoiceState.CONNECTING
        except Exception:
            log.warning("Audio device error — attempting to reinitialize recorder")
            if not self._reinit_recorder():
                await asyncio.sleep(3.0)  # wait before retry

    # ------------------------------------------------------------------
    # Gemini Live session management
    # ------------------------------------------------------------------

    def _init_gemini_client(self):
        """Initialize Google GenAI client for Gemini Live."""
        try:
            from google import genai
            self._client = genai.Client(
                api_key=config.GEMINI_API_KEY,
                http_options={"api_version": "v1alpha"},
            )
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

        # Dynamically load all MCP tools on first use
        self._ensure_tools_loaded()
        tool_decls = self._tool_declarations

        live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            proactivity={"proactive_audio": True},
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
        """Load SOUL.md + USER.md + voice-specific conversation instructions."""
        parts = []
        for identity_path in config.IDENTITY_FILES[:2]:  # SOUL.md + USER.md
            if identity_path.exists():
                try:
                    content = identity_path.read_text(encoding="utf-8").strip()
                    if content:
                        parts.append(content)
                except Exception:
                    pass

        # Voice-specific instructions for multi-turn conversation
        parts.append(
            "VOICE CONVERSATION RULES:\n"
            "- You are in a live voice conversation activated by a wakeword.\n"
            "- IMMEDIATELY greet Brian warmly (e.g. 'Hey Brian! What's up?'). "
            "Do not wait for him to speak first — you start the conversation.\n"
            "- Keep responses concise and conversational — this is spoken audio, "
            "not text. Avoid long lists or dense information.\n"
            "- After responding, wait for the user to speak. Do NOT end the "
            "conversation on your own.\n"
            "- The conversation continues back and forth until the user says "
            "they're done (e.g. 'I'm done', 'goodbye', 'that's all', 'see ya').\n"
            "- When the user indicates they're done, say a brief goodbye and "
            "then call the end_conversation tool to close the session.\n"
            "- Do NOT call end_conversation unless the user explicitly says "
            "they want to stop talking."
        )

        return "\n\n".join(parts) if parts else ""

    def _ensure_tools_loaded(self):
        """Load all MCP tools once, then cache on the instance."""
        if self._tool_declarations is None:
            self._tool_declarations, self._tool_handlers = _load_all_tools()

    # ------------------------------------------------------------------
    # Conversation (4-task pattern from Gemini Live cookbook)
    # ------------------------------------------------------------------

    async def _run_conversation(self):
        """Run a Gemini Live conversation session.

        session.receive() yields messages for ONE model turn then breaks
        (by design — see googleapis/python-genai live.py).  The WebSocket
        stays open.  We call receive() again in a while-loop for multi-turn.

        Official pattern from google-gemini/cookbook:
          while True:
              async for response in session.receive():
                  ...  # one turn
        """
        if self._client is None:
            async with self._state_lock:
                self.state = VoiceState.LISTENING
            return

        self._end_session = False

        try:
            live_config = self._build_live_config()
            async with self._client.aio.live.connect(
                model=config.GEMINI_LIVE_MODEL,
                config=live_config,
            ) as session:
                self._session = session

                # Identity context (SOUL.md, USER.md) is in the system
                # instruction.  Memory preload uses send_client_content which
                # must not be mixed with send_realtime_input per the docs.
                # proactive_audio + system instruction handles the greeting.

                # Run concurrent tasks
                tasks = [
                    asyncio.create_task(self._listen_audio(), name="voice-listen"),
                    asyncio.create_task(self._send_audio(session), name="voice-send"),
                    asyncio.create_task(self._receive_audio(session), name="voice-receive"),
                    asyncio.create_task(self._play_audio(), name="voice-play"),
                ]

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

                for t in done:
                    exc = t.exception()
                    if exc:
                        log.warning(
                            "Voice task '%s' failed: %s: %s",
                            t.get_name(), type(exc).__name__, exc,
                        )
                    else:
                        log.info("Voice task '%s' completed first", t.get_name())

                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

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
                log.warning("Voice session ended without valid start time")
            self._session_start_time = 0.0

            # Save voice transcript to memory (full pipeline)
            if self._transcript:
                await self._save_transcript_to_memory()
            self._transcript = []

            # Clear queues
            for q in (self._send_queue, self._play_queue):
                while True:
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            async with self._state_lock:
                self.state = VoiceState.LISTENING

    async def _save_transcript_to_memory(self):
        """Save the voice conversation transcript through the full memory pipeline.

        Uses process_conversation (L2 vectorstore + L3 graph + daily log)
        for each user→model exchange, same as WhatsApp conversations.
        """
        try:
            from memory.processor import process_conversation

            # Group transcript into user→model pairs
            pairs: list[tuple[str, str]] = []
            user_buf: list[str] = []
            for entry in self._transcript:
                if entry["role"] == "user":
                    user_buf.append(entry["text"])
                elif entry["role"] == "model":
                    user_text = " ".join(user_buf) if user_buf else "(voice)"
                    pairs.append((user_text, entry["text"]))
                    user_buf = []

            if not pairs:
                return

            for user_text, model_text in pairs:
                await process_conversation(
                    user_msg=user_text,
                    assistant_msg=model_text,
                    chat_jid="voice",
                    source="voice",
                )
            log.info(
                "Voice transcript saved to memory (%d exchanges, full pipeline)",
                len(pairs),
            )
        except Exception:
            log.warning("Failed to save voice transcript to memory", exc_info=True)

    async def _listen_audio(self):
        """Read mic frames and push to send queue (drop if full)."""
        consecutive_errors = 0
        while self._running and self.state == VoiceState.CONVERSING and not self._end_session:
            if self.recorder is None:
                log.warning("Audio device lost during conversation — ending session")
                break
            try:
                pcm = await asyncio.wait_for(asyncio.to_thread(self.recorder.read), timeout=5.0)
                consecutive_errors = 0
                # Convert int16 PCM list to bytes for Gemini
                audio_bytes = struct.pack(f"{len(pcm)}h", *pcm)
                try:
                    self._send_queue.put_nowait(audio_bytes)
                except asyncio.QueueFull:
                    pass  # Audio frames are ephemeral — OK to drop
            except asyncio.TimeoutError:
                continue
            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    log.warning("Audio device error (%dx) — ending conversation", consecutive_errors)
                    # Try to reinit for next wakeword session
                    self._reinit_recorder()
                    break
                log.debug("Listen audio error (%d/3)", consecutive_errors)
                await asyncio.sleep(0.2)

    async def _send_audio(self, session):
        """Pull from send queue and stream to Gemini Live."""
        from google.genai import types

        chunks_sent = 0
        while self._running and self.state == VoiceState.CONVERSING and not self._end_session:
            try:
                audio_bytes = await asyncio.wait_for(
                    self._send_queue.get(), timeout=0.5
                )
                await session.send_realtime_input(
                    audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000"),
                )
                chunks_sent += 1
                if chunks_sent == 1:
                    log.info("First audio chunk sent to Gemini (%d bytes)", len(audio_bytes))
                elif chunks_sent % 500 == 0:
                    log.debug("Audio chunks sent: %d", chunks_sent)
            except asyncio.TimeoutError:
                continue
            except Exception:
                log.info("Send audio error after %d chunks", chunks_sent, exc_info=True)
                break
        log.info("Send audio loop exiting after %d chunks", chunks_sent)

    async def _receive_audio(self, session):
        """Receive responses from Gemini Live (audio + tool calls).

        session.receive() yields messages for ONE model turn, then the
        iterator breaks (by design — see google/genai/live.py).  The
        WebSocket stays open.  We call receive() again for each turn.

        Official pattern from google-gemini/cookbook:
          while True:
              async for response in session.receive():
                  ...
        """
        turns = 0
        while self._running and self.state == VoiceState.CONVERSING and not self._end_session:
            try:
                async for response in session.receive():
                    if self._end_session:
                        log.info("end_conversation flag — exiting receive")
                        return

                    server_content = getattr(response, "server_content", None)
                    if server_content:
                        model_turn = getattr(server_content, "model_turn", None)
                        if model_turn and hasattr(model_turn, "parts"):
                            for part in model_turn.parts:
                                inline_data = getattr(part, "inline_data", None)
                                if inline_data and hasattr(inline_data, "data"):
                                    await self._play_queue.put(inline_data.data)

                        turn_complete = getattr(server_content, "turn_complete", False)
                        if turn_complete:
                            await self._play_queue.put(None)
                            turns += 1
                            log.info("Gemini turn %d complete", turns)

                        interrupted = getattr(server_content, "interrupted", False)
                        if interrupted:
                            log.debug("Gemini interrupted at turn %d", turns)
                            while not self._play_queue.empty():
                                try:
                                    self._play_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    break

                        # Capture transcriptions for memory
                        for tx_field, role in [
                            ("output_transcription", "model"),
                            ("input_transcription", "user"),
                        ]:
                            tx = getattr(server_content, tx_field, None)
                            if tx:
                                text = getattr(tx, "text", "") or ""
                                if text.strip():
                                    self._transcript.append({"role": role, "text": text.strip()})

                    tool_call = getattr(response, "tool_call", None)
                    if tool_call:
                        await self._handle_tool_call(session, tool_call)
                        if self._end_session:
                            log.info("end_conversation handled — exiting receive")
                            return

                    session_resumption = getattr(
                        response, "session_resumption_update", None
                    )
                    if session_resumption:
                        new_handle = getattr(session_resumption, "new_handle", None)
                        if new_handle:
                            self._session_handle = new_handle

                # Turn complete — receive() broke out. Loop back for next turn.
                log.debug("Turn %d iterator done, calling receive() for next turn", turns)

            except Exception as exc:
                log.info("Receive loop ended after %d turns: %s", turns, exc)
                break

        log.info("Receive loop exiting after %d turns (end_session=%s)", turns, self._end_session)

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
        """Execute a tool by name — all MCP tools available, same as text channels."""
        # Voice-only tools
        if tool_name == "end_conversation":
            log.info("end_conversation tool called — ending voice session")
            self._end_session = True
            return {"status": "ending"}

        if tool_name == "search_memory":
            try:
                from memory.retriever import retrieve_context
                query = args.get("query", "")
                context = await retrieve_context(query, top_k=5)
                return {"result": context or "No relevant memories found."}
            except Exception as exc:
                return {"error": str(exc)}

        # Dynamic MCP tool dispatch
        self._ensure_tools_loaded()
        handler = self._tool_handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            result = await handler(args)
        except Exception as exc:
            log.error("Voice tool '%s' execution error: %s", tool_name, exc, exc_info=True)
            return {"error": str(exc)}

        # Convert MCP response format {"content": [{"type":"text","text":"..."}]}
        # to a simple dict for Gemini function response
        if isinstance(result, dict) and "content" in result:
            texts = []
            for block in result.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            text = "\n".join(texts)
            if result.get("is_error"):
                return {"error": text}
            return {"result": text}
        return result if isinstance(result, dict) else {"result": str(result)}

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
# Dynamic MCP tool bridge — same tools as text channels
# ---------------------------------------------------------------------------

# Python MCP tool modules to load for voice
_VOICE_TOOL_MODULES = [
    "tools.calendar",
    "tools.gmail",
    "tools.google_people",
    "tools.google_tasks",
    "tools.google_drive",
    "tools.google_meet",
    "tools.imessage",
    "tools.whatsapp",
    "tools.kimi",
    "tools.grok",
    "tools.groq",
]

# Python type → JSON Schema type for Gemini function declarations
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _load_all_tools() -> tuple[list[dict], dict[str, Any]]:
    """Import all MCP tool modules and extract SdkMcpTool instances.

    Returns (declarations, handlers) where declarations are Gemini Live
    function declarations and handlers maps tool_name → async callable.
    """
    from claude_agent_sdk import SdkMcpTool

    declarations: list[dict] = []
    handlers: dict[str, Any] = {}

    for module_path in _VOICE_TOOL_MODULES:
        try:
            mod = importlib.import_module(module_path)
        except Exception:
            log.warning("Voice: failed to import %s", module_path, exc_info=True)
            continue

        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            if not isinstance(obj, SdkMcpTool):
                continue
            if obj.name in config.DISABLED_TOOL_NAMES:
                continue

            # Convert input_schema {param: type} → Gemini properties
            properties = {}
            schema = obj.input_schema
            if isinstance(schema, dict):
                for param, ptype in schema.items():
                    properties[param] = {
                        "type": _TYPE_MAP.get(ptype, "string"),
                        "description": param.replace("_", " "),
                    }

            declarations.append({
                "name": obj.name,
                "description": obj.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                },
            })
            handlers[obj.name] = obj.handler

    # Voice-only tools (not from MCP servers)
    declarations.append({
        "name": "search_memory",
        "description": (
            "Search Molly's memory for relevant past conversations, facts, "
            "and knowledge about Brian. Use this to recall things discussed "
            "before, preferences, events, people, or any stored context."
        ),
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
    })

    declarations.append({
        "name": "end_conversation",
        "description": (
            "End the voice conversation. Call this ONLY when the user "
            "explicitly says they're done talking (e.g. 'I'm done', "
            "'goodbye', 'that's all', 'see ya'). Say a brief goodbye "
            "before calling this."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    })

    log.info(
        "Voice tools loaded: %d declarations (%d MCP + 2 voice-only)",
        len(declarations), len(handlers),
    )
    return declarations, handlers


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
