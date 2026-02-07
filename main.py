import asyncio
import json
import logging
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import config
from agent import handle_message
from approval import ApprovalManager
from commands import handle_command
from database import Database
from heartbeat import run_heartbeat, should_heartbeat
from maintenance import run_maintenance, should_run_maintenance
from whatsapp import WhatsAppClient

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
config.LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_DIR / "molly.log"),
    ],
)
log = logging.getLogger("molly")


# ---------------------------------------------------------------------------
# Service startup checks
# ---------------------------------------------------------------------------
def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def ensure_docker():
    """Verify the Docker daemon is reachable."""
    result = _run(["docker", "info"])
    if result.returncode != 0:
        log.error("Docker daemon is not running. Start Docker Desktop and retry.")
        sys.exit(1)
    log.info("Docker daemon: running")


def ensure_neo4j():
    """Ensure the Neo4j container is running, starting or creating it as needed."""
    # Check if container exists
    result = _run(["docker", "inspect", "--format", "{{.State.Status}}", "neo4j"])

    if result.returncode != 0:
        # Container doesn't exist — create and start it
        log.info("Neo4j container not found. Creating...")
        result = _run([
            "docker", "run", "-d", "--name", "neo4j",
            "-p", "7474:7474", "-p", "7687:7687",
            "-e", f"NEO4J_AUTH={config.NEO4J_USER}/{config.NEO4J_PASSWORD}",
            "-v", f"{Path.home() / '.molly' / 'neo4j'}:/data",
            "--restart", "unless-stopped",
            "neo4j:latest",
        ])
        if result.returncode != 0:
            log.error("Failed to create Neo4j container: %s", result.stderr.strip())
            sys.exit(1)
        log.info("Neo4j container created and started")
    else:
        status = result.stdout.strip()
        if status == "running":
            log.info("Neo4j container: already running")
        else:
            log.info("Neo4j container status: %s — starting...", status)
            result = _run(["docker", "start", "neo4j"])
            if result.returncode != 0:
                log.error("Failed to start Neo4j: %s", result.stderr.strip())
                sys.exit(1)
            log.info("Neo4j container started")

    # Wait for Neo4j to accept bolt connections
    _wait_for_neo4j()


def _wait_for_neo4j(timeout: int = 30):
    """Block until Neo4j responds on bolt://localhost:7687."""
    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", 7687), timeout=2):
                log.info("Neo4j bolt port: ready")
                return
        except OSError:
            time.sleep(1)

    log.error("Neo4j did not become ready within %ds", timeout)
    sys.exit(1)


def ensure_ollama():
    """Ensure Ollama is running and the triage model is available.

    Non-blocking on failure — triage is optional, Molly works without it.
    """
    import urllib.request
    import urllib.error

    def _ollama_api_up() -> bool:
        try:
            req = urllib.request.Request(f"{config.OLLAMA_BASE_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def _model_available() -> bool:
        try:
            req = urllib.request.Request(f"{config.OLLAMA_BASE_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                import json as _json
                data = _json.loads(resp.read())
                model_names = [m.get("name", "") for m in data.get("models", [])]
                # Match both "qwen3:4b" and "qwen3:4b-<hash>" variants
                target = config.TRIAGE_MODEL.split(":")[0]
                return any(target in name for name in model_names)
        except Exception:
            return False

    # Check if Ollama binary exists
    result = _run(["which", "ollama"])
    if result.returncode != 0:
        log.warning("Ollama not found. Installing via Homebrew...")
        result = _run(["brew", "install", "ollama"])
        if result.returncode != 0:
            log.warning("Ollama install failed — triage will be unavailable")
            return

    # Check if Ollama API is up
    if not _ollama_api_up():
        log.info("Ollama not running — starting...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait up to 15 seconds
        deadline = time.time() + 15
        while time.time() < deadline:
            if _ollama_api_up():
                break
            time.sleep(1)
        else:
            log.warning("Ollama failed to start within 15s — triage will be unavailable")
            return

    log.info("Ollama: running")

    # Check if triage model is pulled
    if not _model_available():
        log.info("Pulling triage model %s (this may take a few minutes)...", config.TRIAGE_MODEL)
        result = _run(["ollama", "pull", config.TRIAGE_MODEL], timeout=600)
        if result.returncode != 0:
            log.warning("Failed to pull %s — triage will be unavailable", config.TRIAGE_MODEL)
            return
        log.info("Triage model %s pulled successfully", config.TRIAGE_MODEL)
    else:
        log.info("Triage model %s: available", config.TRIAGE_MODEL)


def ensure_google_auth():
    """Ensure Google OAuth token exists, triggering browser flow if needed.

    Runs synchronously at startup (before the async loop) so the browser
    OAuth consent screen opens immediately on first run.
    """
    if not config.GOOGLE_CLIENT_SECRET.exists():
        log.warning(
            "Google client_secret.json not found at %s — "
            "Calendar and Gmail tools will be unavailable. "
            "See ~/.molly/docs/PHASE-3B.md for setup instructions.",
            config.GOOGLE_CLIENT_SECRET,
        )
        return

    try:
        from tools.google_auth import get_credentials
        creds = get_credentials()
        log.info("Google OAuth: authenticated (%s)", "valid" if creds.valid else "refreshed")
    except Exception:
        log.error("Google OAuth failed — Calendar and Gmail tools will be unavailable", exc_info=True)


def preflight_checks():
    """Run all service checks before Molly starts."""
    log.info("Running preflight checks...")
    ensure_docker()
    ensure_neo4j()
    ensure_ollama()
    ensure_google_auth()
    log.info("Preflight checks passed")


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def load_json(path: Path, default=None):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to load %s: %s", path, e)
    return default if default is not None else {}


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Molly core
# ---------------------------------------------------------------------------
class Molly:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.db = Database(config.DATABASE_PATH)
        self.wa: WhatsAppClient | None = None
        self.sessions: dict[str, str] = {}
        self.registered_chats: dict[str, dict] = {}
        self.state: dict = {}
        self.running = True
        self.start_time: datetime | None = None
        self.last_heartbeat: datetime | None = None
        self.last_maintenance: datetime | None = None
        self._sent_ids: set[str] = set()  # message IDs Molly sent (avoid echo loop)
        self.approvals = ApprovalManager()

    # --- State persistence ---

    def load_state(self):
        self.sessions = load_json(config.SESSIONS_FILE, {})
        self.registered_chats = load_json(config.REGISTERED_CHATS_FILE, {})
        self.state = load_json(config.STATE_FILE, {})

        # Migrate: add "mode" field to existing registered chat entries
        migrated = False
        for jid, info in self.registered_chats.items():
            if "mode" not in info:
                info["mode"] = "respond"
                migrated = True
        if migrated:
            self.save_registered_chats()
            log.info("Migrated registered_chats: added 'mode' field")

        log.info(
            "State loaded: %d sessions, %d registered chats",
            len(self.sessions),
            len(self.registered_chats),
        )

    def save_sessions(self):
        save_json(config.SESSIONS_FILE, self.sessions)

    def save_state(self):
        save_json(config.STATE_FILE, self.state)

    def save_registered_chats(self):
        save_json(config.REGISTERED_CHATS_FILE, self.registered_chats)

    def _is_owner(self, sender_jid: str) -> bool:
        """Check if the sender is Brian (by phone or LID)."""
        user = sender_jid.split("@")[0]
        return user in config.OWNER_IDS

    def _get_chat_mode(self, chat_jid: str) -> str:
        """Determine the processing tier for a chat.

        owner_dm   — DMs where chat JID matches an OWNER_ID
        respond    — registered group, full processing + respond to @Molly
        listen     — monitored group, embed + selective graph, no responses
        store_only — everything else, embed only
        """
        # Owner DMs (self-chat): always full processing
        if chat_jid.split("@")[0] in config.OWNER_IDS:
            return "owner_dm"

        # Check registered chats
        chat_info = self.registered_chats.get(chat_jid)
        if chat_info:
            return chat_info.get("mode", "respond")

        return "store_only"

    # --- WhatsApp callback (runs on neonize thread) ---

    def _on_whatsapp_message(self, msg_data: dict):
        """Thread-safe bridge: push message to the async queue."""
        if self.loop:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, msg_data)

    def _track_send(self, msg_id: str | None):
        """Record a message ID that Molly sent so we skip it on echo."""
        if msg_id:
            self._sent_ids.add(msg_id)

    # --- Message processing ---

    async def process_message(self, msg_data: dict):
        chat_jid = msg_data["chat_jid"]
        content = msg_data["content"]
        sender_jid = msg_data["sender_jid"]
        sender_name = msg_data.get("sender_name", "Unknown")
        group_name = self.registered_chats.get(chat_jid, {}).get("name", chat_jid.split("@")[0])

        # Store ALL messages in SQLite — every group, every DM
        self.db.store_message(
            msg_id=msg_data["msg_id"],
            chat_jid=chat_jid,
            sender=sender_jid,
            sender_name=sender_name,
            content=content,
            timestamp=msg_data["timestamp"],
            is_from_me=msg_data["is_from_me"],
        )

        # Skip messages Molly sent (avoid echo loop)
        msg_id = msg_data["msg_id"]
        if msg_id in self._sent_ids:
            self._sent_ids.discard(msg_id)
            return

        # Determine chat processing tier
        chat_mode = self._get_chat_mode(chat_jid)
        is_owner = self._is_owner(sender_jid)

        # Non-owner messages: passive processing only, never respond
        if not is_owner:
            if content.strip():
                asyncio.create_task(
                    self._process_passive(
                        content, chat_jid, chat_mode, sender_jid,
                        sender_name=sender_name, group_name=group_name,
                    )
                )
            return

        # --- Owner messages from here ---

        # Check if this is a yes/no reply to a pending approval
        if self.approvals.try_resolve(content.strip(), chat_jid):
            return

        # Check for @Molly trigger
        has_trigger = config.TRIGGER_PATTERN.search(content)
        clean_content = (
            config.TRIGGER_PATTERN.sub("", content).strip()
            if has_trigger else content.strip()
        )

        # Commands: always available from owner with @Molly, any chat
        if has_trigger and clean_content:
            first_word = clean_content.split()[0]
            if first_word in config.COMMANDS:
                response = await handle_command(clean_content, chat_jid, self)
                if response:
                    self._track_send(self.wa.send_message(chat_jid, response))
                return

        # Determine if Molly should respond
        should_respond = (
            chat_mode == "owner_dm"  # Self-chat: always respond
            or (chat_mode == "respond" and has_trigger)  # Registered group + @Molly
        )

        if not should_respond or not clean_content:
            # Owner message but not triggering a response — passive processing
            if content.strip():
                asyncio.create_task(
                    self._process_passive(
                        content, chat_jid, chat_mode, sender_jid,
                        sender_name=sender_name, group_name=group_name,
                    )
                )
            return

        # --- Molly is going to respond ---
        self.wa.send_typing(chat_jid)

        try:
            session_id = self.sessions.get(chat_jid)
            response, new_session_id = await handle_message(
                clean_content, chat_jid, session_id,
                approval_manager=self.approvals,
                molly_instance=self,
            )

            if new_session_id:
                self.sessions[chat_jid] = new_session_id
                self.save_sessions()

            if not response:
                return

            # Check if the response contains a prompt-level approval tag
            # (belt-and-suspenders fallback — code-enforced can_use_tool is primary)
            tag = self.approvals.find_approval_tag(response)
            if tag:
                category, description = tag
                visible = self.approvals.strip_approval_tag(response)

                if visible:
                    self._track_send(self.wa.send_message(chat_jid, visible))

                asyncio.create_task(
                    self._approval_flow(category, description, chat_jid, new_session_id)
                )
            else:
                self._track_send(self.wa.send_message(chat_jid, response))

        except Exception:
            log.error("Error processing message in %s", chat_jid, exc_info=True)
            self._track_send(
                self.wa.send_message(
                    chat_jid, "Something went wrong on my end. Try again in a moment."
                )
            )
        finally:
            self.wa.send_typing_stopped(chat_jid)

    # --- Passive processing (all messages Molly doesn't respond to) ---

    async def _process_passive(
        self, content: str, chat_jid: str, chat_mode: str, sender_jid: str,
        sender_name: str = "Unknown", group_name: str = "Unknown",
    ):
        """Background processing for messages Molly doesn't respond to.

        owner_dm / respond: always full processing (embed + graph)
        listen / store_only: run through triage model first
          - urgent: notify Brian via WhatsApp + full extraction
          - relevant: full extraction (embed + graph)
          - background: embed only
          - noise: skip (already stored in SQLite)
        """
        from memory.processor import embed_and_store, extract_to_graph

        # DMs and respond groups: always full processing, skip triage
        if chat_mode in ("owner_dm", "respond"):
            await embed_and_store(content, chat_jid)
            await extract_to_graph(content, chat_jid)
            return

        # listen / store_only: triage first
        from memory.triage import triage_message

        result = await triage_message(content, sender_name, group_name)

        if result is None:
            # Triage unavailable — fall back to old behavior
            await embed_and_store(content, chat_jid)
            if chat_mode == "listen":
                await self._selective_extract(content, chat_jid)
            return

        log.debug(
            "Triage [%s] %s (%.2f): %s",
            chat_mode, result.classification, result.score, result.reason,
        )

        if result.classification == "urgent":
            # Notify Brian + full extraction
            await embed_and_store(content, chat_jid)
            await extract_to_graph(content, chat_jid)
            # Send notification to Brian's DM
            owner_jid = self._get_owner_dm_jid()
            if owner_jid and self.wa:
                preview = content[:200] + "..." if len(content) > 200 else content
                notify_msg = (
                    f"Flagged message in {group_name}\n"
                    f"From: {sender_name}\n"
                    f"{preview}\n\n"
                    f"Reason: {result.reason}"
                )
                self._track_send(self.wa.send_message(owner_jid, notify_msg))

        elif result.classification == "relevant":
            await embed_and_store(content, chat_jid)
            await extract_to_graph(content, chat_jid)

        elif result.classification == "background":
            await embed_and_store(content, chat_jid)

        # noise: already stored in SQLite, skip embed and graph

    def _get_owner_dm_jid(self) -> str | None:
        """Find the owner's DM JID for sending notifications."""
        for jid in self.registered_chats:
            if jid.split("@")[0] in config.OWNER_IDS:
                return jid
        # Fallback: construct from first owner ID
        for owner_id in config.OWNER_IDS:
            return f"{owner_id}@s.whatsapp.net"
        return None

    async def _selective_extract(self, content: str, chat_jid: str):
        """L3: Selective extraction for listen-only chats.

        Only runs full extraction if the message mentions entities
        already tracked in the knowledge graph.
        """
        try:
            from memory.extractor import extract_entities
            from memory.graph import find_matching_entity
            from memory.processor import extract_to_graph

            entities = extract_entities(content)
            if not entities:
                return

            for ent in entities:
                if find_matching_entity(ent["text"], ent["label"]):
                    await extract_to_graph(content, chat_jid)
                    return
        except Exception:
            log.error("Selective extraction failed for %s", chat_jid, exc_info=True)

    # --- Approval flow ---

    async def _approval_flow(
        self, category: str, description: str, chat_jid: str, session_id: str | None,
    ):
        """Background task: wait for approval, then resume the agent session."""
        approved = await self.approvals.request(
            category, description, chat_jid, session_id, self,
        )

        if approved:
            self.wa.send_typing(chat_jid)
            try:
                response, new_session_id = await handle_message(
                    f"Approved. Proceed with: {description}",
                    chat_jid,
                    session_id,
                )
                if new_session_id:
                    self.sessions[chat_jid] = new_session_id
                    self.save_sessions()
                if response:
                    self._track_send(self.wa.send_message(chat_jid, response))
            except Exception:
                log.error("Post-approval execution failed in %s", chat_jid, exc_info=True)
                self._track_send(
                    self.wa.send_message(chat_jid, "Something went wrong executing that action.")
                )
            finally:
                self.wa.send_typing_stopped(chat_jid)
        else:
            self._track_send(
                self.wa.send_message(chat_jid, "Got it, I won't do that.")
            )

    # --- Safe message processing wrapper ---

    async def _safe_process(self, msg_data: dict):
        """Wrapper around process_message that catches and logs exceptions."""
        try:
            await self.process_message(msg_data)
        except Exception:
            log.error("Unhandled error processing message", exc_info=True)

    # --- Main loop ---

    async def run(self):
        self.loop = asyncio.get_running_loop()
        self.start_time = datetime.now()

        # Load persisted state
        self.load_state()

        # Initialize database
        self.db.initialize()

        # Create WhatsApp client
        self.wa = WhatsAppClient(
            config.AUTH_DIR, message_callback=self._on_whatsapp_message
        )

        # Start WhatsApp in background thread (neonize is synchronous)
        wa_thread = threading.Thread(target=self.wa.connect, daemon=True, name="whatsapp")
        wa_thread.start()

        log.info("Molly is starting up. Waiting for WhatsApp connection...")
        log.info("Scan the QR code with your phone to pair.")

        # Start web UI server (Phase 4)
        try:
            import uvicorn
            from web import create_app

            if not config.WEB_AUTH_TOKEN:
                log.warning(
                    "MOLLY_WEB_TOKEN is not set — Web UI has no authentication. "
                    "Set MOLLY_WEB_TOKEN env var for security."
                )
            if config.WEB_HOST == "0.0.0.0" and not config.WEB_AUTH_TOKEN:
                log.warning(
                    "Web UI binding to 0.0.0.0 without auth token — "
                    "anyone on the network can access Molly"
                )

            web_app = create_app(self)
            web_config = uvicorn.Config(
                web_app,
                host=config.WEB_HOST,
                port=config.WEB_PORT,
                log_level="warning",
                ws_max_size=8192,
            )
            web_server = uvicorn.Server(web_config)
            asyncio.create_task(web_server.serve())
            log.info("Web UI started at http://%s:%d", config.WEB_HOST, config.WEB_PORT)
        except ImportError:
            log.warning("uvicorn/fastapi not installed — Web UI disabled")

        # Main processing loop — uses create_task so the queue keeps draining
        # while can_use_tool awaits approval responses from WhatsApp
        while self.running:
            try:
                msg_data = await asyncio.wait_for(
                    self.queue.get(), timeout=config.POLL_INTERVAL
                )
                asyncio.create_task(self._safe_process(msg_data))
            except asyncio.TimeoutError:
                # No message in queue — check scheduled tasks
                if self.wa and self.wa.connected:
                    if should_heartbeat(self.last_heartbeat):
                        self.last_heartbeat = datetime.now()
                        await run_heartbeat(self)
                    if should_run_maintenance(self.last_maintenance):
                        self.last_maintenance = datetime.now()
                        await run_maintenance()
            except asyncio.CancelledError:
                break
            except Exception:
                log.error("Error in main loop", exc_info=True)

        # Cleanup
        self.db.close()
        try:
            from memory.graph import close as close_graph
            close_graph()
        except Exception:
            pass
        log.info("Molly shut down.")

    def shutdown(self, *_args):
        log.info("Shutdown signal received...")
        self.running = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    preflight_checks()

    molly = Molly()

    # Handle graceful shutdown
    signal.signal(signal.SIGINT, molly.shutdown)
    signal.signal(signal.SIGTERM, molly.shutdown)

    try:
        asyncio.run(molly.run())
    except KeyboardInterrupt:
        log.info("Interrupted.")


if __name__ == "__main__":
    main()
