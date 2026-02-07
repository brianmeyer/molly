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


def preflight_checks():
    """Run all service checks before Molly starts."""
    log.info("Running preflight checks...")
    ensure_docker()
    ensure_neo4j()
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

    def _auto_register(self, chat_jid: str, sender_name: str):
        """Register a new chat and persist it."""
        slug = re.sub(r"[^a-z0-9]+", "-", (sender_name or chat_jid.split("@")[0]).lower()).strip("-")
        self.registered_chats[chat_jid] = {
            "name": sender_name or chat_jid.split("@")[0],
            "chat_id": slug,
        }
        self.save_registered_chats()
        log.info("Auto-registered chat: %s as '%s'", chat_jid, slug)

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
        is_from_me = msg_data["is_from_me"]
        is_group = msg_data["is_group"]
        sender_name = msg_data["sender_name"]

        # Store every message in the database
        self.db.store_message(
            msg_id=msg_data["msg_id"],
            chat_jid=chat_jid,
            sender=msg_data["sender_jid"],
            sender_name=sender_name,
            content=content,
            timestamp=msg_data["timestamp"],
            is_from_me=is_from_me,
        )

        # Skip messages Molly sent (avoid echo loop)
        msg_id = msg_data["msg_id"]
        if msg_id in self._sent_ids:
            self._sent_ids.discard(msg_id)
            return

        # Only the owner can trigger Molly
        if not self._is_owner(msg_data["sender_jid"]):
            return

        # Check if this is a yes/no reply to a pending approval.
        # This runs before trigger checks so a bare "yes" is accepted.
        if self.approvals.try_resolve(content.strip(), chat_jid):
            return

        # Self-chat: no @Molly needed. All other chats: require it.
        is_self_chat = chat_jid.split("@")[0] in config.OWNER_IDS
        has_trigger = config.TRIGGER_PATTERN.search(content)

        if not is_self_chat and not has_trigger:
            return

        # Auto-register unknown chats when the owner @Molly's from them
        if chat_jid not in self.registered_chats:
            if not has_trigger:
                return  # first contact with a new chat still requires @Molly
            self._auto_register(chat_jid, sender_name)

        # Strip the @Molly prefix if present
        clean_content = config.TRIGGER_PATTERN.sub("", content).strip() if has_trigger else content.strip()
        if not clean_content:
            return

        # Handle slash commands
        first_word = clean_content.split()[0]
        if first_word in config.COMMANDS:
            response = await handle_command(clean_content, chat_jid, self)
            if response:
                self._track_send(self.wa.send_message(chat_jid, response))
            return

        # Send typing indicator
        self.wa.send_typing(chat_jid)

        try:
            session_id = self.sessions.get(chat_jid)
            response, new_session_id = await handle_message(
                clean_content, chat_jid, session_id
            )

            if new_session_id:
                self.sessions[chat_jid] = new_session_id
                self.save_sessions()

            if not response:
                return

            # Check if the response contains an approval request
            tag = self.approvals.find_approval_tag(response)
            if tag:
                category, description = tag
                visible = self.approvals.strip_approval_tag(response)

                # Send the visible part of the response (Molly's explanation)
                if visible:
                    self._track_send(self.wa.send_message(chat_jid, visible))

                # Run the approval flow as a background task so the main
                # loop keeps processing messages (including the yes/no reply)
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

        # Main processing loop
        while self.running:
            try:
                msg_data = await asyncio.wait_for(
                    self.queue.get(), timeout=config.POLL_INTERVAL
                )
                await self.process_message(msg_data)
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
