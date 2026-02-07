import asyncio
import json
import logging
import re
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path

import config
from agent import handle_message
from commands import handle_command
from database import Database
from heartbeat import run_heartbeat, should_heartbeat
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
        self._sent_ids: set[str] = set()  # message IDs Molly sent (avoid echo loop)

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

            if response:
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
                # No message in queue — check if heartbeat is due
                if self.wa and self.wa.connected:
                    if should_heartbeat(self.last_heartbeat):
                        self.last_heartbeat = datetime.now()
                        await run_heartbeat(self)
            except asyncio.CancelledError:
                break
            except Exception:
                log.error("Error in main loop", exc_info=True)

        # Cleanup
        self.db.close()
        log.info("Molly shut down.")

    def shutdown(self, *_args):
        log.info("Shutdown signal received...")
        self.running = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
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
