import asyncio
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass

import fcntl
import importlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import config
from agent import handle_message
from automations import AutomationEngine
from approval import ApprovalManager
from commands import handle_command
from database import Database
from heartbeat import run_heartbeat, should_heartbeat
from maintenance import run_maintenance, should_run_maintenance
from self_improve import SelfImprovementEngine
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

SURFACED_SIGNAL_WINDOW_SECONDS = 6 * 60 * 60
MAX_RECENT_SURFACED_ITEMS = 100
LAST_RESPONSE_TTL_SECONDS = 300  # 5 minutes
AUTO_CREATE_UNDO_TTL_SECONDS = 24 * 60 * 60
AUTO_CREATE_UNDO_MAX_ENTRIES = 200

_CORRECTION_KEYWORDS = re.compile(
    r"\b(?:no[,.]|that'?s wrong|that is wrong|actually|not (?:that|what)|"
    r"i said|i meant|it'?s actually|wrong|incorrect|that'?s not right|"
    r"you got .{1,20} wrong|that'?s not what)\b",
    re.IGNORECASE,
)


_TOOL_DEPENDENCY_SPECS = [
    {
        "server": "google-calendar",
        "tools": {
            "calendar_list", "calendar_get", "calendar_search",
            "calendar_create", "calendar_update", "calendar_delete",
        },
        "requirements": [
            ("claude_agent_sdk", "claude-agent-sdk"),
            ("google.auth.transport.requests", "google-auth"),
            ("google.oauth2.credentials", "google-auth"),
            ("google_auth_oauthlib.flow", "google-auth-oauthlib"),
            ("googleapiclient.discovery", "google-api-python-client"),
        ],
    },
    {
        "server": "gmail",
        "tools": {
            "gmail_search", "gmail_read", "gmail_draft", "gmail_send", "gmail_reply",
        },
        "requirements": [
            ("claude_agent_sdk", "claude-agent-sdk"),
            ("google.auth.transport.requests", "google-auth"),
            ("google.oauth2.credentials", "google-auth"),
            ("google_auth_oauthlib.flow", "google-auth-oauthlib"),
            ("googleapiclient.discovery", "google-api-python-client"),
        ],
    },
    {
        "server": "google-people",
        "tools": {"people_search", "people_get", "people_list"},
        "requirements": [
            ("claude_agent_sdk", "claude-agent-sdk"),
            ("google.auth.transport.requests", "google-auth"),
            ("google.oauth2.credentials", "google-auth"),
            ("google_auth_oauthlib.flow", "google-auth-oauthlib"),
            ("googleapiclient.discovery", "google-api-python-client"),
        ],
    },
    {
        "server": "google-tasks",
        "tools": {
            "tasks_list", "tasks_list_tasks", "tasks_create",
            "tasks_complete", "tasks_delete",
        },
        "requirements": [
            ("claude_agent_sdk", "claude-agent-sdk"),
            ("google.auth.transport.requests", "google-auth"),
            ("google.oauth2.credentials", "google-auth"),
            ("google_auth_oauthlib.flow", "google-auth-oauthlib"),
            ("googleapiclient.discovery", "google-api-python-client"),
        ],
    },
    {
        "server": "google-drive",
        "tools": {"drive_search", "drive_get", "drive_read"},
        "requirements": [
            ("claude_agent_sdk", "claude-agent-sdk"),
            ("google.auth.transport.requests", "google-auth"),
            ("google.oauth2.credentials", "google-auth"),
            ("google_auth_oauthlib.flow", "google-auth-oauthlib"),
            ("googleapiclient.discovery", "google-api-python-client"),
        ],
    },
    {
        "server": "google-meet",
        "tools": {"meet_list", "meet_get", "meet_transcripts", "meet_recordings"},
        "requirements": [
            ("claude_agent_sdk", "claude-agent-sdk"),
            ("google.auth.transport.requests", "google-auth"),
            ("google.oauth2.credentials", "google-auth"),
            ("google_auth_oauthlib.flow", "google-auth-oauthlib"),
            ("googleapiclient.discovery", "google-api-python-client"),
        ],
    },
    {
        "server": "apple-mcp",
        "tools": {"contacts", "notes", "messages", "mail", "reminders", "calendar", "maps"},
        "requirements": [
            ("claude_agent_sdk", "claude-agent-sdk"),
        ],
        "commands": ["bunx"],
    },
    {
        "server": "imessage",
        "tools": {"imessage_search", "imessage_recent", "imessage_thread", "imessage_unread"},
        "requirements": [
            ("claude_agent_sdk", "claude-agent-sdk"),
        ],
    },
    {
        "server": "whatsapp-history",
        "tools": {"whatsapp_search"},
        "requirements": [
            ("claude_agent_sdk", "claude-agent-sdk"),
        ],
    },
    {
        "server": "kimi",
        "tools": {"kimi_research"},
        "requirements": [
            ("claude_agent_sdk", "claude-agent-sdk"),
            ("httpx", "httpx"),
        ],
    },
    {
        "server": "grok",
        "tools": {"grok_reason"},
        "requirements": [
            ("claude_agent_sdk", "claude-agent-sdk"),
            ("httpx", "httpx"),
            ("xai_sdk", "xai-sdk"),
        ],
    },
]


def _task_done_callback(task: asyncio.Task):
    """Log exceptions from fire-and-forget tasks instead of swallowing them."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        log.error("Background task %s failed: %s", task.get_name(), exc, exc_info=exc)


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


def ensure_triage_model():
    """Preload the local GGUF triage model.

    Non-blocking on failure — triage is optional, Molly works without it.
    """
    from memory.triage import preload_model

    model_path = config.TRIAGE_MODEL_PATH.expanduser()
    if not model_path.exists():
        log.warning("Triage model file not found: %s", model_path)
        return

    try:
        if preload_model():
            log.info("Triage model ready: %s", model_path)
        else:
            log.warning("Triage model failed to load — triage will be unavailable")
    except Exception:
        log.warning("Triage model preload failed — triage will be unavailable", exc_info=True)


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


def ensure_tool_dependencies():
    """Verify MCP tool dependencies are importable; disable tools if missing."""
    disabled_servers: set[str] = set()
    disabled_tools: set[str] = set()

    for spec in _TOOL_DEPENDENCY_SPECS:
        missing_packages: set[str] = set()
        missing_commands: set[str] = set()

        for module_name, package_name in spec.get("requirements", []):
            try:
                importlib.import_module(module_name)
            except Exception:
                missing_packages.add(package_name)

        for command_name in spec.get("commands", []):
            if shutil.which(command_name) is None:
                missing_commands.add(command_name)

        if missing_packages or missing_commands:
            for package_name in sorted(missing_packages):
                log.warning(
                    "Missing dependency package '%s'; disabling MCP server '%s'.",
                    package_name,
                    spec["server"],
                )
            for command_name in sorted(missing_commands):
                log.warning(
                    "Missing command '%s'; disabling MCP server '%s'.",
                    command_name,
                    spec["server"],
                )
            disabled_servers.add(spec["server"])
            disabled_tools.update(spec["tools"])

    config.DISABLED_MCP_SERVERS = disabled_servers
    config.DISABLED_TOOL_NAMES = disabled_tools

    if disabled_servers:
        log.warning(
            "Disabled MCP servers at startup: %s",
            ", ".join(sorted(disabled_servers)),
        )


def _kill_stale_port(port: int):
    """Kill any leftover process binding the given port (e.g. after a crash)."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        my_pid = str(os.getpid())
        for pid in pids:
            if pid and pid != my_pid:
                log.info("Killing stale process %s on port %d", pid, port)
                os.kill(int(pid), signal.SIGTERM)
        if pids:
            time.sleep(0.3)  # give it a moment to release the port
    except Exception:
        log.debug("Could not check for stale port %d holders", port, exc_info=True)


def _acquire_instance_lock() -> int | None:
    """Prevent multiple Molly processes from sharing one WhatsApp session."""
    lock_path = config.STORE_DIR / ".molly.instance.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None

    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def _release_instance_lock(fd: int):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except Exception:
        log.debug("Failed to unlock Molly instance lock", exc_info=True)
    try:
        os.close(fd)
    except Exception:
        log.debug("Failed to close Molly instance lock fd", exc_info=True)


def _clear_pycache(*package_names: str):
    """Remove __pycache__ dirs for given packages to recover from stale bytecode."""
    site_packages = Path(sys.executable).parent.parent / "lib"
    for sp in site_packages.rglob("site-packages"):
        for pkg in package_names:
            pkg_dir = sp / pkg
            if pkg_dir.is_dir():
                for cache_dir in pkg_dir.rglob("__pycache__"):
                    shutil.rmtree(cache_dir, ignore_errors=True)
                # Also invalidate any top-level .pyc
                for pyc in pkg_dir.glob("*.pyc"):
                    pyc.unlink(missing_ok=True)
    # Force re-import on next attempt
    for name in list(sys.modules):
        if any(name == pkg or name.startswith(pkg + ".") for pkg in package_names):
            del sys.modules[name]


def prewarm_ml_models():
    """Prewarm ML models so first message doesn't eat a cold-start penalty.

    Loads embedding (EmbeddingGemma-300M) and GLiNER2 (DeBERTa-v3-large)
    in parallel background threads. Non-blocking on failure — both degrade
    gracefully to lazy loading if this fails.

    Cold start without prewarming: ~18 seconds on first message.
    With prewarming: models are already resident when first message arrives.
    """
    import concurrent.futures

    def _load_embedding():
        try:
            from memory.embeddings import _get_model
            _get_model()
            log.info("Embedding model prewarmed (EmbeddingGemma-300M)")
        except Exception:
            log.warning("Embedding model prewarm failed — will lazy-load on first use", exc_info=True)

    def _load_gliner():
        for attempt in range(2):
            try:
                from memory.extractor import _get_model
                _get_model()
                log.info("GLiNER2 model prewarmed")
                return
            except ImportError:
                if attempt == 0:
                    log.warning("GLiNER2 import failed — clearing bytecode cache and retrying")
                    _clear_pycache("transformers", "gliner", "gliner2")
                else:
                    log.error("GLiNER2 import failed after cache clear — entity extraction will be unavailable", exc_info=True)
            except Exception:
                log.error("GLiNER2 model prewarm failed — entity extraction may be degraded", exc_info=True)
                return

    log.info("Prewarming ML models (embedding + GLiNER2)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="prewarm") as pool:
        futures = [pool.submit(_load_embedding), pool.submit(_load_gliner)]
        concurrent.futures.wait(futures, timeout=120)
    log.info("ML model prewarm complete")


def preflight_checks():
    """Run all service checks before Molly starts."""
    log.info("Running preflight checks...")
    ensure_docker()
    ensure_neo4j()
    ensure_triage_model()
    ensure_tool_dependencies()
    ensure_google_auth()
    prewarm_ml_models()
    try:
        from health import get_health_doctor

        doctor = get_health_doctor()
        doctor.run_abbreviated_preflight()
        log.info("Health preflight: completed")
    except Exception:
        log.warning("Health preflight check failed", exc_info=True)
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
        self._sent_ids: dict[str, float] = {}  # msg_id → timestamp (avoid echo loop)
        self._last_responses: dict[str, tuple[str, float]] = {}  # chat_jid → (response_text, timestamp)
        self._recent_surfaces: list[dict] = []  # recent surfaced notifications for feedback linking
        self._auto_create_undo_map: dict[str, dict] = {}  # notification msg_id -> created resource metadata
        self.approvals = ApprovalManager()
        self.automations = AutomationEngine(self)
        self.self_improvement = SelfImprovementEngine(self)
        self._automation_tick_task: asyncio.Task | None = None
        self._self_improve_tick_task: asyncio.Task | None = None
        self._imessage_mention_task: asyncio.Task | None = None
        self._last_imessage_mention_check: datetime | None = None
        self.exit_code = 0
        self._bg_semaphore = asyncio.Semaphore(12)  # cap concurrent background tasks
        self._bg_tasks: set[asyncio.Task] = set()  # prevent GC before completion

    # --- Background task management ---

    def _spawn_bg(self, coro, *, name: str = "") -> asyncio.Task:
        """Spawn a background task gated by the concurrency semaphore."""
        async def _wrapper():
            async with self._bg_semaphore:
                return await coro
        task = asyncio.create_task(_wrapper(), name=name)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        task.add_done_callback(_task_done_callback)
        return task

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

    @staticmethod
    def _normalize_jid_user(jid: str) -> str:
        """Extract stable user identifier from JIDs (strip @server and :device)."""
        user = (jid or "").split("@", 1)[0]
        return user.split(":", 1)[0]

    def _is_owner(self, sender_jid: str) -> bool:
        """Check if the sender is Brian (by phone or LID)."""
        user = self._normalize_jid_user(sender_jid)
        return user in config.OWNER_IDS

    def _get_chat_mode(self, chat_jid: str) -> str:
        """Determine the processing tier for a chat.

        Returns one of config.CHAT_MODES:
        owner_dm   — DMs where chat JID matches an OWNER_ID
        respond    — registered group, full processing + respond to @Molly
        listen     — monitored group, embed + selective graph, no responses
        store_only — everything else, embed only
        """
        # Owner DMs (self-chat): always full processing
        if self._normalize_jid_user(chat_jid) in config.OWNER_IDS:
            return "owner_dm"

        # Check registered chats
        chat_info = self.registered_chats.get(chat_jid)
        if chat_info:
            mode = chat_info.get("mode", "respond")
            if mode not in config.CHAT_MODES:
                log.warning("Unknown chat mode '%s' for %s, falling back to store_only", mode, chat_jid)
                return "store_only"
            return mode

        return "store_only"

    # --- WhatsApp callback (runs on neonize thread) ---

    def _on_whatsapp_message(self, msg_data: dict):
        """Thread-safe bridge: push message to the async queue."""
        if self.loop:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, msg_data)

    def _track_send(self, msg_id: str | list[str] | None):
        """Record message IDs Molly sent so we skip them on echo."""
        msg_ids = self._coerce_message_ids(msg_id)
        if not msg_ids:
            return

        now = time.time()
        for mid in msg_ids:
            self._sent_ids[mid] = now

        # Prune entries older than 5 minutes
        cutoff = now - 300
        self._sent_ids = {k: v for k, v in self._sent_ids.items() if v > cutoff}

    @staticmethod
    def _coerce_message_ids(msg_id: str | list[str] | None) -> list[str]:
        if isinstance(msg_id, str):
            msg = msg_id.strip()
            return [msg] if msg else []
        if isinstance(msg_id, list):
            return [str(m).strip() for m in msg_id if str(m).strip()]
        return []

    @staticmethod
    def _normalize_signal_source(source: str) -> str:
        src = (source or "").strip().lower()
        if src in {"email", "imessage", "calendar"}:
            return src
        return "calendar"

    def _prune_recent_surfaces(self, now_ts: float):
        cutoff = now_ts - SURFACED_SIGNAL_WINDOW_SECONDS
        self._recent_surfaces = [
            item for item in self._recent_surfaces
            if item.get("surfaced_ts", 0) >= cutoff and not item.get("logged", False)
        ][-MAX_RECENT_SURFACED_ITEMS:]

    def _record_surfaced_item(
        self,
        chat_jid: str,
        source: str,
        surfaced_summary: str,
        sender_pattern: str,
    ):
        now_ts = time.time()
        self._prune_recent_surfaces(now_ts)
        self._recent_surfaces.append(
            {
                "chat_jid": chat_jid,
                "source": self._normalize_signal_source(source),
                "summary": (surfaced_summary or "").strip()[:1000],
                "sender_pattern": (sender_pattern or "").strip()[:500],
                "surfaced_ts": now_ts,
                "logged": False,
            }
        )

    def send_surface_message(
        self,
        chat_jid: str,
        text: str,
        source: str,
        surfaced_summary: str = "",
        sender_pattern: str = "",
    ):
        """Send a surfaced notification and cache metadata for preference learning."""
        if not self.wa:
            return
        msg_id = self.wa.send_message(chat_jid, text)
        self._track_send(msg_id)
        if msg_id:
            self._record_surfaced_item(
                chat_jid=chat_jid,
                source=source,
                surfaced_summary=surfaced_summary or text,
                sender_pattern=sender_pattern,
            )

    @staticmethod
    def _decode_tool_response_payload(tool_response) -> dict:
        if not isinstance(tool_response, dict):
            return {}
        if tool_response.get("is_error"):
            return {}

        content = tool_response.get("content")
        if not isinstance(content, list):
            return {}

        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "text":
                continue
            text = str(part.get("text", "")).strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _format_auto_created_when(raw_when) -> tuple[str, str]:
        if not raw_when:
            return "unspecified date", "unspecified time"

        value = str(raw_when).strip()
        if not value:
            return "unspecified date", "unspecified time"

        if len(value) == 10 and value.count("-") == 2:
            return value, "all day"

        normalized = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is not None:
                dt = dt.astimezone(ZoneInfo(config.TIMEZONE))
            date_text = dt.strftime("%Y-%m-%d")
            time_text = dt.strftime("%I:%M %p").lstrip("0")
            return date_text, time_text
        except Exception:
            if "T" in value:
                date_part, _, time_part = value.partition("T")
                time_part = time_part.split("+", 1)[0].replace("Z", "")
                if date_part:
                    return date_part, time_part or "unspecified time"
            return value, "unspecified time"

    @staticmethod
    def _extract_tool_response_text(tool_response) -> str:
        if not isinstance(tool_response, dict):
            return ""
        content = tool_response.get("content")
        if not isinstance(content, list):
            return ""
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "text":
                continue
            text = str(part.get("text", "")).strip()
            if text:
                return text
        return ""

    def _prune_auto_create_undo_map(self):
        now = time.time()
        cutoff = now - AUTO_CREATE_UNDO_TTL_SECONDS

        fresh: dict[str, dict] = {}
        for notification_id, entry in self._auto_create_undo_map.items():
            if not isinstance(entry, dict):
                continue
            created_ts = float(entry.get("created_ts", 0.0) or 0.0)
            if created_ts and created_ts < cutoff:
                continue
            fresh[notification_id] = entry

        if len(fresh) > AUTO_CREATE_UNDO_MAX_ENTRIES:
            ordered = sorted(
                fresh.items(),
                key=lambda item: float(item[1].get("created_ts", 0.0) or 0.0),
            )
            drop_count = len(fresh) - AUTO_CREATE_UNDO_MAX_ENTRIES
            for notification_id, _ in ordered[:drop_count]:
                fresh.pop(notification_id, None)

        self._auto_create_undo_map = fresh

    def _record_auto_create_undo_entry(
        self,
        notification_id: str,
        chat_jid: str,
        *,
        resource_type: str,
        resource_id: str,
        title: str,
        tasklist_id: str | None = None,
    ):
        notification_id = (notification_id or "").strip()
        resource_id = (resource_id or "").strip()
        if not notification_id or not resource_id:
            return

        self._auto_create_undo_map[notification_id] = {
            "chat_jid": (chat_jid or "").strip(),
            "resource_type": resource_type,
            "resource_id": resource_id,
            "tasklist_id": (tasklist_id or "").strip(),
            "title": (title or "").strip(),
            "created_ts": time.time(),
        }
        self._prune_auto_create_undo_map()

    @staticmethod
    def _extract_undo_request(text: str) -> str | None:
        content = (text or "").strip()
        lowered = content.lower()
        if lowered == "undo":
            return ""
        if lowered.startswith("undo "):
            return content.split(None, 1)[1].strip()
        return None

    def _pop_auto_create_undo_entry(
        self,
        chat_jid: str,
        requested_notification_id: str | None,
    ) -> tuple[str | None, dict | None]:
        chat = (chat_jid or "").strip()
        requested = (requested_notification_id or "").strip()

        self._prune_auto_create_undo_map()

        if requested:
            entry = self._auto_create_undo_map.get(requested)
            if not isinstance(entry, dict):
                return None, None
            if entry.get("chat_jid") != chat:
                return None, None
            return requested, self._auto_create_undo_map.pop(requested)

        best_id = None
        best_entry = None
        best_ts = -1.0
        for notification_id, entry in self._auto_create_undo_map.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("chat_jid") != chat:
                continue
            ts = float(entry.get("created_ts", 0.0) or 0.0)
            if ts > best_ts:
                best_ts = ts
                best_id = notification_id
                best_entry = entry

        if not best_id or not isinstance(best_entry, dict):
            return None, None

        self._auto_create_undo_map.pop(best_id, None)
        return best_id, best_entry

    async def _undo_auto_created_entry(self, entry: dict) -> tuple[bool, str]:
        resource_type = str(entry.get("resource_type") or "").strip().lower()
        resource_id = str(entry.get("resource_id") or "").strip()
        if not resource_id:
            return False, "missing resource id"

        try:
            if resource_type == "calendar":
                from tools.calendar import calendar_delete

                result = await calendar_delete({"event_id": resource_id})
            elif resource_type == "task":
                from tools.google_tasks import tasks_delete

                args = {"task_id": resource_id}
                tasklist_id = str(entry.get("tasklist_id") or "").strip()
                if tasklist_id:
                    args["tasklist_id"] = tasklist_id
                result = await tasks_delete(args)
            else:
                return False, f"unsupported resource type: {resource_type}"
        except Exception as exc:
            log.error("Undo delete failed for %s:%s", resource_type, resource_id, exc_info=True)
            return False, str(exc)

        if isinstance(result, dict) and not result.get("is_error"):
            return True, ""

        detail = self._extract_tool_response_text(result)
        return False, detail or "delete failed"

    async def _maybe_handle_auto_create_undo(self, content: str, chat_jid: str) -> bool:
        requested = self._extract_undo_request(content)
        if requested is None:
            return False
        if not self.wa:
            return True

        requested_id = requested or None
        notification_id, entry = self._pop_auto_create_undo_entry(chat_jid, requested_id)
        if not entry:
            suffix = f" '{requested_id}'" if requested_id else ""
            self._track_send(
                self.wa.send_message(
                    chat_jid,
                    f"No recent auto-created item found for undo{suffix}.",
                )
            )
            return True

        success, detail = await self._undo_auto_created_entry(entry)
        if success:
            item_type = "event" if entry.get("resource_type") == "calendar" else "task"
            title = (entry.get("title") or "").strip() or entry.get("resource_id", "item")
            self._track_send(self.wa.send_message(chat_jid, f"Undid auto-created {item_type}: {title}."))
            return True

        if notification_id:
            self._auto_create_undo_map[notification_id] = entry
            self._prune_auto_create_undo_map()
        item_label = (entry.get("title") or "").strip() or entry.get("resource_id", "item")
        reason = f": {detail}" if detail else "."
        self._track_send(self.wa.send_message(chat_jid, f"Undo failed for {item_label}{reason}"))
        return True

    def notify_auto_created_tool_result(
        self,
        source_chat_jid: str,
        tool_name: str,
        tool_input: dict,
        tool_response,
    ):
        """Notify Brian when an AUTO-tier create action succeeds."""
        if tool_name not in {"calendar_create", "tasks_create"}:
            return
        if not self.wa:
            return
        if isinstance(tool_response, dict) and tool_response.get("is_error"):
            return

        payload = self._decode_tool_response_payload(tool_response)
        title = ""
        date_text = "unspecified date"
        time_text = "unspecified time"
        location = "no location"
        resource_type = "calendar" if tool_name == "calendar_create" else "task"
        resource_id = ""
        tasklist_id = ""

        if tool_name == "calendar_create":
            event = payload.get("event", {}) if isinstance(payload, dict) else {}
            if not isinstance(event, dict):
                event = {}
            title = str(event.get("summary") or tool_input.get("summary") or "Untitled event")
            raw_start = event.get("start") or tool_input.get("start")
            if isinstance(raw_start, dict):
                raw_start = raw_start.get("dateTime") or raw_start.get("date")
            date_text, time_text = self._format_auto_created_when(raw_start)
            location = str(event.get("location") or tool_input.get("location") or "no location")
            resource_id = str(event.get("id") or "").strip()
        else:
            created = payload.get("created", {}) if isinstance(payload, dict) else {}
            if not isinstance(created, dict):
                created = {}
            title = str(created.get("title") or tool_input.get("title") or "Untitled task")
            raw_due = created.get("due") or tool_input.get("due")
            date_text, time_text = self._format_auto_created_when(raw_due)
            location = str(tool_input.get("location") or "no location")
            resource_id = str(created.get("id") or "").strip()
            tasklist_id = str(tool_input.get("tasklist_id") or "@default").strip()

        if not resource_id:
            log.warning("Skipping auto-create notification without resource id for %s", tool_name)
            return

        title = " ".join(title.split()) or "Untitled item"
        location = " ".join(location.split()) or "no location"
        message = (
            f"Auto-created: {title} on {date_text} at {time_text} at {location}. "
            "Reply 'undo' to remove."
        )

        target_jid = self._get_owner_dm_jid() or source_chat_jid
        notification_msg = self.wa.send_message(target_jid, message)
        self._track_send(notification_msg)

        for notification_id in self._coerce_message_ids(notification_msg):
            self._record_auto_create_undo_entry(
                notification_id,
                target_jid,
                resource_type=resource_type,
                resource_id=resource_id,
                title=title,
                tasklist_id=tasklist_id,
            )

    def _get_group_participant_names(self, chat_jid: str) -> list[str]:
        names: list[str] = []
        cached_members = self.registered_chats.get(chat_jid, {}).get("members", [])
        for member in cached_members:
            name = (member.get("name", "") or "").strip()
            if name:
                names.append(name)

        deduped = list(dict.fromkeys(names))
        return deduped

    def build_agent_chat_context(self, chat_jid: str, is_group: bool) -> str | None:
        """Return extra prompt context for group chats."""
        if not is_group:
            return None

        participants = self._get_group_participant_names(chat_jid)
        participant_names = ", ".join(participants[:20]) if participants else "group participants"
        return (
            f"You are responding in a group chat with [{participant_names}]. "
            "Reply here directly — do not attempt to message individuals separately."
        )

    async def _log_preference_signal_if_dismissive(self, chat_jid: str, owner_feedback: str):
        feedback = owner_feedback.strip()
        if not feedback:
            return

        # Find the most recent un-logged surfaced item for this chat first
        # (skip LLM call entirely if nothing to link feedback to)
        now_ts = time.time()
        self._prune_recent_surfaces(now_ts)

        surfaced = None
        for item in reversed(self._recent_surfaces):
            if item.get("chat_jid") == chat_jid and not item.get("logged", False):
                surfaced = item
                break

        if not surfaced:
            return

        surfaced_summary = surfaced.get("summary", "a proactive notification")
        surfaced_source = surfaced.get("source", "unknown")

        # LLM classification with full context about what was surfaced
        from memory.triage import classify_local_async

        prompt = (
            "You are analyzing a user's reply to a notification from their AI assistant.\n\n"
            f"The assistant surfaced this {surfaced_source} notification:\n"
            f"\"{surfaced_summary[:300]}\"\n\n"
            f"The user replied:\n"
            f"\"{feedback[:300]}\"\n\n"
            "Is the user expressing they do NOT want this type of notification? "
            "Look for: rejection, annoyance, asking to stop, saying it's not useful.\n"
            "A simple acknowledgement like 'ok' or 'thanks' is NOT dismissive.\n"
            "Respond YES or NO."
        )
        result = await classify_local_async(prompt, "")
        if not result.strip().upper().startswith("YES"):
            return

        sender_pattern = surfaced.get("sender_pattern", "")
        feedback_pattern = "feedback_pattern:llm_classified_dismissive"
        if sender_pattern:
            sender_pattern = f"{sender_pattern} | {feedback_pattern}"
        else:
            sender_pattern = feedback_pattern

        try:
            from memory.retriever import get_vectorstore
            vs = get_vectorstore()
            vs.log_preference_signal(
                source=surfaced.get("source", "calendar"),
                surfaced_summary=surfaced.get("summary", ""),
                sender_pattern=sender_pattern,
                owner_feedback=feedback,
            )
            surfaced["logged"] = True
            log.info("preference-signal: logged dismissive feedback (LLM classified)")
        except Exception:
            log.debug("Failed to log preference signal", exc_info=True)

    async def _detect_and_log_correction(self, chat_jid: str, owner_message: str):
        """Detect when the owner corrects Molly and log it for learning."""
        message = owner_message.strip()
        if not message:
            return

        # Prune stale last-response entries (>5 min old)
        now = time.time()
        cutoff = now - LAST_RESPONSE_TTL_SECONDS
        self._last_responses = {
            k: v for k, v in self._last_responses.items() if v[1] > cutoff
        }

        # Need a recent Molly response to compare against
        last = self._last_responses.get(chat_jid)
        if not last:
            return
        molly_response, _ts = last

        # Step 1: keyword heuristic — fast rejection
        match = _CORRECTION_KEYWORDS.search(message)
        if not match:
            return

        # Step 2: LLM confirmation — avoid false positives
        try:
            from memory.triage import classify_local_async

            prompt = (
                f"The assistant previously responded with: '{molly_response[:300]}'. "
                f"The user then said: '{message[:300]}'. "
                "Is the user correcting or contradicting the assistant's response? "
                "Respond YES or NO."
            )
            result = await classify_local_async(prompt, "")
            if not result.strip().upper().startswith("YES"):
                return
        except Exception:
            log.debug("Correction LLM classification failed", exc_info=True)
            return

        # Step 3: log the correction
        try:
            from memory.retriever import get_vectorstore
            vs = get_vectorstore()
            vs.log_correction(
                context=f"chat:{chat_jid}",
                molly_output=molly_response[:500],
                user_correction=message[:500],
                pattern=match.group(0),
            )
            log.info("Logged correction from owner in %s", chat_jid)
        except Exception:
            log.debug("Failed to log correction", exc_info=True)

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
            del self._sent_ids[msg_id]
            return

        # Phase 6: evaluate message-triggered automations in background
        self._spawn_bg(
            self.automations.on_message(msg_data),
            name=f"automation-msg:{chat_jid[:20]}",
        )

        # Determine chat processing tier
        chat_mode = self._get_chat_mode(chat_jid)
        is_owner = self._is_owner(sender_jid)

        # Non-owner messages: passive processing only, never respond
        if not is_owner:
            if content.strip():
                self._spawn_bg(
                    self._process_passive(
                        content, chat_jid, chat_mode, sender_jid,
                        sender_name=sender_name, group_name=group_name,
                    ),
                    name=f"passive:{chat_jid[:20]}",
                )
            return

        # --- Owner messages from here ---

        # Check if this is a yes/no reply to a pending approval
        if self.approvals.try_resolve(content.strip(), chat_jid):
            return

        # Owner safety net for AUTO-tier create tools (calendar/task)
        if await self._maybe_handle_auto_create_undo(content.strip(), chat_jid):
            return

        # Passive data collection only: log dismissive feedback to surfaced items
        self._spawn_bg(
            self._log_preference_signal_if_dismissive(chat_jid, content),
            name=f"dismissive-feedback:{chat_jid[:20]}",
        )

        # Passive data collection: detect corrections to Molly's responses
        self._spawn_bg(
            self._detect_and_log_correction(chat_jid, content),
            name=f"correction-detect:{chat_jid[:20]}",
        )

        # Check for @Molly trigger
        has_trigger = config.TRIGGER_PATTERN.search(content)
        clean_content = (
            config.TRIGGER_PATTERN.sub("", content).strip()
            if has_trigger else content.strip()
        )

        # Explicit owner request to capture a workflow as a skill.
        if (
            clean_content
            and self.self_improvement.should_trigger_owner_skill_phrase(clean_content)
            and (chat_mode == "owner_dm" or has_trigger)
        ):
            await self.self_improvement.propose_skill_from_owner_phrase(clean_content)
            return

        # Commands: available from owner with @Molly (any chat) or directly in DMs
        cmd_text = clean_content
        is_dm_command = (
            chat_mode == "owner_dm"
            and content.strip().startswith("/")
        )
        if is_dm_command:
            cmd_text = content.strip()

        if (has_trigger or is_dm_command) and cmd_text:
            first_word = cmd_text.split()[0]
            if first_word in config.COMMANDS:
                response = await handle_command(cmd_text, chat_jid, self)
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
                self._spawn_bg(
                    self._process_passive(
                        content, chat_jid, chat_mode, sender_jid,
                        sender_name=sender_name, group_name=group_name,
                    ),
                    name=f"passive-owner:{chat_jid[:20]}",
                )
            return

        # --- Molly is going to respond ---
        self.wa.send_typing(chat_jid)

        try:
            session_id = self.sessions.get(chat_jid)
            chat_context = self.build_agent_chat_context(
                chat_jid,
                bool(msg_data.get("is_group")),
            )
            response, new_session_id = await handle_message(
                clean_content, chat_jid, session_id,
                approval_manager=self.approvals,
                molly_instance=self,
                source="whatsapp",
                chat_context=chat_context,
            )

            if new_session_id:
                self.sessions[chat_jid] = new_session_id
                self.save_sessions()

            if not response:
                return

            # Track last response for correction detection
            self._last_responses[chat_jid] = (response, time.time())

            # Check if the response contains a prompt-level approval tag
            # (belt-and-suspenders fallback — code-enforced can_use_tool is primary)
            tag = self.approvals.find_approval_tag(response)
            if tag:
                category, description = tag
                visible = self.approvals.strip_approval_tag(response)

                if visible:
                    self._track_send(self.wa.send_message(chat_jid, visible))

                self._spawn_bg(
                    self._approval_flow(category, description, chat_jid, new_session_id),
                    name=f"approval:{chat_jid[:20]}",
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
            await asyncio.gather(
                embed_and_store(content, chat_jid),
                extract_to_graph(content, chat_jid),
            )
            return

        # listen / store_only: triage first
        from memory.triage import triage_message

        result = await triage_message(content, sender_name, group_name, chat_jid=chat_jid)

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
            await asyncio.gather(
                embed_and_store(content, chat_jid),
                extract_to_graph(content, chat_jid),
            )
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
                self.send_surface_message(
                    owner_jid,
                    notify_msg,
                    source="whatsapp",
                    surfaced_summary=f"{sender_name}: {preview}",
                    sender_pattern=f"group:{group_name};sender:{sender_name}",
                )

        elif result.classification == "relevant":
            await asyncio.gather(
                embed_and_store(content, chat_jid),
                extract_to_graph(content, chat_jid),
            )

        elif result.classification == "background":
            await embed_and_store(content, chat_jid)

        # noise: already stored in SQLite, skip embed and graph

    def _get_owner_dm_jid(self) -> str | None:
        """Find the owner's DM JID for sending notifications."""
        for jid in self.registered_chats:
            if self._normalize_jid_user(jid) in config.OWNER_IDS:
                return jid
        # Fallback: construct from first owner ID
        for owner_id in config.OWNER_IDS:
            return f"{owner_id}@s.whatsapp.net"
        return None

    def _should_check_imessage_mentions(self) -> bool:
        """Check if it's time to poll for @molly mentions in iMessages."""
        now = datetime.now()
        hour = now.hour
        if hour < config.ACTIVE_HOURS[0] or hour >= config.ACTIVE_HOURS[1]:
            return False
        if self._last_imessage_mention_check is None:
            return True
        elapsed = (now - self._last_imessage_mention_check).total_seconds()
        return elapsed >= config.IMESSAGE_MENTION_POLL_INTERVAL

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
                chat_context = self.build_agent_chat_context(
                    chat_jid,
                    chat_jid.endswith("@g.us"),
                )
                response, new_session_id = await handle_message(
                    f"Approved. Proceed with: {description}",
                    chat_jid,
                    session_id,
                    source="whatsapp",
                    chat_context=chat_context,
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

    # --- Timeout wrapper for background tasks ---

    async def _run_with_timeout(self, coro, name: str, timeout: int):
        """Run a coroutine with a timeout cap. Logs warning on timeout."""
        try:
            await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("%s exceeded %ds timeout — cancelled", name, timeout)
        except Exception:
            log.error("%s failed", name, exc_info=True)

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
        await self.automations.initialize()

        # Start WhatsApp in background thread (neonize is synchronous)
        wa_thread = threading.Thread(target=self.wa.connect, daemon=True, name="whatsapp")
        wa_thread.start()

        log.info("Molly is starting up. Waiting for WhatsApp connection...")
        log.info("Scan the QR code with your phone to pair.")

        await self.self_improvement.initialize()

        if not self.running:
            self.db.close()
            try:
                from memory.graph import close as close_graph
                close_graph()
            except Exception:
                pass
            log.info("Molly shutdown requested during startup.")
            return self.exit_code

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
            # Kill any stale process holding our port (e.g. from a crash/kernel panic)
            _kill_stale_port(config.WEB_PORT)

            web_server = uvicorn.Server(web_config)

            async def _serve_web():
                try:
                    await web_server.serve()
                except SystemExit:
                    log.warning("Web UI exited (port %d likely in use) — continuing without it", config.WEB_PORT)

            web_task = asyncio.create_task(_serve_web(), name="web-ui")
            web_task.add_done_callback(_task_done_callback)
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
                task = asyncio.create_task(
                    self._safe_process(msg_data),
                    name=f"process:{msg_data.get('chat_jid', '')[:20]}",
                )
                task.add_done_callback(_task_done_callback)
            except asyncio.TimeoutError:
                # No message in queue — check scheduled tasks (run in background)
                if self.wa and self.wa.connected:
                    if (
                        self._automation_tick_task is None
                        or self._automation_tick_task.done()
                    ):
                        self._automation_tick_task = self._spawn_bg(
                            self.automations.tick(),
                            name="automations-tick",
                        )
                    if (
                        self._self_improve_tick_task is None
                        or self._self_improve_tick_task.done()
                    ):
                        self._self_improve_tick_task = self._spawn_bg(
                            self.self_improvement.tick(),
                            name="self-improvement-tick",
                        )
                    if should_heartbeat(self.last_heartbeat):
                        self.last_heartbeat = datetime.now()
                        self._spawn_bg(
                            self._run_with_timeout(
                                run_heartbeat(self), "heartbeat", timeout=120,
                            ),
                            name="heartbeat",
                        )
                    # Fast poll: iMessage @molly mentions (every 60s)
                    if (
                        self._imessage_mention_task is None
                        or self._imessage_mention_task.done()
                    ):
                        if self._should_check_imessage_mentions():
                            self._last_imessage_mention_check = datetime.now()
                            from heartbeat import _check_imessage_mentions
                            self._imessage_mention_task = self._spawn_bg(
                                self._run_with_timeout(
                                    _check_imessage_mentions(self),
                                    "imessage-mentions",
                                    timeout=config.APPROVAL_TIMEOUT + 120,
                                ),
                                name="imessage-mentions",
                            )
                    if should_run_maintenance(self.last_maintenance):
                        self.last_maintenance = datetime.now()
                        self._spawn_bg(
                            self._run_with_timeout(
                                run_maintenance(molly=self), "maintenance", timeout=1800,
                            ),
                            name="maintenance",
                        )
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
        return self.exit_code

    def shutdown(self, *_args):
        log.info("Shutdown signal received...")
        self.running = False

    def request_restart(self, reason: str = ""):
        reason_text = reason.strip()
        if reason_text:
            log.info("Restart requested: %s", reason_text)
        else:
            log.info("Restart requested")
        self.exit_code = config.MOLLY_RESTART_EXIT_CODE
        self.running = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    instance_lock_fd = _acquire_instance_lock()
    if instance_lock_fd is None:
        log.error(
            "Another Molly instance is already running (lock: %s). Exiting.",
            config.STORE_DIR / ".molly.instance.lock",
        )
        sys.exit(1)

    try:
        preflight_checks()

        molly = Molly()

        # Handle graceful shutdown
        signal.signal(signal.SIGINT, molly.shutdown)
        signal.signal(signal.SIGTERM, molly.shutdown)

        # SIGHUP → graceful restart (exit code 42, supervisor loop restarts)
        def _restart_on_hup(*_args):
            molly.request_restart("SIGHUP received")

        signal.signal(signal.SIGHUP, _restart_on_hup)

        exit_code = 0
        try:
            exit_code = asyncio.run(molly.run()) or 0
        except KeyboardInterrupt:
            log.info("Interrupted.")
            exit_code = 130

        sys.exit(int(exit_code))
    finally:
        _release_instance_lock(instance_lock_fd)


if __name__ == "__main__":
    main()
