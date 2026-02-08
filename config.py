import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".molly" / "credentials" / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_vip_contacts() -> list[dict]:
    """Load VIP contacts from env as JSON list, fallback to empty list.

    Expected format:
      MOLLY_VIP_CONTACTS='[{"email":"a@b.com","name":"Manager"}]'
    """
    raw = os.getenv("MOLLY_VIP_CONTACTS", "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except Exception:
        pass
    return []

# Identity
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Molly")
TRIGGER_PATTERN = re.compile(rf"(?:^|\s)@{ASSISTANT_NAME}\b", re.IGNORECASE)

# Commands
COMMANDS = {
    "/help", "/clear", "/memory", "/graph", "/forget",
    "/status", "/pending", "/register", "/unregister", "/groups",
    "/skills", "/skill", "/digest", "/automations",
}

# Chat processing modes (tiered classification)
# owner_dm  — DMs where chat JID matches OWNER_IDS: full processing + always respond
# respond   — registered groups: full processing + respond to @Molly from owner
# listen    — monitored groups: embed + selective graph extraction, no responses
# store_only — everything else: embed only, no graph, no responses
CHAT_MODES = {"owner_dm", "respond", "listen", "store_only"}

# Claude
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "opus")

# Owner — only this user can trigger Molly (phone + LID)
OWNER_IDS = set(filter(None, os.getenv("OWNER_IDS", "").split(",")))

# Web UI (Phase 4)
WEB_HOST = os.getenv("MOLLY_WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("MOLLY_WEB_PORT", "8080"))
WEB_AUTH_TOKEN = os.getenv("MOLLY_WEB_TOKEN", "")  # simple bearer token

# Email monitoring (Phase 4)
EMAIL_POLL_INTERVAL = 600  # 10 minutes

# Timing
POLL_INTERVAL = 2  # seconds (queue drain timeout)
HEARTBEAT_INTERVAL = 1800  # 30 minutes
ACTIVE_HOURS = (8, 22)  # 8am to 10pm
TIMEZONE = os.getenv("MOLLY_TIMEZONE", "America/New_York")
AUTOMATION_TICK_INTERVAL = int(os.getenv("MOLLY_AUTOMATION_TICK_INTERVAL", "20"))  # seconds

# Quiet hours / proactive automations (Phase 6)
QUIET_HOURS_START = os.getenv("MOLLY_QUIET_HOURS_START", "22:00")
QUIET_HOURS_END = os.getenv("MOLLY_QUIET_HOURS_END", "07:00")
QUIET_HOURS_TIMEZONE = os.getenv("MOLLY_QUIET_HOURS_TIMEZONE", TIMEZONE)
QUIET_HOURS_VIP_BYPASS = _env_bool("MOLLY_QUIET_HOURS_VIP_BYPASS", True)
QUIET_HOURS_URGENT_BYPASS = _env_bool("MOLLY_QUIET_HOURS_URGENT_BYPASS", True)
VIP_CONTACTS = _load_vip_contacts()
AUTOMATION_PROPOSAL_COOLDOWN = int(os.getenv("MOLLY_AUTOMATION_PROPOSAL_COOLDOWN", "86400"))
AUTOMATION_MIN_PATTERN_COUNT = int(os.getenv("MOLLY_AUTOMATION_MIN_PATTERN_COUNT", "3"))
EMAIL_TRIAGE_INTERVAL = int(os.getenv("MOLLY_EMAIL_TRIAGE_INTERVAL", "600"))

# Paths
PROJECT_ROOT = Path(__file__).parent
STORE_DIR = PROJECT_ROOT / "store"
DATA_DIR = PROJECT_ROOT / "data"
WORKSPACE = Path(os.getenv("MOLLY_WORKSPACE", Path.home() / ".molly" / "workspace"))
LOG_DIR = Path.home() / ".molly" / "logs"
AUTOMATIONS_DIR = WORKSPACE / "automations"

# Files
DATABASE_PATH = STORE_DIR / "messages.db"
MOLLYGRAPH_PATH = STORE_DIR / "mollygraph.db"
AUTH_DIR = STORE_DIR / "auth"
SESSIONS_FILE = DATA_DIR / "sessions.json"
REGISTERED_CHATS_FILE = DATA_DIR / "registered_chats.json"
STATE_FILE = DATA_DIR / "state.json"
AUTOMATIONS_STATE_FILE = AUTOMATIONS_DIR / "state.json"

# Identity files (loaded every turn)
IDENTITY_FILES = [
    WORKSPACE / "SOUL.md",
    WORKSPACE / "USER.md",
    WORKSPACE / "AGENTS.md",
    WORKSPACE / "MEMORY.md",
]
HEARTBEAT_FILE = WORKSPACE / "HEARTBEAT.md"
SKILLS_DIR = WORKSPACE / "skills"

# Approval system — three-tier action classification
ACTION_TIERS = {
    "AUTO": {
        # Read-only, local, safe — execute immediately
        "Read", "Glob", "Grep", "WebSearch", "WebFetch", "Task",
        # Google read-only (Phase 3B)
        "calendar_list", "calendar_get", "calendar_search",
        "gmail_search", "gmail_read",
        # Apple read-only (Phase 3C)
        "contacts_search", "contacts_get", "contacts_list", "contacts_recent",
        "imessage_search", "imessage_recent", "imessage_thread", "imessage_unread",
        # WhatsApp history (Phase 4)
        "whatsapp_search",
        # External models (Phase 5)
        "kimi_research", "grok_reason",
    },
    "CONFIRM": {
        # Shell access — requires Brian's approval
        "Bash",
        # External writes, file modifications
        "Write", "Edit",
        # Google writes (Phase 3B)
        "gmail_send", "gmail_draft", "gmail_reply",
        "calendar_create", "calendar_update", "calendar_delete",
    },
    "BLOCKED": {
        "gmail_delete", "account_settings", "credential_access",
        "share_document",
    },
}

# Runtime preflight toggles (set in main.py at startup)
DISABLED_MCP_SERVERS: set[str] = set()
DISABLED_TOOL_NAMES: set[str] = set()

# Paths within workspace/memory/ are auto-approved for Write/Edit
# (daily logs, deep knowledge files). Identity files still require approval.
AUTO_APPROVE_PATHS = {"memory/"}

APPROVAL_TIMEOUT = 600  # seconds (10 minutes)

# Google OAuth (Phase 3B)
GOOGLE_CREDENTIALS_DIR = Path.home() / ".molly" / "credentials"
GOOGLE_CLIENT_SECRET = GOOGLE_CREDENTIALS_DIR / "client_secret.json"
GOOGLE_TOKEN = GOOGLE_CREDENTIALS_DIR / "token.json"
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
]

# Apple local databases (Phase 3C)
CONTACTS_DB = Path.home() / "Library" / "Application Support" / "AddressBook" / "AddressBook-v22.abcddb"
IMESSAGE_DB = Path.home() / "Library" / "Messages" / "chat.db"

# External model APIs (Phase 5)
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "")
MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL = "https://api.x.ai/v1"

# Local triage model (Qwen3-4B GGUF via llama-cpp-python)
TRIAGE_MODEL_PATH = Path(
    os.getenv(
        "TRIAGE_MODEL_PATH",
        Path.home() / ".molly" / "models" / "Qwen_Qwen3-4B-Q4_K_M.gguf",
    )
).expanduser()
TRIAGE_TIMEOUT = 30  # seconds per triage call
TRIAGE_CONTEXT_ENTITIES = 20  # top N entities by strength for triage context
TRIAGE_GPU_LAYERS = int(os.getenv("TRIAGE_GPU_LAYERS", "-1"))  # -1 = use all available GPU layers
TRIAGE_N_CTX = int(os.getenv("TRIAGE_N_CTX", "4096"))
TRIAGE_N_THREADS = int(os.getenv("TRIAGE_N_THREADS", str(os.cpu_count() or 8)))

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
