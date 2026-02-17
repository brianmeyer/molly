import json
import logging as _logging
import os
import re
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".molly" / "credentials" / ".env")

_log = _logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    """Parse an integer env var with safe fallback on invalid input."""
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw.strip())
        except ValueError:
            value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float(name: str, default: float) -> float:
    """Parse a float env var with safe fallback on invalid input."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


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
    "/skills", "/skill", "/digest", "/automations", "/followups",
    "/commitments", "/health",
    "/upgrade", "/downgrade", "/mute", "/tiers",
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
OWNER_NAME = os.getenv("OWNER_NAME", "Brian")

if not OWNER_IDS:
    _log.warning("OWNER_IDS is empty — Molly will not respond to any messages")

# Web UI (Phase 4)
WEB_HOST = os.getenv("MOLLY_WEB_HOST", "127.0.0.1")
WEB_PORT = _env_int("MOLLY_WEB_PORT", 8080)
WEB_AUTH_TOKEN = os.getenv("MOLLY_WEB_TOKEN", "")  # simple bearer token
GATEWAY_WEBHOOK_TOKEN = os.getenv("MOLLY_GATEWAY_WEBHOOK_TOKEN", "")  # webhook auth

# WhatsApp outbound rendering
WHATSAPP_PLAIN_RENDER = _env_bool("MOLLY_WHATSAPP_PLAIN_RENDER", True)
WHATSAPP_PROMPT_GUARDRAILS = _env_bool("MOLLY_WHATSAPP_PROMPT_GUARDRAILS", True)
WHATSAPP_CHUNKING_ENABLED = _env_bool("MOLLY_WHATSAPP_CHUNKING_ENABLED", True)
WHATSAPP_CHUNK_CHARS = _env_int("MOLLY_WHATSAPP_CHUNK_CHARS", 1400)

# Email monitoring (Phase 4)
EMAIL_POLL_INTERVAL = 600  # 10 minutes

# Timing
POLL_INTERVAL = 2  # seconds (queue drain timeout)
HEARTBEAT_INTERVAL = 1800  # 30 minutes
ACTIVE_HOURS = (8, 22)  # 8am to 10pm
TIMEZONE = os.getenv("MOLLY_TIMEZONE", "America/New_York")
AUTOMATION_TICK_INTERVAL = _env_int("MOLLY_AUTOMATION_TICK_INTERVAL", 20)  # seconds

# iMessage @molly mention polling (fast check, separate from heartbeat)
IMESSAGE_MENTION_POLL_INTERVAL = _env_int("MOLLY_IMESSAGE_MENTION_POLL_INTERVAL", 60)  # seconds
IMESSAGE_MENTION_CONTEXT_COUNT = _env_int("MOLLY_IMESSAGE_MENTION_CONTEXT_COUNT", 8)

# Quiet hours / proactive automations (Phase 6)
QUIET_HOURS_START = os.getenv("MOLLY_QUIET_HOURS_START", "22:00")
QUIET_HOURS_END = os.getenv("MOLLY_QUIET_HOURS_END", "07:00")
QUIET_HOURS_TIMEZONE = os.getenv("MOLLY_QUIET_HOURS_TIMEZONE", TIMEZONE)
QUIET_HOURS_VIP_BYPASS = _env_bool("MOLLY_QUIET_HOURS_VIP_BYPASS", True)
QUIET_HOURS_URGENT_BYPASS = _env_bool("MOLLY_QUIET_HOURS_URGENT_BYPASS", True)
VIP_CONTACTS = _load_vip_contacts()
AUTOMATION_PROPOSAL_COOLDOWN = _env_int("MOLLY_AUTOMATION_PROPOSAL_COOLDOWN", 86400)
AUTOMATION_MIN_PATTERN_COUNT = _env_int("MOLLY_AUTOMATION_MIN_PATTERN_COUNT", 3)
EMAIL_TRIAGE_INTERVAL = _env_int("MOLLY_EMAIL_TRIAGE_INTERVAL", 600)

# Gateway daily cap (max outbound messages per day per gateway job cycle)
GATEWAY_DAILY_MESSAGE_CAP = _env_int("MOLLY_GATEWAY_DAILY_MESSAGE_CAP", 5, minimum=1)

# Feature flags — kill switches for proactive automation (V3 safety).
# ON by default; dedup + undo verified. Override via env var to disable.
AUTO_CALENDAR_EXTRACTION_ENABLED = _env_bool("MOLLY_AUTO_CALENDAR_EXTRACTION_ENABLED", True)
AUTO_TASK_EXTRACTION_ENABLED = _env_bool("MOLLY_AUTO_TASK_EXTRACTION_ENABLED", True)

# Paths
PROJECT_ROOT = Path(__file__).parent
STORE_DIR = PROJECT_ROOT / "store"
DATA_DIR = PROJECT_ROOT / "data"
WORKSPACE = Path(os.getenv("MOLLY_WORKSPACE", str(Path.home() / ".molly" / "workspace"))).expanduser()
LOG_DIR = Path.home() / ".molly" / "logs"
AUTOMATIONS_DIR = WORKSPACE / "automations"
SANDBOX_DIR = WORKSPACE / "sandbox"
WORKSPACE_STORE_DIR = WORKSPACE / "store"

# Files
DATABASE_PATH = STORE_DIR / "messages.db"
MOLLYGRAPH_PATH = STORE_DIR / "mollygraph.db"
AUTH_DIR = STORE_DIR / "auth"
SESSIONS_FILE = DATA_DIR / "sessions.json"
REGISTERED_CHATS_FILE = DATA_DIR / "registered_chats.json"
STATE_FILE = DATA_DIR / "state.json"
AUTOMATIONS_STATE_FILE = AUTOMATIONS_DIR / "state.json"
UNDO_MAP_FILE = DATA_DIR / "undo_map.json"
MAINTENANCE_STATE_FILE = WORKSPACE_STORE_DIR / "maintenance_state.json"
HEARTBEAT_CHECKPOINT_FILE = WORKSPACE_STORE_DIR / "heartbeat_checkpoint.json"
RELATION_SCHEMA_FILE = WORKSPACE / "config" / "relation_schema.yaml"

# Identity files (loaded every turn)
IDENTITY_FILES = [
    WORKSPACE / "SOUL.md",
    WORKSPACE / "USER.md",
    WORKSPACE / "AGENTS.md",
    WORKSPACE / "MEMORY.md",
]
HEARTBEAT_FILE = WORKSPACE / "HEARTBEAT.md"
SKILLS_DIR = WORKSPACE / "skills"

# Phase 7: Self-improvement
SELF_EDIT_ENABLED = _env_bool("MOLLY_SELF_EDIT_ENABLED", True)
SELF_EDIT_MAX_PATCH_LINES = _env_int("MOLLY_SELF_EDIT_MAX_PATCH_LINES", 200)
SELF_EDIT_AUTO_ROLLBACK_WINDOW = _env_int("MOLLY_SELF_EDIT_AUTO_ROLLBACK_WINDOW", 300)
MOLLY_RESTART_EXIT_CODE = _env_int("MOLLY_RESTART_EXIT_CODE", 42)
SELF_EDIT_PROTECTED_FILES = {
    "approval.py",
}
SELF_EDIT_PROTECTED_IDENTITY = {
    "SOUL.md",
}
GLINER_FINETUNE_MIN_EXAMPLES = _env_int("MOLLY_GLINER_FINETUNE_MIN_EXAMPLES", 500)
GLINER_FINETUNE_BENCHMARK_THRESHOLD = _env_float(
    "MOLLY_GLINER_FINETUNE_BENCHMARK_THRESHOLD", 0.05
)
GLINER_FULL_FINETUNE_MIN_EXAMPLES = _env_int(
    "MOLLY_GLINER_FULL_FINETUNE_MIN_EXAMPLES", 2000
)
GLINER_LORA_PLATEAU_WINDOW = _env_int("MOLLY_GLINER_LORA_PLATEAU_WINDOW", 3)
GLINER_LORA_PLATEAU_EPSILON = _env_float(
    "MOLLY_GLINER_LORA_PLATEAU_EPSILON", 0.01
)
WEEKLY_ASSESSMENT_DAY = os.getenv("MOLLY_WEEKLY_ASSESSMENT_DAY", "sunday").strip().lower()
WEEKLY_ASSESSMENT_HOUR = _env_int("MOLLY_WEEKLY_ASSESSMENT_HOUR", 3)
WEEKLY_ASSESSMENT_DIR = WORKSPACE / "memory" / "weekly"
EMAIL_DIGEST_QUEUE_DIR = WORKSPACE / "memory" / "email_digest_queue"
TOOL_GAP_MIN_FAILURES = _env_int("MOLLY_TOOL_GAP_MIN_FAILURES", 5, minimum=1)
TOOL_GAP_WINDOW_DAYS = _env_int("MOLLY_TOOL_GAP_WINDOW_DAYS", 7, minimum=1)

# Phase 4: Evolution Engine
CODE_LOOP_ENABLED = _env_bool("MOLLY_CODE_LOOP_ENABLED", True)

# Phase 5A: Orchestrator + Workers
ORCHESTRATOR_ENABLED = _env_bool("MOLLY_ORCHESTRATOR_ENABLED", True)
ORCHESTRATOR_TIMEOUT = _env_int("MOLLY_ORCHESTRATOR_TIMEOUT", 15, minimum=5)
MAX_CONCURRENT_WORKERS = _env_int("MOLLY_MAX_CONCURRENT_WORKERS", 3, minimum=1)
KIMI_TRIAGE_MODEL = os.getenv("MOLLY_KIMI_TRIAGE_MODEL", "kimi-k2.5").strip()
GEMINI_TRIAGE_FALLBACK = os.getenv("MOLLY_GEMINI_TRIAGE_FALLBACK", "gemini-2.5-flash-lite").strip()
WORKER_MODEL_FAST = os.getenv("MOLLY_WORKER_MODEL_FAST", "claude-haiku-4-5").strip()
WORKER_MODEL_DEFAULT = os.getenv("MOLLY_WORKER_MODEL_DEFAULT", "claude-sonnet-4-5").strip()
WORKER_MODEL_DEEP = os.getenv("MOLLY_WORKER_MODEL_DEEP", "claude-opus-4-6").strip()
WORKER_TIMEOUT_DEFAULT = _env_int("MOLLY_WORKER_TIMEOUT_DEFAULT", 45, minimum=5)
WORKER_TIMEOUT_RESEARCH = _env_int("MOLLY_WORKER_TIMEOUT_RESEARCH", 60, minimum=10)

# Phase 7: Health Doctor
HEALTH_REPORT_DIR = WORKSPACE / "memory" / "health"
HEALTH_REPORT_RETENTION_DAYS = _env_int("MOLLY_HEALTH_REPORT_RETENTION_DAYS", 30)
HEALTH_YELLOW_ESCALATION_DAYS = _env_int("MOLLY_HEALTH_YELLOW_ESCALATION_DAYS", 3)
HEALTH_POST_DEPLOY_CHECK = _env_bool("MOLLY_HEALTH_POST_DEPLOY_CHECK", True)
HEALTH_PIPELINE_WINDOW_HOURS = _env_int("MOLLY_HEALTH_PIPELINE_WINDOW_HOURS", 24)
HEALTH_ENTITY_SAMPLE_SIZE = _env_int("MOLLY_HEALTH_ENTITY_SAMPLE_SIZE", 20)
HEALTH_DUPLICATE_THRESHOLD = _env_int("MOLLY_HEALTH_DUPLICATE_THRESHOLD", 3)

# Track F rollout guards (pre-prod audit). Defaults are report-only/safe.
TRACK_F_REPORT_ONLY = _env_bool("MOLLY_TRACK_F_REPORT_ONLY", True)
TRACK_F_ENFORCE_PARSER_COMPAT = _env_bool("MOLLY_TRACK_F_ENFORCE_PARSER_COMPAT", True)
TRACK_F_ENFORCE_SKILL_TELEMETRY = _env_bool("MOLLY_TRACK_F_ENFORCE_SKILL_TELEMETRY", True)
TRACK_F_ENFORCE_FOUNDRY_INGESTION = _env_bool("MOLLY_TRACK_F_ENFORCE_FOUNDRY_INGESTION", True)
TRACK_F_ENFORCE_PROMOTION_DRIFT = _env_bool("MOLLY_TRACK_F_ENFORCE_PROMOTION_DRIFT", True)
TRACK_F_AUDIT_DIR = Path(
    os.getenv(
        "MOLLY_TRACK_F_AUDIT_DIR",
        str(PROJECT_ROOT / "store" / "audits" / "track-f"),
    )
).expanduser()
TRACK_F_SKILL_TELEMETRY_WINDOW_DAYS = _env_int(
    "MOLLY_TRACK_F_SKILL_TELEMETRY_WINDOW_DAYS", 14, minimum=1
)
TRACK_F_FOUNDRY_INGESTION_WINDOW_HOURS = _env_int(
    "MOLLY_TRACK_F_FOUNDRY_INGESTION_WINDOW_HOURS", 24, minimum=1
)
TRACK_F_FOUNDRY_INGESTION_MIN_EVENTS = _env_int(
    "MOLLY_TRACK_F_FOUNDRY_INGESTION_MIN_EVENTS", 1, minimum=1
)
TRACK_F_PROMOTION_DRIFT_WINDOW_DAYS = _env_int(
    "MOLLY_TRACK_F_PROMOTION_DRIFT_WINDOW_DAYS", 30, minimum=1
)
TRACK_F_PROMOTION_DRIFT_MAX_PENDING = _env_int(
    "MOLLY_TRACK_F_PROMOTION_DRIFT_MAX_PENDING", 5, minimum=0
)
TRACK_F_PROMOTION_DRIFT_MIN_RATE = min(
    1.0,
    max(0.0, _env_float("MOLLY_TRACK_F_PROMOTION_DRIFT_MIN_RATE", 0.30)),
)

# Approval system — three-tier action classification
ACTION_TIERS = {
    "AUTO": {
        # Read-only, local, safe
        "Read", "Glob", "Grep", "WebSearch", "WebFetch", "Task",
        # Shell access, file modifications
        "Bash", "Write", "Edit",
        # Google read (Phase 3B)
        "calendar_list", "calendar_get", "calendar_search",
        "gmail_search", "gmail_read",
        # Google write (Phase 3B)
        "gmail_send", "gmail_draft", "gmail_reply",
        "calendar_create", "calendar_update", "calendar_delete",
        # Google People (read-only)
        "people_search", "people_get", "people_list",
        # Google Tasks (full CRUD)
        "tasks_list", "tasks_list_tasks", "tasks_create",
        "tasks_complete", "tasks_delete",
        # Google Drive (read-only)
        "drive_search", "drive_get", "drive_read",
        # Google Meet (read-only)
        "meet_list", "meet_get", "meet_transcripts", "meet_recordings",
        # Apple MCP tools (Phase 3C)
        "contacts", "reminders", "notes", "messages", "mail", "calendar", "maps",
        "imessage_search", "imessage_recent", "imessage_thread", "imessage_unread",
        # WhatsApp history (Phase 4)
        "whatsapp_search",
        # External models (Phase 5)
        "kimi_research", "grok_reason", "groq_reason",
        # Browser MCP (Phase 5C) — mostly AUTO for reservations, lookups, etc.
        "browser_navigate", "browser_snapshot", "browser_take_screenshot",
        "browser_click", "browser_type", "browser_fill_form",
        "browser_select_option", "browser_hover", "browser_drag",
        "browser_press_key", "browser_evaluate", "browser_run_code",
        "browser_navigate_back", "browser_wait_for",
        "browser_console_messages", "browser_network_requests",
        "browser_tabs", "browser_resize", "browser_handle_dialog",
        "browser_file_upload", "browser_close", "browser_install",
    },
    "CONFIRM": set(),
    "BLOCKED": {
        "gmail_delete", "account_settings", "credential_access",
        "share_document",
    },
}

# Runtime preflight toggles (set in main.py at startup)
# Protected by _runtime_lock for thread safety.
_runtime_lock = threading.Lock()
DISABLED_MCP_SERVERS: set[str] = set()
DISABLED_TOOL_NAMES: set[str] = set()


def disable_mcp_server(name: str) -> None:
    """Thread-safe add to DISABLED_MCP_SERVERS."""
    with _runtime_lock:
        DISABLED_MCP_SERVERS.add(name)


def disable_tool(name: str) -> None:
    """Thread-safe add to DISABLED_TOOL_NAMES."""
    with _runtime_lock:
        DISABLED_TOOL_NAMES.add(name)


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
    "https://www.googleapis.com/auth/contacts.readonly",
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/meetings.space.readonly",
]

# Apple local databases (Phase 3C)
CONTACTS_DB = Path.home() / "Library" / "Application Support" / "AddressBook" / "AddressBook-v22.abcddb"
IMESSAGE_DB = Path.home() / "Library" / "Messages" / "chat.db"

# External model APIs (Phase 5)
# Note: Claude Agent SDK uses Max subscription — no ANTHROPIC_API_KEY needed
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Optional: codegen backend
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "")
MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL = "https://api.x.ai/v1"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)

# Contract audit layer (Track D)
CONTRACT_AUDIT_LLM_ENABLED = _env_bool("MOLLY_CONTRACT_AUDIT_LLM_ENABLED", True)
CONTRACT_AUDIT_LLM_BLOCKING = _env_bool("MOLLY_CONTRACT_AUDIT_LLM_BLOCKING", True)
CONTRACT_AUDIT_NIGHTLY_MODEL = os.getenv(
    "MOLLY_CONTRACT_AUDIT_NIGHTLY_MODEL", "kimi"
).strip().lower()
CONTRACT_AUDIT_WEEKLY_MODEL = os.getenv(
    "MOLLY_CONTRACT_AUDIT_WEEKLY_MODEL", "opus"
).strip().lower()
CONTRACT_AUDIT_MODEL_TIMEOUT_SECONDS = _env_int(
    "MOLLY_CONTRACT_AUDIT_MODEL_TIMEOUT_SECONDS", 45, minimum=5
)
CONTRACT_AUDIT_KIMI_MODEL = os.getenv(
    "MOLLY_CONTRACT_AUDIT_KIMI_MODEL", "kimi-k2.5"
).strip()
CONTRACT_AUDIT_GEMINI_MODEL = os.getenv(
    "MOLLY_CONTRACT_AUDIT_GEMINI_MODEL", "gemini-2.5-flash"
).strip()

# Relationship quality audit (nightly Step 4b)
REL_AUDIT_MODEL_ENABLED = _env_bool("MOLLY_REL_AUDIT_MODEL_ENABLED", True)
REL_AUDIT_KIMI_MODEL = os.getenv("MOLLY_REL_AUDIT_KIMI_MODEL", "kimi-k2.5").strip()
REL_AUDIT_LOW_CONFIDENCE_THRESHOLD = _env_float(
    "MOLLY_REL_AUDIT_LOW_CONFIDENCE_THRESHOLD", 0.35
)
REL_AUDIT_RELATED_TO_WARN_THRESHOLD = _env_int(
    "MOLLY_REL_AUDIT_RELATED_TO_WARN_THRESHOLD", 3
)
REL_AUDIT_MAX_MODEL_BATCH = _env_int(
    "MOLLY_REL_AUDIT_MAX_MODEL_BATCH", 30
)
REL_AUDIT_AUTO_FIX_ENABLED = _env_bool("MOLLY_REL_AUDIT_AUTO_FIX_ENABLED", True)

# Local triage model (Qwen3-4B GGUF via llama-cpp-python)
TRIAGE_MODEL_PATH = Path(
    os.getenv(
        "TRIAGE_MODEL_PATH",
        Path.home() / ".molly" / "models" / "Qwen_Qwen3-4B-Q4_K_M.gguf",
    )
).expanduser()
TRIAGE_TIMEOUT = 30  # seconds per triage call
TRIAGE_CONTEXT_ENTITIES = 20  # top N entities by strength for triage context
TRIAGE_GPU_LAYERS = _env_int("TRIAGE_GPU_LAYERS", -1)  # -1 = use all available GPU layers
TRIAGE_N_CTX = _env_int("TRIAGE_N_CTX", 4096)
TRIAGE_N_THREADS = _env_int("TRIAGE_N_THREADS", os.cpu_count() or 8)

# Phase 5C: Voice loop (Porcupine wakeword + Gemini Live)
VOICE_ENABLED = _env_bool("MOLLY_VOICE_ENABLED", True)
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY", "")
PORCUPINE_MODEL_PATH = os.getenv(
    "MOLLY_PORCUPINE_MODEL_PATH",
    str(Path.home() / ".molly" / "models" / "molly_mac.ppn"),
)
GEMINI_LIVE_MODEL = os.getenv(
    "MOLLY_GEMINI_LIVE_MODEL",
    "gemini-2.5-flash-native-audio-preview-12-2025",
).strip()
VOICE_MAX_SESSION_MINUTES = _env_int("MOLLY_VOICE_MAX_SESSION_MINUTES", 10, minimum=1)
VOICE_DAILY_BUDGET_MINUTES = _env_int("MOLLY_VOICE_DAILY_BUDGET_MINUTES", 60, minimum=1)
VOICE_DEVICE_INDEX = _env_int("MOLLY_VOICE_DEVICE_INDEX", -1)
VOICE_SENSITIVITY = _env_float("MOLLY_VOICE_SENSITIVITY", 0.5)
VOICE_PRELOAD_ENABLED = _env_bool("MOLLY_VOICE_PRELOAD_ENABLED", True)

# Phase 5C: Browser MCP
BROWSER_MCP_ENABLED = _env_bool("MOLLY_BROWSER_MCP_ENABLED", True)
BROWSER_PROFILE_DIR = Path(
    os.getenv("MOLLY_BROWSER_PROFILE_DIR", str(Path.home() / ".molly" / "browser_profile"))
).expanduser()

# Phase 5C: Qwen3 LoRA fine-tuning (triage + email classification)
QWEN_LORA_ENABLED = _env_bool("MOLLY_QWEN_LORA_ENABLED", True)
QWEN_LORA_MIN_EXAMPLES = _env_int("MOLLY_QWEN_LORA_MIN_EXAMPLES", 500)

# Phase 5C: Plugin architecture
PLUGIN_DIR = Path(os.getenv("MOLLY_PLUGIN_DIR", str(PROJECT_ROOT / "plugins"))).expanduser()
PLUGIN_ENABLED = _env_bool("MOLLY_PLUGIN_ENABLED", True)

# Phase 5C: Docker sandbox
DOCKER_SANDBOX_ENABLED = _env_bool("MOLLY_DOCKER_SANDBOX_ENABLED", True)
DOCKER_SANDBOX_IMAGE = os.getenv("MOLLY_DOCKER_SANDBOX_IMAGE", "python:3.12-slim")
DOCKER_SANDBOX_TIMEOUT = _env_int("MOLLY_DOCKER_SANDBOX_TIMEOUT", 60, minimum=10)
DOCKER_SANDBOX_MEMORY = os.getenv("MOLLY_DOCKER_SANDBOX_MEMORY", "256m")

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
if not NEO4J_PASSWORD:
    _log.warning("NEO4J_PASSWORD is empty — graph features will fail to authenticate")
