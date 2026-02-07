import os
import re
from pathlib import Path

# Identity
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Molly")
TRIGGER_PATTERN = re.compile(rf"(?:^|\s)@{ASSISTANT_NAME}\b", re.IGNORECASE)

# Commands
COMMANDS = {"/help", "/clear", "/memory", "/graph", "/forget", "/status"}

# Claude
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "opus")
ALLOWED_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch"]

# Owner — only this user can trigger Molly (phone + LID)
OWNER_IDS = {
    "15550001234",
    "52660963176533",
}

# Timing
POLL_INTERVAL = 2  # seconds (queue drain timeout)
HEARTBEAT_INTERVAL = 1800  # 30 minutes
ACTIVE_HOURS = (8, 22)  # 8am to 10pm

# Paths
PROJECT_ROOT = Path(__file__).parent
STORE_DIR = PROJECT_ROOT / "store"
DATA_DIR = PROJECT_ROOT / "data"
WORKSPACE = Path(os.getenv("MOLLY_WORKSPACE", Path.home() / ".molly" / "workspace"))
LOG_DIR = Path.home() / ".molly" / "logs"

# Files
DATABASE_PATH = STORE_DIR / "messages.db"
MOLLYGRAPH_PATH = STORE_DIR / "mollygraph.db"
AUTH_DIR = STORE_DIR / "auth"
SESSIONS_FILE = DATA_DIR / "sessions.json"
REGISTERED_CHATS_FILE = DATA_DIR / "registered_chats.json"
STATE_FILE = DATA_DIR / "state.json"

# Identity files (loaded every turn)
IDENTITY_FILES = [
    WORKSPACE / "SOUL.md",
    WORKSPACE / "USER.md",
    WORKSPACE / "AGENTS.md",
    WORKSPACE / "MEMORY.md",
]
HEARTBEAT_FILE = WORKSPACE / "HEARTBEAT.md"

# Approval system
REQUIRES_APPROVAL = {
    "send_email",           # Sending emails or messages to external recipients
    "send_message_external",# Sending messages to people other than Brian
    "api_write",            # POST/PUT/DELETE to external APIs
    "bash_destructive",     # rm, kill, drop, truncate, or other destructive commands
    "modify_identity",      # Editing SOUL.md, AGENTS.md, or other identity files
    "install_package",      # Installing or removing system packages
    "file_delete",          # Deleting files outside the workspace
    "calendar_modify",      # Creating, updating, or deleting calendar events
}

APPROVED_ACTIONS: set[str] = set()  # Pre-approved categories (bypass approval)

APPROVAL_TIMEOUT = 300  # seconds (5 minutes)

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme")
