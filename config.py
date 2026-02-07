import os
import re
from pathlib import Path

# Identity
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Molly")
TRIGGER_PATTERN = re.compile(rf"(?:^|\s)@{ASSISTANT_NAME}\b", re.IGNORECASE)

# Commands
COMMANDS = {"/clear", "/memory", "/graph", "/forget", "/status"}

# Claude
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "opus")
ALLOWED_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch"]

# Owner — only this user can trigger Molly (phone + LID)
OWNER_IDS = {
    "15857332025",
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
