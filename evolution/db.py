"""SQLite schema and connection management for evolution.db.

Separate from mollygraph.db — evolution-specific tables (bandit, trajectories,
DGM state, judges, memory, causal) live here.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

_DB_DIR = Path.home() / ".molly" / "data"
_DB_PATH = _DB_DIR / "evolution.db"

_SCHEMA_SQL = """\
-- Thompson Sampling arms
CREATE TABLE IF NOT EXISTS bandit_arms (
    arm_id TEXT PRIMARY KEY,
    successes INTEGER DEFAULT 0,
    failures INTEGER DEFAULT 0,
    pulls INTEGER DEFAULT 0,
    total_reward REAL DEFAULT 0.0,
    last_pulled REAL DEFAULT 0,
    created_at REAL DEFAULT (unixepoch())
);

-- Full interaction trajectories
CREATE TABLE IF NOT EXISTS trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL DEFAULT (unixepoch()),
    arm_id TEXT,
    task_hash TEXT,
    action TEXT,
    reward REAL,
    outcome_score REAL,
    process_score REAL,
    safety_score REAL,
    cost_penalty REAL,
    diversity_bonus REAL,
    latency_seconds REAL,
    tokens_used INTEGER,
    tools_used INTEGER,
    git_commit TEXT,
    FOREIGN KEY (arm_id) REFERENCES bandit_arms(arm_id)
);

-- DGM state machine (single row)
CREATE TABLE IF NOT EXISTS dgm_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    state TEXT DEFAULT 'idle',
    proposal_json TEXT,
    test_results_json TEXT,
    shadow_results_json TEXT,
    guard_results_json TEXT,
    git_branch TEXT,
    started_at REAL,
    updated_at REAL
);

-- Judge evaluation records
CREATE TABLE IF NOT EXISTS judge_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL DEFAULT (unixepoch()),
    trajectory_id INTEGER,
    judge_model TEXT,
    functional_score REAL,
    efficiency_score REAL,
    final_score REAL,
    tie_broken INTEGER DEFAULT 0,
    FOREIGN KEY (trajectory_id) REFERENCES trajectories(id)
);

-- Episodic experience bank (raw interactions with decay)
CREATE TABLE IF NOT EXISTS experiences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL DEFAULT (unixepoch()),
    embedding BLOB,
    task_hash TEXT,
    task_class TEXT,
    reward REAL,
    confidence REAL DEFAULT 1.0,
    decay_weight REAL DEFAULT 1.0,
    content_json TEXT
);

-- Meta-guideline bank (distilled rules from experiences)
CREATE TABLE IF NOT EXISTS guidelines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL DEFAULT (unixepoch()),
    rule_text TEXT,
    source_exp_ids TEXT,
    activation_count INTEGER DEFAULT 0,
    last_activated REAL
);

-- Causal self-tracking (change -> consequence mapping)
CREATE TABLE IF NOT EXISTS causal_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL DEFAULT (unixepoch()),
    commit_hash TEXT,
    file_changed TEXT,
    test_file TEXT,
    test_name TEXT,
    baseline_result TEXT,
    post_result TEXT,
    outcome_delta REAL
);

-- Shadow evaluation results (one per patch evaluation)
CREATE TABLE IF NOT EXISTS shadow_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL DEFAULT (unixepoch()),
    proposal_id TEXT,
    avg_reward_before REAL,
    avg_reward_after REAL,
    error_rate_before REAL,
    error_rate_after REAL,
    latency_p95_before REAL,
    latency_p95_after REAL,
    golden_pass_rate REAL,
    is_improvement INTEGER DEFAULT 0,
    guard_passed INTEGER DEFAULT 0
);

-- Guard violation log
CREATE TABLE IF NOT EXISTS guard_violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL DEFAULT (unixepoch()),
    proposal_id TEXT,
    guard_name TEXT,
    threshold REAL,
    actual_value REAL,
    severity TEXT DEFAULT 'warning'
);

-- Proposal lifecycle audit log (every state transition)
CREATE TABLE IF NOT EXISTS proposal_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL DEFAULT (unixepoch()),
    proposal_id TEXT,
    old_state TEXT,
    new_state TEXT,
    note TEXT DEFAULT '',
    proposal_json TEXT
);

-- Code generation results (per-request tracking)
CREATE TABLE IF NOT EXISTS codegen_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL DEFAULT (unixepoch()),
    proposal_id TEXT,
    backend TEXT,
    success INTEGER DEFAULT 0,
    latency_seconds REAL,
    tokens_used INTEGER DEFAULT 0,
    patch_strategy TEXT,
    error TEXT DEFAULT '',
    patch_count INTEGER DEFAULT 0
);

-- Performance indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_trajectories_timestamp ON trajectories(timestamp);
CREATE INDEX IF NOT EXISTS idx_trajectories_arm_id ON trajectories(arm_id);
CREATE INDEX IF NOT EXISTS idx_experiences_task_hash ON experiences(task_hash);
CREATE INDEX IF NOT EXISTS idx_experiences_reward ON experiences(reward);
CREATE INDEX IF NOT EXISTS idx_shadow_results_proposal_id ON shadow_results(proposal_id);
CREATE INDEX IF NOT EXISTS idx_guard_violations_proposal_id ON guard_violations(proposal_id);
CREATE INDEX IF NOT EXISTS idx_proposal_history_proposal_id ON proposal_history(proposal_id);
CREATE INDEX IF NOT EXISTS idx_proposal_history_timestamp ON proposal_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_codegen_results_proposal_id ON codegen_results(proposal_id);
CREATE INDEX IF NOT EXISTS idx_codegen_results_backend ON codegen_results(backend);
"""


def ensure_schema() -> None:
    """Create all evolution tables if they don't already exist."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(_SCHEMA_SQL)
        # Seed DGM state row if missing
        conn.execute(
            "INSERT OR IGNORE INTO dgm_state (id, state) VALUES (1, 'idle')"
        )
        conn.commit()
    finally:
        conn.close()
    log.debug("evolution.db schema ensured at %s", _DB_PATH)


def get_connection() -> sqlite3.Connection:
    """Return a sqlite3 connection to evolution.db (creates schema if needed)."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
