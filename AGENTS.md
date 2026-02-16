# Agent Instructions for Molly Codebase

## For Codex Agents Executing Refactor Phases

### READ FIRST
- Architecture overview: `~/.molly/workspace/memory/projects/refactor/overview.md`
- Phase specs: `~/.molly/workspace/memory/projects/refactor/phase-{N}-*.md`

### CRITICAL: Commit Requirements

Most phase tasks require **modifying existing files** (main.py, heartbeat.py, config.py, memory/triage.py, etc.), not just creating new ones.

**Before committing, you MUST verify:**

1. `git status` shows ZERO unstaged or untracked files related to your work
2. `git add` ALL modified files AND new files together
3. `git diff --cached --stat` shows BOTH new AND modified files in the same commit
4. Never leave work in `git stash` -- if you stash to resolve conflicts, pop and commit immediately

**After committing, verify:**
```bash
git diff HEAD~1 --stat
```
If you only see new files (all `create mode 100644`), your commit is **incomplete** -- go back and add the modified existing files.

### Test Requirements

- Virtual env: `source .venv/bin/activate`
- Set PYTHONPATH: `PYTHONPATH=/Users/brianmeyer/molly`
- Run tests: `python -m pytest tests/ -q --tb=short`
- Run from repo root: `/Users/brianmeyer/molly` (NOT from a worktree)
- Pre-existing failures in test_dedup_engine, test_relationship_audit, test_phase4 (web imports), test_preference_signals are known -- don't try to fix these

### Python Style

- Python 3.11+, asyncio throughout
- Type hints on public functions
- Logging via `logging.getLogger(__name__)`
- Use `from utils import atomic_write_json, load_json` for file I/O (Phase 2+)
- Use `from db_pool import get_connection, get_async_connection` for SQLite (Phase 2+)
