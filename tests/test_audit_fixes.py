"""Tests for audit fix issues #1-#8."""

import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent


def _read_source(filename: str) -> str:
    """Read a Python source file without importing it."""
    return (ROOT / filename).read_text()


# ---------------------------------------------------------------------------
# #1: maintenance.py no longer uses bypassPermissions
# ---------------------------------------------------------------------------
class TestMaintenanceNoBypass(unittest.TestCase):
    """Verify maintenance.py does NOT use bypassPermissions or MCP tools."""

    def setUp(self):
        self.source = _read_source("maintenance.py")

    def test_no_bypass_permissions_in_code(self):
        """maintenance.py must not have permission_mode='bypassPermissions' in code."""
        # Check that bypassPermissions is not used as a value assignment
        self.assertNotIn('permission_mode="bypassPermissions"', self.source)
        self.assertNotIn("permission_mode='bypassPermissions'", self.source)

    def test_no_mcp_servers(self):
        """maintenance.py must not reference mcp_servers."""
        self.assertNotIn("mcp_servers=", self.source)

    def test_no_allowed_tools_config(self):
        """maintenance.py must not use config.ALLOWED_TOOLS."""
        self.assertNotIn("config.ALLOWED_TOOLS", self.source)

    def test_no_gmail_or_calendar_imports(self):
        """maintenance.py must not import MCP tool servers."""
        self.assertNotIn("from tools.calendar import", self.source)
        self.assertNotIn("from tools.gmail import", self.source)
        self.assertNotIn("from tools.contacts import", self.source)
        self.assertNotIn("from tools.imessage import", self.source)

    def test_has_direct_graph_operations(self):
        """maintenance.py must call graph operations directly."""
        self.assertIn("_run_strength_decay", self.source)
        self.assertIn("_run_orphan_cleanup", self.source)
        self.assertIn("_run_dedup_sweep", self.source)
        self.assertIn("_run_blocklist_cleanup", self.source)

    def test_run_maintenance_accepts_molly(self):
        """run_maintenance must accept a molly parameter."""
        self.assertIn("async def run_maintenance(molly=None)", self.source)

    def test_sends_summary_to_owner(self):
        """run_maintenance must send a WhatsApp summary to owner."""
        self.assertIn("_get_owner_dm_jid", self.source)
        self.assertIn("send_message", self.source)


# ---------------------------------------------------------------------------
# #2: Canonical names in relationships
# ---------------------------------------------------------------------------
class TestCanonicalNames(unittest.TestCase):
    """Verify processor.py uses canonical names for relationships."""

    def setUp(self):
        self.source = _read_source("memory/processor.py")

    def test_raw_to_canonical_mapping_exists(self):
        """extract_to_graph must build a raw→canonical name mapping."""
        self.assertIn("raw_to_canonical", self.source)

    def test_relationships_use_canonical(self):
        """Relationships must use canonical names, not raw extracted names."""
        self.assertIn('raw_to_canonical.get(rel["head"]', self.source)
        self.assertIn('raw_to_canonical.get(rel["tail"]', self.source)

    def test_no_raw_head_tail(self):
        """Must NOT use rel['head'] directly for upsert_relationship."""
        # The old pattern was: head_name=rel["head"]
        # Should NOT appear as a direct argument to upsert_relationship
        lines = self.source.split("\n")
        for line in lines:
            if "upsert_relationship" in line or "head_name=" in line:
                self.assertNotIn('head_name=rel["head"]', line)


# ---------------------------------------------------------------------------
# #3: Heartbeat/maintenance run as background tasks
# ---------------------------------------------------------------------------
class TestBackgroundTasks(unittest.TestCase):
    """Verify heartbeat and maintenance run as background tasks with timeout."""

    def setUp(self):
        self.source = _read_source("main.py")

    def test_no_direct_await_heartbeat(self):
        """Main loop must NOT await run_heartbeat directly."""
        # The old pattern was: await run_heartbeat(self)
        # in the main while loop. Now it should be in create_task
        lines = self.source.split("\n")
        in_main_loop = False
        for line in lines:
            if "while self.running" in line:
                in_main_loop = True
            if in_main_loop:
                stripped = line.strip()
                if stripped.startswith("await run_heartbeat"):
                    self.fail("Found direct 'await run_heartbeat' in main loop")
                if stripped.startswith("await run_maintenance"):
                    self.fail("Found direct 'await run_maintenance' in main loop")

    def test_timeout_wrapper_exists(self):
        """Molly must have a _run_with_timeout method."""
        self.assertIn("async def _run_with_timeout", self.source)

    def test_heartbeat_timeout_is_2_min(self):
        """Heartbeat timeout must be 120 seconds."""
        self.assertIn("timeout=120", self.source)

    def test_maintenance_timeout_is_30_min(self):
        """Maintenance timeout must be 1800 seconds."""
        self.assertIn("timeout=1800", self.source)


# ---------------------------------------------------------------------------
# #4: Fire-and-forget tasks log exceptions
# ---------------------------------------------------------------------------
class TestTaskCallbacks(unittest.TestCase):
    """Verify fire-and-forget tasks have done callbacks."""

    def setUp(self):
        self.source = _read_source("main.py")

    def test_task_done_callback_defined(self):
        """main.py must define _task_done_callback."""
        self.assertIn("def _task_done_callback", self.source)

    def test_callback_checks_exception(self):
        """Callback must check task.exception()."""
        self.assertIn("task.exception()", self.source)

    def test_all_create_task_have_callback(self):
        """Every create_task call should have a corresponding add_done_callback."""
        # Count create_task calls and add_done_callback calls
        create_count = self.source.count("asyncio.create_task(")
        callback_count = self.source.count("add_done_callback(")
        # Every create_task should have a callback (except inside _run_with_timeout)
        self.assertGreaterEqual(callback_count, create_count - 1)

    def test_agent_post_processing_has_callback(self):
        """agent.py post-processing task must have a callback."""
        agent_source = _read_source("agent.py")
        self.assertIn("add_done_callback", agent_source)


# ---------------------------------------------------------------------------
# #5: Sync model inference wrapped in executor
# ---------------------------------------------------------------------------
class TestExecutorWrapping(unittest.TestCase):
    """Verify embed and extract calls use run_in_executor."""

    def setUp(self):
        self.source = _read_source("memory/processor.py")

    def test_embed_uses_executor(self):
        """embed_and_store must use run_in_executor for embed()."""
        self.assertIn("run_in_executor", self.source)
        # Should find run_in_executor called with embed
        self.assertIn("run_in_executor(None, embed, content)", self.source)

    def test_extract_uses_executor(self):
        """extract_to_graph must use run_in_executor for extract()."""
        self.assertIn("run_in_executor(None, extract, content)", self.source)

    def test_asyncio_imported(self):
        """processor.py must import asyncio."""
        self.assertIn("import asyncio", self.source)


# ---------------------------------------------------------------------------
# #6: Heartbeat targets owner DM
# ---------------------------------------------------------------------------
class TestHeartbeatTarget(unittest.TestCase):
    """Verify heartbeat sends to owner DM, not first registered chat."""

    def setUp(self):
        self.source = _read_source("heartbeat.py")

    def test_uses_owner_dm_jid(self):
        """run_heartbeat must call _get_owner_dm_jid."""
        self.assertIn("_get_owner_dm_jid", self.source)

    def test_no_next_iter_registered(self):
        """run_heartbeat must NOT use next(iter(molly.registered_chats))."""
        self.assertNotIn("next(iter(molly.registered_chats))", self.source)


# ---------------------------------------------------------------------------
# #7: Commands in DMs without @Molly
# ---------------------------------------------------------------------------
class TestDMCommands(unittest.TestCase):
    """Verify slash commands work in DMs without @Molly prefix."""

    def setUp(self):
        self.source = _read_source("main.py")

    def test_dm_command_detection(self):
        """process_message must check for DM commands without trigger."""
        self.assertIn("is_dm_command", self.source)

    def test_dm_command_checks_owner_dm(self):
        """DM command check must verify chat_mode is owner_dm."""
        self.assertIn('chat_mode == "owner_dm"', self.source)

    def test_dm_command_checks_slash(self):
        """DM command check must verify message starts with /."""
        self.assertIn('startswith("/")', self.source)


# ---------------------------------------------------------------------------
# #8: Dynamic skill trigger parsing
# ---------------------------------------------------------------------------
class TestDynamicTriggers(unittest.TestCase):
    """Verify skill triggers are parsed from markdown, not hardcoded."""

    def setUp(self):
        self.source = _read_source("skills.py")

    def test_no_hardcoded_trigger_list(self):
        """skills.py must not have the old hardcoded regex patterns."""
        self.assertNotIn('"daily-digest", re.compile(', self.source)
        self.assertNotIn('"meeting-prep", re.compile(', self.source)

    def test_has_trigger_parsing_functions(self):
        """skills.py must have dynamic trigger parsing functions."""
        self.assertIn("def _extract_trigger_phrases", self.source)
        self.assertIn("def _phrase_to_regex", self.source)
        self.assertIn("def _build_trigger_patterns", self.source)

    def test_extract_trigger_phrases(self):
        """_extract_trigger_phrases should find quoted strings."""
        from skills import _extract_trigger_phrases

        text = '- "daily digest", "morning briefing", "what\'s on today"'
        phrases = _extract_trigger_phrases(text)
        self.assertEqual(len(phrases), 3)
        self.assertIn("daily digest", phrases)
        self.assertIn("morning briefing", phrases)

    def test_phrase_to_regex_basic(self):
        """_phrase_to_regex should convert simple phrases."""
        from skills import _phrase_to_regex

        pattern = _phrase_to_regex("daily digest")
        self.assertTrue(re.search(pattern, "daily digest", re.IGNORECASE))
        self.assertTrue(re.search(pattern, "daily  digest", re.IGNORECASE))

    def test_phrase_to_regex_placeholder(self):
        """_phrase_to_regex should handle [placeholder] tokens."""
        from skills import _phrase_to_regex

        pattern = _phrase_to_regex("draft an email to [person]")
        self.assertTrue(re.search(pattern, "draft an email to John", re.IGNORECASE))

    def test_build_trigger_patterns(self):
        """_build_trigger_patterns should build patterns from loaded skills."""
        from skills import _build_trigger_patterns, reload_skills

        reload_skills()
        patterns = _build_trigger_patterns()
        self.assertGreater(len(patterns), 0)

        for name, pattern in patterns:
            self.assertIsInstance(name, str)
            self.assertIsInstance(pattern, re.Pattern)

    def test_match_skills_finds_digest(self):
        """match_skills should match 'what's on today' to daily-digest."""
        from skills import match_skills, reload_skills

        reload_skills()
        matched = match_skills("what's on today")
        names = [s.name for s in matched]
        self.assertIn("daily-digest", names)

    def test_match_skills_finds_email(self):
        """match_skills should match email drafting phrases."""
        from skills import match_skills, reload_skills

        reload_skills()
        matched = match_skills("draft an email to John about the project")
        names = [s.name for s in matched]
        self.assertIn("email-draft", names)

    def test_match_skills_finds_research(self):
        """match_skills should match research phrases."""
        from skills import match_skills, reload_skills

        reload_skills()
        matched = match_skills("research AI agent security")
        names = [s.name for s in matched]
        self.assertIn("research-brief", names)


if __name__ == "__main__":
    unittest.main()
