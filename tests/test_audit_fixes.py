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


# ---------------------------------------------------------------------------
# #9: Task tool in AUTO tier (sub-agent routing unblocked)
# ---------------------------------------------------------------------------
class TestTaskToolTier(unittest.TestCase):
    """Verify Task tool is in AUTO tier so sub-agents can be invoked."""

    def setUp(self):
        self.source = _read_source("config.py")

    def test_task_in_auto_tier(self):
        """Task must be in ACTION_TIERS AUTO set."""
        # Find the AUTO section and check for Task
        self.assertIn('"Task"', self.source)
        # Verify it's in the AUTO block, not CONFIRM or BLOCKED
        lines = self.source.split("\n")
        in_auto = False
        for line in lines:
            if '"AUTO"' in line:
                in_auto = True
            if in_auto and '"Task"' in line:
                break
            if in_auto and ('"CONFIRM"' in line or '"BLOCKED"' in line):
                self.fail("Task not found in AUTO tier before CONFIRM/BLOCKED")

    def test_no_allowed_tools_list(self):
        """ALLOWED_TOOLS should be removed (dead config)."""
        self.assertNotIn("ALLOWED_TOOLS", self.source)


# ---------------------------------------------------------------------------
# #10: Timestamp format consistency
# ---------------------------------------------------------------------------
class TestTimestampFormat(unittest.TestCase):
    """Verify timestamps are stored in ISO format."""

    def setUp(self):
        self.source = _read_source("whatsapp.py")

    def test_uses_isoformat(self):
        """whatsapp.py must convert timestamps to ISO format."""
        self.assertIn("isoformat()", self.source)

    def test_no_bare_str_timestamp(self):
        """Must not use bare str(info.Timestamp) for storage."""
        self.assertNotIn("timestamp = str(info.Timestamp)", self.source)


# ---------------------------------------------------------------------------
# #11: Approval edit flow returns instruction
# ---------------------------------------------------------------------------
class TestApprovalEditFlow(unittest.TestCase):
    """Verify edit responses return the edit instruction, not just False."""

    def setUp(self):
        self.approval_source = _read_source("approval.py")
        self.agent_source = _read_source("agent.py")

    def test_edit_returns_instruction(self):
        """Edit branch must return result[1] (the instruction string)."""
        self.assertIn("return result[1]", self.approval_source)

    def test_edit_not_returns_false(self):
        """Edit branch must NOT return False."""
        lines = self.approval_source.split("\n")
        for i, line in enumerate(lines):
            if 'result[0] == "edit"' in line:
                # Check the next few lines don't return False
                for j in range(i + 1, min(i + 5, len(lines))):
                    if "return False" in lines[j]:
                        self.fail("Edit branch still returns False")
                    if "return result[1]" in lines[j]:
                        break

    def test_agent_handles_edit_string(self):
        """agent.py must handle edit instruction as a string result."""
        self.assertIn("isinstance(result, str)", self.agent_source)
        self.assertIn("edit instructions", self.agent_source)


# ---------------------------------------------------------------------------
# #12: Source parameter in handle_message
# ---------------------------------------------------------------------------
class TestSourceParameter(unittest.TestCase):
    """Verify handle_message accepts source and callers pass it."""

    def setUp(self):
        self.agent_source = _read_source("agent.py")
        self.main_source = _read_source("main.py")
        self.web_source = _read_source("web.py")
        self.terminal_source = _read_source("terminal.py")

    def test_handle_message_has_source_param(self):
        """handle_message must accept a source parameter."""
        self.assertIn('source: str = "unknown"', self.agent_source)

    def test_default_not_whatsapp(self):
        """Default source must NOT be 'whatsapp'."""
        self.assertNotIn('source: str = "whatsapp"', self.agent_source)

    def test_main_passes_whatsapp(self):
        """main.py must pass source='whatsapp'."""
        self.assertIn('source="whatsapp"', self.main_source)

    def test_web_passes_web(self):
        """web.py must pass source='web'."""
        self.assertIn('source="web"', self.web_source)

    def test_terminal_passes_terminal(self):
        """terminal.py must pass source='terminal'."""
        self.assertIn('source="terminal"', self.terminal_source)

    def test_post_processing_passes_source(self):
        """agent.py post-processing must forward source to process_conversation."""
        self.assertIn("source=source", self.agent_source)


# ---------------------------------------------------------------------------
# #13: Graph extraction in heartbeat paths
# ---------------------------------------------------------------------------
class TestHeartbeatGraphExtraction(unittest.TestCase):
    """Verify iMessage and email heartbeat paths extract to graph."""

    def setUp(self):
        self.source = _read_source("heartbeat.py")

    def test_imessage_imports_extract(self):
        """iMessage heartbeat must import extract_to_graph."""
        # Find the _check_imessage function area
        self.assertIn("extract_to_graph", self.source)

    def test_imessage_calls_extract(self):
        """iMessage heartbeat must call extract_to_graph for relevant messages."""
        # Look for extract_to_graph with imessage source
        self.assertIn('source="imessage"', self.source)
        lines = self.source.split("\n")
        found_imessage_extract = False
        for line in lines:
            if "extract_to_graph" in line and "imessage" in line:
                found_imessage_extract = True
                break
        self.assertTrue(found_imessage_extract, "extract_to_graph not called with imessage source")

    def test_email_calls_extract(self):
        """Email heartbeat must call extract_to_graph for relevant emails."""
        lines = self.source.split("\n")
        found_email_extract = False
        for line in lines:
            if "extract_to_graph" in line and "email" in line:
                found_email_extract = True
                break
        self.assertTrue(found_email_extract, "extract_to_graph not called with email source")


# ---------------------------------------------------------------------------
# #14: Email high-water mark timing
# ---------------------------------------------------------------------------
class TestEmailHighWaterMark(unittest.TestCase):
    """Verify email high-water mark is updated AFTER processing."""

    def setUp(self):
        self.source = _read_source("heartbeat.py")

    def test_hw_after_processing_loop(self):
        """email_heartbeat_hw update must come after the for msg_ref loop."""
        lines = self.source.split("\n")
        in_check_email = False
        for_loop_line = None
        hw_update_line = None
        for i, line in enumerate(lines):
            if "async def _check_email" in line:
                in_check_email = True
            if not in_check_email:
                continue
            if "for msg_ref in messages:" in line:
                for_loop_line = i
            if 'email_heartbeat_hw' in line and "now" in line and for_loop_line is not None:
                # There should be a hw update after the for loop
                # (one early return for no-messages case is OK, but the main one must be after)
                if i > for_loop_line:
                    hw_update_line = i
        self.assertIsNotNone(for_loop_line, "for msg_ref loop not found in _check_email")
        self.assertIsNotNone(hw_update_line, "email_heartbeat_hw update not found after processing loop")
        self.assertGreater(hw_update_line, for_loop_line)


if __name__ == "__main__":
    unittest.main()
