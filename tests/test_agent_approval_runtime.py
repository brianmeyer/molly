import asyncio
import inspect
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "memory.processor" not in sys.modules:
    processor_stub = types.ModuleType("memory.processor")

    async def _noop_process_conversation(*_args, **_kwargs):
        return None

    processor_stub.process_conversation = _noop_process_conversation
    sys.modules["memory.processor"] = processor_stub

if "memory.retriever" not in sys.modules:
    retriever_stub = types.ModuleType("memory.retriever")
    async def _noop_retrieve_context(*_args, **_kwargs):
        return ""
    retriever_stub.retrieve_context = _noop_retrieve_context

    class _NoopVectorStore:
        def log_skill_execution(self, *_args, **_kwargs):
            return None

    retriever_stub.get_vectorstore = lambda: _NoopVectorStore()
    sys.modules["memory.retriever"] = retriever_stub

import agent
from approval import ApprovalManager, OWNER_PRIMARY_WHATSAPP_JID, RequestApprovalState
from claude_agent_sdk import CLIConnectionError, PermissionResultAllow, PermissionResultDeny


class _FakeWA:
    def __init__(self, fail_jids: set[str] | None = None, always_fail: bool = False):
        self.sent: list[tuple[str, str]] = []
        self.fail_jids = set(fail_jids or set())
        self.always_fail = always_fail

    def send_message(self, chat_jid: str, text: str):
        self.sent.append((chat_jid, text))
        if self.always_fail or chat_jid in self.fail_jids:
            return None
        return {"chat_jid": chat_jid}


class _FakeMolly:
    def __init__(
        self,
        owner_jid: str = "123@s.whatsapp.net",
        wa: _FakeWA | None = None,
    ):
        self.wa = wa or _FakeWA()
        self.tasks: list[asyncio.Task] = []
        self.owner_jid = owner_jid
        self.registered_chats = {owner_jid: {}}

    def _track_send(self, send_result):
        if inspect.isawaitable(send_result):
            self.tasks.append(asyncio.create_task(send_result))

    def _get_owner_dm_jid(self) -> str:
        return self.owner_jid


class _CancelAwareApproval:
    def __init__(self):
        self.cancel_calls: list[str] = []

    def cancel_pending(self, chat_jid: str):
        self.cancel_calls.append(chat_jid)


class _FakeVectorStore:
    def __init__(self, should_raise: bool = False):
        self.should_raise = should_raise
        self.skill_logs: list[dict[str, str]] = []

    def log_skill_execution(
        self,
        skill_name: str,
        trigger: str,
        outcome: str,
        user_approval: str = "",
        edits_made: str = "",
    ):
        if self.should_raise:
            raise RuntimeError("write failed")
        self.skill_logs.append(
            {
                "skill_name": skill_name,
                "trigger": trigger,
                "outcome": outcome,
                "user_approval": user_approval,
                "edits_made": edits_made,
            }
        )


class TestRequestScopedApproval(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = ApprovalManager()
        self.molly = _FakeMolly()
        self.chat = "123@s.whatsapp.net"

    async def asyncTearDown(self):
        if self.molly.tasks:
            await asyncio.gather(*self.molly.tasks, return_exceptions=True)

    async def test_non_whatsapp_reroute_logs_expected_message(self):
        with self.assertLogs("approval", level="INFO") as logs:
            resolved = self.manager._resolve_response_chat_jid("web:8d94e06e", self.molly)

        self.assertEqual(resolved, OWNER_PRIMARY_WHATSAPP_JID)
        self.assertTrue(
            any(
                "[approval INFO] - Rerouting approval to WhatsApp (original chat: web:8d94e06e)"
                in entry
                for entry in logs.output
            )
        )

    async def test_coalesces_concurrent_bash_approvals(self):
        state = RequestApprovalState()
        tasks = [
            asyncio.create_task(
                self.manager.request_tool_approval(
                    "Bash",
                    {"command": "echo hello"},
                    self.chat,
                    self.molly,
                    request_state=state,
                )
            )
            for _ in range(5)
        ]

        await asyncio.sleep(0.05)
        self.assertEqual(len(self.molly.wa.sent), 1)
        self.assertTrue(self.manager.has_pending(self.chat))

        consumed = self.manager.try_resolve("YES", self.chat)
        self.assertTrue(consumed)

        results = await asyncio.gather(*tasks)
        self.assertEqual(results, [True, True, True, True, True])
        self.assertEqual(state.tool_asks, 5)
        self.assertEqual(state.prompts_sent, 1)
        self.assertEqual(state.auto_approved, 4)
        self.assertFalse(state.approved_all_confirm)

    async def test_all_only_applies_within_request_state(self):
        state = RequestApprovalState()
        first = asyncio.create_task(
            self.manager.request_tool_approval(
                "Bash",
                {"command": "ls"},
                self.chat,
                self.molly,
                request_state=state,
            )
        )

        await asyncio.sleep(0.05)
        self.assertEqual(len(self.molly.wa.sent), 1)
        consumed = self.manager.try_resolve("ALL", self.chat)
        self.assertTrue(consumed)
        self.assertTrue(await first)
        self.assertTrue(state.approved_all_confirm)
        self.assertEqual(state.tool_asks, 1)
        self.assertEqual(state.prompts_sent, 1)
        self.assertEqual(state.auto_approved, 0)

        # Same request state: no second prompt for another CONFIRM tool.
        second = await self.manager.request_tool_approval(
            "Write",
            {"file_path": "/tmp/test.txt", "content": "hello"},
            self.chat,
            self.molly,
            request_state=state,
        )
        self.assertTrue(second)
        self.assertEqual(len(self.molly.wa.sent), 1)

        # New request state: approval is required again.
        new_state = RequestApprovalState()
        third = asyncio.create_task(
            self.manager.request_tool_approval(
                "Bash",
                {"command": "pwd"},
                self.chat,
                self.molly,
                request_state=new_state,
            )
        )
        await asyncio.sleep(0.05)
        self.assertEqual(len(self.molly.wa.sent), 2)
        consumed = self.manager.try_resolve("YES", self.chat)
        self.assertTrue(consumed)
        self.assertTrue(await third)

    async def test_deny_propagates_to_coalesced_waiters(self):
        state = RequestApprovalState()
        tasks = [
            asyncio.create_task(
                self.manager.request_tool_approval(
                    "Bash",
                    {"command": "echo deny"},
                    self.chat,
                    self.molly,
                    request_state=state,
                )
            )
            for _ in range(4)
        ]

        await asyncio.sleep(0.05)
        self.assertEqual(len(self.molly.wa.sent), 1)
        consumed = self.manager.try_resolve("NO", self.chat)
        self.assertTrue(consumed)

        results = await asyncio.gather(*tasks)
        self.assertEqual(results, [False, False, False, False])
        self.assertEqual(state.tool_asks, 4)
        self.assertEqual(state.prompts_sent, 1)
        self.assertEqual(state.auto_approved, 0)

    async def test_timeout_is_coalesced_to_one_prompt(self):
        state = RequestApprovalState()
        tasks = []
        with patch("approval.config.APPROVAL_TIMEOUT", 0.05):
            for _ in range(3):
                tasks.append(
                    asyncio.create_task(
                        self.manager.request_tool_approval(
                            "Bash",
                            {"command": "sleep 1"},
                            self.chat,
                            self.molly,
                            request_state=state,
                        )
                    )
                )
            results = await asyncio.gather(*tasks, return_exceptions=True)

        normalized: list[bool] = []
        for result in results:
            if isinstance(result, asyncio.CancelledError):
                normalized.append(False)
                continue
            if isinstance(result, Exception):
                raise result
            normalized.append(bool(result))

        self.assertEqual(normalized, [False, False, False])
        self.assertEqual(state.tool_asks, 3)
        self.assertEqual(state.prompts_sent, 1)
        self.assertEqual(state.auto_approved, 0)
        self.assertIn("Bash", state.denied_tools)

    async def test_web_request_routes_to_owner_whatsapp_for_resolution(self):
        owner_chat = OWNER_PRIMARY_WHATSAPP_JID
        self.molly.owner_jid = owner_chat
        self.molly.registered_chats = {owner_chat: {}}
        web_chat = "web:8d94e06e"
        state = RequestApprovalState()

        approval_task = asyncio.create_task(
            self.manager.request_tool_approval(
                "Bash",
                {"command": "pwd"},
                web_chat,
                self.molly,
                request_state=state,
            )
        )

        await asyncio.sleep(0.05)
        self.assertEqual(len(self.molly.wa.sent), 1)
        routed_chat, _message = self.molly.wa.sent[0]
        self.assertEqual(routed_chat, owner_chat)
        self.assertTrue(self.manager.has_pending(web_chat))
        self.assertTrue(self.manager.has_pending(owner_chat))

        consumed = self.manager.try_resolve("YES", owner_chat)
        self.assertTrue(consumed)
        self.assertTrue(await approval_task)
        self.assertFalse(self.manager.has_pending(web_chat))

    async def test_imessage_request_routes_to_primary_owner_whatsapp(self):
        owner_chat = OWNER_PRIMARY_WHATSAPP_JID
        self.molly.owner_jid = owner_chat
        self.molly.registered_chats = {owner_chat: {}}
        imessage_chat = "imessage:thread:abc123"
        state = RequestApprovalState()

        approval_task = asyncio.create_task(
            self.manager.request_tool_approval(
                "Bash",
                {"command": "pwd"},
                imessage_chat,
                self.molly,
                request_state=state,
            )
        )

        await asyncio.sleep(0.05)
        self.assertEqual(len(self.molly.wa.sent), 1)
        routed_chat, _ = self.molly.wa.sent[0]
        self.assertEqual(routed_chat, OWNER_PRIMARY_WHATSAPP_JID)

        consumed = self.manager.try_resolve("YES", OWNER_PRIMARY_WHATSAPP_JID)
        self.assertTrue(consumed)
        self.assertTrue(await approval_task)

    async def test_whatsapp_request_stays_on_origin_chat(self):
        wa_chat = "15857332025@s.whatsapp.net"
        state = RequestApprovalState()

        approval_task = asyncio.create_task(
            self.manager.request_tool_approval(
                "Bash",
                {"command": "pwd"},
                wa_chat,
                self.molly,
                request_state=state,
            )
        )

        await asyncio.sleep(0.05)
        self.assertEqual(len(self.molly.wa.sent), 1)
        routed_chat, _ = self.molly.wa.sent[0]
        self.assertEqual(routed_chat, wa_chat)

        consumed = self.manager.try_resolve("YES", wa_chat)
        self.assertTrue(consumed)
        self.assertTrue(await approval_task)

    async def test_send_failure_retries_owner_fallback_before_timeout(self):
        fallback_owner = "52660963176533@lid"
        failing_wa = _FakeWA(fail_jids={OWNER_PRIMARY_WHATSAPP_JID})
        self.molly = _FakeMolly(owner_jid=fallback_owner, wa=failing_wa)
        self.chat = "web:8d94e06e"
        state = RequestApprovalState()

        approval_task = asyncio.create_task(
            self.manager.request_tool_approval(
                "Bash",
                {"command": "pwd"},
                self.chat,
                self.molly,
                request_state=state,
            )
        )

        await asyncio.sleep(0.05)
        sent_targets = [jid for jid, _ in self.molly.wa.sent]
        self.assertIn(OWNER_PRIMARY_WHATSAPP_JID, sent_targets)
        self.assertIn(fallback_owner, sent_targets)
        self.assertGreaterEqual(len(sent_targets), 2)

        consumed = self.manager.try_resolve("YES", fallback_owner)
        self.assertTrue(consumed)
        self.assertTrue(await approval_task)

    async def test_delivery_failure_returns_false_without_waiting_for_timeout(self):
        self.molly = _FakeMolly(
            owner_jid=OWNER_PRIMARY_WHATSAPP_JID,
            wa=_FakeWA(always_fail=True),
        )
        web_chat = "web:8d94e06e"
        state = RequestApprovalState()

        t0 = time.monotonic()
        with patch("approval.config.APPROVAL_TIMEOUT", 5):
            result = await self.manager.request_tool_approval(
                "Bash",
                {"command": "pwd"},
                web_chat,
                self.molly,
                request_state=state,
            )
        elapsed = time.monotonic() - t0

        self.assertFalse(result)
        self.assertLess(elapsed, 1.0)
        self.assertFalse(self.manager.has_pending(web_chat))
        self.assertEqual(state.prompts_sent, 0)

    async def test_custom_approval_accepts_edit_with_space_format(self):
        task = asyncio.create_task(
            self.manager.request_custom_approval(
                category="self-improve-skill",
                description="proposal",
                chat_jid=self.chat,
                molly=self.molly,
                required_keyword="YES",
                allow_edit=True,
            )
        )
        await asyncio.sleep(0.05)
        consumed = self.manager.try_resolve("EDIT trigger: use weekly summary flow", self.chat)
        self.assertTrue(consumed)
        result = await task
        self.assertEqual(result, "trigger: use weekly summary flow")

    async def test_custom_approval_can_return_denial_reason(self):
        task = asyncio.create_task(
            self.manager.request_custom_approval(
                category="self-improve-skill",
                description="proposal",
                chat_jid=self.chat,
                molly=self.molly,
                required_keyword="YES",
                allow_edit=True,
                return_reasoned_denial=True,
            )
        )
        await asyncio.sleep(0.05)
        consumed = self.manager.try_resolve("NO: keep current version", self.chat)
        self.assertTrue(consumed)
        result = await task
        self.assertEqual(result, ("deny", "keep current version"))


class TestAgentRuntimeAndRetry(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        agent._CHAT_RUNTIMES.clear()

    async def test_retries_once_on_transport_close(self):
        call_count = 0
        approval = _CancelAwareApproval()

        async def fake_query(_runtime, _turn_prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise CLIConnectionError("ProcessTransport is not ready for writing")
            return "Recovered", "session-recovered"

        with patch("agent.load_identity_stack", return_value="identity"), \
                patch("agent.retrieve_context", new=AsyncMock(return_value="")), \
                patch("agent.match_skills", return_value=[]), \
                patch("agent.get_skill_context", return_value=""), \
                patch("agent._ensure_connected_runtime", new=AsyncMock()), \
                patch("agent._disconnect_runtime", new=AsyncMock()) as disconnect_mock, \
                patch("agent._query_with_client", side_effect=fake_query), \
                patch("agent.process_conversation", new=AsyncMock()):
            response, session_id = await agent.handle_message(
                "hello",
                "chat-retry",
                approval_manager=approval,
            )

        self.assertEqual(response, "Recovered")
        self.assertEqual(session_id, "session-recovered")
        self.assertEqual(call_count, 2)
        self.assertGreaterEqual(disconnect_mock.await_count, 1)
        self.assertEqual(approval.cancel_calls, ["chat-retry"])

    async def test_serializes_same_chat_requests(self):
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        query_calls = 0

        async def fake_query(_runtime, _turn_prompt):
            nonlocal query_calls
            query_calls += 1
            if query_calls == 1:
                first_started.set()
                await release_first.wait()
                return "first-response", "session-1"
            return "second-response", "session-2"

        with patch("agent.load_identity_stack", return_value="identity"), \
                patch("agent.retrieve_context", new=AsyncMock(return_value="")), \
                patch("agent.match_skills", return_value=[]), \
                patch("agent.get_skill_context", return_value=""), \
                patch("agent._ensure_connected_runtime", new=AsyncMock()), \
                patch("agent._disconnect_runtime", new=AsyncMock()), \
                patch("agent._query_with_client", side_effect=fake_query), \
                patch("agent.process_conversation", new=AsyncMock()):
            first_task = asyncio.create_task(agent.handle_message("msg-1", "chat-queue"))
            await asyncio.wait_for(first_started.wait(), timeout=1)

            second_task = asyncio.create_task(agent.handle_message("msg-2", "chat-queue"))
            await asyncio.sleep(0.05)
            self.assertEqual(query_calls, 1, "Second request should wait for the chat lock")

            release_first.set()
            first_result, second_result = await asyncio.gather(first_task, second_task)

        self.assertEqual(first_result[0], "first-response")
        self.assertEqual(second_result[0], "second-response")
        self.assertEqual(query_calls, 2)

    async def test_retry_clears_denied_tools_before_second_attempt(self):
        observed_denied_sets: list[set[str]] = []

        async def fake_query(runtime, _turn_prompt):
            observed_denied_sets.append(set(runtime.request_state.denied_tools))
            if len(observed_denied_sets) == 1:
                runtime.request_state.denied_tools.add("Bash")
                raise CLIConnectionError("Stream closed")
            return "Recovered", "session-ok"

        with patch("agent.load_identity_stack", return_value="identity"), \
                patch("agent.retrieve_context", new=AsyncMock(return_value="")), \
                patch("agent.match_skills", return_value=[]), \
                patch("agent.get_skill_context", return_value=""), \
                patch("agent._ensure_connected_runtime", new=AsyncMock()), \
                patch("agent._disconnect_runtime", new=AsyncMock()), \
                patch("agent._query_with_client", side_effect=fake_query), \
                patch("agent.process_conversation", new=AsyncMock()):
            response, session_id = await agent.handle_message("retry", "chat-denied-reset")

        self.assertEqual(response, "Recovered")
        self.assertEqual(session_id, "session-ok")
        self.assertEqual(observed_denied_sets, [set(), set()])


class TestSkillExecutionObservability(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        agent._CHAT_RUNTIMES.clear()

    async def test_logs_success_outcome_for_matched_skills(self):
        fake_vs = _FakeVectorStore()
        matched = [
            type("Skill", (), {"name": "daily-digest"})(),
            type("Skill", (), {"name": "meeting-prep"})(),
        ]

        with patch("agent.load_identity_stack", return_value="identity"), \
                patch("agent.retrieve_context", new=AsyncMock(return_value="")), \
                patch("agent.match_skills", return_value=matched), \
                patch("agent.get_skill_context", return_value="skill-context"), \
                patch("agent._ensure_connected_runtime", new=AsyncMock()), \
                patch("agent._disconnect_runtime", new=AsyncMock()), \
                patch("agent._query_with_client", return_value=("ok", "session-skills")), \
                patch("agent.process_conversation", new=AsyncMock()), \
                patch("memory.retriever.get_vectorstore", return_value=fake_vs):
            response, session_id = await agent.handle_message(
                "what's on today?",
                "chat-skill-success",
                source="whatsapp",
            )
            await asyncio.sleep(0.05)

        self.assertEqual(response, "ok")
        self.assertEqual(session_id, "session-skills")
        self.assertEqual(len(fake_vs.skill_logs), 2)
        self.assertEqual(
            {row["skill_name"] for row in fake_vs.skill_logs},
            {"daily-digest", "meeting-prep"},
        )
        self.assertTrue(all(row["outcome"] == "success" for row in fake_vs.skill_logs))

    async def test_logs_failure_outcome_for_matched_skills(self):
        fake_vs = _FakeVectorStore()
        matched = [type("Skill", (), {"name": "daily-digest"})()]

        with patch("agent.load_identity_stack", return_value="identity"), \
                patch("agent.retrieve_context", new=AsyncMock(return_value="")), \
                patch("agent.match_skills", return_value=matched), \
                patch("agent.get_skill_context", return_value="skill-context"), \
                patch("agent._ensure_connected_runtime", new=AsyncMock()), \
                patch("agent._disconnect_runtime", new=AsyncMock()), \
                patch("agent._query_with_client", side_effect=RuntimeError("boom")), \
                patch("agent.process_conversation", new=AsyncMock()), \
                patch("memory.retriever.get_vectorstore", return_value=fake_vs):
            response, _session_id = await agent.handle_message(
                "run digest",
                "chat-skill-failure",
                source="whatsapp",
            )
            await asyncio.sleep(0.05)

        self.assertTrue(response.startswith("Something went wrong"))
        self.assertEqual(len(fake_vs.skill_logs), 1)
        self.assertEqual(fake_vs.skill_logs[0]["skill_name"], "daily-digest")
        self.assertEqual(fake_vs.skill_logs[0]["outcome"], "failure")
        self.assertIn("RuntimeError", fake_vs.skill_logs[0]["edits_made"])

    async def test_skill_logging_failure_never_breaks_response(self):
        fake_vs = _FakeVectorStore(should_raise=True)
        matched = [type("Skill", (), {"name": "daily-digest"})()]

        with patch("agent.load_identity_stack", return_value="identity"), \
                patch("agent.retrieve_context", new=AsyncMock(return_value="")), \
                patch("agent.match_skills", return_value=matched), \
                patch("agent.get_skill_context", return_value="skill-context"), \
                patch("agent._ensure_connected_runtime", new=AsyncMock()), \
                patch("agent._disconnect_runtime", new=AsyncMock()), \
                patch("agent._query_with_client", return_value=("Recovered", "session-ok")), \
                patch("agent.process_conversation", new=AsyncMock()), \
                patch("memory.retriever.get_vectorstore", return_value=fake_vs):
            response, session_id = await agent.handle_message(
                "what's on today?",
                "chat-skill-log-failure",
                source="whatsapp",
            )
            await asyncio.sleep(0.05)

        self.assertEqual(response, "Recovered")
        self.assertEqual(session_id, "session-ok")


class TestToolCheckerNegativeCases(unittest.IsolatedAsyncioTestCase):
    async def test_blocked_tool_not_overridden_by_all_grant(self):
        state = RequestApprovalState(approved_all_confirm=True)
        checker = agent.make_tool_checker(
            approval_manager=object(),
            molly=object(),
            chat_jid="chat",
            request_state=state,
        )

        result = await checker("mcp__gmail__gmail_delete", {"message_id": "1"}, None)
        self.assertIsInstance(result, PermissionResultDeny)
        self.assertEqual(result.behavior, "deny")
        self.assertIn("blocked", result.message.lower())


class TestChatRuntimeEviction(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        agent._CHAT_RUNTIMES.clear()

    async def test_evicts_idle_runtime(self):
        runtime = agent._ChatRuntime(
            last_used_monotonic=time.monotonic() - (agent._CHAT_RUNTIME_IDLE_SECONDS + 5),
        )
        agent._CHAT_RUNTIMES["idle-chat"] = runtime

        evicted = await agent._evict_stale_chat_runtimes()
        self.assertEqual(evicted, 1)
        self.assertNotIn("idle-chat", agent._CHAT_RUNTIMES)

    async def test_does_not_evict_locked_runtime(self):
        runtime = agent._ChatRuntime(
            last_used_monotonic=time.monotonic() - (agent._CHAT_RUNTIME_IDLE_SECONDS + 5),
        )
        await runtime.lock.acquire()
        try:
            agent._CHAT_RUNTIMES["busy-chat"] = runtime
            evicted = await agent._evict_stale_chat_runtimes()
        finally:
            runtime.lock.release()

        self.assertEqual(evicted, 0)
        self.assertIn("busy-chat", agent._CHAT_RUNTIMES)


class TestAgentUsesClaudeSDKClient(unittest.TestCase):
    def test_agent_uses_claude_sdk_client_path(self):
        source = (PROJECT_ROOT / "agent.py").read_text()
        self.assertIn("ClaudeSDKClient", source)
        self.assertNotIn("query(prompt=", source)
        self.assertIn("stderr=_handle_sdk_stderr", source)

    def test_approval_metric_log_format(self):
        state = RequestApprovalState(
            request_id="abc123",
            tool_asks=5,
            prompts_sent=1,
            auto_approved=4,
            approved_all_confirm=False,
        )
        with patch("agent.log.info") as info_mock:
            agent._emit_approval_metrics(state)

        info_mock.assert_called_once()
        template = info_mock.call_args[0][0]
        values = info_mock.call_args[0][1:]
        rendered = template % values
        self.assertIn("request_id=abc123", rendered)
        self.assertIn("tool_asks=5", rendered)
        self.assertIn("prompts_sent=1", rendered)
        self.assertIn("auto_approved=4", rendered)
        self.assertIn("all_grant=false", rendered)

    def test_stderr_suppression_handler(self):
        with patch("agent.log.debug") as debug_mock, patch("agent.log.warning") as warn_mock:
            agent._handle_sdk_stderr("error: Stream closed")
            agent._handle_sdk_stderr("regular stderr line")

        debug_mock.assert_called_once()
        warn_mock.assert_called_once()


class TestDependencyPins(unittest.TestCase):
    def test_claude_agent_sdk_minimum_pin(self):
        requirements = (PROJECT_ROOT / "requirements.txt").read_text()
        self.assertIn("claude-agent-sdk>=0.1.33", requirements)


class TestSafeBashAutoApproval(unittest.IsolatedAsyncioTestCase):
    """Tests for BUG-08: safe Bash command auto-approval."""

    # ---- Unit tests for is_safe_bash_command ----

    def test_simple_safe_commands_are_approved(self):
        """Basic read-only commands should be classified as safe."""
        from approval import is_safe_bash_command

        safe_commands = [
            "ls", "ls -la", "ls -la /tmp",
            "pwd",
            "cat /etc/hosts",
            "head -n 10 file.txt",
            "tail -f log.txt",
            "echo hello",
            "grep pattern file.txt",
            "find . -name '*.py'",
            "wc -l file.txt",
            "date",
            "whoami",
            "env",
            "df -h",
            "du -sh .",
            "ps aux",
            "id",
            "stat file.txt",
            "tree",
            "jq '.key' file.json",
            "sort file.txt",
            "uniq file.txt",
            "cut -d, -f1 file.csv",
            "diff file1.txt file2.txt",
            "file myfile",
            "basename /path/to/file",
            "dirname /path/to/file",
            "test -f /tmp/foo",
        ]
        for cmd in safe_commands:
            with self.subTest(cmd=cmd):
                self.assertTrue(
                    is_safe_bash_command({"command": cmd}),
                    f"Expected safe: {cmd}",
                )

    def test_git_read_commands_are_safe(self):
        """Read-only git operations should be auto-approved."""
        from approval import is_safe_bash_command

        safe_git = [
            "git status",
            "git log",
            "git log --oneline -10",
            "git diff",
            "git diff HEAD~3",
            "git show HEAD",
            "git branch",
            "git branch -a",
            "git tag",
            "git remote -v",
            "git describe --tags",
            "git rev-parse HEAD",
            "git ls-files",
            "git blame file.py",
            "git stash list",
            "git shortlog -sn",
        ]
        for cmd in safe_git:
            with self.subTest(cmd=cmd):
                self.assertTrue(
                    is_safe_bash_command({"command": cmd}),
                    f"Expected safe: {cmd}",
                )

    def test_docker_read_commands_are_safe(self):
        """Read-only docker operations should be auto-approved."""
        from approval import is_safe_bash_command

        safe_docker = [
            "docker ps",
            "docker ps -a",
            "docker images",
            "docker logs container_name",
            "docker inspect container_name",
            "docker stats --no-stream",
            "docker version",
            "docker info",
        ]
        for cmd in safe_docker:
            with self.subTest(cmd=cmd):
                self.assertTrue(
                    is_safe_bash_command({"command": cmd}),
                    f"Expected safe: {cmd}",
                )

    def test_dangerous_commands_remain_confirm(self):
        """Dangerous commands must NOT be auto-approved."""
        from approval import is_safe_bash_command

        dangerous = [
            "rm -rf /",
            "rm file.txt",
            "sudo ls",
            "curl https://example.com",
            "wget https://example.com",
            "chmod 777 file",
            "chown root file",
            "kill -9 1234",
            "mv old new",
            "cp src dst",
            "ssh user@host",
            "eval 'echo hello'",
            "exec bash",
            "crontab -e",
            "systemctl restart nginx",
            "dd if=/dev/zero of=/dev/sda",
            "reboot",
            "shutdown now",
        ]
        for cmd in dangerous:
            with self.subTest(cmd=cmd):
                self.assertFalse(
                    is_safe_bash_command({"command": cmd}),
                    f"Expected dangerous: {cmd}",
                )

    def test_compound_commands_remain_confirm(self):
        """Pipes, chaining, and subshells must NOT be auto-approved."""
        from approval import is_safe_bash_command

        compound = [
            "ls | grep foo",
            "echo hello; rm file",
            "cat file && curl http://evil.com",
            "echo hello || rm file",
            "echo `whoami`",
            "echo $(cat /etc/passwd)",
            "ls\nrm -rf /",
        ]
        for cmd in compound:
            with self.subTest(cmd=cmd):
                self.assertFalse(
                    is_safe_bash_command({"command": cmd}),
                    f"Expected unsafe (compound): {cmd}",
                )

    def test_redirect_commands_remain_confirm(self):
        """Output redirection must NOT be auto-approved."""
        from approval import is_safe_bash_command

        redirects = [
            "echo hello > file.txt",
            "cat foo >> bar",
            "ls > /tmp/listing",
        ]
        for cmd in redirects:
            with self.subTest(cmd=cmd):
                self.assertFalse(
                    is_safe_bash_command({"command": cmd}),
                    f"Expected unsafe (redirect): {cmd}",
                )

    def test_unknown_commands_remain_confirm(self):
        """Unrecognized commands must NOT be auto-approved."""
        from approval import is_safe_bash_command

        unknown = [
            "mycustomtool --flag",
            "python script.py",
            "node server.js",
            "make build",
            "cargo run",
        ]
        for cmd in unknown:
            with self.subTest(cmd=cmd):
                self.assertFalse(
                    is_safe_bash_command({"command": cmd}),
                    f"Expected unsafe (unknown): {cmd}",
                )

    def test_git_write_commands_remain_confirm(self):
        """Git write operations should NOT be auto-approved."""
        from approval import is_safe_bash_command

        git_writes = [
            "git commit -m 'test'",
            "git push",
            "git push origin main",
            "git checkout main",
            "git merge branch",
            "git rebase main",
            "git reset --hard HEAD",
            "git clean -fd",
            "git stash",
            "git stash pop",
            "git add .",
            "git add file.py",
            "git pull",
            "git clone https://example.com",
        ]
        for cmd in git_writes:
            with self.subTest(cmd=cmd):
                self.assertFalse(
                    is_safe_bash_command({"command": cmd}),
                    f"Expected unsafe (git write): {cmd}",
                )

    def test_empty_and_none_inputs(self):
        """Edge cases: empty, None, missing command key."""
        from approval import is_safe_bash_command

        self.assertFalse(is_safe_bash_command(None))
        self.assertFalse(is_safe_bash_command({}))
        self.assertFalse(is_safe_bash_command({"command": ""}))
        self.assertFalse(is_safe_bash_command({"command": "   "}))
        self.assertFalse(is_safe_bash_command({"not_command": "ls"}))

    def test_malformed_quoting_remains_confirm(self):
        """Malformed shell quoting should fail safely to CONFIRM."""
        from approval import is_safe_bash_command

        self.assertFalse(is_safe_bash_command({"command": "echo 'unterminated"}))
        self.assertFalse(is_safe_bash_command({"command": 'cat "no close'}))

    def test_process_substitution_rejected(self):
        """Process substitution <(...) and >(...) must be rejected."""
        from approval import is_safe_bash_command

        self.assertFalse(is_safe_bash_command({"command": "cat <(echo test)"}))
        self.assertFalse(is_safe_bash_command({"command": "diff <(ls dir1) <(ls dir2)"}))

    def test_input_redirect_rejected(self):
        """Input redirection < must be rejected."""
        from approval import is_safe_bash_command

        self.assertFalse(is_safe_bash_command({"command": "cat < /etc/shadow"}))
        self.assertFalse(is_safe_bash_command({"command": "grep pattern < file.txt"}))
        self.assertFalse(is_safe_bash_command({"command": "cat <<< 'here-string'"}))

    def test_stderr_redirect_rejected(self):
        """Stderr redirects (2>, &>) must be rejected."""
        from approval import is_safe_bash_command

        self.assertFalse(is_safe_bash_command({"command": "cat /etc/passwd 2> /tmp/x"}))
        self.assertFalse(is_safe_bash_command({"command": "echo hello &> /tmp/both"}))

    def test_version_check_commands_are_safe(self):
        """Version check commands should be auto-approved."""
        from approval import is_safe_bash_command

        version_cmds = [
            "npm --version",
            "pip --version",
            "pip3 --version",
            "git --version",
            "docker --version",
            "python --version",
            "python3 --version",
            "node --version",
        ]
        for cmd in version_cmds:
            with self.subTest(cmd=cmd):
                self.assertTrue(
                    is_safe_bash_command({"command": cmd}),
                    f"Expected safe: {cmd}",
                )

    def test_unusual_whitespace_handled(self):
        """Commands with tabs and extra spaces should work."""
        from approval import is_safe_bash_command

        self.assertTrue(is_safe_bash_command({"command": "ls    -la"}))
        self.assertTrue(is_safe_bash_command({"command": "  git   status  "}))

    def test_null_byte_in_command_rejected(self):
        """Null bytes in commands should be rejected."""
        from approval import is_safe_bash_command

        self.assertFalse(is_safe_bash_command({"command": "ls\x00rm -rf /"}))

    def test_command_substitution_variants_rejected(self):
        """All forms of command substitution should be rejected."""
        from approval import is_safe_bash_command

        self.assertFalse(is_safe_bash_command({"command": "echo $(whoami)"}))
        self.assertFalse(is_safe_bash_command({"command": "echo `whoami`"}))
        self.assertFalse(is_safe_bash_command({"command": "echo ${PATH}"}))

    def test_find_exec_rejected(self):
        """find -exec enables arbitrary command execution and must be rejected."""
        from approval import is_safe_bash_command

        dangerous_finds = [
            "find . -exec python3 -c 'import os' {} +",
            "find . -exec node -e 'process.exit()' {} +",
            "find . -execdir make -f evil {} +",
            "find . -ok rm {} +",
            "find . -okdir rm {} +",
        ]
        for cmd in dangerous_finds:
            with self.subTest(cmd=cmd):
                self.assertFalse(
                    is_safe_bash_command({"command": cmd}),
                    f"Expected unsafe (find -exec): {cmd}",
                )

    def test_find_without_exec_is_safe(self):
        """Normal find without -exec should remain safe."""
        from approval import is_safe_bash_command

        safe_finds = [
            "find . -name '*.py'",
            "find . -type f -mtime +30",
            "find /tmp -maxdepth 1",
        ]
        for cmd in safe_finds:
            with self.subTest(cmd=cmd):
                self.assertTrue(
                    is_safe_bash_command({"command": cmd}),
                    f"Expected safe: {cmd}",
                )

    def test_non_string_command_types_rejected(self):
        """Non-string command values should be rejected, not crash."""
        from approval import is_safe_bash_command

        self.assertFalse(is_safe_bash_command({"command": 123}))
        self.assertFalse(is_safe_bash_command({"command": ["ls"]}))
        self.assertFalse(is_safe_bash_command({"command": None}))
        self.assertFalse(is_safe_bash_command({"command": {"nested": "dict"}}))

    # ---- Unit tests for is_bash_workspace_safe ----

    def test_workspace_mkdir_is_safe(self):
        """mkdir within workspace should be auto-approved."""
        from approval import is_bash_workspace_safe

        import config

        ws = str(config.WORKSPACE)
        safe = [
            f"mkdir {ws}/memory/new_dir",
            f"mkdir -p {ws}/memory/deep/nested/dir",
            f"mkdir {ws}/skills/new_skill",
            f"mkdir {ws}/foundry/observations/2024",
            f"mkdir -p {ws}/sandbox/test",
        ]
        for cmd in safe:
            with self.subTest(cmd=cmd):
                self.assertTrue(
                    is_bash_workspace_safe({"command": cmd}),
                    f"Expected workspace-safe: {cmd}",
                )

    def test_workspace_touch_is_safe(self):
        """touch within workspace should be auto-approved."""
        from approval import is_bash_workspace_safe

        import config

        ws = str(config.WORKSPACE)
        safe = [
            f"touch {ws}/memory/file.json",
            f"touch {ws}/skills/new_skill.md",
            f"touch {ws}/foundry/observations/new.jsonl",
        ]
        for cmd in safe:
            with self.subTest(cmd=cmd):
                self.assertTrue(
                    is_bash_workspace_safe({"command": cmd}),
                    f"Expected workspace-safe: {cmd}",
                )

    def test_workspace_write_outside_workspace_rejected(self):
        """mkdir/touch outside workspace must be CONFIRM."""
        from approval import is_bash_workspace_safe

        unsafe = [
            "mkdir /tmp/something",
            "touch /tmp/file.txt",
            "mkdir /etc/new_dir",
            "touch /etc/new_file",
        ]
        for cmd in unsafe:
            with self.subTest(cmd=cmd):
                self.assertFalse(
                    is_bash_workspace_safe({"command": cmd}),
                    f"Expected unsafe (outside workspace): {cmd}",
                )

    def test_workspace_write_path_escape_rejected(self):
        """Path traversal (../) must not escape workspace."""
        from approval import is_bash_workspace_safe

        import config

        ws = str(config.WORKSPACE)
        escapes = [
            f"mkdir {ws}/../../../etc/evil",
            f"touch {ws}/memory/../../SOUL.md",
        ]
        for cmd in escapes:
            with self.subTest(cmd=cmd):
                self.assertFalse(
                    is_bash_workspace_safe({"command": cmd}),
                    f"Expected unsafe (path escape): {cmd}",
                )

    def test_workspace_disallowed_commands_rejected(self):
        """rm/cp/mv to workspace should NOT be auto-approved."""
        from approval import is_bash_workspace_safe

        import config

        ws = str(config.WORKSPACE)
        disallowed = [
            f"rm {ws}/memory/file.json",
            f"cp src {ws}/memory/dst",
            f"mv old {ws}/memory/new",
        ]
        for cmd in disallowed:
            with self.subTest(cmd=cmd):
                self.assertFalse(
                    is_bash_workspace_safe({"command": cmd}),
                    f"Expected unsafe (disallowed command): {cmd}",
                )

    def test_workspace_bare_mkdir_no_path_rejected(self):
        """mkdir with no path argument should be rejected."""
        from approval import is_bash_workspace_safe

        self.assertFalse(is_bash_workspace_safe({"command": "mkdir"}))
        self.assertFalse(is_bash_workspace_safe({"command": "mkdir -p"}))
        self.assertFalse(is_bash_workspace_safe({"command": "touch"}))

    def test_workspace_compound_rejected(self):
        """Compound commands targeting workspace must still be CONFIRM."""
        from approval import is_bash_workspace_safe

        import config

        ws = str(config.WORKSPACE)
        self.assertFalse(is_bash_workspace_safe({"command": f"mkdir {ws}/a; rm -rf /"}))
        self.assertFalse(is_bash_workspace_safe({"command": f"touch {ws}/f | cat"}))

    def test_workspace_tilde_outside_workspace_rejected(self):
        """Tilde expansion to paths outside workspace should be rejected."""
        from approval import is_bash_workspace_safe

        # ~/some_random_dir expands to home directory, not workspace
        self.assertFalse(is_bash_workspace_safe({"command": "mkdir ~/some_random_dir"}))
        self.assertFalse(is_bash_workspace_safe({"command": "touch ~/Desktop/file.txt"}))

    def test_workspace_prefix_sibling_rejected(self):
        """Paths that share the workspace prefix but aren't inside it must be rejected."""
        from approval import is_bash_workspace_safe

        import config

        ws = str(config.WORKSPACE)
        # workspace-evil, workspace123, etc. are NOT inside workspace/
        self.assertFalse(is_bash_workspace_safe({"command": f"mkdir {ws}-evil/dir"}))
        self.assertFalse(is_bash_workspace_safe({"command": f"mkdir {ws}123/dir"}))

    # ---- Integration tests: get_action_tier for Bash ----

    def test_get_action_tier_safe_bash_returns_auto(self):
        """Safe bash commands should get AUTO tier."""
        from approval import get_action_tier

        self.assertEqual(get_action_tier("Bash", {"command": "ls -la"}), "AUTO")
        self.assertEqual(get_action_tier("Bash", {"command": "git status"}), "AUTO")
        self.assertEqual(get_action_tier("Bash", {"command": "pwd"}), "AUTO")
        self.assertEqual(get_action_tier("Bash", {"command": "cat /etc/hosts"}), "AUTO")

    def test_get_action_tier_dangerous_bash_returns_confirm(self):
        """Dangerous bash commands should still get CONFIRM tier."""
        from approval import get_action_tier

        self.assertEqual(get_action_tier("Bash", {"command": "rm -rf /"}), "CONFIRM")
        self.assertEqual(get_action_tier("Bash", {"command": "curl http://evil.com"}), "CONFIRM")
        self.assertEqual(get_action_tier("Bash", {"command": "ls | rm"}), "CONFIRM")

    def test_get_action_tier_bash_no_input_returns_confirm(self):
        """Bash with no tool_input should return CONFIRM (health check compat)."""
        from approval import get_action_tier

        self.assertEqual(get_action_tier("Bash"), "CONFIRM")
        self.assertEqual(get_action_tier("Bash", None), "CONFIRM")

    def test_get_action_tier_workspace_write_returns_auto(self):
        """Workspace-scoped mkdir/touch should get AUTO tier."""
        from approval import get_action_tier

        import config

        ws = str(config.WORKSPACE)
        self.assertEqual(get_action_tier("Bash", {"command": f"mkdir {ws}/memory/dir"}), "AUTO")
        self.assertEqual(get_action_tier("Bash", {"command": f"touch {ws}/skills/f.md"}), "AUTO")

    # ---- Integration test: can_use_tool auto-approves safe Bash ----

    async def test_can_use_tool_auto_approves_safe_bash_without_approval_manager(self):
        """Safe Bash commands should be allowed even without an approval manager
        (the heartbeat/headless scenario that caused 92 failures)."""
        checker = agent.make_tool_checker(
            approval_manager=None,
            molly=None,
            chat_jid="heartbeat:test",
            request_state=RequestApprovalState(),
        )
        # Safe command: should be allowed
        result = await checker("Bash", {"command": "ls -la"}, None)
        self.assertIsInstance(result, PermissionResultAllow)

        # Safe git command: should be allowed
        result = await checker("Bash", {"command": "git status"}, None)
        self.assertIsInstance(result, PermissionResultAllow)

        # Dangerous command: should be denied (no approval manager)
        result = await checker("Bash", {"command": "rm -rf /"}, None)
        self.assertIsInstance(result, PermissionResultDeny)

    async def test_can_use_tool_safe_bash_skips_whatsapp_approval(self):
        """Safe Bash commands should not trigger WhatsApp approval flow."""
        from approval import ApprovalManager

        manager = ApprovalManager()
        molly = _FakeMolly()
        state = RequestApprovalState()
        checker = agent.make_tool_checker(
            approval_manager=manager,
            molly=molly,
            chat_jid="123@s.whatsapp.net",
            request_state=state,
        )

        result = await checker("Bash", {"command": "pwd"}, None)
        self.assertIsInstance(result, PermissionResultAllow)
        # No WhatsApp messages should have been sent
        self.assertEqual(len(molly.wa.sent), 0)


if __name__ == "__main__":
    unittest.main()
