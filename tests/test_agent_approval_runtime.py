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
    retriever_stub.retrieve_context = lambda *_args, **_kwargs: ""

    class _NoopVectorStore:
        def log_skill_execution(self, *_args, **_kwargs):
            return None

    retriever_stub.get_vectorstore = lambda: _NoopVectorStore()
    sys.modules["memory.retriever"] = retriever_stub

import agent
from approval import ApprovalManager, OWNER_PRIMARY_WHATSAPP_JID, RequestApprovalState
from claude_agent_sdk import CLIConnectionError, PermissionResultDeny


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
        wa_chat = "15550001234@s.whatsapp.net"
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
                patch("agent.retrieve_context", return_value=""), \
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
                patch("agent.retrieve_context", return_value=""), \
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
                patch("agent.retrieve_context", return_value=""), \
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
                patch("agent.retrieve_context", return_value=""), \
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
                patch("agent.retrieve_context", return_value=""), \
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
                patch("agent.retrieve_context", return_value=""), \
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


if __name__ == "__main__":
    unittest.main()
