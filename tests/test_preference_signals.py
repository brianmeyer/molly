import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (PROJECT_ROOT / path).read_text()


class TestPreferenceSignalSchema(unittest.TestCase):
    def test_vectorstore_has_preference_fields(self):
        src = _read("memory/vectorstore.py")
        self.assertIn("source TEXT", src)
        self.assertIn("surfaced_summary TEXT", src)
        self.assertIn("sender_pattern TEXT", src)
        self.assertIn("owner_feedback TEXT", src)
        self.assertIn("def log_preference_signal", src)


class TestDismissiveFeedbackHook(unittest.TestCase):
    def test_main_logs_dismissive_owner_feedback(self):
        src = _read("main.py")
        self.assertNotIn("DISMISSIVE_FEEDBACK_PATTERNS", src)
        self.assertIn("def _log_preference_signal_if_dismissive", src)
        self.assertIn("from memory.triage import classify_local_async", src)
        self.assertIn("_log_preference_signal_if_dismissive(chat_jid, content)", src)

    def test_triage_has_local_classifier_helper(self):
        src = _read("memory/triage.py")
        self.assertIn("def classify_local(prompt: str, text: str) -> str:", src)
        self.assertIn("async def classify_local_async(prompt: str, text: str) -> str:", src)
        self.assertIn("run_in_executor", src)


class TestSurfacingMetadata(unittest.TestCase):
    def test_heartbeat_uses_surface_send_helper(self):
        src = _read("heartbeat.py")
        self.assertIn("def _send_surface(", src)
        self.assertIn('source="email"', src)
        self.assertIn('source="imessage"', src)


if __name__ == "__main__":
    unittest.main()
