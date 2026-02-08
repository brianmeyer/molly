import unittest

from formatting import render_for_whatsapp, split_for_whatsapp


class TestWhatsAppFormatting(unittest.TestCase):
    def test_render_converts_markdown_table_to_bullets(self):
        text = (
            "## System Status\n\n"
            "| Service | State |\n"
            "| --- | --- |\n"
            "| API | Green |\n"
            "| DB | Yellow |\n"
        )

        rendered = render_for_whatsapp(text)

        self.assertIn("System Status", rendered)
        self.assertIn("- Service: API; State: Green", rendered)
        self.assertIn("- Service: DB; State: Yellow", rendered)
        self.assertNotIn("| --- |", rendered)
        self.assertNotIn("## ", rendered)

    def test_render_converts_pipe_table_without_outer_pipes(self):
        text = "Name | Status\n--- | ---\nAPI | Green"
        rendered = render_for_whatsapp(text)
        self.assertIn("- Name: API; Status: Green", rendered)

    def test_render_converts_links_and_code(self):
        text = (
            "See [docs](https://example.com/path).\n"
            "```python\nprint('ok')\n```\n"
            "Use `flag` for dry runs."
        )

        rendered = render_for_whatsapp(text)

        self.assertIn("docs: https://example.com/path", rendered)
        self.assertIn("print('ok')", rendered)
        self.assertIn("Use flag for dry runs.", rendered)
        self.assertNotIn("```", rendered)

    def test_split_for_whatsapp_respects_length_limit(self):
        text = ("alpha " * 120).strip()
        chunks = split_for_whatsapp(text, max_chars=220)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 220)

    def test_split_merges_orphan_heading_chunk(self):
        text = "Plan:\n\n" + " ".join(["This is a long paragraph for chunking sanity."] * 40)
        chunks = split_for_whatsapp(text, max_chars=280)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0].startswith("Plan:\n\nThis is a long paragraph"))
        self.assertLessEqual(len(chunks[0]), 280)


if __name__ == "__main__":
    unittest.main()
