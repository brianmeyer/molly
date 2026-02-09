"""Contract tests for dynamic skill trigger parsing."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import skills


class TestSkillTriggerContract(unittest.TestCase):
    """Verify trigger parsing supports both legacy and YAML skill contracts."""

    def _write_skill(self, skills_dir: Path, name: str, content: str) -> None:
        (skills_dir / f"{name}.md").write_text(content)

    def tearDown(self):
        # Reset module-level caches to the real configured skills dir between tests.
        skills.reload_skills()

    def test_yaml_front_matter_only_skill_matches(self):
        """A skill with only YAML front matter triggers should still match."""
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp)
            self._write_skill(
                skills_dir,
                "yaml-only",
                (
                    "---\n"
                    "name: yaml-only\n"
                    "triggers:\n"
                    '  - "run yaml flow"\n'
                    "  - /yamlrun\n"
                    "---\n"
                    "\n"
                    "# Skill: YAML only\n"
                ),
            )

            with patch("skills.config.SKILLS_DIR", skills_dir):
                skills.reload_skills()

                phrase_matches = [s.name for s in skills.match_skills("please run yaml flow now")]
                command_matches = [s.name for s in skills.match_skills("do /yamlrun next")]

                self.assertIn("yaml-only", phrase_matches)
                self.assertIn("yaml-only", command_matches)

    def test_mixed_mode_precedence_and_dedup(self):
        """Mixed-mode files should apply deterministic precedence and de-dup."""
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp)
            self._write_skill(
                skills_dir,
                "mixed-mode",
                (
                    "---\n"
                    "name: mixed-mode\n"
                    "triggers:\n"
                    '  - "from yaml"\n'
                    '  - "shared trigger"\n'
                    "  - /same\n"
                    "---\n"
                    "\n"
                    "# Skill: Mixed\n"
                    "\n"
                    "## Trigger\n"
                    '- "from markdown"\n'
                    '- "shared trigger"\n'
                    '- "FROM YAML"\n'
                    "- `/same` command\n"
                    "- `/legacy` command\n"
                ),
            )

            with patch("skills.config.SKILLS_DIR", skills_dir):
                skills.reload_skills()
                skill = skills.get_skill("mixed-mode")
                self.assertIsNotNone(skill)

                trigger_values = skills._collect_skill_trigger_values(skill)
                expected_values = [
                    "from yaml",
                    "shared trigger",
                    "/same",
                    "from markdown",
                    "/legacy",
                ]
                self.assertEqual(trigger_values, expected_values)

                pattern_map = dict(skills._build_trigger_patterns())
                self.assertIn("mixed-mode", pattern_map)
                pattern_parts = pattern_map["mixed-mode"].pattern.split("|")
                expected_parts = [skills._phrase_to_regex(value) for value in expected_values]
                self.assertEqual(pattern_parts, expected_parts)

                matched_yaml = [s.name for s in skills.match_skills("please do from yaml first")]
                matched_markdown = [s.name for s in skills.match_skills("please do from markdown first")]
                matched_command = [s.name for s in skills.match_skills("run /legacy now")]
                self.assertIn("mixed-mode", matched_yaml)
                self.assertIn("mixed-mode", matched_markdown)
                self.assertIn("mixed-mode", matched_command)


if __name__ == "__main__":
    unittest.main()
