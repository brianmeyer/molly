import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import skills


SKILL_TEMPLATE = """## Trigger
- "{trigger}"

## Required Tools
- none

## Steps
1. Do the thing.

## Guardrails
- Keep output concise.
"""


class TestSkillsHotReload(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="skills-hot-reload-"))
        self.skills_dir = self.temp_dir / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir_patcher = patch.object(config, "SKILLS_DIR", self.skills_dir)
        self.skills_dir_patcher.start()
        self.addCleanup(self.skills_dir_patcher.stop)
        self._reset_skill_state()

    def tearDown(self):
        self._reset_skill_state()

    def _reset_skill_state(self):
        with skills._state_lock:
            skills._skills_cache = None
            skills._trigger_patterns = None
            skills._skills_snapshot = None
            skills._last_reload_status = "cold"

    def _write_skill(self, name: str, trigger: str):
        path = self.skills_dir / f"{name}.md"
        path.write_text(SKILL_TEMPLATE.format(trigger=trigger))

    def test_check_for_changes_swaps_cache_and_patterns_atomically(self):
        self._write_skill("alpha", "alpha trigger")
        skills.reload_skills()

        old_cache = skills._skills_cache
        old_patterns = skills._trigger_patterns

        self._write_skill("beta", "beta trigger")
        changed = skills.check_for_changes()

        self.assertTrue(changed)
        self.assertIsNot(skills._skills_cache, old_cache)
        self.assertIsNot(skills._trigger_patterns, old_patterns)
        self.assertIn("beta", skills._skills_cache)
        self.assertIn("beta", [s.name for s in skills.match_skills("beta trigger now")])

    def test_malformed_skill_rolls_back_to_previous_state(self):
        self._write_skill("alpha", "alpha trigger")
        skills.reload_skills()

        old_cache = skills._skills_cache
        old_patterns = skills._trigger_patterns

        # Invalid UTF-8 should fail strict rebuild during hot-reload.
        (self.skills_dir / "broken.md").write_bytes(b"\xff\xfe\xfd")
        changed = skills.check_for_changes()

        self.assertFalse(changed)
        self.assertIs(skills._skills_cache, old_cache)
        self.assertIs(skills._trigger_patterns, old_patterns)
        self.assertEqual("failed", skills.get_reload_status())
        self.assertNotIn("broken", skills._skills_cache)
        self.assertIn("alpha", [s.name for s in skills.match_skills("alpha trigger now")])

    def test_pending_files_are_excluded_from_matcher(self):
        self._write_skill("active", "active trigger")
        self._write_skill("shadow.pending", "pending-only trigger")
        self._write_skill("shadow.pending-edit", "pending-edit-only trigger")

        skills.reload_skills()
        loaded = {entry["name"] for entry in skills.list_skills()}

        self.assertIn("active", loaded)
        self.assertNotIn("shadow.pending", loaded)
        self.assertNotIn("shadow.pending-edit", loaded)
        self.assertEqual([], [s.name for s in skills.match_skills("pending-only trigger")])
        self.assertEqual([], [s.name for s in skills.match_skills("pending-edit-only trigger")])
        self.assertIn("active", [s.name for s in skills.match_skills("active trigger")])


if __name__ == "__main__":
    unittest.main()
