import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import heartbeat


class TestHeartbeatSkillHotReload(unittest.IsolatedAsyncioTestCase):
    async def test_run_heartbeat_checks_hot_reload_each_cycle(self):
        heartbeat._skill_reload_count = 0
        molly = MagicMock()
        molly.cancel_event = None
        molly._get_owner_dm_jid.return_value = None

        with patch("heartbeat._check_imessages", new=AsyncMock()) as mock_imessages, patch(
            "heartbeat._check_email", new=AsyncMock()
        ) as mock_email, patch("skills.check_for_changes", return_value=True) as mock_check, patch(
            "skills.get_reload_status", return_value="reloaded"
        ), patch.object(heartbeat.log, "info") as mock_info:
            await heartbeat.run_heartbeat(molly)

        mock_check.assert_called_once()
        mock_imessages.assert_awaited_once()
        mock_email.assert_awaited_once()
        self.assertEqual(heartbeat._skill_reload_count, 1)
        mock_info.assert_any_call(
            "Heartbeat skill hot-reload: status=%s total_reloads=%d",
            "reloaded",
            1,
        )


if __name__ == "__main__":
    unittest.main()
