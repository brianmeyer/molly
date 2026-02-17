import unittest
from unittest.mock import patch

import config
from main import Molly


class TestOwnerJidNormalization(unittest.TestCase):
    def _new_molly(self) -> Molly:
        molly = object.__new__(Molly)
        molly.registered_chats = {}
        return molly

    def test_is_owner_accepts_device_suffix(self):
        molly = self._new_molly()
        with patch.object(config, "OWNER_IDS", {"15551234567", "99900000000000"}):
            self.assertTrue(molly._is_owner("99900000000000:10@lid"))
            self.assertTrue(molly._is_owner("15551234567:9@s.whatsapp.net"))
            self.assertFalse(molly._is_owner("99999999999:3@lid"))

    def test_owner_dm_chat_mode_accepts_device_suffix(self):
        molly = self._new_molly()
        with patch.object(config, "OWNER_IDS", {"99900000000000"}):
            self.assertEqual(molly._get_chat_mode("99900000000000:10@lid"), "owner_dm")
