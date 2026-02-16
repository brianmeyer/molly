import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class TestMainInstanceLock(unittest.TestCase):
    def test_instance_lock_blocks_second_acquire(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(main.config, "STORE_DIR", Path(tmp)):
                first_fd = main._acquire_instance_lock()
                self.assertIsNotNone(first_fd)
                try:
                    second_fd = main._acquire_instance_lock()
                    self.assertIsNone(second_fd)
                finally:
                    main._release_instance_lock(first_fd)

                third_fd = main._acquire_instance_lock()
                self.assertIsNotNone(third_fd)
                main._release_instance_lock(third_fd)
