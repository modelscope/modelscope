import importlib
import sys
import unittest
from unittest import mock


class LegacyCacheGuardTest(unittest.TestCase):
    """Tests for the modelscope-hub capability guard in the shim.

    Network-free: only probes whether the loaded modelscope-hub exposes the
    legacy-cache auto-detection capability and warns once when it does not.
    """

    def setUp(self):
        # NOTE: ``modelscope.hub.snapshot_download`` the *attribute* is shadowed
        # by the re-exported function in ``modelscope.hub.__init__``; use
        # importlib to obtain the actual submodule object.
        sd = importlib.import_module('modelscope.hub.snapshot_download')
        self.sd = sd
        # Reset the fire-once probe cache before each test.
        sd._legacy_cache_capability = None

    def tearDown(self):
        self.sd._legacy_cache_capability = None

    def test_capability_present_no_warning(self):
        sd = self.sd
        # The real modelscope-hub DownloadManager has _find_legacy_repo_dir.
        with mock.patch.object(sd.logger, 'warning') as warn:
            sd._warn_if_legacy_cache_detection_unavailable()
        self.assertTrue(sd._legacy_cache_capability)
        warn.assert_not_called()

    def test_capability_absent_warns_once(self):
        sd = self.sd

        class _OldDownloadManager:  # lacks _find_legacy_repo_dir
            pass

        fake_module = mock.MagicMock()
        fake_module.DownloadManager = _OldDownloadManager

        with mock.patch.dict(sys.modules,
                             {'modelscope_hub._download': fake_module}):
            with mock.patch.object(sd.logger, 'warning') as warn:
                sd._warn_if_legacy_cache_detection_unavailable()
                sd._warn_if_legacy_cache_detection_unavailable()

        self.assertFalse(sd._legacy_cache_capability)
        self.assertEqual(warn.call_count, 1)  # fire-once


if __name__ == '__main__':
    unittest.main()
