import os
import subprocess
import sys
import unittest

from modelscope.utils.file_utils import get_modelscope_cache_dir


class TestScanCacheCommand(unittest.TestCase):
    """Test cases for scancache command in ModelScope CLI."""

    def setUp(self):
        """Set up for tests."""
        self.fake_cache_dir = '/fake/cache/path'

    def test_scan_default_dir(self):
        cmd = 'python -m modelscope.cli.cli scan-cache'
        stat, output = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)
        self.assertIn('Done', output)

    def test_scan_given_dir(self):
        cmd = f'python -m modelscope.cli.cli scan-cache --dir {get_modelscope_cache_dir()}'
        stat, output = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)
        self.assertIn('Done', output)

    def test_scan_not_exist_dir(self):
        cmd = f'python -m modelscope.cli.cli scan-cache --dir {self.fake_cache_dir}'
        stat, output = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)
        self.assertIn('not found', output)


if __name__ == '__main__':
    unittest.main()
