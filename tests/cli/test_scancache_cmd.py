import os
import subprocess
import sys
import unittest

from modelscope.utils.file_utils import get_modelscope_cache_dir


class TestScanCacheCommand(unittest.TestCase):
    """Test cases for scancache command in ModelScope CLI."""

    @classmethod
    def setUpClass(cls):
        """Set up for tests."""
        # download one file to ensure the cache directory exists
        model_id = 'Qwen/Qwen3-0.6B'
        cmd = f'python -m modelscope.cli.cli download --model {model_id} README.md'
        subprocess.getstatusoutput(cmd)

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
        cmd = 'python -m modelscope.cli.cli scan-cache --dir /fake/cache/path'
        stat, output = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)
        self.assertIn('not found', output)


if __name__ == '__main__':
    unittest.main()
