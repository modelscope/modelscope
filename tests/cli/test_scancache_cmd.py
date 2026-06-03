import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest import mock

from modelscope.hub.constants import TEMPORARY_FOLDER_NAME
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


class TestClearCacheCommand(unittest.TestCase):
    """clear-cache must resolve the right cache root in both scenarios:
      * MODELSCOPE_CACHE set   -> $MODELSCOPE_CACHE/{models,datasets}/<id>
      * MODELSCOPE_CACHE unset -> ~/.cache/modelscope/hub/{models,datasets}/<id>
    """

    def _clear_and_assert(self, root, kind, entity_id):
        from modelscope.cli.clearcache import ClearCacheCMD
        sub = {'model': 'models', 'dataset': 'datasets'}[kind]
        entity_dir = os.path.join(root, sub, entity_id)
        temp_dir = os.path.join(root, sub, TEMPORARY_FOLDER_NAME, entity_id)
        os.makedirs(entity_dir)
        os.makedirs(temp_dir)
        args = Namespace(
            model=entity_id if kind == 'model' else None,
            dataset=entity_id if kind == 'dataset' else None)
        with mock.patch('builtins.input', return_value='Y'):
            ClearCacheCMD(args).execute()
        self.assertFalse(os.path.exists(entity_dir))
        self.assertFalse(os.path.exists(temp_dir))

    def test_env_cache(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ,
                                {'MODELSCOPE_CACHE': tmp}, clear=False):
            for kind in ('model', 'dataset'):
                with self.subTest(kind=kind):
                    self._clear_and_assert(tmp, kind, f'org/{kind}')

    def test_default_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                k: v
                for k, v in os.environ.items() if k != 'MODELSCOPE_CACHE'
            }
            env['HOME'] = tmp
            with mock.patch.dict(os.environ, env, clear=True):
                self._clear_and_assert(
                    os.path.join(tmp, '.cache', 'modelscope', 'hub'), 'model',
                    'org/repo')


if __name__ == '__main__':
    unittest.main()
