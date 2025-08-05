# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope import snapshot_download
from modelscope.hub.cache_manager import scan_cache_dir
from modelscope.hub.errors import CacheNotFound
from modelscope.utils.file_utils import get_modelscope_cache_dir
from modelscope.utils.logger import get_logger

logger = get_logger()


class HubScanCacheTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up for tests."""
        # download one file to ensure the cache directory exists
        model_id = 'Qwen/Qwen3-0.6B'
        snapshot_download(model_id, allow_file_pattern='README.md')

    def test_scan_default_dir(self):
        """Test scanning the default cache directory."""
        try:
            res_info = scan_cache_dir()
            table_str = res_info.export_as_table()
            lines = table_str.split('\n')
            if len(lines) <= 20:
                logger.info(table_str)
            else:
                limited_table = '\n'.join(
                    lines[:10]
                    + [f'... ({len(lines) - 20} lines omitted) ...']
                    + lines[-10:])
                logger.info(limited_table)
        except Exception as e:
            self.fail(f'Scanning default cache directory failed: {e}')

    def test_scan_given_dir(self):
        """Test scanning a given cache directory."""
        try:
            scan_cache_dir(get_modelscope_cache_dir())
            logger.info('Done')
        except Exception as e:
            self.fail(f'Scanning given cache directory failed: {e}')

    def test_scan_not_exist_dir(self):
        """Test scanning a non-existent cache directory."""
        with self.assertRaises(CacheNotFound):
            scan_cache_dir('/non/existent/path')


if __name__ == '__main__':
    unittest.main()
