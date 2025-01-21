# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.hub.api import HubApi
from modelscope.utils.logger import get_logger

logger = get_logger()
logger.setLevel('DEBUG')
DEFAULT_GIT_PATH = 'git'
download_model_file_name = 'test.bin'


class FileExistsTest(unittest.TestCase):

    def test_file_exists(self):
        api = HubApi()
        self.assertTrue(
            api.file_exists('iic/gte_Qwen2-7B-instruct', 'added_tokens.json'))
        self.assertTrue(
            api.file_exists('iic/gte_Qwen2-7B-instruct',
                            '1_Pooling/config.json'))


if __name__ == '__main__':
    unittest.main()
