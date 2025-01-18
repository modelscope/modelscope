# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
import uuid
from os.path import expanduser

from requests import delete

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility
from modelscope.hub.errors import NotExistError
from modelscope.hub.file_download import model_file_download
from modelscope.hub.git import GitCommandWrapper
from modelscope.hub.repository import Repository
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import (TEST_ACCESS_TOKEN1,
                                         TEST_MODEL_CHINESE_NAME,
                                         TEST_MODEL_ORG, delete_credential)

logger = get_logger()
logger.setLevel('DEBUG')
DEFAULT_GIT_PATH = 'git'
download_model_file_name = 'test.bin'


class FileExistsTest(unittest.TestCase):

    def test_file_exsists(self):
        api = HubApi()
        self.assertTrue(api.file_exists('iic/gte_Qwen2-7B-instruct', 'added_tokens.json'))
        self.assertTrue(api.file_exists('iic/gte_Qwen2-7B-instruct', '1_Pooling/config.json'))


if __name__ == '__main__':
    unittest.main()
