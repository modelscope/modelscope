# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shutil
from codecs import ignore_errors
from os.path import expanduser

from modelscope.hub.constants import DEFAULT_CREDENTIALS_PATH

# for user citest and sdkdev
TEST_ACCESS_TOKEN1 = os.environ['TEST_ACCESS_TOKEN_CITEST']
TEST_ACCESS_TOKEN2 = os.environ['TEST_ACCESS_TOKEN_SDKDEV']

TEST_MODEL_CHINESE_NAME = '内部测试模型'
TEST_MODEL_ORG = 'citest'


def delete_credential():
    path_credential = expanduser(DEFAULT_CREDENTIALS_PATH)
    shutil.rmtree(path_credential, ignore_errors=True)
