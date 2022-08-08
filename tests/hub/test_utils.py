import os
import shutil
from codecs import ignore_errors
from os.path import expanduser

from modelscope.hub.constants import DEFAULT_CREDENTIALS_PATH

# for user citest and sdkdev
TEST_ACCESS_TOKEN1 = 'RGZZdkh2Z3BlMFU1VktjUkdIcUJtdjdqdnhQUEQrUVROdVBjclAzUGVycHFhU1BFZFBIaGtUOHB1eHQ2OTV3dQ=='
TEST_ACCESS_TOKEN2 = 'dFpadllseTZQbHlyK0E4amQxVC84a2RtZHdkUVhmMUl3M1VXZXU4dS9GZlRuVmFUTW5yQm8yTENYWEw2SVh0Uw=='

TEST_MODEL_CHINESE_NAME = '内部测试模型'
TEST_MODEL_ORG = 'citest'


def delete_credential():
    path_credential = expanduser(DEFAULT_CREDENTIALS_PATH)
    shutil.rmtree(path_credential, ignore_errors=True)
