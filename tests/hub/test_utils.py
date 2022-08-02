import os
import shutil
from codecs import ignore_errors
from os.path import expanduser

from modelscope.hub.constants import DEFAULT_CREDENTIALS_PATH

# for user citest and sdkdev
TEST_ACCESS_TOKEN1 = 'OVAzNU9aZ2FYbXFhdGNzZll6VHRtalQ0T1BpZTNGeWVhMkxSSGpTSzU0dkM5WE5ObDFKdFRQWGc2U2ZIdjdPdg=='
TEST_ACCESS_TOKEN2 = 'aXRocHhGeG0rNXRWQWhBSnJpTTZUQ0RDbUlkcUJRS1dQR2lNb0xIa0JjRDBrT1JKYklZV05DVzROTTdtamxWcg=='

TEST_MODEL_CHINESE_NAME = '内部测试模型'
TEST_MODEL_ORG = 'citest'


def delete_credential():
    path_credential = expanduser(DEFAULT_CREDENTIALS_PATH)
    shutil.rmtree(path_credential, ignore_errors=True)
