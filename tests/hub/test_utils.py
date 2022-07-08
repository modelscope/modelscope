import os
import shutil
from codecs import ignore_errors
from os.path import expanduser

TEST_USER_NAME1 = 'citest'
TEST_USER_NAME2 = 'sdkdev'
TEST_PASSWORD = '12345678'

TEST_MODEL_CHINESE_NAME = '内部测试模型'
TEST_MODEL_ORG = 'citest'


def delete_credential():
    path_credential = expanduser('~/.modelscope/credentials')
    shutil.rmtree(path_credential, ignore_errors=True)


def delete_stored_git_credential(user):
    credential_path = expanduser('~/.git-credentials')
    if os.path.exists(credential_path):
        with open(credential_path, 'r+') as f:
            lines = f.readlines()
            lines = [line for line in lines if user not in line]
            f.seek(0)
            f.write(''.join(lines))
            f.truncate()
