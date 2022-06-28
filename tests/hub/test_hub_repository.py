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
from modelscope.hub.repository import Repository
from modelscope.utils.logger import get_logger

logger = get_logger()
logger.setLevel('DEBUG')
USER_NAME = 'maasadmin'
PASSWORD = '12345678'

model_chinese_name = '达摩卡通化模型'
model_org = 'unittest'
DEFAULT_GIT_PATH = 'git'

download_model_file_name = 'mnist-12.onnx'


def delete_credential():
    path_credential = expanduser('~/.modelscope/credentials')
    shutil.rmtree(path_credential)


def delete_stored_git_credential(user):
    credential_path = expanduser('~/.git-credentials')
    if os.path.exists(credential_path):
        with open(credential_path, 'r+') as f:
            lines = f.readlines()
            for line in lines:
                if user in line:
                    lines.remove(line)
            f.seek(0)
            f.write(''.join(lines))
            f.truncate()


class HubRepositoryTest(unittest.TestCase):

    def setUp(self):
        self.api = HubApi()
        # note this is temporary before official account management is ready
        self.api.login(USER_NAME, PASSWORD)
        self.model_name = uuid.uuid4().hex
        self.model_id = '%s/%s' % (model_org, self.model_name)
        self.api.create_model(
            model_id=self.model_id,
            visibility=ModelVisibility.PUBLIC,  # 1-private, 5-public
            license=Licenses.APACHE_V2,
            chinese_name=model_chinese_name,
        )
        temporary_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(temporary_dir, self.model_name)

    def tearDown(self):
        self.api.delete_model(model_id=self.model_id)

    def test_clone_repo(self):
        Repository(self.model_dir, clone_from=self.model_id)
        assert os.path.exists(os.path.join(self.model_dir, 'README.md'))

    def test_clone_public_model_without_token(self):
        delete_credential()
        delete_stored_git_credential(USER_NAME)
        Repository(self.model_dir, clone_from=self.model_id)
        assert os.path.exists(os.path.join(self.model_dir, 'README.md'))
        self.api.login(USER_NAME, PASSWORD)  # re-login for delete

    def test_push_all(self):
        repo = Repository(self.model_dir, clone_from=self.model_id)
        assert os.path.exists(os.path.join(self.model_dir, 'README.md'))
        os.chdir(self.model_dir)
        os.system("echo '111'>%s" % os.path.join(self.model_dir, 'add1.py'))
        os.system("echo '222'>%s" % os.path.join(self.model_dir, 'add2.py'))
        repo.push('test')
        add1 = model_file_download(self.model_id, 'add1.py')
        assert os.path.exists(add1)
        add2 = model_file_download(self.model_id, 'add2.py')
        assert os.path.exists(add2)


if __name__ == '__main__':
    unittest.main()
