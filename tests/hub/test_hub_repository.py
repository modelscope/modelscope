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
from .test_utils import (TEST_ACCESS_TOKEN1, TEST_MODEL_CHINESE_NAME,
                         TEST_MODEL_ORG, delete_credential)

logger = get_logger()
logger.setLevel('DEBUG')
DEFAULT_GIT_PATH = 'git'
download_model_file_name = 'test.bin'


class HubRepositoryTest(unittest.TestCase):

    def setUp(self):
        self.old_cwd = os.getcwd()
        self.api = HubApi()
        self.api.login(TEST_ACCESS_TOKEN1)
        self.model_name = 'repo-%s' % (uuid.uuid4().hex)
        self.model_id = '%s/%s' % (TEST_MODEL_ORG, self.model_name)
        self.revision = 'v0.1_test_revision'
        self.api.create_model(
            model_id=self.model_id,
            visibility=ModelVisibility.PUBLIC,
            license=Licenses.APACHE_V2,
            chinese_name=TEST_MODEL_CHINESE_NAME,
        )
        temporary_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(temporary_dir, self.model_name)

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.api.delete_model(model_id=self.model_id)

    def test_clone_repo(self):
        Repository(self.model_dir, clone_from=self.model_id)
        assert os.path.exists(os.path.join(self.model_dir, ModelFile.README))

    def test_clone_public_model_without_token(self):
        delete_credential()
        Repository(self.model_dir, clone_from=self.model_id)
        assert os.path.exists(os.path.join(self.model_dir, ModelFile.README))
        self.api.login(TEST_ACCESS_TOKEN1)  # re-login for delete

    def test_push_all(self):
        repo = Repository(self.model_dir, clone_from=self.model_id)
        assert os.path.exists(os.path.join(self.model_dir, ModelFile.README))
        os.chdir(self.model_dir)
        lfs_file1 = 'test1.bin'
        lfs_file2 = 'test2.bin'
        os.system("echo '111'>%s" % os.path.join(self.model_dir, 'add1.py'))
        os.system("echo '222'>%s" % os.path.join(self.model_dir, 'add2.py'))
        os.system("echo 'lfs'>%s" % os.path.join(self.model_dir, lfs_file1))
        os.system("echo 'lfs2'>%s" % os.path.join(self.model_dir, lfs_file2))
        repo.push('test')
        repo.tag_and_push(self.revision, 'Test revision')
        add1 = model_file_download(self.model_id, 'add1.py', self.revision)
        assert os.path.exists(add1)
        add2 = model_file_download(self.model_id, 'add2.py', self.revision)
        assert os.path.exists(add2)
        # check lfs files.
        git_wrapper = GitCommandWrapper()
        lfs_files = git_wrapper.list_lfs_files(self.model_dir)
        assert lfs_file1 in lfs_files
        assert lfs_file2 in lfs_files


if __name__ == '__main__':
    unittest.main()
