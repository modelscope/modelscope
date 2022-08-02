# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import tempfile
import unittest
import uuid

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility
from modelscope.hub.errors import GitError
from modelscope.hub.repository import Repository
from modelscope.utils.constant import ModelFile
from .test_utils import (TEST_ACCESS_TOKEN1, TEST_ACCESS_TOKEN2,
                         TEST_MODEL_CHINESE_NAME, TEST_MODEL_ORG)

DEFAULT_GIT_PATH = 'git'


@unittest.skip(
    "Access token is always change, we can't login with same access token, so skip!"
)
class HubPrivateRepositoryTest(unittest.TestCase):

    def setUp(self):
        self.old_cwd = os.getcwd()
        self.api = HubApi()
        # note this is temporary before official account management is ready
        self.token, _ = self.api.login(TEST_ACCESS_TOKEN1)
        self.model_name = uuid.uuid4().hex
        self.model_id = '%s/%s' % (TEST_MODEL_ORG, self.model_name)
        self.api.create_model(
            model_id=self.model_id,
            visibility=ModelVisibility.PRIVATE,  # 1-private, 5-public
            license=Licenses.APACHE_V2,
            chinese_name=TEST_MODEL_CHINESE_NAME,
        )

    def tearDown(self):
        self.api.login(TEST_ACCESS_TOKEN1)
        os.chdir(self.old_cwd)
        self.api.delete_model(model_id=self.model_id)

    def test_clone_private_repo_no_permission(self):
        token, _ = self.api.login(TEST_ACCESS_TOKEN2)
        temporary_dir = tempfile.mkdtemp()
        local_dir = os.path.join(temporary_dir, self.model_name)
        with self.assertRaises(GitError) as cm:
            Repository(local_dir, clone_from=self.model_id, auth_token=token)

        print(cm.exception)
        assert not os.path.exists(os.path.join(local_dir, ModelFile.README))

    def test_clone_private_repo_has_permission(self):
        temporary_dir = tempfile.mkdtemp()
        local_dir = os.path.join(temporary_dir, self.model_name)
        Repository(local_dir, clone_from=self.model_id, auth_token=self.token)
        assert os.path.exists(os.path.join(local_dir, ModelFile.README))

    def test_initlize_repo_multiple_times(self):
        temporary_dir = tempfile.mkdtemp()
        local_dir = os.path.join(temporary_dir, self.model_name)
        repo1 = Repository(
            local_dir, clone_from=self.model_id, auth_token=self.token)
        print(repo1.model_dir)
        assert os.path.exists(os.path.join(local_dir, ModelFile.README))
        repo2 = Repository(
            local_dir, clone_from=self.model_id,
            auth_token=self.token)  # skip clone
        print(repo2.model_dir)
        assert repo1.model_dir == repo2.model_dir


if __name__ == '__main__':
    unittest.main()
