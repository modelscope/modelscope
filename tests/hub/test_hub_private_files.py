# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import tempfile
import unittest
import uuid

from requests.exceptions import HTTPError

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility
from modelscope.hub.errors import GitError
from modelscope.hub.file_download import model_file_download
from modelscope.hub.repository import Repository
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.constant import ModelFile

USER_NAME = 'maasadmin'
PASSWORD = '12345678'
USER_NAME2 = 'sdkdev'

model_chinese_name = '达摩卡通化模型'
model_org = 'unittest'


class HubPrivateFileDownloadTest(unittest.TestCase):

    def setUp(self):
        self.old_cwd = os.getcwd()
        self.api = HubApi()
        # note this is temporary before official account management is ready
        self.token, _ = self.api.login(USER_NAME, PASSWORD)
        self.model_name = uuid.uuid4().hex
        self.model_id = '%s/%s' % (model_org, self.model_name)
        self.api.create_model(
            model_id=self.model_id,
            visibility=ModelVisibility.PRIVATE,  # 1-private, 5-public
            license=Licenses.APACHE_V2,
            chinese_name=model_chinese_name,
        )

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.api.delete_model(model_id=self.model_id)

    def test_snapshot_download_private_model(self):
        snapshot_path = snapshot_download(self.model_id)
        assert os.path.exists(os.path.join(snapshot_path, ModelFile.README))

    def test_snapshot_download_private_model_no_permission(self):
        self.token, _ = self.api.login(USER_NAME2, PASSWORD)
        with self.assertRaises(HTTPError):
            snapshot_download(self.model_id)
        self.api.login(USER_NAME, PASSWORD)

    def test_download_file_private_model(self):
        file_path = model_file_download(self.model_id, ModelFile.README)
        assert os.path.exists(file_path)

    def test_download_file_private_model_no_permission(self):
        self.token, _ = self.api.login(USER_NAME2, PASSWORD)
        with self.assertRaises(HTTPError):
            model_file_download(self.model_id, ModelFile.README)
        self.api.login(USER_NAME, PASSWORD)

    def test_snapshot_download_local_only(self):
        with self.assertRaises(ValueError):
            snapshot_download(self.model_id, local_files_only=True)
        snapshot_path = snapshot_download(self.model_id)
        assert os.path.exists(os.path.join(snapshot_path, ModelFile.README))
        snapshot_path = snapshot_download(self.model_id, local_files_only=True)
        assert os.path.exists(snapshot_path)

    def test_file_download_local_only(self):
        with self.assertRaises(ValueError):
            model_file_download(
                self.model_id, ModelFile.README, local_files_only=True)
        file_path = model_file_download(self.model_id, ModelFile.README)
        assert os.path.exists(file_path)
        file_path = model_file_download(
            self.model_id, ModelFile.README, local_files_only=True)
        assert os.path.exists(file_path)


if __name__ == '__main__':
    unittest.main()
