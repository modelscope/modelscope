# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import tempfile
import unittest
import uuid

from modelscope.hub.api import HubApi, ModelScopeConfig
from modelscope.hub.constants import Licenses, ModelVisibility
from modelscope.hub.file_download import model_file_download
from modelscope.hub.repository import Repository
from modelscope.hub.snapshot_download import snapshot_download

USER_NAME = 'maasadmin'
PASSWORD = '12345678'

model_chinese_name = '达摩卡通化模型'
model_org = 'unittest'
DEFAULT_GIT_PATH = 'git'

download_model_file_name = 'test.bin'


class HubOperationTest(unittest.TestCase):

    def setUp(self):
        self.old_cwd = os.getcwd()
        self.api = HubApi()
        # note this is temporary before official account management is ready
        self.api.login(USER_NAME, PASSWORD)
        self.model_name = uuid.uuid4().hex
        self.model_id = '%s/%s' % (model_org, self.model_name)
        self.api.create_model(
            model_id=self.model_id,
            chinese_name=model_chinese_name,
            visibility=ModelVisibility.PUBLIC,
            license=Licenses.APACHE_V2)
        temporary_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(temporary_dir, self.model_name)
        repo = Repository(self.model_dir, clone_from=self.model_id)
        os.chdir(self.model_dir)
        os.system("echo 'testtest'>%s"
                  % os.path.join(self.model_dir, 'test.bin'))
        repo.push('add model', all_files=True)

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.api.delete_model(model_id=self.model_id)

    def test_model_repo_creation(self):
        # change to proper model names before use
        try:
            info = self.api.get_model(model_id=self.model_id)
            assert info['Name'] == self.model_name
        except KeyError as ke:
            if ke.args[0] == 'name':
                print(f'model {self.model_name} already exists, ignore')
            else:
                raise

    def test_download_single_file(self):
        downloaded_file = model_file_download(
            model_id=self.model_id, file_path=download_model_file_name)
        assert os.path.exists(downloaded_file)
        mdtime1 = os.path.getmtime(downloaded_file)
        # download again
        downloaded_file = model_file_download(
            model_id=self.model_id, file_path=download_model_file_name)
        mdtime2 = os.path.getmtime(downloaded_file)
        assert mdtime1 == mdtime2

    def test_snapshot_download(self):
        snapshot_path = snapshot_download(model_id=self.model_id)
        downloaded_file_path = os.path.join(snapshot_path,
                                            download_model_file_name)
        assert os.path.exists(downloaded_file_path)
        mdtime1 = os.path.getmtime(downloaded_file_path)
        # download again
        snapshot_path = snapshot_download(model_id=self.model_id)
        mdtime2 = os.path.getmtime(downloaded_file_path)
        assert mdtime1 == mdtime2


if __name__ == '__main__':
    unittest.main()
