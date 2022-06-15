# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

from maas_hub.maas_api import MaasApi
from maas_hub.repository import Repository

USER_NAME = 'maasadmin'
PASSWORD = '12345678'


class HubOperationTest(unittest.TestCase):

    def setUp(self):
        self.api = MaasApi()
        # note this is temporary before official account management is ready
        self.api.login(USER_NAME, PASSWORD)

    @unittest.skip('to be used for local test only')
    def test_model_repo_creation(self):
        # change to proper model names before use
        model_name = 'cv_unet_person-image-cartoon_compound-models'
        model_chinese_name = '达摩卡通化模型'
        model_org = 'damo'
        try:
            self.api.create_model(
                owner=model_org,
                name=model_name,
                chinese_name=model_chinese_name,
                visibility=5,  # 1-private, 5-public
                license='apache-2.0')
        # TODO: support proper name duplication checking
        except KeyError as ke:
            if ke.args[0] == 'name':
                print(f'model {self.model_name} already exists, ignore')
            else:
                raise

    # Note that this can be done via git operation once model repo
    # has been created. Git-Op is the RECOMMENDED model upload approach
    @unittest.skip('to be used for local test only')
    def test_model_upload(self):
        local_path = '/path/to/local/model/directory'
        assert osp.exists(local_path), 'Local model directory not exist.'
        repo = Repository(local_dir=local_path)
        repo.push_to_hub(commit_message='Upload model files')


if __name__ == '__main__':
    unittest.main()
