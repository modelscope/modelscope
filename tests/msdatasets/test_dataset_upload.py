# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
import zipfile

from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import ModelFile
from modelscope.utils.test_utils import test_level

KEY_EXTRACTED = 'extracted'


class DatasetUploadTest(unittest.TestCase):

    def setUp(self):
        self.old_dir = os.getcwd()
        self.dataset_name = 'small_coco_for_test'
        self.dataset_file_name = self.dataset_name
        self.prepared_dataset_name = 'pets_small'
        self.token = os.getenv('TEST_UPLOAD_MS_TOKEN')
        error_msg = 'The modelscope token can not be empty, please set env variable: TEST_UPLOAD_MS_TOKEN'
        self.assertIsNotNone(self.token, msg=error_msg)
        from modelscope.hub.api import HubApi
        from modelscope.hub.api import ModelScopeConfig
        self.api = HubApi()
        self.api.login(self.token)

        # get user info
        self.namespace, _ = ModelScopeConfig.get_user_info()

        self.temp_dir = tempfile.mkdtemp()
        self.test_work_dir = os.path.join(self.temp_dir, self.dataset_name)
        self.test_meta_dir = os.path.join(self.test_work_dir, 'meta')
        if not os.path.exists(self.test_work_dir):
            os.makedirs(self.test_work_dir)

    def tearDown(self):
        os.chdir(self.old_dir)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print('The test dir successfully removed!')

    @staticmethod
    def get_raw_downloaded_file_path(extracted_path):
        raw_downloaded_file_path = ''
        raw_data_dir = os.path.abspath(
            os.path.join(extracted_path, '../../..'))
        for root, dirs, files in os.walk(raw_data_dir):
            if KEY_EXTRACTED in dirs:
                for file in files:
                    curr_file_path = os.path.join(root, file)
                    if zipfile.is_zipfile(curr_file_path):
                        raw_downloaded_file_path = curr_file_path
        return raw_downloaded_file_path

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_ds_upload(self):
        # Get the prepared data from hub, using default modelscope namespace
        ms_ds_train = MsDataset.load(self.prepared_dataset_name, split='train')
        config_res = ms_ds_train._hf_ds.config_kwargs
        extracted_path = config_res.get('split_config').get('train')
        raw_zipfile_path = self.get_raw_downloaded_file_path(extracted_path)

        MsDataset.upload(
            object_name=self.dataset_file_name + '.zip',
            local_file_path=raw_zipfile_path,
            dataset_name=self.dataset_name,
            namespace=self.namespace)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_ds_clone_meta(self):
        MsDataset.clone_meta(
            dataset_work_dir=self.test_meta_dir,
            dataset_id=os.path.join(self.namespace, self.dataset_name))

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_ds_upload_meta(self):
        # Clone dataset meta repo first.
        MsDataset.clone_meta(
            dataset_work_dir=self.test_meta_dir,
            dataset_id=os.path.join(self.namespace, self.dataset_name))

        with open(os.path.join(self.test_meta_dir, ModelFile.README),
                  'a') as f:
            f.write('\nThis is a line for unit test.')

        MsDataset.upload_meta(
            dataset_work_dir=self.test_meta_dir,
            commit_message='Update for unit test.')


if __name__ == '__main__':
    unittest.main()
