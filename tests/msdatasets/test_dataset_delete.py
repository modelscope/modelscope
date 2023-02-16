# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest
import zipfile

from modelscope.msdatasets import MsDataset
from modelscope.utils import logger as logging
from modelscope.utils.test_utils import test_level

logger = logging.get_logger()

KEY_EXTRACTED = 'extracted'
EXPECTED_MSG = 'success'


class DatasetDeleteTest(unittest.TestCase):

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
        if not os.path.exists(self.test_work_dir):
            os.makedirs(self.test_work_dir)

    def tearDown(self):
        os.chdir(self.old_dir)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        logger.info(
            f'Temporary directory {self.temp_dir} successfully removed!')

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

    def upload_test_file(self):
        # Get the prepared data from hub, using default modelscope namespace
        ms_ds_train = MsDataset.load(self.prepared_dataset_name, split='train')
        config_res = ms_ds_train._hf_ds.config_kwargs
        extracted_path = config_res.get('split_config').get('train')
        raw_zipfile_path = self.get_raw_downloaded_file_path(extracted_path)

        object_name = self.dataset_file_name + '_for_del.zip'
        MsDataset.upload(
            object_name=object_name,
            local_file_path=raw_zipfile_path,
            dataset_name=self.dataset_name,
            namespace=self.namespace)

        return object_name

    def upload_test_dir(self):
        ms_ds_train = MsDataset.load(self.prepared_dataset_name, split='train')
        config_train = ms_ds_train._hf_ds.config_kwargs
        extracted_path_train = config_train.get('split_config').get('train')

        object_name = 'train_for_del'
        MsDataset.upload(
            object_name=object_name,
            local_file_path=os.path.join(extracted_path_train,
                                         'Pets/images/train'),
            dataset_name=self.dataset_name,
            namespace=self.namespace)

        return object_name + '/'

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_ds_delete_object(self):

        # upload prepared data
        file_name = self.upload_test_file()
        dir_name = self.upload_test_dir()

        # delete object
        del_file_msg = MsDataset.delete(
            object_name=file_name,
            dataset_name=self.dataset_name,
            namespace=self.namespace)
        del_dir_msg = MsDataset.delete(
            object_name=dir_name,
            dataset_name=self.dataset_name,
            namespace=self.namespace)

        assert all([del_file_msg == EXPECTED_MSG, del_dir_msg == EXPECTED_MSG])


if __name__ == '__main__':
    unittest.main()
