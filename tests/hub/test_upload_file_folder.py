# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import struct
import tempfile
import unittest
import uuid

import json

from modelscope import HubApi
from modelscope.utils.constant import REPO_TYPE_DATASET, REPO_TYPE_MODEL
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import TEST_ACCESS_TOKEN1
from modelscope.utils.test_utils import TEST_MODEL_ORG as TEST_ORG
from modelscope.utils.test_utils import delete_credential, test_level

logger = get_logger()


class TestUploadFileFolder(unittest.TestCase):

    def setUp(self):
        self.api = HubApi()
        self.api.login(TEST_ACCESS_TOKEN1)

        self.repo_id_model: str = f'{TEST_ORG}/test_upload_file_folder_model_{uuid.uuid4().hex[-6:]}'
        self.repo_id_dataset: str = f'{TEST_ORG}/test_upload_file_folder_dataset_{uuid.uuid4().hex[-6:]}'

        self.work_dir = tempfile.mkdtemp()
        self.model_file_path = f'{self.work_dir}/test_model.bin'
        self.dataset_file_path = f'{self.work_dir}/test_data.jsonl'

        logger.info(f'Work directory: {self.work_dir}')

        self.api.create_repo(
            repo_id=self.repo_id_model,
            repo_type=REPO_TYPE_MODEL,
            exist_ok=True)
        self.api.create_repo(
            repo_id=self.repo_id_dataset,
            repo_type=REPO_TYPE_DATASET,
            exist_ok=True)

        self._construct_file()

    def tearDown(self):

        # Remove repositories
        self.api.delete_repo(
            repo_id=self.repo_id_model, repo_type=REPO_TYPE_MODEL)
        self.api.delete_repo(
            repo_id=self.repo_id_dataset, repo_type=REPO_TYPE_DATASET)

        # Clean up the temporary credentials
        delete_credential()

        # Clean up the temporary directory
        shutil.rmtree(self.work_dir)

    def _construct_file(self):

        # Construct data
        data_list = [
            {
                'id': 1,
                'value': 3.14
            },
            {
                'id': 2,
                'value': 2.71
            },
            {
                'id': 3,
                'value': 3.69
            },
            {
                'id': 4,
                'value': 9.31
            },
            {
                'id': 5,
                'value': 1.21
            },
        ]

        with open(self.model_file_path, 'wb') as f:
            for entry in data_list:
                packed_data = struct.pack('if', entry['id'], entry['value'])
                f.write(packed_data)
        logger.info(f'Constructed model file: {self.model_file_path}')

        with open(self.dataset_file_path, 'w') as f:
            for entry in data_list:
                f.write(json.dumps(entry) + '\n')
        logger.info(f'Constructed dataset file: {self.dataset_file_path}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_upload_file_folder(self):
        """
        Test uploading file/folder to the model/dataset repository.
        """

        commit_info_upload_file_model = self.api.upload_file(
            path_or_fileobj=self.model_file_path,
            path_in_repo=os.path.basename(self.model_file_path),
            repo_id=self.repo_id_model,
            repo_type=REPO_TYPE_MODEL,
            commit_message='Add model file for CI_TEST',
        )
        self.assertTrue(commit_info_upload_file_model is not None)

        commit_info_upload_file_dataset = self.api.upload_file(
            path_or_fileobj=self.dataset_file_path,
            path_in_repo=os.path.basename(self.dataset_file_path),
            repo_id=self.repo_id_dataset,
            repo_type=REPO_TYPE_DATASET,
            commit_message='Add dataset file for CI_TEST',
        )
        self.assertTrue(commit_info_upload_file_dataset is not None)

        commit_info_upload_folder_model = self.api.upload_folder(
            repo_id=self.repo_id_model,
            folder_path=self.work_dir,
            path_in_repo='test_data',
            repo_type=REPO_TYPE_MODEL,
            commit_message='Add model folder for CI_TEST',
        )
        self.assertTrue(commit_info_upload_folder_model is not None)

        commit_info_upload_folder_dataset = self.api.upload_folder(
            repo_id=self.repo_id_dataset,
            folder_path=self.work_dir,
            path_in_repo='test_data',
            repo_type=REPO_TYPE_DATASET,
            commit_message='Add dataset folder for CI_TEST',
        )
        self.assertTrue(commit_info_upload_folder_dataset is not None)
