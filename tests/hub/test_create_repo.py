# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest
import uuid

from modelscope import HubApi
from modelscope.utils.constant import REPO_TYPE_DATASET, REPO_TYPE_MODEL
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import TEST_ACCESS_TOKEN1
from modelscope.utils.test_utils import TEST_MODEL_ORG as TEST_ORG
from modelscope.utils.test_utils import delete_credential, test_level

logger = get_logger()


class TestCreateRepo(unittest.TestCase):

    def setUp(self):
        self.api = HubApi()
        self.api.login(TEST_ACCESS_TOKEN1)

        self.repo_id_model: str = f'{TEST_ORG}/test_create_repo_model_{uuid.uuid4().hex[-6:]}'
        self.repo_id_dataset: str = f'{TEST_ORG}/test_create_repo_dataset_{uuid.uuid4().hex[-6:]}'

    def tearDown(self):
        self.api.delete_repo(
            repo_id=self.repo_id_model, repo_type=REPO_TYPE_MODEL)
        self.api.delete_repo(
            repo_id=self.repo_id_dataset, repo_type=REPO_TYPE_DATASET)
        delete_credential()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_create_repo(self):

        logger.info(
            f'TEST: Creating repo {self.repo_id_model} and {self.repo_id_dataset} ...'
        )

        try:
            self.api.create_repo(
                repo_id=self.repo_id_model,
                repo_type=REPO_TYPE_MODEL,
                exist_ok=True)
        except Exception as e:
            logger.error(f'Failed to create repo {self.repo_id_model} !')
            raise e

        try:
            self.api.create_repo(
                repo_id=self.repo_id_dataset,
                repo_type=REPO_TYPE_DATASET,
                exist_ok=True)
        except Exception as e:
            logger.error(f'Failed to create repo {self.repo_id_dataset} !')
            raise e

        logger.info(
            f'TEST: Created repo {self.repo_id_model} and {self.repo_id_dataset} successfully !'
        )
