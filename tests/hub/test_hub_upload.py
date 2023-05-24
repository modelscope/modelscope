# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import time
import unittest
import uuid

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility
from modelscope.hub.errors import GitError, HTTPError, NotLoginException
from modelscope.hub.push_to_hub import push_to_hub, push_to_hub_async
from modelscope.hub.repository import Repository
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import (TEST_ACCESS_TOKEN1, TEST_MODEL_ORG,
                                         delete_credential, test_level)

logger = get_logger()


class HubUploadTest(unittest.TestCase):

    def setUp(self):
        logger.info('SetUp')
        self.api = HubApi()
        self.user = TEST_MODEL_ORG
        logger.info(self.user)
        self.create_model_name = '%s/%s_%s' % (self.user, 'test_model_upload',
                                               uuid.uuid4().hex)
        logger.info('create %s' % self.create_model_name)
        temporary_dir = tempfile.mkdtemp()
        self.work_dir = temporary_dir
        self.model_dir = os.path.join(temporary_dir, self.create_model_name)
        self.finetune_path = os.path.join(self.work_dir, 'finetune_path')
        self.repo_path = os.path.join(self.work_dir, 'repo_path')
        os.mkdir(self.finetune_path)
        os.system("echo '{}'>%s"
                  % os.path.join(self.finetune_path, ModelFile.CONFIGURATION))
        os.environ['MODELSCOPE_TRAIN_ID'] = 'test-id'

    def tearDown(self):
        logger.info('TearDown')
        shutil.rmtree(self.model_dir, ignore_errors=True)
        try:
            self.api.delete_model(model_id=self.create_model_name)
        except Exception:
            pass

    def test_upload_exits_repo_master(self):
        logger.info('basic test for upload!')
        self.api.login(TEST_ACCESS_TOKEN1)
        self.api.create_model(
            model_id=self.create_model_name,
            visibility=ModelVisibility.PUBLIC,
            license=Licenses.APACHE_V2)
        os.system("echo '111'>%s"
                  % os.path.join(self.finetune_path, 'add1.py'))
        self.api.push_model(
            model_id=self.create_model_name, model_dir=self.finetune_path)
        Repository(model_dir=self.repo_path, clone_from=self.create_model_name)
        assert os.path.exists(os.path.join(self.repo_path, 'add1.py'))
        shutil.rmtree(self.repo_path, ignore_errors=True)
        os.system("echo '222'>%s"
                  % os.path.join(self.finetune_path, 'add2.py'))
        self.api.push_model(
            model_id=self.create_model_name,
            model_dir=self.finetune_path,
            revision='new_revision/version1')
        Repository(
            model_dir=self.repo_path,
            clone_from=self.create_model_name,
            revision='new_revision/version1')
        assert os.path.exists(os.path.join(self.repo_path, 'add2.py'))
        shutil.rmtree(self.repo_path, ignore_errors=True)
        os.system("echo '333'>%s"
                  % os.path.join(self.finetune_path, 'add3.py'))
        self.api.push_model(
            model_id=self.create_model_name,
            model_dir=self.finetune_path,
            revision='new_revision/version2',
            commit_message='add add3.py')
        Repository(
            model_dir=self.repo_path,
            clone_from=self.create_model_name,
            revision='new_revision/version2')
        assert os.path.exists(os.path.join(self.repo_path, 'add2.py'))
        assert os.path.exists(os.path.join(self.repo_path, 'add3.py'))
        shutil.rmtree(self.repo_path, ignore_errors=True)
        add4_path = os.path.join(self.finetune_path, 'temp')
        os.mkdir(add4_path)
        os.system("echo '444'>%s" % os.path.join(add4_path, 'add4.py'))
        self.api.push_model(
            model_id=self.create_model_name,
            model_dir=self.finetune_path,
            revision='new_revision/version1')
        Repository(
            model_dir=self.repo_path,
            clone_from=self.create_model_name,
            revision='new_revision/version1')
        assert os.path.exists(os.path.join(add4_path, 'add4.py'))
        shutil.rmtree(self.repo_path, ignore_errors=True)
        assert os.path.exists(os.path.join(self.finetune_path, 'add3.py'))
        os.remove(os.path.join(self.finetune_path, 'add3.py'))
        self.api.push_model(
            model_id=self.create_model_name,
            model_dir=self.finetune_path,
            revision='new_revision/version1')
        Repository(
            model_dir=self.repo_path,
            clone_from=self.create_model_name,
            revision='new_revision/version1')
        assert not os.path.exists(os.path.join(self.repo_path, 'add3.py'))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_upload_non_exists_repo(self):
        logger.info('test upload non exists repo!')
        self.api.login(TEST_ACCESS_TOKEN1)
        os.system("echo '111'>%s"
                  % os.path.join(self.finetune_path, 'add1.py'))
        self.api.push_model(
            model_id=self.create_model_name,
            model_dir=self.finetune_path,
            revision='new_model_new_revision',
            visibility=ModelVisibility.PUBLIC,
            license=Licenses.APACHE_V2)
        Repository(
            model_dir=self.repo_path,
            clone_from=self.create_model_name,
            revision='new_model_new_revision')
        assert os.path.exists(os.path.join(self.repo_path, 'add1.py'))
        shutil.rmtree(self.repo_path, ignore_errors=True)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_upload_without_token(self):
        logger.info('test upload without login!')
        self.api.login(TEST_ACCESS_TOKEN1)
        delete_credential()
        with self.assertRaises(NotLoginException):
            self.api.push_model(
                model_id=self.create_model_name,
                model_dir=self.finetune_path,
                visibility=ModelVisibility.PUBLIC,
                license=Licenses.APACHE_V2)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_upload_invalid_repo(self):
        logger.info('test upload to invalid repo!')
        self.api.login(TEST_ACCESS_TOKEN1)
        with self.assertRaises((HTTPError, GitError)):
            self.api.push_model(
                model_id='%s/%s' % ('speech_tts', 'invalid_model_test'),
                model_dir=self.finetune_path,
                visibility=ModelVisibility.PUBLIC,
                license=Licenses.APACHE_V2)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_push_to_hub(self):
        ret = push_to_hub(
            repo_name=self.create_model_name,
            output_dir=self.finetune_path,
            token=TEST_ACCESS_TOKEN1)
        self.assertTrue(ret is True)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_push_to_hub_async(self):
        future = push_to_hub_async(
            repo_name=self.create_model_name,
            output_dir=self.finetune_path,
            token=TEST_ACCESS_TOKEN1)
        while not future.done():
            time.sleep(1)
        self.assertTrue(future.result())


if __name__ == '__main__':
    unittest.main()
