# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import tempfile
import time
import unittest
import uuid
from datetime import datetime

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility
from modelscope.hub.errors import NotExistError, NoValidRevisionError
from modelscope.hub.file_download import model_file_download
from modelscope.hub.repository import Repository
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import (TEST_ACCESS_TOKEN1,
                                         TEST_MODEL_CHINESE_NAME,
                                         TEST_MODEL_ORG)

logger = get_logger()
logger.setLevel('DEBUG')
download_model_file_name = 'test.bin'
download_model_file_name2 = 'test2.bin'


class HubRevisionTest(unittest.TestCase):

    def setUp(self):
        self.api = HubApi()
        self.api.login(TEST_ACCESS_TOKEN1)
        self.model_name = 'rv-%s' % (uuid.uuid4().hex)
        self.model_id = '%s/%s' % (TEST_MODEL_ORG, self.model_name)
        self.revision = 'v0.1_test_revision'
        self.revision2 = 'v0.2_test_revision'
        self.api.create_model(
            model_id=self.model_id,
            visibility=ModelVisibility.PUBLIC,
            license=Licenses.APACHE_V2,
            chinese_name=TEST_MODEL_CHINESE_NAME,
        )

    def tearDown(self):
        self.api.delete_model(model_id=self.model_id)

    def prepare_repo_data(self):
        temporary_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(temporary_dir, self.model_name)
        self.repo = Repository(self.model_dir, clone_from=self.model_id)
        os.system("echo 'testtest'>%s"
                  % os.path.join(self.model_dir, download_model_file_name))
        self.repo.push('add model')
        self.repo.tag_and_push(self.revision, 'Test revision')

    def test_no_tag(self):
        # no tag will download master
        snapshot_download(self.model_id, None)
        # not specified tag will use master
        model_file_download(self.model_id, ModelFile.README)

        # specified master branch
        snapshot_download(self.model_id, 'master')

    def test_with_only_one_tag(self):
        self.prepare_repo_data()
        with tempfile.TemporaryDirectory() as temp_cache_dir:
            snapshot_path = snapshot_download(
                self.model_id, cache_dir=temp_cache_dir)
            assert os.path.exists(
                os.path.join(snapshot_path, download_model_file_name))
        with tempfile.TemporaryDirectory() as temp_cache_dir:
            file_path = model_file_download(
                self.model_id, ModelFile.README, cache_dir=temp_cache_dir)
            assert os.path.exists(file_path)

    def add_new_file_and_tag(self):
        os.system("echo 'testtest'>%s"
                  % os.path.join(self.model_dir, download_model_file_name2))
        self.repo.push('add new file')
        self.repo.tag_and_push(self.revision2, 'Test revision')

    def test_snapshot_download_different_revision(self):
        self.prepare_repo_data()
        t1 = datetime.now().isoformat(sep=' ', timespec='seconds')
        logger.info('First time stamp: %s' % t1)
        snapshot_path = snapshot_download(self.model_id, self.revision)
        assert os.path.exists(
            os.path.join(snapshot_path, download_model_file_name))
        self.add_new_file_and_tag()
        with tempfile.TemporaryDirectory() as temp_cache_dir:
            time.sleep(11)
            snapshot_path = snapshot_download(
                self.model_id,
                revision=self.revision,
                cache_dir=temp_cache_dir)
            assert os.path.exists(
                os.path.join(snapshot_path, download_model_file_name))
            assert not os.path.exists(
                os.path.join(snapshot_path, download_model_file_name2))
        with tempfile.TemporaryDirectory() as temp_cache_dir:
            time.sleep(11)
            snapshot_path = snapshot_download(
                self.model_id,
                revision=self.revision2,
                cache_dir=temp_cache_dir)
            assert os.path.exists(
                os.path.join(snapshot_path, download_model_file_name))
            assert os.path.exists(
                os.path.join(snapshot_path, download_model_file_name2))

    def test_file_download_different_revision(self):
        self.prepare_repo_data()
        t1 = datetime.now().isoformat(sep=' ', timespec='seconds')
        logger.info('First time stamp: %s' % t1)
        file_path = model_file_download(self.model_id,
                                        download_model_file_name,
                                        self.revision)
        assert os.path.exists(file_path)
        self.add_new_file_and_tag()
        with tempfile.TemporaryDirectory() as temp_cache_dir:
            file_path = model_file_download(
                self.model_id,
                download_model_file_name,
                revision=self.revision,
                cache_dir=temp_cache_dir)
            assert os.path.exists(file_path)
            with self.assertRaises(NotExistError):
                model_file_download(
                    self.model_id,
                    download_model_file_name2,
                    revision=self.revision,
                    cache_dir=temp_cache_dir)

        with tempfile.TemporaryDirectory() as temp_cache_dir:
            file_path = model_file_download(
                self.model_id,
                download_model_file_name,
                revision=self.revision2,
                cache_dir=temp_cache_dir)
            print('Downloaded file path: %s' % file_path)
            assert os.path.exists(file_path)
            file_path = model_file_download(
                self.model_id,
                download_model_file_name2,
                revision=self.revision2,
                cache_dir=temp_cache_dir)
            assert os.path.exists(file_path)


if __name__ == '__main__':
    unittest.main()
