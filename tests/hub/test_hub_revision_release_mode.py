# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import tempfile
import time
import unittest
import uuid
from datetime import datetime
from unittest import mock

from modelscope import version
from modelscope.hub.api import HubApi
from modelscope.hub.constants import (MODELSCOPE_SDK_DEBUG, Licenses,
                                      ModelVisibility)
from modelscope.hub.errors import NotExistError
from modelscope.hub.file_download import model_file_download
from modelscope.hub.repository import Repository
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import (TEST_ACCESS_TOKEN1,
                                         TEST_MODEL_CHINESE_NAME,
                                         TEST_MODEL_ORG)

logger = get_logger()
logger.setLevel('DEBUG')
download_model_file_name = 'test.bin'
download_model_file_name2 = 'test2.bin'


@unittest.skip('temporarily skip')
class HubRevisionTest(unittest.TestCase):

    def setUp(self):
        self.api = HubApi()
        self.api.login(TEST_ACCESS_TOKEN1)
        self.model_name = 'rvr-%s' % (uuid.uuid4().hex)
        self.model_id = '%s/%s' % (TEST_MODEL_ORG, self.model_name)
        self.revision = 'v0.1_test_revision'
        self.revision2 = 'v0.2_test_revision'
        self.api.create_model(
            model_id=self.model_id,
            visibility=ModelVisibility.PUBLIC,
            license=Licenses.APACHE_V2,
            chinese_name=TEST_MODEL_CHINESE_NAME,
        )
        names_to_remove = {MODELSCOPE_SDK_DEBUG}
        self.modified_environ = {
            k: v
            for k, v in os.environ.items() if k not in names_to_remove
        }

    def tearDown(self):
        self.api.delete_model(model_id=self.model_id)

    def prepare_repo_data(self):
        temporary_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(temporary_dir, self.model_name)
        self.repo = Repository(self.model_dir, clone_from=self.model_id)
        os.system("echo 'testtest'>%s"
                  % os.path.join(self.model_dir, download_model_file_name))
        self.repo.push('add model')

    def prepare_repo_data_and_tag(self):
        self.prepare_repo_data()
        self.repo.tag_and_push(self.revision, 'Test revision')

    def add_new_file_and_tag_to_repo(self):
        os.system("echo 'testtest'>%s"
                  % os.path.join(self.model_dir, download_model_file_name2))
        self.repo.push('add new file')
        self.repo.tag_and_push(self.revision2, 'Test revision')

    def add_new_file_and_branch_to_repo(self, branch_name):
        os.system("echo 'testtest'>%s"
                  % os.path.join(self.model_dir, download_model_file_name2))
        self.repo.push('add new file', remote_branch=branch_name)

    def test_dev_mode_default_master(self):
        with mock.patch.dict(os.environ, self.modified_environ, clear=True):
            self.prepare_repo_data()  # no tag, default get master
            with tempfile.TemporaryDirectory() as temp_cache_dir:
                snapshot_path = snapshot_download(
                    self.model_id, cache_dir=temp_cache_dir)
                assert os.path.exists(
                    os.path.join(snapshot_path, download_model_file_name))
            with tempfile.TemporaryDirectory() as temp_cache_dir:
                file_path = model_file_download(
                    self.model_id,
                    download_model_file_name,
                    cache_dir=temp_cache_dir)
                assert os.path.exists(file_path)

    def test_dev_mode_specify_branch(self):
        with mock.patch.dict(os.environ, self.modified_environ, clear=True):
            self.prepare_repo_data()  # no tag, default get master
            branch_name = 'test'
            self.add_new_file_and_branch_to_repo(branch_name)
            with tempfile.TemporaryDirectory() as temp_cache_dir:
                snapshot_path = snapshot_download(
                    self.model_id,
                    revision=branch_name,
                    cache_dir=temp_cache_dir)
                assert os.path.exists(
                    os.path.join(snapshot_path, download_model_file_name))
            with tempfile.TemporaryDirectory() as temp_cache_dir:
                file_path = model_file_download(
                    self.model_id,
                    download_model_file_name,
                    revision=branch_name,
                    cache_dir=temp_cache_dir)
                assert os.path.exists(file_path)

    def test_snapshot_download_revision(self):
        with mock.patch.dict(os.environ, self.modified_environ, clear=True):
            self.prepare_repo_data_and_tag()
            t1 = datetime.now().isoformat(sep=' ', timespec='seconds')
            logger.info('First time: %s' % t1)
            time.sleep(10)
            self.add_new_file_and_tag_to_repo()
            t2 = datetime.now().isoformat(sep=' ', timespec='seconds')
            logger.info('Second time: %s' % t2)
            # set
            release_datetime_backup = version.__release_datetime__
            logger.info('Origin __release_datetime__: %s'
                        % version.__release_datetime__)
            try:
                logger.info('Setting __release_datetime__ to: %s' % t1)
                version.__release_datetime__ = t1
                with tempfile.TemporaryDirectory() as temp_cache_dir:
                    snapshot_path = snapshot_download(
                        self.model_id, cache_dir=temp_cache_dir)
                    assert os.path.exists(
                        os.path.join(snapshot_path, download_model_file_name))
                    assert not os.path.exists(
                        os.path.join(snapshot_path, download_model_file_name2))
                version.__release_datetime__ = t2
                logger.info('Setting __release_datetime__ to: %s' % t2)
                with tempfile.TemporaryDirectory() as temp_cache_dir:
                    snapshot_path = snapshot_download(
                        self.model_id, cache_dir=temp_cache_dir)
                    assert os.path.exists(
                        os.path.join(snapshot_path, download_model_file_name))
                    assert os.path.exists(
                        os.path.join(snapshot_path, download_model_file_name2))
            finally:
                version.__release_datetime__ = release_datetime_backup

    def test_snapshot_download_revision_user_set_revision(self):
        with mock.patch.dict(os.environ, self.modified_environ, clear=True):
            self.prepare_repo_data_and_tag()
            t1 = datetime.now().isoformat(sep=' ', timespec='seconds')
            logger.info('First time: %s' % t1)
            time.sleep(10)
            self.add_new_file_and_tag_to_repo()
            t2 = datetime.now().isoformat(sep=' ', timespec='seconds')
            logger.info('Secnod time: %s' % t2)
            # set
            release_datetime_backup = version.__release_datetime__
            logger.info('Origin __release_datetime__: %s'
                        % version.__release_datetime__)
            try:
                logger.info('Setting __release_datetime__ to: %s' % t1)
                version.__release_datetime__ = t1
                with tempfile.TemporaryDirectory() as temp_cache_dir:
                    snapshot_path = snapshot_download(
                        self.model_id,
                        revision=self.revision,
                        cache_dir=temp_cache_dir)
                    assert os.path.exists(
                        os.path.join(snapshot_path, download_model_file_name))
                    assert not os.path.exists(
                        os.path.join(snapshot_path, download_model_file_name2))
                with tempfile.TemporaryDirectory() as temp_cache_dir:
                    snapshot_path = snapshot_download(
                        self.model_id,
                        revision=self.revision2,
                        cache_dir=temp_cache_dir)
                    assert os.path.exists(
                        os.path.join(snapshot_path, download_model_file_name))
                    assert os.path.exists(
                        os.path.join(snapshot_path, download_model_file_name2))
            finally:
                version.__release_datetime__ = release_datetime_backup

    def test_file_download_revision(self):
        with mock.patch.dict(os.environ, self.modified_environ, clear=True):
            self.prepare_repo_data_and_tag()
            t1 = datetime.now().isoformat(sep=' ', timespec='seconds')
            logger.info('First time stamp: %s' % t1)
            time.sleep(10)
            self.add_new_file_and_tag_to_repo()
            t2 = datetime.now().isoformat(sep=' ', timespec='seconds')
            logger.info('Second time: %s' % t2)
            release_datetime_backup = version.__release_datetime__
            logger.info('Origin __release_datetime__: %s'
                        % version.__release_datetime__)
            try:
                version.__release_datetime__ = t1
                logger.info('Setting __release_datetime__ to: %s' % t1)
                with tempfile.TemporaryDirectory() as temp_cache_dir:
                    file_path = model_file_download(
                        self.model_id,
                        download_model_file_name,
                        cache_dir=temp_cache_dir)
                    assert os.path.exists(file_path)
                    with self.assertRaises(NotExistError):
                        model_file_download(
                            self.model_id,
                            download_model_file_name2,
                            cache_dir=temp_cache_dir)
                version.__release_datetime__ = t2
                logger.info('Setting __release_datetime__ to: %s' % t2)
                with tempfile.TemporaryDirectory() as temp_cache_dir:
                    file_path = model_file_download(
                        self.model_id,
                        download_model_file_name,
                        cache_dir=temp_cache_dir)
                    assert os.path.exists(file_path)
                    file_path = model_file_download(
                        self.model_id,
                        download_model_file_name2,
                        cache_dir=temp_cache_dir)
                    assert os.path.exists(file_path)
            finally:
                version.__release_datetime__ = release_datetime_backup

    def test_file_download_revision_user_set_revision(self):
        with mock.patch.dict(os.environ, self.modified_environ, clear=True):
            self.prepare_repo_data_and_tag()
            t1 = datetime.now().isoformat(sep=' ', timespec='seconds')
            logger.info('First time stamp: %s' % t1)
            time.sleep(10)
            self.add_new_file_and_tag_to_repo()
            t2 = datetime.now().isoformat(sep=' ', timespec='seconds')
            logger.info('Second time: %s' % t2)
            release_datetime_backup = version.__release_datetime__
            logger.info('Origin __release_datetime__: %s'
                        % version.__release_datetime__)
            try:
                version.__release_datetime__ = t1
                logger.info('Setting __release_datetime__ to: %s' % t1)
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
                    assert os.path.exists(file_path)
                    file_path = model_file_download(
                        self.model_id,
                        download_model_file_name2,
                        revision=self.revision2,
                        cache_dir=temp_cache_dir)
                    assert os.path.exists(file_path)
            finally:
                version.__release_datetime__ = release_datetime_backup


if __name__ == '__main__':
    unittest.main()
