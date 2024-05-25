import os
import os.path as osp
import shutil
import subprocess
import tempfile
import unittest
import uuid

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility
from modelscope.hub.repository import Repository
from modelscope.utils.test_utils import (TEST_ACCESS_TOKEN1,
                                         TEST_MODEL_CHINESE_NAME,
                                         TEST_MODEL_ORG)

DEFAULT_GIT_PATH = 'git'
download_model_file_name = 'test.bin'


class DownloadCMDTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.api = HubApi()
        self.api.login(TEST_ACCESS_TOKEN1)
        self.model_name = 'op-%s' % (uuid.uuid4().hex)
        self.model_id = '%s/%s' % (TEST_MODEL_ORG, self.model_name)
        self.revision = 'v0.1_test_revision'
        self.api.create_model(
            model_id=self.model_id,
            visibility=ModelVisibility.PUBLIC,
            license=Licenses.APACHE_V2,
            chinese_name=TEST_MODEL_CHINESE_NAME,
        )
        self.prepare_case()

    def prepare_case(self):
        temporary_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(temporary_dir, self.model_name)
        repo = Repository(self.model_dir, clone_from=self.model_id)
        os.system("echo 'testtest'>%s"
                  % os.path.join(self.model_dir, download_model_file_name))
        repo.push('add model')
        repo.tag_and_push(self.revision, 'Test revision')

    def tearDown(self):
        self.api.delete_model(model_id=self.model_id)
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_download(self):
        cmd = f'python -m modelscope.cli.cli download --model {self.model_id}'
        stat, output = subprocess.getstatusoutput(cmd)
        self.assertEqual(stat, 0)

    def test_download_with_cache(self):
        cmd = f'python -m modelscope.cli.cli download --model {self.model_id} --cache_dir {self.tmp_dir}'
        stat, output = subprocess.getstatusoutput(cmd)
        if stat != 0:
            print(output)
        self.assertEqual(stat, 0)
        self.assertTrue(
            osp.exists(
                f'{self.tmp_dir}/{self.model_id}/{download_model_file_name}'))

    def test_download_with_revision(self):
        cmd = f'python -m modelscope.cli.cli download --model {self.model_id} --revision {self.revision}'
        stat, output = subprocess.getstatusoutput(cmd)
        if stat != 0:
            print(output)
        self.assertEqual(stat, 0)


if __name__ == '__main__':
    unittest.main()
