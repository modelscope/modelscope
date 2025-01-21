import os
import os.path as osp
import shutil
import subprocess
import tempfile
import unittest
import uuid

from modelscope.hub.api import HubApi
from modelscope.utils.test_utils import TEST_ACCESS_TOKEN1, TEST_MODEL_ORG

os.environ['MKL_THREADING_LAYER'] = 'GNU'


class ModelUploadCMDTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        print(self.tmp_dir)
        self.api = HubApi()
        self.api.login(TEST_ACCESS_TOKEN1)
        self.task_name = 'task-%s' % (uuid.uuid4().hex)
        self.model_name = 'op-%s' % (uuid.uuid4().hex)
        self.model_id = '%s/%s' % (TEST_MODEL_ORG, self.model_name)
        print(self.tmp_dir, self.task_name, self.model_name)

    def tearDown(self):
        self.api.delete_model(model_id=self.model_id)
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_upload_modelcard(self):
        cmd = f'python -m modelscope.cli.cli pipeline --action create --task_name {self.task_name} ' \
              f'--save_file_path {self.tmp_dir} --configuration_path {self.tmp_dir}'
        stat, output = subprocess.getstatusoutput(cmd)
        if stat != 0:
            print(output)

        cmd = f'python {self.tmp_dir}/ms_wrapper.py'
        stat, output = subprocess.getstatusoutput(cmd)
        if stat != 0:
            print(output)
        self.assertEqual(stat, 0)

        cmd = f'python -m modelscope.cli.cli modelcard --action upload -tk {TEST_ACCESS_TOKEN1} ' \
              f'--model_id {self.model_id} --model_dir {self.tmp_dir}'
        stat, output = subprocess.getstatusoutput(cmd)
        if stat != 0:
            print(output)


if __name__ == '__main__':
    unittest.main()
