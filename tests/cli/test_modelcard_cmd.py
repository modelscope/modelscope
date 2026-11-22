import argparse
import ntpath
import os
import shutil
import subprocess
import tempfile
import unittest
import uuid
from unittest import mock

from modelscope.cli.modelcard import ModelCardCMD
from modelscope.hub.api import HubApi
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import TEST_ACCESS_TOKEN1, TEST_MODEL_ORG

logger = get_logger()

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
        try:
            self.api.delete_model(model_id=self.model_id)
        except Exception as e:
            logger.warning(f'Failed to delete model {self.model_id}: {e}')

        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    @unittest.skip('Pipeline wrapper generation issue, not hub-related')
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


class ModelCardCMDModelIdTest(unittest.TestCase):

    def _build_cmd(self, model_id, group_id='damo'):
        args = argparse.Namespace(
            access_token=None, group_id=group_id, model_id=model_id)
        # Use Windows path rules to check that the model id and the url
        # are built with '/' on every platform.
        hub_api = mock.patch('modelscope.cli.modelcard.HubApi')
        endpoint = mock.patch(
            'modelscope.cli.modelcard.get_endpoint',
            return_value='https://www.modelscope.cn')
        with hub_api, endpoint, mock.patch('os.path', ntpath):
            return ModelCardCMD(args)

    def test_model_id_joined_with_group_id(self):
        cmd = self._build_cmd('my-model', group_id='my-org')
        self.assertEqual(cmd.model_id, 'my-org/my-model')
        self.assertEqual(cmd.url, 'https://www.modelscope.cn/my-org/my-model')

    def test_full_model_id_kept(self):
        cmd = self._build_cmd('my-org/my-model')
        self.assertEqual(cmd.model_id, 'my-org/my-model')
        self.assertEqual(cmd.url, 'https://www.modelscope.cn/my-org/my-model')


if __name__ == '__main__':
    unittest.main()
