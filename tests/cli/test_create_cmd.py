import subprocess
import unittest
import uuid

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses
from modelscope.utils.test_utils import (TEST_ACCESS_TOKEN1,
                                         TEST_MODEL_CHINESE_NAME,
                                         TEST_MODEL_ORG, test_level)


class CreateCMDTest(unittest.TestCase):
    """
    Repository creation tests for the ModelScope CLI.

    Usage:
        modelscope create <repo_id> --token <token> --repo_type <model/dataset> --visibility <public/internal/private> --chinese_name <chinese_name> --license <license>  # noqa: E501
    """

    def setUp(self):
        print(f'Running test {type(self).__name__}.{self._testMethodName}')

        tmp_suffix: str = uuid.uuid4().hex[:4]
        self.repo_id: str = f'{TEST_MODEL_ORG}/test_create_model_{tmp_suffix}'
        self.repo_type: str = 'model'
        self.visibility: str = 'private'
        self.chinese_name: str = f'{TEST_MODEL_CHINESE_NAME}_{tmp_suffix}'
        self.license: str = Licenses.MIT
        self.token: str = TEST_ACCESS_TOKEN1

    def tearDown(self):
        api = HubApi()
        api.login(self.token)

        try:
            api.delete_model(model_id=self.repo_id)
        except Exception as e:
            print(f'Error deleting model {self.repo_id}: {e}')

        print(f'Test {type(self).__name__}.{self._testMethodName} finished')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_create_repo_cmd(self):

        cmd: str = f'python -m modelscope.cli.cli create {self.repo_id} --token {self.token} --repo_type {self.repo_type} --visibility {self.visibility} --chinese_name {self.chinese_name} --license {self.license}'  # noqa: E501

        stat, output = subprocess.getstatusoutput(cmd)
        self.assertEqual(
            stat, 0, msg=f'Command failed: {cmd}\nOutput: {output}')


if __name__ == '__main__':
    unittest.main()
