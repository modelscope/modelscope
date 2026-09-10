import unittest
from unittest import mock

from modelscope.hub.api import HubApi
from modelscope.utils.constant import REPO_TYPE_MODEL


class HubApiDeleteFilesCompatTest(unittest.TestCase):

    def test_delete_patterns_are_available_from_modelscope_hub_api(self):
        api = HubApi()
        expected = {
            'deleted_files': ['config.json'],
            'failed_files': [],
            'total_files': 1,
        }

        with mock.patch.object(
                api._api, 'delete_files',
                return_value=expected) as delete_files:
            result = api.delete_files(
                repo_id='owner/repo',
                repo_type=REPO_TYPE_MODEL,
                delete_patterns='*.json',
                revision='main')

        delete_files.assert_called_once_with(
            'owner/repo',
            REPO_TYPE_MODEL,
            file_paths=None,
            delete_patterns='*.json',
            commit_message=None,
            revision='main')
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
