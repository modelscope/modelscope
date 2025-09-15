# Copyright (c) Alibaba, Inc. and its affiliates.
# yapf: disable

import datetime
import unittest
import zoneinfo
from unittest.mock import Mock, patch

from modelscope.hub.api import DatasetInfo, HubApi, ModelInfo
from modelscope.utils.constant import REPO_TYPE_DATASET, REPO_TYPE_MODEL
from modelscope.utils.repo_utils import DetailedCommitInfo
from modelscope.utils.test_utils import test_level


class HubRepoInfoTest(unittest.TestCase):

    def setUp(self):
        self.api = HubApi()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    @patch.object(HubApi, 'get_model')
    @patch.object(HubApi, 'list_repo_commits')
    def test_model_info(self, mock_list_repo_commits, mock_get_model):
        # Setup mock responses
        mock_get_model.return_value = {
            'Id': 123,
            'Name': 'demo-model',
            'ChineseName': '测试模型',
            'Description': 'A test model',
            'Tasks': [{
                'Name': 'text-classification',
                'Description': 'A test task'
            }],
            'Tags': ['nlp', 'text']
        }

        # Mock commit history response
        commit = DetailedCommitInfo(
            id='abc123',
            short_id='abc12',
            title='Initial commit',
            message='Initial commit',
            author_name='Test User',
            authored_date=None,
            author_email='test@example.com',
            committed_date=None,
            committer_name='Test User',
            committer_email='test@example.com',
            created_at=None)
        commits_response = Mock()
        commits_response.commits = [commit]
        mock_list_repo_commits.return_value = commits_response

        # Call the method
        info = self.api.model_info(
            repo_id='demo/model', revision='master', endpoint=None)

        # Verify results
        self.assertEqual(info.id, 123)
        self.assertEqual(info.name, 'demo-model')
        self.assertEqual(info.author, 'demo')
        self.assertEqual(info.chinese_name, '测试模型')
        self.assertEqual(info.description, 'A test model')
        self.assertEqual(info.tasks, [{
            'Name': 'text-classification',
            'Description': 'A test task'
        }])
        self.assertEqual(info.tags, ['nlp', 'text'])
        self.assertEqual(info.sha, 'abc123')
        self.assertEqual(info.last_commit, commit.to_dict())

        # Verify correct method calls
        mock_get_model.assert_called_once_with(
            model_id='demo/model', revision='master', endpoint=None)
        mock_list_repo_commits.assert_called_once_with(
            repo_id='demo/model',
            repo_type=REPO_TYPE_MODEL,
            revision='master',
            endpoint=None)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    @patch.object(HubApi, 'get_dataset')
    @patch.object(HubApi, 'list_repo_commits')
    def test_dataset_info(self, mock_list_repo_commits, mock_get_dataset):
        # Setup mock responses
        mock_get_dataset.return_value = {
            'Id': 456,
            'Name': 'demo-dataset',
            'ChineseName': '演示数据集',
            'Description': 'A test dataset',
            'Tags': [
                {
                    'Name': 'nlp',
                    'Color': 'blue'
                },
                {
                    'Name': 'text',
                    'Color': 'green'
                }
            ]
        }

        # Mock commit history response
        commits = [
            DetailedCommitInfo(
                id='c1',
                short_id='c1',
                title='Update data',
                message='Update data',
                author_name='Test User',
                authored_date=None,
                author_email='test@example.com',
                committed_date=None,
                committer_name='Test User',
                committer_email='test@example.com',
                created_at=None),
            DetailedCommitInfo(
                id='c2',
                short_id='c2',
                title='Initial commit',
                message='Initial commit',
                author_name='Test User',
                authored_date=None,
                author_email='test@example.com',
                committed_date=1756284063,
                committer_name='Test User',
                committer_email='test@example.com',
                created_at=None)
        ]
        commits_response = Mock()
        commits_response.commits = commits
        mock_list_repo_commits.return_value = commits_response

        # Call the method
        info = self.api.dataset_info('demo/dataset')

        # Verify results
        self.assertEqual(info.id, 456)
        self.assertEqual(info.name, 'demo-dataset')
        self.assertEqual(info.author, 'demo')
        self.assertEqual(info.chinese_name, '演示数据集')
        self.assertEqual(info.description, 'A test dataset')
        self.assertEqual(info.tags, [{
            'Name': 'nlp',
            'Color': 'blue'
        }, {
            'Name': 'text',
            'Color': 'green'
        }])
        self.assertEqual(info.sha, 'c1')
        self.assertEqual(info.last_commit, commits[0].to_dict())

        # Verify correct method calls
        mock_get_dataset.assert_called_once_with(
            dataset_id='demo/dataset', revision=None, endpoint=None)
        mock_list_repo_commits.assert_called_once_with(
            repo_id='demo/dataset',
            repo_type=REPO_TYPE_DATASET,
            revision=None,
            endpoint=None)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    @patch.object(HubApi, 'model_info')
    @patch.object(HubApi, 'dataset_info')
    def test_repo_info_model(self, mock_dataset_info, mock_model_info):
        # Setup mock response
        model_info = ModelInfo(
            id=123, name='demo-model', description='A test model')
        mock_model_info.return_value = model_info

        # Call the method with model type
        info = self.api.repo_info(
            repo_id='demo/model', revision='master', endpoint=None)

        # Verify results
        self.assertEqual(info, model_info)
        mock_model_info.assert_called_once_with(
            repo_id='demo/model', revision='master', endpoint=None)
        mock_dataset_info.assert_not_called()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    @patch.object(HubApi, 'model_info')
    @patch.object(HubApi, 'dataset_info')
    def test_repo_info_dataset(self, mock_dataset_info, mock_model_info):
        # Setup mock response
        dataset_info = DatasetInfo(
            id=456, name='demo-dataset', description='A test dataset')
        mock_dataset_info.return_value = dataset_info

        # Call the method with dataset type
        info = self.api.repo_info(
            repo_id='demo/dataset',
            repo_type=REPO_TYPE_DATASET,
            revision='master',
            endpoint=None)

        # Verify results
        self.assertEqual(info, dataset_info)
        mock_dataset_info.assert_called_once_with(
            repo_id='demo/dataset', revision='master', endpoint=None)
        mock_model_info.assert_not_called()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_model_info_class_comprehensive(self):
        """Test ModelInfo class initialization and properties."""
        model_data = {
            'Id': 123,
            'Name': 'demo-model',
            'author': 'demo',
            'ChineseName': '演示模型',
            'Description': 'A test model',
            'Tasks': [{
                'Name': 'text-classification',
                'Description': 'A test task'
            }],
            'Tags': ['nlp', 'text'],
            'CreatedTime': '2023-01-01T00:00:00Z',
            'Visibility': 5,
            'IsPublished': 1,
            'IsOnline': 1,
            'License': 'Apache-2.0',
            'Downloads': 100,
            'Stars': 50,
            'Architectures': ['transformer'],
            'ModelType': ['nlp']
        }

        # Create mock commits
        commit = DetailedCommitInfo(
            id='abc123',
            short_id='abc12',
            title='Initial commit',
            message='Initial commit',
            author_name='Test User',
            authored_date=None,
            author_email='test@example.com',
            committed_date=None,
            committer_name='Test User',
            committer_email='test@example.com',
            created_at=None)
        commits = Mock()
        commits.commits = [commit]

        # Create ModelInfo instance
        model_info = ModelInfo(**model_data, commits=commits)

        # Verify properties
        self.assertEqual(model_info.id, 123)
        self.assertEqual(model_info.name, 'demo-model')
        self.assertEqual(model_info.author, 'demo')
        self.assertEqual(model_info.chinese_name, '演示模型')
        self.assertEqual(model_info.description, 'A test model')
        self.assertEqual(model_info.tasks, [{
            'Name': 'text-classification',
            'Description': 'A test task'
        }])
        self.assertEqual(model_info.tags, ['nlp', 'text'])
        self.assertEqual(
            model_info.created_at,
            datetime.datetime(
                2023, 1, 1, 8, 0, 0).replace(tzinfo=zoneinfo.ZoneInfo('Asia/Shanghai')))
        self.assertEqual(model_info.sha, 'abc123')
        self.assertEqual(model_info.last_commit, commit.to_dict())
        self.assertEqual(model_info.visibility, 5)
        self.assertEqual(model_info.is_published, 1)
        self.assertEqual(model_info.is_online, 1)
        self.assertEqual(model_info.license, 'Apache-2.0')
        self.assertEqual(model_info.downloads, 100)
        self.assertEqual(model_info.likes, 50)
        self.assertEqual(model_info.architectures, ['transformer'])
        self.assertEqual(model_info.model_type, ['nlp'])

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_dataset_info_class(self):
        # Test DatasetInfo class initialization and properties
        dataset_data = {
            'Id': 456,
            'Name': 'demo-dataset',
            'Owner': 'demo',
            'ChineseName': '演示数据集',
            'Description': 'A test dataset',
            'Tags': [
                {
                    'Name': 'nlp',
                    'Color': 'blue'
                },
                {
                    'Name': 'text',
                    'Color': 'green'
                }
            ],
            'GmtCreate': 1755752511,
            'Visibility': 5,
            'License': 'MIT',
            'Downloads': 200,
            'Likes': 75
        }

        # Create mock commits
        commit = DetailedCommitInfo(
            id='c1',
            short_id='c1',
            title='Initial commit',
            message='Initial commit',
            author_name='Test User',
            authored_date=None,
            author_email='test@example.com',
            committed_date='2024-09-18T06:20:05Z',
            committer_name='Test User',
            committer_email='test@example.com',
            created_at=None)
        commits = Mock()
        commits.commits = [commit]

        # Create DatasetInfo instance
        dataset_info = DatasetInfo(**dataset_data, commits=commits)

        # Verify properties
        self.assertEqual(dataset_info.id, 456)
        self.assertEqual(dataset_info.name, 'demo-dataset')
        self.assertEqual(dataset_info.author, 'demo')
        self.assertEqual(dataset_info.chinese_name, '演示数据集')
        self.assertEqual(dataset_info.description, 'A test dataset')
        self.assertEqual(
            dataset_info.tags,
            [
                {
                    'Name': 'nlp',
                    'Color': 'blue'
                },
                {
                    'Name': 'text',
                    'Color': 'green'
                }
            ]
        )
        self.assertEqual(
            dataset_info.created_at,
            datetime.datetime(
                2025, 8, 21, 13, 1, 51).replace(tzinfo=zoneinfo.ZoneInfo('Asia/Shanghai')))
        self.assertEqual(dataset_info.sha, 'c1')
        self.assertEqual(
            dataset_info.last_modified,
            datetime.datetime(
                2024, 9, 18, 14, 20, 5).replace(tzinfo=zoneinfo.ZoneInfo('Asia/Shanghai')))
        self.assertEqual(dataset_info.last_commit, commit.to_dict())
        self.assertEqual(dataset_info.visibility, 5)
        self.assertEqual(dataset_info.license, 'MIT')
        self.assertEqual(dataset_info.downloads, 200)
        self.assertEqual(dataset_info.likes, 75)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_model_info_empty_commits(self):
        """Test ModelInfo with empty commits."""
        model_data = {'Id': 123, 'Name': 'demo-model', 'author': 'demo'}

        # Create ModelInfo with no commits
        model_info = ModelInfo(**model_data, commits=None)

        # Verify commit-related fields are None
        self.assertIsNone(model_info.sha)
        self.assertIsNone(model_info.last_commit)
        self.assertIsNone(model_info.last_modified)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_dataset_info_empty_commits(self):
        """Test DatasetInfo with empty commits."""
        dataset_data = {'Id': 456, 'Name': 'demo-dataset', 'Owner': 'demo'}

        # Create DatasetInfo with no commits
        dataset_info = DatasetInfo(**dataset_data, commits=None)

        # Verify commit-related fields are None
        self.assertIsNone(dataset_info.sha)
        self.assertIsNone(dataset_info.last_commit)
        self.assertIsNone(dataset_info.last_modified)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_detailed_commit_info_to_dict(self):
        """Test DetailedCommitInfo to_dict method."""
        commit = DetailedCommitInfo(
            id='abc123',
            short_id='abc12',
            title='Test commit',
            message='Test commit message',
            author_name='Test Author',
            authored_date=datetime.datetime(2023, 1, 1, 0, 0, 0),
            author_email='test@example.com',
            committed_date=datetime.datetime(2023, 1, 1, 0, 0, 0),
            committer_name='Test Committer',
            committer_email='committer@example.com',
            created_at=datetime.datetime(2023, 1, 1, 0, 0, 0))

        result = commit.to_dict()

        expected = {
            'id': 'abc123',
            'short_id': 'abc12',
            'title': 'Test commit',
            'message': 'Test commit message',
            'author_name': 'Test Author',
            'authored_date': datetime.datetime(2023, 1, 1, 0, 0, 0),
            'author_email': 'test@example.com',
            'committed_date': datetime.datetime(2023, 1, 1, 0, 0, 0),
            'committer_name': 'Test Committer',
            'committer_email': 'committer@example.com',
            'created_at': datetime.datetime(2023, 1, 1, 0, 0, 0)
        }

        self.assertEqual(result, expected)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_real_model_repo_info(self):
        """Test getting real model repository information without mocks."""
        # Use a real model repository
        model_repo_id = 'black-forest-labs/FLUX.1-Krea-dev'

        # Get repository information
        info = self.api.repo_info(
            repo_id=model_repo_id, repo_type=REPO_TYPE_MODEL)

        # Basic validation
        self.assertIsNotNone(info)
        self.assertEqual(info.author, 'black-forest-labs')
        self.assertEqual(info.name, 'FLUX.1-Krea-dev')

        # Check commit information
        self.assertIsNotNone(info.sha)
        if hasattr(info, 'last_commit') and info.last_commit:
            self.assertIn('id', info.last_commit)
            self.assertIn('title', info.last_commit)

        # Print some information for debugging
        print(f'\nModel Info for {model_repo_id}:')
        print(f'ID: {info.id}')
        print(f'Name: {info.name}')
        print(f'Author: {info.author}')
        print(f'SHA: {info.sha}')
        if hasattr(info, 'last_modified'):
            print(f'Last Modified: {info.last_modified}')

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_real_dataset_repo_info(self):
        """Test getting real dataset repository information without mocks."""
        # Use a real dataset repository
        dataset_repo_id = 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT'

        # Get repository information
        info = self.api.repo_info(
            repo_id=dataset_repo_id, repo_type=REPO_TYPE_DATASET)

        # Basic validation
        self.assertIsNotNone(info)
        self.assertEqual(info.author, 'swift')
        self.assertTrue('Chinese-Qwen3' in info.name)

        # Check commit information
        self.assertIsNotNone(info.sha)
        if hasattr(info, 'last_commit') and info.last_commit:
            self.assertIn('id', info.last_commit)
            self.assertIn('title', info.last_commit)

        # Print some information for debugging
        print(f'\nDataset Info for {dataset_repo_id}:')
        print(f'ID: {info.id}')
        print(f'Name: {info.name}')
        print(f'Author: {info.author}')
        print(f'SHA: {info.sha}')
        if hasattr(info, 'last_modified'):
            print(f'Last Modified: {info.last_modified}')


if __name__ == '__main__':
    unittest.main()
