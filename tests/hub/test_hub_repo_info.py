# Copyright (c) Alibaba, Inc. and its affiliates.
# yapf: disable

import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from modelscope_hub.config import HubConfig
from modelscope_hub.types import RepoInfo
from tests.hub.test_hub_revision import _RepoTransport

from modelscope.hub.api import HubApi
from modelscope.hub.info import BlobLfsInfo, DatasetInfo, ModelInfo
from modelscope.utils.constant import REPO_TYPE_DATASET, REPO_TYPE_MODEL
from modelscope.utils.repo_utils import DetailedCommitInfo


class HubRepoInfoTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.transport = _RepoTransport()
        config = HubConfig(
            endpoint=self.transport.endpoint,
            token='',
            config_dir=Path(self.temp.name) / 'home',
            cache_dir=Path(self.temp.name) / 'hub')
        for patcher in [
                patch('modelscope_hub.config._default_config', config),
                patch('requests.sessions.Session.send', side_effect=self.transport.send)
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.api = HubApi(endpoint=self.transport.endpoint, token='')

    def test_model_info(self):
        detail = self.api.model_info(self.transport.repo_id)
        self.assertEqual(detail, self.transport.model)
        self.assertEqual(detail['versions'], [{'version_id': 7}])
        legacy = self.api.get_model(self.transport.repo_id)
        self.assertEqual(legacy['Id'], 123)
        self.assertEqual(legacy['Name'], 'revision-model')
        self.assertTrue(all('/openapi/v1/models/' in request.url
                            for request in self.transport.requests))

    def test_dataset_info(self):
        detail = self.api.dataset_info(self.transport.repo_id)
        self.assertEqual(detail, self.transport.dataset)
        self.assertEqual(detail['primary_subject_area'], 'biology')
        legacy = self.api.get_dataset(self.transport.repo_id)
        self.assertIsInstance(legacy, RepoInfo)
        self.assertEqual(legacy.id, 456)
        self.assertTrue(all('/openapi/v1/datasets/' in request.url
                            for request in self.transport.requests))

    def test_get_repo_model(self):
        info = self.api.get_repo(self.transport.repo_id, REPO_TYPE_MODEL, revision='master')
        self.assertIsInstance(info, RepoInfo)
        self.assertEqual(info.id, 123)
        self.assertEqual(info.owner, 'test')
        self.assertEqual(info.name, 'revision-model')
        self.assertEqual(info.model_category, 'aigc')

    def test_get_repo_dataset(self):
        info = self.api.get_repo(self.transport.repo_id, REPO_TYPE_DATASET, revision='master')
        self.assertIsInstance(info, RepoInfo)
        self.assertEqual(info.id, 456)
        self.assertEqual(info.owner, 'test')
        self.assertEqual(info.dataset_category, 'science')

    def test_model_info_class_comprehensive(self):
        """Test ModelInfo class initialization and properties."""
        import zoneinfo
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
            'ModelType': ['nlp'],
            'Siblings': [
                {
                    'Path': 'model-1.bin',
                    'Size': 100,
                    'Sha256': '1234567890',
                    'Type': 'blob'
                },
                {
                    'Path': 'model-2.bin',
                    'Size': 200,
                    'Sha256': '1234567891',
                    'Type': 'blob'
                }
            ]
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
        self.assertEqual(model_info.siblings[0].__dict__, {
            'rfilename': 'model-1.bin',
            'size': 100,
            'blob_id': '1234567890',
            'type': 'blob',
            'sha': None,
            'last_modified': None,
            'lfs': BlobLfsInfo(
                size=100,
                sha256='1234567890'
            )
        })
        self.assertEqual(model_info.siblings[1].__dict__, {
            'rfilename': 'model-2.bin',
            'size': 200,
            'blob_id': '1234567891',
            'type': 'blob',
            'sha': None,
            'last_modified': None,
            'lfs': BlobLfsInfo(
                size=200,
                sha256='1234567891'
            )
        })

    def test_dataset_info_class(self):
        # Test DatasetInfo class initialization and properties
        import zoneinfo
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
        siblings = [
            {
                'Path': 'data-1.json',
                'Size': 100,
                'Sha256': '1234567890',
                'Type': 'blob'
            }
        ]

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
        dataset_info = DatasetInfo(**dataset_data, commits=commits, siblings=siblings)

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
        self.assertEqual(dataset_info.siblings[0].__dict__, {
            'rfilename': 'data-1.json',
            'size': 100,
            'blob_id': '1234567890',
            'type': 'blob',
            'sha': None,
            'last_modified': None,
            'lfs': BlobLfsInfo(
                size=100,
                sha256='1234567890'
            )
        })

    def test_model_info_empty_commits(self):
        """Test ModelInfo with empty commits."""
        model_data = {'Id': 123, 'Name': 'demo-model', 'author': 'demo'}

        # Create ModelInfo with no commits
        model_info = ModelInfo(**model_data, commits=None)

        # Verify commit-related fields are None
        self.assertIsNone(model_info.sha)
        self.assertIsNone(model_info.last_commit)
        self.assertIsNone(model_info.last_modified)

    def test_dataset_info_empty_commits(self):
        """Test DatasetInfo with empty commits."""
        dataset_data = {'Id': 456, 'Name': 'demo-dataset', 'Owner': 'demo'}

        # Create DatasetInfo with no commits
        dataset_info = DatasetInfo(**dataset_data, commits=None)

        # Verify commit-related fields are None
        self.assertIsNone(dataset_info.sha)
        self.assertIsNone(dataset_info.last_commit)
        self.assertIsNone(dataset_info.last_modified)

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


if __name__ == '__main__':
    unittest.main()
