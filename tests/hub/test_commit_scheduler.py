import os
import shutil
import tempfile
import time
import unittest
import uuid
from io import SEEK_END
from pathlib import Path
from unittest.mock import MagicMock, patch

from modelscope.hub.api import HubApi
from modelscope.hub.commit_scheduler import CommitScheduler, PartialFileIO
from modelscope.hub.errors import NotExistError
from modelscope.hub.repository import Repository
from modelscope.utils.constant import DEFAULT_REPOSITORY_REVISION
from modelscope.utils.repo_utils import CommitInfo, CommitOperationAdd
from modelscope.utils.test_utils import (TEST_ACCESS_TOKEN1, TEST_MODEL_ORG,
                                         delete_credential, test_level)


class TestCommitScheduler(unittest.TestCase):
    """Test suite for ModelScope CommitScheduler functionality."""

    def setUp(self) -> None:
        """Set up test environment with temporary directories and mock API."""
        self.api = HubApi()
        self.repo_name = f'test-commit-scheduler-{uuid.uuid4().hex[:8]}'
        self.repo_id = f'{TEST_MODEL_ORG}/{self.repo_name}'

        # Create temporary cache directory
        self.cache_dir = Path(tempfile.mkdtemp())
        self.watched_folder = self.cache_dir / 'watched_folder'
        self.watched_folder.mkdir(exist_ok=True, parents=True)

        # Initialize scheduler reference for cleanup
        self.scheduler = None

    def tearDown(self) -> None:
        """Clean up test resources."""
        # Stop scheduler if it exists
        if self.scheduler is not None:
            try:
                self.scheduler.stop()
            except Exception:
                pass

        # Clean up temporary directories
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)

        # Try to delete test repo (may not exist for mocked tests)
        try:
            if hasattr(self, 'api') and TEST_ACCESS_TOKEN1:
                self.api.login(TEST_ACCESS_TOKEN1)
                self.api.delete_repo(repo_id=self.repo_id, repo_type='dataset')
        except Exception:
            pass

        delete_credential()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_invalid_folder_path_nonexistent(self) -> None:
        """Test that CommitScheduler raises error for non-existent folder."""
        nonexistent_path = self.cache_dir / 'nonexistent'

        with self.assertRaises(ValueError) as cm:
            CommitScheduler(
                repo_id=self.repo_id,
                folder_path=nonexistent_path,
                interval=1,
                hub_api=self.api,
                repo_type='dataset',
                token=TEST_ACCESS_TOKEN1)
        self.assertIn('does not exist', str(cm.exception))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_invalid_interval(self) -> None:
        """Test that CommitScheduler raises error for invalid interval values."""
        # Test zero interval
        with self.assertRaises(ValueError) as cm:
            CommitScheduler(
                repo_id=self.repo_id,
                folder_path=self.watched_folder,
                interval=0,
                hub_api=self.api,
                repo_type='dataset',
                token=TEST_ACCESS_TOKEN1)
        self.assertIn('positive', str(cm.exception))

        # Test negative interval
        with self.assertRaises(ValueError) as cm:
            CommitScheduler(
                repo_id=self.repo_id,
                folder_path=self.watched_folder,
                interval=-1,
                hub_api=self.api,
                repo_type='dataset',
                token=TEST_ACCESS_TOKEN1)
        self.assertIn('positive', str(cm.exception))

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_initialization_with_defaults(self) -> None:
        """Test CommitScheduler initialization with default parameters."""
        with patch.object(HubApi, 'create_repo') as mock_create:
            mock_create.return_value = f'https://modelscope.cn/datasets/{self.repo_id}'

            self.scheduler = CommitScheduler(
                repo_id=self.repo_id,
                folder_path=self.watched_folder,
                hub_api=self.api,
                repo_type='dataset',
                token=TEST_ACCESS_TOKEN1)

            # Check default values
            self.assertEqual(self.scheduler.repo_id, self.repo_id)
            self.assertEqual(self.scheduler.folder_path,
                             self.watched_folder.resolve())
            self.assertEqual(self.scheduler.interval, 5)  # default 5 minutes
            self.assertEqual(self.scheduler.path_in_repo, '')
            self.assertEqual(self.scheduler.revision,
                             DEFAULT_REPOSITORY_REVISION)
            self.assertIsInstance(self.scheduler.last_uploaded, dict)

            # Verify create_repo was called
            mock_create.assert_called_once()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_initialization_with_custom_parameters(self) -> None:
        """Test CommitScheduler initialization with custom parameters."""
        custom_interval = 10
        custom_path_in_repo = 'custom/path'
        custom_revision = 'develop'
        allow_patterns = ['*.txt', '*.json']
        ignore_patterns = ['*.log', 'temp/*']

        with patch.object(HubApi, 'create_repo') as mock_create:
            mock_create.return_value = f'https://modelscope.cn/datasets/{self.repo_id}'

            self.scheduler = CommitScheduler(
                repo_id=self.repo_id,
                folder_path=self.watched_folder,
                interval=custom_interval,
                path_in_repo=custom_path_in_repo,
                revision=custom_revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                private=True,
                hub_api=self.api,
                repo_type='dataset',
                token=TEST_ACCESS_TOKEN1)

            # Check custom values
            self.assertEqual(self.scheduler.interval, custom_interval)
            self.assertEqual(self.scheduler.path_in_repo, custom_path_in_repo)
            self.assertEqual(self.scheduler.revision, custom_revision)
            self.assertEqual(self.scheduler.allow_patterns, allow_patterns)

            # Check ignore patterns include git folders
            expected_ignore = ignore_patterns + [
                '.git', '.git/*', '*/.git', '**/.git/**'
            ]
            self.assertEqual(self.scheduler.ignore_patterns, expected_ignore)

            # Verify create_repo was called with correct visibility
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            self.assertEqual(call_args.kwargs.get('visibility'), 'private')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    @patch.object(CommitScheduler, 'commit_scheduled_changes')
    def test_mocked_scheduler_execution(self, mock_push: MagicMock) -> None:
        """Test scheduler with mocked commit_scheduled_changes method."""
        mock_push.return_value = CommitInfo(
            commit_url=
            'https://modelscope.cn/datasets/test_scheduler_unit/commit/test123',
            commit_message='Test commit',
            commit_description='',
            oid='test123')

        with patch.object(HubApi, 'create_repo'):
            self.scheduler = CommitScheduler(
                repo_id=self.repo_id,
                folder_path=self.watched_folder,
                interval=1 / 60 / 10,  # every 0.1s for fast testing
                hub_api=self.api,
                repo_type='dataset',
                token=TEST_ACCESS_TOKEN1)

            # Wait for at least a couple of scheduler cycles
            time.sleep(0.5)

            # Should have been called multiple times
            self.assertGreater(len(mock_push.call_args_list), 1)

            # Check the last future result
            if hasattr(self.scheduler,
                       'last_future') and self.scheduler.last_future:
                result = self.scheduler.last_future.result()
                self.assertEqual(result.oid, 'test123')
                self.assertEqual(result.commit_message, 'Test commit')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_scheduler_stop(self) -> None:
        """Test stopping the scheduler."""
        with patch.object(HubApi, 'create_repo'):
            self.scheduler = CommitScheduler(
                repo_id=self.repo_id,
                folder_path=self.watched_folder,
                interval=1,  # 1 minute
                hub_api=self.api,
                repo_type='dataset',
                token=TEST_ACCESS_TOKEN1)

            # Scheduler should be running
            self.assertFalse(self.scheduler._CommitScheduler__stopped)

            # Stop the scheduler
            self.scheduler.stop()

            # Scheduler should be stopped
            self.assertTrue(self.scheduler._CommitScheduler__stopped)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_context_manager(self) -> None:
        """Test CommitScheduler as context manager."""
        file_path = self.watched_folder / 'test_file.txt'

        with patch.object(HubApi, 'create_repo'), \
             patch.object(CommitScheduler, 'commit_scheduled_changes') as mock_push:
            mock_push.return_value = CommitInfo(
                commit_url=
                'https://modelscope.cn/datasets/test_scheduler_unit/commit/test123',
                commit_message='Test commit',
                commit_description='',
                oid='test123')

            with CommitScheduler(
                    repo_id=self.repo_id,
                    folder_path=self.watched_folder,
                    interval=5,  # 5 minutes - won't trigger during test
                    hub_api=self.api,
                    repo_type='dataset',
                    token=TEST_ACCESS_TOKEN1) as scheduler:
                # Write a file inside the context
                file_path.write_text('test content')

                # Reference for later assertions
                self.scheduler = scheduler

            # After exiting context, scheduler should be stopped
            self.assertTrue(self.scheduler._CommitScheduler__stopped)

            # Should have triggered commit_scheduled_changes on exit
            mock_push.assert_called()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trigger_manual_commit(self) -> None:
        """Test manually triggering a commit."""
        with patch.object(HubApi, 'create_repo'), \
             patch.object(CommitScheduler, 'commit_scheduled_changes') as mock_push:
            mock_push.return_value = CommitInfo(
                commit_url=
                'https://modelscope.cn/datasets/test_scheduler_unit/commit/test123',
                commit_message='Manual commit',
                commit_description='',
                oid='manual123')

            self.scheduler = CommitScheduler(
                repo_id=self.repo_id,
                folder_path=self.watched_folder,
                interval=60,  # Long interval to avoid auto-trigger
                hub_api=self.api,
                repo_type='dataset',
                token=TEST_ACCESS_TOKEN1)
            future = self.scheduler.trigger()
            result = future.result()

            # Verify the result
            self.assertEqual(result.oid, 'manual123')
            self.assertEqual(result.commit_message, 'Manual commit')
            mock_push.assert_called()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_stopped_scheduler_no_push(self) -> None:
        """Test that stopped scheduler doesn't perform push operations."""
        with patch.object(HubApi, 'create_repo'):

            self.scheduler = CommitScheduler(
                repo_id=self.repo_id,
                folder_path=self.watched_folder,
                interval=60,
                hub_api=self.api,
                repo_type='dataset',
                token=TEST_ACCESS_TOKEN1)

            # Stop the scheduler immediately
            self.scheduler.stop()

            # Try to trigger after stopping
            future = self.scheduler.trigger()
            result = future.result()

            # Result should be None for stopped scheduler
            self.assertIsNone(result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_sync_local_folder_to_hub(self) -> None:
        """Test sync local folder to remote repo."""
        hub_cache = self.cache_dir / 'hub'
        file_path = self.watched_folder / 'file.txt'
        bin_path = self.watched_folder / 'file.bin'
        git_path = self.watched_folder / '.git'
        self.scheduler = CommitScheduler(
            folder_path=self.watched_folder,
            repo_id=self.repo_id,
            interval=1 / 60,
            hub_api=self.api,
            repo_type='dataset',
            token=TEST_ACCESS_TOKEN1)

        # 1 push to hub triggered (empty commit not pushed)
        time.sleep(0.5)

        # write content to files
        with file_path.open('a') as f:
            f.write('first line\n')
        with bin_path.open('a') as f:
            f.write('binary content')
        with git_path.open('a') as f:
            f.write('git content\n')

        # 2 push to hub triggered (2 commit + 1 ignored)
        time.sleep(2)
        self.scheduler.last_future.result()

        # new content in file
        with file_path.open('a') as f:
            f.write('second line\n')

        # 1 push to hub triggered (1 commit)
        time.sleep(1)
        self.scheduler.last_future.result()

        with bin_path.open('a') as f:
            f.write(' updated')

        # 5 push to hub triggered (1 commit)
        time.sleep(5)  # wait for every threads/uploads to complete
        self.scheduler.stop()
        self.scheduler.last_future.result()

        # wait for 20 seconds for repository to be updated
        time.sleep(20)

        # 4 commits expected (initial commit + 3 push to hub)
        repo_id = self.scheduler.repo_id
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = Repository(
                model_dir=tmp_dir,
                clone_from=f'{self.scheduler.repo_type}s/{repo_id}',
                auth_token=TEST_ACCESS_TOKEN1)
            git_wrapper = repo.git_wrapper
            log_args = ['-C', tmp_dir, 'log', '--format=%H']
            response = git_wrapper._run_git_command(*log_args)
            commits = response.stdout.decode('utf8').strip().split('\n')

            # Get last commit message
            last_commit = commits[0]
            msg_args = ['-C', tmp_dir, 'log', '-1', '--format=%B', last_commit]
            msg_response = git_wrapper._run_git_command(*msg_args)
            last_msg = msg_response.stdout.decode('utf8').strip()

            # Exclude dataset_infos.json commit if present
            commit_count = len(commits)
            if 'dataset_infos.json' in last_msg:
                commit_count -= 1
                commits = commits[1:]  # Remove the dataset_infos commit

            self.assertEqual(commit_count, 5)

        def _download(filename: str, revision: str) -> Path:
            from modelscope.hub.file_download import _repo_file_download
            return Path(
                _repo_file_download(
                    repo_id=repo_id,
                    file_path=filename,
                    revision=revision,
                    cache_dir=hub_cache,
                    repo_type='dataset'))

        # Check file.txt consistency
        txt_push = _download(filename='file.txt', revision='master')
        self.assertEqual(txt_push.read_text(), 'first line\nsecond line\n')

        # Check file.bin consistency
        bin_push = _download(filename='file.bin', revision='master')
        self.assertEqual(bin_push.read_text(), 'binary content updated')

        # Check .git file was not uploaded (should be ignored)
        with self.assertRaises(NotExistError):
            _download(filename='.git', revision='master')


class TestPartialFileIO(unittest.TestCase):
    """Test suite for PartialFileIO functionality."""

    def setUp(self) -> None:
        """Set up test environment with temporary file."""
        self.cache_dir = Path(tempfile.mkdtemp())
        self.test_file = self.cache_dir / 'file.txt'
        self.test_file.write_text('123456789abcdef')  # 15 bytes

    def tearDown(self) -> None:
        """Clean up test resources."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_partial_file_read_with_limit(self) -> None:
        """Test reading partial file with size limit."""
        partial_file = PartialFileIO(self.test_file, size_limit=5)

        # Should read only first 5 bytes
        content = partial_file.read()
        self.assertEqual(content, b'12345')

        # Second read should return empty
        content = partial_file.read()
        self.assertEqual(content, b'')

        partial_file.close()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_partial_file_read_by_chunks(self) -> None:
        """Test reading partial file in chunks."""
        partial_file = PartialFileIO(self.test_file, size_limit=8)

        # Read in chunks
        chunk1 = partial_file.read(3)
        self.assertEqual(chunk1, b'123')

        chunk2 = partial_file.read(3)
        self.assertEqual(chunk2, b'456')

        chunk3 = partial_file.read(3)
        self.assertEqual(chunk3, b'78')  # Only 2 bytes left within limit

        chunk4 = partial_file.read(3)
        self.assertEqual(chunk4, b'')  # Nothing left

        partial_file.close()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_partial_file_read_oversized_chunk(self) -> None:
        """Test reading more than size limit in one go."""
        partial_file = PartialFileIO(self.test_file, size_limit=5)

        # Request more than limit
        content = partial_file.read(20)
        self.assertEqual(content, b'12345')  # Should return only up to limit

        partial_file.close()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_partial_file_len(self) -> None:
        """Test __len__ method returns correct size limit."""
        partial_file = PartialFileIO(self.test_file, size_limit=7)
        self.assertEqual(len(partial_file), 7)
        partial_file.close()

        # Size limit larger than actual file should be capped
        large_partial = PartialFileIO(self.test_file, size_limit=100)
        self.assertEqual(len(large_partial), 15)  # Actual file size
        large_partial.close()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_partial_file_seek_and_tell(self) -> None:
        """Test seek and tell operations."""
        partial_file = PartialFileIO(self.test_file, size_limit=10)

        # Initial position
        self.assertEqual(partial_file.tell(), 0)

        # Read some bytes
        partial_file.read(3)
        self.assertEqual(partial_file.tell(), 3)

        # Seek to beginning
        partial_file.seek(0)
        self.assertEqual(partial_file.tell(), 0)

        # Seek to specific position
        partial_file.seek(5)
        self.assertEqual(partial_file.tell(), 5)

        # Seek beyond limit should be capped
        partial_file.seek(50)
        self.assertEqual(partial_file.tell(), 10)  # Capped at size limit

        # Seek from end
        partial_file.seek(-2, SEEK_END)
        self.assertEqual(partial_file.tell(), 8)  # 10-2

        partial_file.close()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_partial_file_not_implemented_methods(self) -> None:
        """Test that unsupported methods raise NotImplementedError."""
        partial_file = PartialFileIO(self.test_file, size_limit=5)

        # These methods should not be implemented
        with self.assertRaises(NotImplementedError):
            partial_file.readline()

        with self.assertRaises(NotImplementedError):
            partial_file.write(b'test')

        partial_file.close()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_partial_file_repr(self) -> None:
        """Test string representation of PartialFileIO."""
        partial_file = PartialFileIO(self.test_file, size_limit=5)
        repr_str = repr(partial_file)

        self.assertEqual(
            repr_str,
            f'<PartialFileIO file_path={self.test_file} size_limit=5>')

        partial_file.close()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_file_modification_after_creation(self) -> None:
        """Test that file modifications after PartialFileIO creation don't affect size limit."""
        partial_file = PartialFileIO(self.test_file, size_limit=20)

        # Original length is capped at file size
        self.assertEqual(len(partial_file), 15)

        with self.test_file.open('ab') as f:
            f.write(b'additional_content')

        # Size limit should remain the same (captured at creation)
        self.assertEqual(len(partial_file), 15)

        # Content read should still be limited to original size
        content = partial_file.read()
        self.assertEqual(len(content), 15)
        self.assertEqual(content, b'123456789abcdef')

        partial_file.close()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_high_size_limit(self) -> None:
        """Test size limit larger than file size."""
        file = PartialFileIO(self.test_file, size_limit=20)
        with self.test_file.open('ab') as f:
            f.write(b'ghijkl')

        # File size limit is truncated to the actual file size at instance creation (not on the fly)
        self.assertEqual(len(file), 15)
        self.assertEqual(file._size_limit, 15)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_with_commit_operation_add(self) -> None:
        """Test with CommitOperationAdd."""
        op_truncated = CommitOperationAdd(
            path_or_fileobj=PartialFileIO(self.test_file, size_limit=5),
            path_in_repo='test_file.txt')
        self.assertEqual(op_truncated.upload_info.size, 5)
        self.assertEqual(op_truncated.upload_info.sample, b'12345')

        with op_truncated.as_file() as f:
            self.assertEqual(f.read(), b'12345')

        # Full file
        op_full = CommitOperationAdd(
            path_or_fileobj=PartialFileIO(self.test_file, size_limit=9),
            path_in_repo='test_file.txt')
        self.assertEqual(op_full.upload_info.size, 9)
        self.assertEqual(op_full.upload_info.sample, b'123456789')

        with op_full.as_file() as f:
            self.assertEqual(f.read(), b'123456789')

        # Truncated file has a different hash than the full file
        self.assertNotEqual(op_truncated.upload_info.sha256,
                            op_full.upload_info.sha256)


if __name__ == '__main__':
    unittest.main()
