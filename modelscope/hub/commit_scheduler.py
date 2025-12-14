# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2022-present, the HuggingFace Inc. team.
import atexit
import contextlib
import os
import time
import types
from concurrent.futures import Future, ThreadPoolExecutor
from io import SEEK_END, SEEK_SET, BytesIO
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional, Union

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Visibility
from modelscope.utils.constant import DEFAULT_REPOSITORY_REVISION
from modelscope.utils.logger import get_logger
from modelscope.utils.repo_utils import CommitInfo, RepoUtils

logger = get_logger()

IGNORE_GIT_FOLDER_PATTERNS = ['.git', '.git/*', '*/.git', '**/.git/**']


@contextlib.contextmanager
def patch_upload_folder_for_scheduler(scheduler_instance):
    """Patch upload_folder for CommitScheduler"""
    api = scheduler_instance.api
    original_prepare = api._prepare_upload_folder

    def patched_prepare_upload_folder(
        api_self,
        folder_path_or_files: Union[str, Path, List[str], List[Path]],
        path_in_repo: str,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
    ) -> List[Union[tuple, list]]:
        """
        Patched version that supports incremental updates for CommitScheduler.
        """
        with scheduler_instance.lock:
            if isinstance(folder_path_or_files, list):
                raise ValueError(
                    'Uploading multiple files or folders is not supported for scheduled commit.'
                )
            elif os.path.isfile(folder_path_or_files):
                raise ValueError(
                    'Uploading file is not supported for scheduled commit.')
            else:
                folder_path = Path(folder_path_or_files).expanduser().resolve()

            logger.debug('Listing files to upload for scheduled commit.')
            relpath_to_abspath = {
                path.relative_to(folder_path).as_posix(): path
                for path in sorted(folder_path.glob('**/*')) if path.is_file()
            }
            prefix = f"{path_in_repo.strip('/')}/" if path_in_repo else ''

            prepared_repo_objects = []
            files_to_track = {}

            for relpath in RepoUtils.filter_repo_objects(
                    relpath_to_abspath.keys(),
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns):
                local_path = relpath_to_abspath[relpath]
                stat = local_path.stat()
                if scheduler_instance.last_uploaded.get(
                        local_path
                ) is None or scheduler_instance.last_uploaded[
                        local_path] != stat.st_mtime:
                    partial_file = PartialFileIO(local_path, stat.st_size)
                    prepared_repo_objects.append(
                        (prefix + relpath, partial_file))
                    files_to_track[local_path] = stat.st_mtime

            scheduler_instance._pending_tracker_updates = files_to_track

            if not prepared_repo_objects:
                logger.debug(
                    'No changed files to upload for scheduled commit.')

        return prepared_repo_objects

    try:
        api._prepare_upload_folder = types.MethodType(
            patched_prepare_upload_folder, api)
        yield
    finally:
        api._prepare_upload_folder = original_prepare


class CommitScheduler:
    """
    A scheduler that automatically uploads a local folder to ModelScope Hub at
    specified intervals (e.g., every 5 minutes).

    It's recommended to use the scheduler as a context manager to ensure proper
    cleanup and final commit execution when your script completes. Alternatively,
    you can manually stop the scheduler using the `stop` method.

    Args:
        repo_id (`str`):
            The id of the repo to commit to.
        folder_path (`str` or `Path`):
            Local folder path that will be monitored and uploaded periodically.
        interval (`int` or `float`, *optional*):
            Time interval in minutes between each upload operation. Defaults to 5 minutes.
        path_in_repo (`str`, *optional*):
            Target directory path within the repository, such as `"models/"`.
            If not specified, files are uploaded to the repository root.
        repo_type (`str`, *optional*):
            Repository type for the target repo. Defaults to `model`.
        revision (`str`, *optional*):
            Target branch or revision for commits. Defaults to `master`.
        visibility (`str`, *optional*):
            The visibility of the repo,
            could be `public`, `private`, `internal`, default to `public`.
        token (`str`, *optional*):
            The token to use to commit to the repo. Defaults to the token saved on the machine.
        allow_patterns (`List[str]` or `str`, *optional*):
            File patterns to include in uploads. Only files matching these patterns will be uploaded.
        ignore_patterns (`List[str]` or `str`, *optional*):
            File patterns to exclude from uploads. Files matching these patterns will be skipped.
        hub_api (`HubApi`, *optional*):
            Custom [`HubApi`] instance for Hub operations. Allows for customized
            configurations like user agent or token settings.

    Example:
    ```py
    >>> from pathlib import Path
    >>> from modelscope.hub import CommitScheduler

    # Create scheduler with 10-minute intervals
    >>> data_file = Path("workspace/experiment.log")
    >>> scheduler = CommitScheduler(
    ...     repo_id="my_experiments",
    ...     repo_type="dataset",
    ...     folder_path=data_file.parent,
    ...     interval=10
    ... )

    >>> with data_file.open("a") as f:
    ...     f.write("experiment started")

    # Later in the workflow...
    >>> with data_file.open("a") as f:
    ...     f.write("experiment completed")
    ```

    Context manager usage:
    ```py
    >>> from pathlib import Path
    >>> from modelscope.hub import CommitScheduler

    >>> with CommitScheduler(
    ...     repo_id="my_experiments",
    ...     repo_type="dataset",
    ...     folder_path="workspace",
    ...     interval=10
    ... ) as scheduler:
    ...     log_file = Path("workspace/progress.log")
    ...     with log_file.open("a") as f:
    ...         f.write("starting process")
    ...     # ... perform work ...
    ...     with log_file.open("a") as f:
    ...         f.write("process finished")

    # Scheduler automatically stops and performs final upload
    ```
    """

    def __init__(
        self,
        *,
        repo_id: str,
        folder_path: Union[str, Path],
        interval: Union[int, float] = 5,
        path_in_repo: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = DEFAULT_REPOSITORY_REVISION,
        visibility: Optional[str] = Visibility.PUBLIC,
        token: Optional[str] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        hub_api: Optional[HubApi] = None,
    ) -> None:
        self.api = hub_api or HubApi()

        self.folder_path = Path(folder_path).expanduser().resolve()
        if not self.folder_path.exists():
            raise ValueError(f'Folder path does not exist: {folder_path}')

        self.path_in_repo = path_in_repo or ''
        self.allow_patterns = allow_patterns

        if ignore_patterns is None:
            ignore_patterns = []
        elif isinstance(ignore_patterns, str):
            ignore_patterns = [ignore_patterns]
        self.ignore_patterns = ignore_patterns + IGNORE_GIT_FOLDER_PATTERNS

        self.repo_url = self.api.create_repo(
            repo_id=repo_id,
            token=token,
            repo_type=repo_type,
            visibility=visibility,
            exist_ok=True,
            create_default_config=False,
        )
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revision = revision
        self.token = token

        # Keep track of already uploaded files
        self.last_uploaded: Dict[Path, float] = {}

        if interval <= 0:
            raise ValueError(
                f'"interval" must be a positive integer, not "{interval}".')
        self.lock = Lock()
        self.interval = interval
        self.__stopped = False

        logger.info(
            f'Scheduled job to push {self.folder_path} to {self.repo_id} at an interval of {self.interval} minutes.'
        )
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._scheduler_thread = Thread(
            target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        atexit.register(self.commit_scheduled_changes)

    def stop(self) -> None:
        """Stop the scheduler."""
        self.__stopped = True

    def __enter__(self) -> 'CommitScheduler':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.trigger().result()
        self.stop()
        return

    def _run_scheduler(self) -> None:
        while not self.__stopped:
            self.last_future = self.trigger()
            time.sleep(self.interval * 60)

    def trigger(self) -> Future:
        """Trigger a background commit and return a future."""
        return self.executor.submit(self._commit_scheduled_changes)

    def _commit_scheduled_changes(self) -> Optional[CommitInfo]:
        if self.__stopped:
            return None

        logger.info('(Background) scheduled commit triggered.')
        try:
            value = self.commit_scheduled_changes()
            return value
        except Exception as e:
            logger.error(f'Error while pushing to Hub: {e}')
            raise

    def commit_scheduled_changes(self) -> Optional[CommitInfo]:
        """Push folder to the Hub and return commit info if changes are found."""
        try:
            self._pending_tracker_updates = {}
            with patch_upload_folder_for_scheduler(self):
                commit_info = self.api.upload_folder(
                    repo_id=self.repo_id,
                    folder_path=self.folder_path,
                    path_in_repo=self.path_in_repo,
                    commit_message='Scheduled Commit',
                    token=self.token,
                    repo_type=self.repo_type,
                    allow_patterns=self.allow_patterns,
                    ignore_patterns=self.ignore_patterns,
                    revision=self.revision,
                )

            if commit_info is None:
                logger.debug(
                    'No changed files to upload for scheduled commit.')
                return None

            with self.lock:
                if hasattr(self, '_pending_tracker_updates'):
                    self.last_uploaded.update(self._pending_tracker_updates)
                    logger.debug(
                        f'Updated modification tracker for {len(self._pending_tracker_updates)} files.'
                    )
                    del self._pending_tracker_updates

            return commit_info

        except Exception as e:
            # Treat "No files to upload" as a normal ― no-change ― situation instead of an error.
            if 'No files to upload' in str(e):
                logger.debug(
                    'No changed files to upload for scheduled commit.')
                return None

            if hasattr(self, '_pending_tracker_updates'):
                del self._pending_tracker_updates
            logger.error(f'Error during scheduled commit: {e}')
            raise


class PartialFileIO(BytesIO):
    """A file-like object that reads only the first part of a file."""

    def __init__(self, file_path: Union[str, Path], size_limit: int) -> None:
        self._file_path = Path(file_path)
        self._file = None
        self._size_limit = size_limit
        self.open()

    def open(self) -> None:
        """Open the file and initialize size limit."""
        if self._file is not None:
            return
        try:
            self._file = self._file_path.open('rb')
            self._size_limit = min(
                self._size_limit or float('inf'),
                os.fstat(self._file.fileno()).st_size)
        except OSError as e:
            logger.error(f'Failed to open file {self._file_path}: {e}')
            raise

    def close(self) -> None:
        """Close the file if it's open."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()
        return super().__del__()

    def __repr__(self) -> str:
        return f'<PartialFileIO file_path={self._file_path} size_limit={self._size_limit}>'

    def __len__(self) -> int:
        return self._size_limit

    def __getattribute__(self, name: str):
        if name.startswith('_') or name in {
                'read', 'tell', 'seek', 'close', 'open'
        }:  # only 5 public methods supported
            return super().__getattribute__(name)
        raise NotImplementedError(f"PartialFileIO does not support '{name}'.")

    def tell(self) -> int:
        return self._file.tell()

    def seek(self, __offset: int, __whence: int = SEEK_SET) -> int:
        """Seek to a position in the file, but never beyond size_limit."""
        if __whence == SEEK_END:
            __offset = len(self) + __offset
            __whence = SEEK_SET

        pos = self._file.seek(__offset, __whence)
        if pos > self._size_limit:
            return self._file.seek(self._size_limit)
        return pos

    def read(self, __size: Optional[int] = -1) -> bytes:
        """Read at most _size bytes from the current position."""
        current = self.tell()
        if __size is None or __size < 0:
            # Read until file limit
            truncated_size = self._size_limit - current
        else:
            # Read until file limit or __size
            truncated_size = min(__size, self._size_limit - current)
        return self._file.read(truncated_size)
