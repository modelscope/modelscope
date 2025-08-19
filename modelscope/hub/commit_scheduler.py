# Copyright (c) Alibaba, Inc. and its affiliates.

import atexit
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from io import SEEK_END, SEEK_SET, BytesIO
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional, Union

from modelscope.hub.api import HubApi
from modelscope.utils.logger import get_logger
from modelscope.utils.repo_utils import (CommitInfo, CommitOperationAdd,
                                         RepoUtils)

logger = get_logger()

IGNORE_GIT_FOLDER_PATTERNS = ['.git', '.git/*', '*/.git', '**/.git/**']


@dataclass(frozen=True)
class _FileToUpload:
    """Information about a file to upload."""

    local_path: Path
    path_in_repo: str
    size_limit: int
    last_modified: float


class CommitScheduler:
    """Scheduler to regularly push a local folder to ModelScope Hub."""

    def __init__(
        self,
        *,
        repo_id: str,
        folder_path: Union[str, Path],
        interval: Union[int, float] = 5,
        path_in_repo: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[str] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        squash_history: bool = False,
        hub_api: Optional['HubApi'] = None,
    ) -> None:
        self.api = hub_api or HubApi()

        # Folder
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

        repo_url = self.api.create_repo(
            repo_id=repo_id,
            token=token,
            repo_type=repo_type,
            visibility='private' if private else 'public',
            exist_ok=True,
            create_default_config=False,
        )
        self.repo_id = repo_url.repo_id
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
        self.squash_history = squash_history

        logger.info(
            f'Scheduled job to push {self.folder_path} to {self.repo_id} at a interval of {self.interval} minutes.'
        )
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._scheduler_thread = Thread(
            target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        atexit.register(self.push_to_hub)

        self.__stopped = False

    def stop(self) -> None:
        """Stop the scheduler."""
        self.__stopped = True

    def _run_scheduler(self) -> None:
        while True:
            self.last_future = self.trigger()
            time.sleep(self.interval * 60)
            if self.__stopped:
                break

    def trigger(self) -> Future:
        """Trigger a background commit and return a future."""
        return self.executor.submit(self._push_to_hub)

    def _push_to_hub(self) -> Optional[CommitInfo]:
        if self.__stopped:
            return None

        logger.info('(Background) scheduled commit triggered.')
        try:
            value = self.push_to_hub()
            if value and self.squash_history and hasattr(
                    self.api, 'super_squash_history'):
                logger.info('(Background) squashing repo history.')
                try:
                    self.api.super_squash_history(
                        repo_id=self.repo_id,
                        repo_type=self.repo_type,
                        branch=self.revision)
                except Exception as e:
                    logger.error(f'Error while squashing history: {e}')
                    # Don't raise here as the commit was successful
            return value
        except Exception as e:
            logger.error(f'Error while pushing to Hub: {e}')
            raise

    def push_to_hub(self) -> Optional[CommitInfo]:
        """Push folder to the Hub and return commit info if changes are found."""
        try:
            files = []
            for path in self.folder_path.rglob('*'):
                if not path.is_file():
                    continue
                relpath = path.relative_to(self.folder_path).as_posix()
                if not RepoUtils.filter_repo_objects(
                    [relpath],
                        allow_patterns=self.allow_patterns,
                        ignore_patterns=self.ignore_patterns):
                    continue
                files.append((relpath, path))
        except Exception as e:
            logger.error(f'Error while scanning files: {e}')
            raise

        with self.lock:
            try:
                prefix = f'{self.path_in_repo.strip("/")}/' if self.path_in_repo else ''
                files_to_upload: List[_FileToUpload] = []

                for relpath, local_path in files:
                    try:
                        stat = local_path.stat()
                        if self.last_uploaded.get(
                                local_path) is None or self.last_uploaded[
                                    local_path] != stat.st_mtime:
                            files_to_upload.append(
                                _FileToUpload(
                                    local_path=local_path,
                                    path_in_repo=prefix + relpath,
                                    size_limit=stat.st_size,
                                    last_modified=stat.st_mtime))
                    except OSError as e:
                        logger.warning(
                            f'Failed to stat file {local_path}: {e}')
                        continue

                if not files_to_upload:
                    logger.debug(
                        'Dropping schedule commit: no changed file to upload.')
                    return None
            finally:
                pass

        add_operations = []
        for file in files_to_upload:
            try:
                add_operations.append(
                    CommitOperationAdd(
                        path_or_fileobj=PartialFileIO(
                            file.local_path, size_limit=file.size_limit),
                        path_in_repo=file.path_in_repo,
                    ))
            except Exception as e:
                logger.warning(
                    f'Failed to create operation for {file.local_path}: {e}')
                continue

        if not add_operations:
            logger.debug('No valid operations to perform.')
            return None

        try:
            logger.debug('Uploading files for scheduled commit.')
            commit_info = self.api.create_commit(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                operations=add_operations,
                commit_message='Scheduled Commit',
                revision=self.revision,
                token=self.token,
            )
        except Exception as e:
            logger.error(f'Error during commit: {e}')
            raise

        for file in files_to_upload:
            self.last_uploaded[file.local_path] = file.last_modified

        return commit_info


class PartialFileIO(BytesIO):
    """A file-like object that reads only the first part of a file."""

    def __init__(self, file_path: Union[str, Path], size_limit: int) -> None:
        self._file_path = Path(file_path)
        self._file = None
        self._size_limit = None
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

    def __enter__(self) -> 'PartialFileIO':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'<PartialFileIO file_path={self._file_path} size_limit={self._size_limit}>'

    def __len__(self) -> int:
        if self._size_limit is None:
            self.open()
        return self._size_limit

    def __getattribute__(self, name: str):
        if name.startswith('_') or name in {
                'read', 'tell', 'seek', 'close', 'open', '__enter__',
                '__exit__'
        }:
            return super().__getattribute__(name)
        raise NotImplementedError(
            f'{self.__class__.__name__} does not support {name}')

    def tell(self) -> int:
        if self._file is None:
            self.open()
        return self._file.tell()

    def seek(self, __offset: int, __whence: int = SEEK_SET) -> int:
        """Seek to a position in the file, but never beyond size_limit."""
        if self._file is None:
            self.open()

        if __whence == SEEK_END:
            __offset = len(self) + __offset
            __whence = SEEK_SET

        target_pos = __offset if __whence == SEEK_SET else self.tell(
        ) + __offset
        target_pos = min(target_pos, self._size_limit)

        return self._file.seek(target_pos, SEEK_SET)

    def read(self, __size: Optional[int] = -1) -> bytes:
        """Read at most size_limit bytes from the current position."""
        if self._file is None:
            self.open()

        current = self.tell()
        if __size is None or __size < 0:
            truncated_size = self._size_limit - current
        else:
            truncated_size = min(__size, self._size_limit - current)
        return self._file.read(truncated_size)
