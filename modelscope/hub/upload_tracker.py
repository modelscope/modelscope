# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unified file-level upload tracker.

Merges hash cache and upload progress into a single .ms_upload_cache file
with per-file status tracking, eliminating batch-granularity issues.
"""
import os
import re
import tempfile
import threading
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import json
import requests

from modelscope.utils.logger import get_logger

logger = get_logger()

# Legacy progress file name (for backward-compat detection only)
_LEGACY_PROGRESS_FILE = '.ms_upload_progress'

# Current cache format version
_TRACKER_VERSION = 3


class FileStatus:
    """Single-character status codes for compact JSON storage."""
    UPLOADED = 'u'  # Blob uploaded, not yet committed
    COMMITTED = 'c'  # Successfully committed to repo
    FAILED = 'f'  # Upload or commit failed


class ErrorCategory(str, Enum):
    """Classification of upload/commit errors for retry strategy."""
    TRANSIENT_NETWORK = 'transient_network'
    TRANSIENT_SERVER = 'transient_server'
    THROTTLED = 'throttled'
    AUTH_FAILED = 'auth_failed'
    NOT_FOUND = 'not_found'
    FILE_INVALID = 'file_invalid'
    UNKNOWN = 'unknown'

    @property
    def is_retryable(self) -> bool:
        return self not in (
            ErrorCategory.AUTH_FAILED,
            ErrorCategory.NOT_FOUND,
            ErrorCategory.FILE_INVALID,
        )


def classify_error(error: Exception) -> ErrorCategory:
    """Classify an exception into a retry category.

    Returns an ErrorCategory that indicates whether the error is transient
    (retryable) or permanent, and what kind of failure occurred.
    """
    error_str = str(error).lower()

    # ---- Specific OS error subclasses (check BEFORE generic IOError) ----
    if isinstance(error, FileNotFoundError):
        return ErrorCategory.FILE_INVALID
    if isinstance(error, PermissionError):
        return ErrorCategory.FILE_INVALID

    # Network / connection errors
    if isinstance(error, (ConnectionError, TimeoutError)):
        return ErrorCategory.TRANSIENT_NETWORK

    # requests HTTP errors (check response status code)
    if isinstance(error, requests.exceptions.HTTPError):
        resp = getattr(error, 'response', None)
        if resp is not None:
            status = resp.status_code
            if status == 429:
                return ErrorCategory.THROTTLED
            if status in (401, 403):
                return ErrorCategory.AUTH_FAILED
            if status == 404:
                return ErrorCategory.NOT_FOUND
            if status >= 500:
                return ErrorCategory.TRANSIENT_SERVER
        return ErrorCategory.UNKNOWN

    # ValueError from _commit_with_retry (wraps HTTP status in message)
    if isinstance(error, ValueError):
        if '429' in error_str:
            return ErrorCategory.THROTTLED
        if '401' in error_str or '403' in error_str:
            return ErrorCategory.AUTH_FAILED
        if '404' in error_str:
            return ErrorCategory.NOT_FOUND
        if re.search(r'(?:http[/\s]*)?5\d{2}|server.*error', error_str):
            return ErrorCategory.TRANSIENT_SERVER
        return ErrorCategory.UNKNOWN

    # Generic file / IO errors
    if isinstance(error, (IOError, OSError)):
        if 'size changed' in error_str or 'no such file' in error_str:
            return ErrorCategory.FILE_INVALID
        if 'permission' in error_str or 'access denied' in error_str:
            return ErrorCategory.FILE_INVALID
        return ErrorCategory.TRANSIENT_NETWORK

    # Fallback: check common patterns in error message
    if 'timeout' in error_str or 'timed out' in error_str:
        return ErrorCategory.TRANSIENT_NETWORK
    if 'connection' in error_str:
        return ErrorCategory.TRANSIENT_NETWORK

    return ErrorCategory.UNKNOWN


class UploadTracker:
    """Unified file-level upload tracker.

    Replaces both UploadHashCache (.ms_upload_cache) and
    UploadProgress (.ms_upload_progress) with a single file that tracks
    per-file hash and upload status.

    File format (version 3):
        {
            "version": 3,
            "repo_id": "user/repo",
            "files": {
                "path|mtime|size": {"hash": "...", "size": 123, "status": "c"},
                ...
            }
        }

    Status values:
        "c" = committed (blob uploaded AND committed to repo)
        "u" = uploaded (blob uploaded, NOT yet committed)
        "f" = failed
        (no status field) = hash cached only, upload not attempted

    Thread safety: all mutations are protected by a lock.
    Persistence: atomic write via temp file + rename.
    """

    def __init__(self, cache_path: Union[str, Path], repo_id: str):
        self._path = Path(cache_path)
        self._repo_id = repo_id
        self._files: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._dirty = False
        self._load()

    @staticmethod
    def _make_key(rel_path: str, mtime: float, size: int) -> str:
        """Build cache key from file metadata (same format as legacy UploadHashCache)."""
        return f'{rel_path}|{mtime}|{size}'

    # ---- Hash cache interface (replaces UploadHashCache) ----

    def get_hash(self, rel_path: str, mtime: float,
                 size: int) -> Optional[dict]:
        """Get cached hash info for a file.

        Returns dict compatible with legacy UploadHashCache.get():
            {'file_path_or_obj': rel_path, 'file_hash': ..., 'file_size': ...}
        or None if not cached or file has changed.
        """
        key = self._make_key(rel_path, mtime, size)
        with self._lock:
            entry = self._files.get(key)
        if entry is None or 'hash' not in entry:
            return None
        return {
            'file_path_or_obj': rel_path,
            'file_hash': entry['hash'],
            'file_size': entry['size'],
        }

    def put_hash(self, rel_path: str, mtime: float, size: int,
                 hash_info: dict):
        """Store computed hash info for a file.

        Args:
            hash_info: dict with 'file_hash' and 'file_size' keys.
        """
        key = self._make_key(rel_path, mtime, size)
        with self._lock:
            entry = self._files.get(key, {})
            entry['hash'] = hash_info['file_hash']
            entry['size'] = hash_info['file_size']
            # Preserve existing status if any
            self._files[key] = entry
            self._dirty = True

    # ---- Status tracking interface (replaces UploadProgress) ----

    def is_committed(self, rel_path: str, mtime: float, size: int) -> bool:
        """Check if a file is committed (with matching mtime and size)."""
        key = self._make_key(rel_path, mtime, size)
        with self._lock:
            entry = self._files.get(key)
        return entry is not None and entry.get(
            'status') == FileStatus.COMMITTED

    def get_status(self, rel_path: str, mtime: float,
                   size: int) -> Optional[str]:
        """Get file status, or None if not tracked."""
        key = self._make_key(rel_path, mtime, size)
        with self._lock:
            entry = self._files.get(key)
        return entry.get('status') if entry else None

    def mark_uploaded(self, rel_path: str, mtime: float, size: int):
        """Mark a file as blob-uploaded (not yet committed)."""
        key = self._make_key(rel_path, mtime, size)
        with self._lock:
            if key in self._files:
                self._files[key]['status'] = FileStatus.UPLOADED
                self._dirty = True

    def mark_committed_batch(self, file_keys: List[Tuple[str, float, int]]):
        """Mark multiple files as committed after a successful commit.

        Args:
            file_keys: list of (rel_path, mtime, size) tuples.
        """
        with self._lock:
            for rel_path, mtime, size in file_keys:
                key = self._make_key(rel_path, mtime, size)
                if key in self._files:
                    self._files[key]['status'] = FileStatus.COMMITTED
            self._dirty = True

    def mark_failed(self,
                    rel_path: str,
                    mtime: float,
                    size: int,
                    error_type: str = ''):
        """Mark a file as failed with optional error classification."""
        key = self._make_key(rel_path, mtime, size)
        with self._lock:
            if key in self._files:
                self._files[key]['status'] = FileStatus.FAILED
                if error_type:
                    self._files[key]['error_type'] = error_type
            else:
                entry = {'status': FileStatus.FAILED}
                if error_type:
                    entry['error_type'] = error_type
                self._files[key] = entry
            self._dirty = True

    # ---- Persistence ----

    def save(self):
        """Atomically save tracker state to disk."""
        with self._lock:
            if not self._dirty:
                return
            data = {
                'version': _TRACKER_VERSION,
                'repo_id': self._repo_id,
                'files': {k: dict(v)
                          for k, v in self._files.items()},
            }
            self._dirty = False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._path.parent), suffix='.tmp')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
                os.replace(tmp_path, str(self._path))
            except BaseException:
                os.unlink(tmp_path)
                raise
        except Exception as e:
            logger.warning(f'Failed to save upload tracker: {e}')

    def clear(self):
        """Delete the tracker file."""
        try:
            self._path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f'Failed to delete tracker file: {e}')
        with self._lock:
            self._files.clear()
            self._dirty = False

    def _load(self):
        """Load tracker state from disk, handling format migration."""
        if not self._path.exists():
            self._check_legacy_progress()
            return

        try:
            with open(self._path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f'Failed to load upload tracker, starting fresh: {e}')
            return

        version = data.get('version')
        if version is None:
            # v1: legacy hash-only format from UploadHashCache
            self._migrate_v1(data)
            return

        if version < _TRACKER_VERSION:
            logger.warning(
                f'Upload tracker version {version} is older than current '
                f'{_TRACKER_VERSION}. Data will be migrated on next save.')

        # v3+: validate repo_id
        stored_repo = data.get('repo_id', '')
        if stored_repo and stored_repo != self._repo_id:
            logger.warning(
                f'Tracker repo_id mismatch (cached: {stored_repo}, '
                f'current: {self._repo_id}), ignoring stale tracker.')
            return

        self._files = data.get('files', {})
        committed_count = sum(1 for e in self._files.values()
                              if e.get('status') == FileStatus.COMMITTED)
        if committed_count > 0:
            logger.info(f'Upload tracker loaded: {len(self._files)} entries, '
                        f'{committed_count} committed.')

        self._check_legacy_progress()

    def _migrate_v1(self, data: dict):
        """Migrate from legacy hash-only format (UploadHashCache v1).

        Old format: {"rel_path|mtime|size": {"file_hash": "...", "file_size": 123}}
        New format: {"rel_path|mtime|size": {"hash": "...", "size": 123}}

        Status is NOT set during migration -- cached hashes do not imply
        the file was committed (conservative approach).
        """
        migrated = {}
        for key, value in data.items():
            if isinstance(value, dict) and 'file_hash' in value:
                migrated[key] = {
                    'hash': value['file_hash'],
                    'size': value.get('file_size', 0),
                }
        self._files = migrated
        self._dirty = True  # will save in new format on next save()
        if migrated:
            logger.info(
                f'Migrated {len(migrated)} entries from legacy hash cache format.'
            )

    def _check_legacy_progress(self):
        """Warn if legacy .ms_upload_progress file exists."""
        legacy_path = self._path.parent / _LEGACY_PROGRESS_FILE
        if legacy_path.exists():
            logger.warning(
                f'Legacy upload progress file detected: {legacy_path}. '
                f'This file is no longer used. You may delete it safely.')


class NullTracker:
    """No-op tracker for when caching is disabled.

    Implements the same interface as UploadTracker but does nothing,
    eliminating 'if tracker is not None' checks throughout api.py.
    """

    def get_hash(self, rel_path: str, mtime: float, size: int) -> None:
        return None

    def put_hash(self, rel_path: str, mtime: float, size: int,
                 hash_info: dict):
        pass

    def is_committed(self, rel_path: str, mtime: float, size: int) -> bool:
        return False

    def get_status(self, rel_path: str, mtime: float, size: int):
        return None

    def mark_uploaded(self, rel_path: str, mtime: float, size: int):
        pass

    def mark_committed_batch(self, file_keys):
        pass

    def mark_failed(self,
                    rel_path: str,
                    mtime: float,
                    size: int,
                    error_type: str = ''):
        pass

    def save(self):
        pass

    def clear(self):
        pass
