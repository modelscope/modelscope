# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional, Union

import json

from modelscope.utils.logger import get_logger

logger = get_logger()

UPLOAD_HASH_CACHE_FILE = '.ms_upload_cache'


class UploadHashCache:
    """Persistent local hash cache for upload_folder resume.

    Stores SHA256 hashes keyed by (relative_path, mtime, size) to skip
    re-hashing unchanged files on retry/resume. Thread-safe for concurrent
    put() calls from multiple upload threads.

    Cache is stored as JSON at {folder_path}/.ms_upload_cache, co-located
    with the upload source for portability.
    """

    def __init__(self, cache_path: Union[str, Path]):
        """Initialize cache.

        Args:
            cache_path: Path to the cache file (typically folder/.ms_upload_cache).
        """
        self._cache_path = Path(cache_path)
        self._cache: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    @staticmethod
    def _make_key(rel_path: str, mtime: float, size: int) -> str:
        """Build cache lookup key from file metadata."""
        return f'{rel_path}|{mtime}|{size}'

    def get(self, rel_path: str, mtime: float, size: int) -> Optional[dict]:
        """Return cached hash info or None if not cached / stale.

        Args:
            rel_path: Relative path of the file within the upload folder.
            mtime: File modification time (os.stat st_mtime).
            size: File size in bytes.

        Returns:
            Dict with file_hash and file_size, or None.
        """
        key = self._make_key(rel_path, mtime, size)
        with self._lock:
            entry = self._cache.get(key)
        if entry is None:
            return None
        # Reconstruct the hash_info dict expected by callers
        return {
            'file_path_or_obj': rel_path,
            'file_hash': entry['file_hash'],
            'file_size': entry['file_size'],
        }

    def put(self, rel_path: str, mtime: float, size: int, hash_info: dict):
        """Store hash info for a file. Thread-safe.

        Args:
            rel_path: Relative path of the file.
            mtime: File modification time.
            size: File size in bytes.
            hash_info: Dict from compute_file_hash with file_hash and file_size.
        """
        key = self._make_key(rel_path, mtime, size)
        entry = {
            'file_hash': hash_info['file_hash'],
            'file_size': hash_info['file_size'],
        }
        with self._lock:
            self._cache[key] = entry

    def save(self):
        """Persist cache to disk via atomic write (temp file + rename).

        Safe against crashes -- either the old or new file is present,
        never a partial write.
        """
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = dict(self._cache)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._cache_path.parent),
                prefix='.ms_upload_cache_tmp_')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
                os.replace(tmp_path, str(self._cache_path))
            except BaseException:
                os.unlink(tmp_path)
                raise
            logger.info(
                f'Hash cache saved: {len(data)} entries -> {self._cache_path}')
            if not self._cache_path.exists():
                logger.warning(
                    f'Hash cache file not found after save: {self._cache_path}'
                )
        except Exception as e:
            logger.warning(
                f'Failed to save hash cache to {self._cache_path}: {e}')

    def _load(self):
        """Load cache from disk. Tolerates missing or corrupt file."""
        if not self._cache_path.exists():
            return
        try:
            with open(self._cache_path, 'r') as f:
                self._cache = json.load(f)
            logger.info(
                f'Hash cache loaded: {len(self._cache)} entries from {self._cache_path}'
            )
        except Exception as e:
            logger.warning(f'Failed to load hash cache, starting fresh: {e}')
            self._cache = {}
