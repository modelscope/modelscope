# Copyright (c) Alibaba, Inc. and its affiliates.

import hashlib
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import json

from modelscope.utils.logger import get_logger

logger = get_logger()

UPLOAD_CHECKPOINT_FILE = '.ms_upload_checkpoint'


class UploadCheckpoint:
    """Tracks committed batch indices for upload_folder resume.

    Stored as JSON at {folder_path}/.ms_upload_checkpoint. On resume,
    already-committed batches are skipped. Validates repo_id to prevent
    cross-repo confusion.
    """

    def __init__(self, checkpoint_path: Union[str, Path], repo_id: str):
        """Initialize checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
            repo_id: Repository ID for validation on resume.
        """
        self._path = Path(checkpoint_path)
        self._repo_id = repo_id
        self._committed_batches: Set[int] = set()
        self._fingerprint: Optional[str] = None
        self._load()

    @staticmethod
    def compute_fingerprint(items: List[Tuple[str, str]], ) -> str:
        """Compute a fingerprint from (file_path_in_repo, file_hash) pairs.

        Used to detect when the operations set changes between runs,
        invalidating stale batch indices.
        """
        parts = [f'{path}|{fhash}' for path, fhash in sorted(items)]
        return hashlib.sha256('||'.join(parts).encode()).hexdigest()

    def validate_fingerprint(self, fingerprint: str):
        """Validate operations fingerprint. Reset if mismatch.

        If the fingerprint differs from the stored one, the file set has
        changed and old batch indices are invalid.
        """
        if (self._fingerprint is not None
                and self._fingerprint != fingerprint):
            logger.warning(
                'Operations fingerprint changed since last checkpoint, '
                'resetting committed batches.')
            self._committed_batches.clear()
        self._fingerprint = fingerprint

    def is_batch_committed(self, batch_index: int) -> bool:
        """Check if a batch has already been committed."""
        return batch_index in self._committed_batches

    def mark_batch_committed(self, batch_index: int):
        """Mark a batch as committed and persist immediately.

        Saves to disk right away so progress survives crashes.
        """
        self._committed_batches.add(batch_index)
        self._save()

    def clear(self):
        """Remove checkpoint file."""
        self._committed_batches.clear()
        self._fingerprint = None
        try:
            if self._path.exists():
                self._path.unlink()
                logger.info(f'Upload checkpoint cleared: {self._path}')
        except Exception as e:
            logger.warning(f'Failed to remove checkpoint file: {e}')

    def _load(self):
        """Load checkpoint from disk. Invalidates if repo_id mismatches."""
        if not self._path.exists():
            return
        try:
            with open(self._path, 'r') as f:
                data = json.load(f)
            # Validate repo_id to prevent cross-repo confusion
            if data.get('repo_id') != self._repo_id:
                logger.warning(
                    f'Checkpoint repo_id mismatch '
                    f'(cached: {data.get("repo_id")}, current: {self._repo_id}), '
                    f'ignoring stale checkpoint.')
                return
            self._fingerprint = data.get('fingerprint')
            self._committed_batches = set(data.get('committed_batches', []))
            if self._committed_batches:
                logger.info(
                    f'Upload checkpoint loaded: {len(self._committed_batches)} '
                    f'batch(es) already committed.')
        except Exception as e:
            logger.warning(f'Failed to load checkpoint, starting fresh: {e}')
            self._committed_batches = set()

    def _save(self):
        """Atomic persist via temp file + rename."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'repo_id': self._repo_id,
                'fingerprint': self._fingerprint,
                'committed_batches': sorted(self._committed_batches),
            }
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._path.parent), prefix='.ms_upload_ckpt_tmp_')
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(data, f)
                os.replace(tmp_path, str(self._path))
            except BaseException:
                os.unlink(tmp_path)
                raise
            logger.info(
                f'Checkpoint saved: batches {sorted(self._committed_batches)} -> {self._path}'
            )
        except Exception as e:
            logger.warning(f'Failed to save checkpoint to {self._path}: {e}')
