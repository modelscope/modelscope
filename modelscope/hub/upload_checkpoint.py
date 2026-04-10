# Copyright (c) Alibaba, Inc. and its affiliates.

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

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
        self._batch_fingerprints: Dict[int, str] = {}
        self._load()

    @staticmethod
    def compute_fingerprint(items: List[Tuple[str, str]], ) -> str:
        """Compute a fingerprint from (file_path_in_repo, metadata) pairs.

        Used to detect when a batch's file set changes between runs,
        invalidating stale batch indices. The metadata element is
        typically 'mtime|size' but can be any string that changes
        when the file content changes. Called per-batch to produce
        individual batch fingerprints.
        """
        parts = [f'{path}|{fhash}' for path, fhash in sorted(items)]
        return hashlib.sha256('||'.join(parts).encode()).hexdigest()

    def validate_batch_fingerprint(self, batch_idx: int,
                                   fingerprint: str) -> bool:
        """Check if a committed batch's fingerprint still matches.

        Returns True if batch is committed and fingerprint matches (safe to skip).
        If committed but fingerprint mismatches, clears the batch's committed status.
        """
        if batch_idx not in self._committed_batches:
            return False
        stored_fp = self._batch_fingerprints.get(batch_idx)
        if stored_fp == fingerprint:
            return True
        # Fingerprint mismatch — invalidate this batch only
        self._committed_batches.discard(batch_idx)
        self._batch_fingerprints.pop(batch_idx, None)
        self._save()
        return False

    def is_batch_committed(self, batch_index: int) -> bool:
        """Check if a batch has already been committed."""
        return batch_index in self._committed_batches

    def mark_batch_committed(self, batch_idx: int, fingerprint: str):
        """Mark a batch as committed with its fingerprint and persist."""
        self._committed_batches.add(batch_idx)
        self._batch_fingerprints[batch_idx] = fingerprint
        self._save()

    def clear(self):
        """Remove checkpoint file."""
        self._committed_batches.clear()
        self._batch_fingerprints.clear()
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
            self._batch_fingerprints = {
                int(k): v
                for k, v in data.get('batch_fingerprints', {}).items()
            }
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
                'batch_fingerprints':
                {str(k): v
                 for k, v in self._batch_fingerprints.items()},
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
