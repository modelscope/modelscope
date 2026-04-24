# Copyright (c) Alibaba, Inc. and its affiliates.

import threading
from typing import List, Tuple

from modelscope.utils.logger import get_logger

logger = get_logger()


class BatchTracker:
    """Thread-safe tracker for pre-assigned upload batches.

    Files are assigned to batches by sorted index (file_index // batch_size).
    Upload threads record results; main thread waits for batches in order.
    """

    def __init__(self, total_files: int, batch_size: int):
        self._batch_size = batch_size
        self._num_batches = (total_files
                             - 1) // batch_size + 1 if total_files > 0 else 0
        self._batch_results: List[List[dict]] = [
            [] for _ in range(self._num_batches)
        ]
        self._batch_failures: List[List[tuple]] = [
            [] for _ in range(self._num_batches)
        ]
        self._batch_expected: List[int] = []
        for i in range(self._num_batches):
            start = i * batch_size
            end = min(start + batch_size, total_files)
            self._batch_expected.append(end - start)
        self._batch_events: List[threading.Event] = [
            threading.Event() for _ in range(self._num_batches)
        ]
        self._lock = threading.Lock()

    @property
    def num_batches(self) -> int:
        return self._num_batches

    def batch_index(self, file_index: int) -> int:
        return file_index // self._batch_size

    def record_success(self, file_index: int, result: dict):
        idx = self.batch_index(file_index)
        with self._lock:
            self._batch_results[idx].append(result)
            if self._is_batch_complete(idx):
                self._batch_events[idx].set()

    def record_failure(self, file_index: int, item: tuple, error: Exception):
        idx = self.batch_index(file_index)
        with self._lock:
            self._batch_failures[idx].append((item, error))
            if self._is_batch_complete(idx):
                self._batch_events[idx].set()

    def mark_file_skipped(self, file_index: int):
        """Mark a file as skipped (already committed).

        Decrements the batch's expected count so _is_batch_complete
        uses the correct target. When all files in a batch are skipped,
        the batch event is set automatically.
        """
        idx = self.batch_index(file_index)
        with self._lock:
            self._batch_expected[idx] -= 1
            if self._is_batch_complete(idx):
                self._batch_events[idx].set()

    def wait_for_batch(self, batch_idx: int) -> Tuple[List[dict], List[tuple]]:
        """Wait for a batch to complete.

        Blocks indefinitely until all files in the batch have reported
        success or failure. Per-blob timeouts (UPLOAD_BLOB_TIMEOUT)
        prevent individual uploads from hanging forever.

        Args:
            batch_idx: Index of the batch to wait for.

        Returns:
            Tuple of (successful_results, failures).
        """
        self._batch_events[batch_idx].wait()
        with self._lock:
            return list(self._batch_results[batch_idx]), list(
                self._batch_failures[batch_idx])

    def _is_batch_complete(self, batch_idx: int) -> bool:
        """Must be called under self._lock."""
        count = len(self._batch_results[batch_idx]) + len(
            self._batch_failures[batch_idx])
        return count >= self._batch_expected[batch_idx]
