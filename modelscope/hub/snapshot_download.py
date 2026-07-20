"""Snapshot download — shim preserving the legacy positional-arg signature.

Delegates to ``modelscope_hub.compat`` while keeping ``revision``, ``cache_dir``
and friends accessible as positional arguments for backward compatibility.
"""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union

from modelscope_hub.compat.snapshot_download import \
    dataset_snapshot_download as _compat_dataset_snapshot_download
from modelscope_hub.compat.snapshot_download import \
    snapshot_download as _compat_snapshot_download

if TYPE_CHECKING:
    from .callback import ProgressCallback

__all__ = ['snapshot_download', 'dataset_snapshot_download']


def snapshot_download(
    model_id: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    user_agent: Optional[Union[Dict, str]] = None,
    local_files_only: Optional[bool] = False,
    cookies=None,
    ignore_file_pattern: Optional[Union[str, List[str]]] = None,
    allow_file_pattern: Optional[Union[str, List[str]]] = None,
    local_dir: Optional[str] = None,
    allow_patterns: Optional[Union[List[str], str]] = None,
    ignore_patterns: Optional[Union[List[str], str]] = None,
    max_workers: Optional[int] = None,
    repo_id: Optional[str] = None,
    repo_type: Optional[str] = None,
    progress_callbacks: Optional[List[Type[ProgressCallback]]] = None,
    token: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> str:
    """Download a complete repo snapshot.

    Preserves the legacy positional-argument signature for backward
    compatibility while delegating to ``modelscope_hub.compat``.
    ``progress_callbacks`` is a list of :class:`ProgressCallback` subclasses
    (not instances), each instantiated per file to report download progress.
    """
    return _compat_snapshot_download(
        model_id=model_id,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_dir=local_dir,
        allow_file_pattern=allow_file_pattern,
        ignore_file_pattern=ignore_file_pattern,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        max_workers=max_workers if max_workers is not None else 4,
        cookies=cookies,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        endpoint=endpoint,
        local_files_only=bool(local_files_only)
        if local_files_only is not None else False,
        user_agent=user_agent,
        progress_callbacks=progress_callbacks,
    )


def dataset_snapshot_download(
    dataset_id: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    local_dir: Optional[str] = None,
    allow_file_pattern: Optional[Union[str, List[str]]] = None,
    ignore_file_pattern: Optional[Union[str, List[str]]] = None,
    allow_patterns: Optional[Union[List[str], str]] = None,
    ignore_patterns: Optional[Union[List[str], str]] = None,
    max_workers: Optional[int] = None,
    cookies=None,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> str:
    """Download a dataset repo snapshot (legacy positional-arg signature)."""
    effective_id = dataset_id or repo_id
    return _compat_dataset_snapshot_download(
        dataset_id=effective_id,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_dir=local_dir,
        allow_file_pattern=allow_file_pattern,
        ignore_file_pattern=ignore_file_pattern,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        max_workers=max_workers if max_workers is not None else 4,
        cookies=cookies,
        token=token,
        endpoint=endpoint,
    )
