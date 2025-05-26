"""Contains utilities to manage the ModelScope cache directory."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Union

from modelscope.hub.errors import CacheNotFound, CorruptedCacheException
from modelscope.hub.utils.caching import ModelFileSystemCache
from modelscope.utils.constant import REPO_TYPE_DATASET, REPO_TYPE_MODEL
from modelscope.utils.file_utils import get_default_modelscope_cache_dir
from modelscope.utils.logger import get_logger
from .utils.utils import convert_readable_size, format_timesince, tabulate

logger = get_logger()

# List of OS-created helper files that need to be ignored
FILES_TO_IGNORE = ['.DS_Store', '._____temp']


@dataclass(frozen=True)
class CachedFileInfo:
    """Frozen data structure holding information about a single cached file.

    Args:
        file_name (`str`):
            Name of the file. Example: `config.json`.
        file_path (`Path`):
            Path of the file in the `snapshots` directory. The file path is a symlink
            referring to a blob in the `blobs` folder.
        blob_path (`Path`):
            Path of the blob file. This is equivalent to `file_path.resolve()`.
        size_on_disk (`int`):
            Size of the blob file in bytes.
        blob_last_accessed (`float`):
            Timestamp of the last time the blob file has been accessed (from any
            revision).
        blob_last_modified (`float`):
            Timestamp of the last time the blob file has been modified/created.
    """

    file_name: str
    file_path: Path
    blob_path: Path
    size_on_disk: int

    blob_last_accessed: float
    blob_last_modified: float

    @property
    def blob_last_accessed_str(self) -> str:
        """
        (property) Timestamp of the last time the blob file has been accessed (from any
        revision), returned as a human-readable string.

        Example: "2 weeks ago".
        """
        return format_timesince(self.blob_last_accessed)

    @property
    def blob_last_modified_str(self) -> str:
        """
        (property) Timestamp of the last time the blob file has been modified, returned
        as a human-readable string.

        Example: "2 weeks ago".
        """
        return format_timesince(self.blob_last_modified)

    @property
    def size_on_disk_str(self) -> str:
        """
        (property) Size of the blob file as a human-readable string.

        Example: "42.2K".
        """
        return convert_readable_size(self.size_on_disk)


@dataclass(frozen=True)
class CachedRevisionInfo:
    """Frozen data structure holding information about a revision.

    Args:
        commit_hash (`str`):
            Hash of the revision (unique).
            Example: `"9338f7b671827df886678df2bdd7cc7b4f36dffd"`.
        snapshot_path (`Path`):
            Path to the revision directory in the `snapshots` folder. It contains the
            exact tree structure as the repo on the Hub.
        files: (`FrozenSet[CachedFileInfo]`):
            Set of [`~CachedFileInfo`] describing all files contained in the snapshot.
        refs (`FrozenSet[str]`):
            Set of `refs` pointing to this revision. If the revision has no `refs`, it
            is considered detached.
            Example: `{"main", "2.4.0"}` or `{"refs/pr/1"}`.
        size_on_disk (`int`):
            Sum of the blob file sizes that are symlink-ed by the revision.
        last_modified (`float`):
            Timestamp of the last time the revision has been created/modified.
    """

    commit_hash: str
    snapshot_path: Path
    size_on_disk: int
    files: FrozenSet[CachedFileInfo]
    refs: FrozenSet[str]

    last_modified: float

    @property
    def last_modified_str(self) -> str:
        """
        (property) Timestamp of the last time the revision has been modified, returned
        as a human-readable string.

        Example: "2 weeks ago".
        """
        return format_timesince(self.last_modified)

    @property
    def size_on_disk_str(self) -> str:
        """
        (property) Sum of the blob file sizes as a human-readable string.

        Example: "42.2K".
        """
        return convert_readable_size(self.size_on_disk)

    @property
    def nb_files(self) -> int:
        """
        (property) Total number of files in the revision.
        """
        return len(self.files)


@dataclass(frozen=True)
class CachedRepoInfo:
    """Frozen data structure holding information about a cached repository.

    Args:
        repo_id (`str`):
            Repo id of the repo on the Hub. Example: `"damo/bert-base-chinese"`.
        repo_type (`Literal["dataset", "model"]`):
            Type of the cached repo.
        repo_path (`Path`):
            Local path to the cached repo.
        size_on_disk (`int`):
            Sum of the blob file sizes in the cached repo.
        nb_files (`int`):
            Total number of blob files in the cached repo.
        revisions (`FrozenSet[CachedRevisionInfo]`):
            Set of [`~CachedRevisionInfo`] describing all revisions cached in the repo.
        last_accessed (`float`):
            Timestamp of the last time a blob file of the repo has been accessed.
        last_modified (`float`):
            Timestamp of the last time a blob file of the repo has been modified/created.
    """

    repo_id: str
    repo_type: str
    repo_path: Path
    size_on_disk: int
    nb_files: int
    revisions: FrozenSet[CachedRevisionInfo]

    last_accessed: float
    last_modified: float

    @property
    def last_accessed_str(self) -> str:
        """
        (property) Last time a blob file of the repo has been accessed, returned as a
        human-readable string.

        Example: "2 weeks ago".
        """
        return format_timesince(self.last_accessed)

    @property
    def last_modified_str(self) -> str:
        """
        (property) Last time a blob file of the repo has been modified, returned as a
        human-readable string.

        Example: "2 weeks ago".
        """
        return format_timesince(self.last_modified)

    @property
    def size_on_disk_str(self) -> str:
        """
        (property) Sum of the blob file sizes as a human-readable string.

        Example: "42.2K".
        """
        return convert_readable_size(self.size_on_disk)

    @property
    def refs(self) -> Dict[str, CachedRevisionInfo]:
        """
        (property) Mapping between `refs` and revision data structures.
        """
        return {
            ref: revision
            for revision in self.revisions for ref in revision.refs
        }


@dataclass(frozen=True)
class ModelScopeCacheInfo:
    """Frozen data structure holding information about the entire cache-system.

    This data structure is returned by [`scan_cache_dir`] and is immutable.

    Args:
        size_on_disk (`int`):
            Sum of all valid repo sizes in the cache-system.
        repos (`FrozenSet[CachedRepoInfo]`):
            Set of [`~CachedRepoInfo`] describing all valid cached repos found on the
            cache-system while scanning.
        warnings (`List[CorruptedCacheException]`):
            List of [`~CorruptedCacheException`] that occurred while scanning the cache.
            Those exceptions are captured so that the scan can continue. Corrupted repos
            are skipped from the scan.
    """

    size_on_disk: int
    repos: FrozenSet[CachedRepoInfo]
    warnings: List[CorruptedCacheException]

    @property
    def size_on_disk_str(self) -> str:
        """
        (property) Sum of all valid repo sizes in the cache-system as a human-readable
        string.
        """
        return convert_readable_size(self.size_on_disk)

    def export_as_table(self) -> str:
        """Generate a detailed table from the [`ModelScopeCacheInfo`] object.

        Returns a table with a row per repo and revision (thus multiple rows can appear for a single repo), with columns
        "repo_id", "repo_type", "revision", "size_on_disk", "nb_files", "last_modified", "refs", "local_path".

        Example:
        ```py
        >>> from modelscope.hub.utils import scan_cache_dir

        >>> ms_cache_info = scan_cache_dir()
        ModelScopeCacheInfo(...)

        >>> print(ms_cache_info.export_as_table())
        REPO ID                                             REPO TYPE REVISION                                      SIZE ON DISK NB FILES LAST_MODIFIED REFS LOCAL PATH
        --------------------------------------------------- --------- -------------------------------------------- ------------ -------- ------------- ---- -------------------------------------------------------------
        damo/bert-base-chinese                              model     9338f7b671827df886678df2bdd7cc7b4f36dffd             2.7M        5 1 week ago    main ~/.cache/modelscope/hub/models--damo--bert-base-chinese/...
        damo/structured-bert                                model     76f64c2173c6ff1941f5ca08a23fa36611276874             8.8K        1 1 week ago    main ~/.cache/modelscope/hub/models--damo--structured-bert/...
        damo/t5-base                                        model     d78aea13fa7ecd06c29e3e46195d6341255065d5           893.8M        4 7 months ago  main ~/.cache/modelscope/hub/models--damo--t5-base/...
        ```

        Returns:
            `str`: The table as a string.
        """  # noqa: E501

        def format_repo_revision(repo: CachedRepoInfo,
                                 revision: CachedRevisionInfo) -> List[str]:
            """Format a single repo and revision into a list of strings for tabulation."""
            return [
                repo.repo_id,
                repo.repo_type,
                revision.commit_hash,
                '{:>12}'.format(revision.size_on_disk_str),
                revision.nb_files,
                revision.last_modified_str,
                ', '.join(sorted(revision.refs)),
                str(revision.snapshot_path),
            ]

        column_headers = [
            'REPO ID',
            'REPO TYPE',
            'REVISION',
            'SIZE ON DISK',
            'NB FILES',
            'LAST_MODIFIED',
            'REFS',
            'LOCAL PATH',
        ]

        table_data = [
            format_repo_revision(repo, revision)
            for repo in sorted(self.repos, key=lambda repo: repo.repo_path)
            for revision in sorted(
                repo.revisions, key=lambda revision: revision.commit_hash)
        ]

        return tabulate(
            rows=table_data,
            headers=column_headers,
        )


def scan_cache_dir(
        cache_dir: Optional[Union[str, Path]] = None) -> ModelScopeCacheInfo:
    """Scan the entire ModelScope cache-system and return a [`ModelScopeCacheInfo`] structure.

    Use `scan_cache_dir` to programmatically scan your cache-system. The cache
    will be scanned repo by repo. If a repo is corrupted, a [`~CorruptedCacheException`]
    will be thrown internally but captured and returned in the [`~ModelScopeCacheInfo`]
    structure. Only valid repos get a proper report.

    ```py
    >>> from modelscope.hub.utils import scan_cache_dir

    >>> ms_cache_info = scan_cache_dir()
    ModelScopeCacheInfo(
        size_on_disk=3398085269,
        repos=frozenset({
            CachedRepoInfo(
                repo_id='damo/t5-small',
                repo_type='model',
                repo_path=PosixPath(...),
                size_on_disk=970726914,
                nb_files=11,
                revisions=frozenset({
                    CachedRevisionInfo(
                        commit_hash='d78aea13fa7ecd06c29e3e46195d6341255065d5',
                        size_on_disk=970726339,
                        snapshot_path=PosixPath(...),
                        files=frozenset({
                            CachedFileInfo(
                                file_name='config.json',
                                size_on_disk=1197
                                file_path=PosixPath(...),
                                blob_path=PosixPath(...),
                            ),
                            CachedFileInfo(...),
                            ...
                        }),
                    ),
                    CachedRevisionInfo(...),
                    ...
                }),
            ),
            CachedRepoInfo(...),
            ...
        }),
        warnings=[
            CorruptedCacheException("Snapshots dir doesn't exist in cached repo: ..."),
            CorruptedCacheException(...),
            ...
        ],
    )
    ```

    Args:
        cache_dir (`str` or `Path`, `optional`):
            Cache directory to scan. Defaults to the default ModelScope cache directory.

    Raises:
        `CacheNotFound`: If the cache directory does not exist.
        `ValueError`: If the cache directory is a file, instead of a directory.

    Returns: a [`ModelScopeCacheInfo`] object.
    """
    if cache_dir is None:
        cache_dir = get_default_modelscope_cache_dir()

    cache_dir = Path(cache_dir).expanduser().resolve()
    if not cache_dir.exists():
        raise CacheNotFound(
            f'Cache directory not found: {cache_dir}. Please use `cache_dir` argument or set `MODELSCOPE_CACHE` environment variable.',  # noqa: E501
            cache_dir=cache_dir,
        )

    if cache_dir.is_file():
        raise ValueError(
            f'Scan cache expects a directory but found a file: {cache_dir}. Please use `cache_dir` argument or set `MODELSCOPE_CACHE` environment variable.'  # noqa: E501
        )

    repos: Set[CachedRepoInfo] = set()
    warnings: List[CorruptedCacheException] = []

    # ModelScope structure is different - we need to look in models/ and datasets/ directories
    model_dir = cache_dir / 'models'
    dataset_dir = cache_dir / 'datasets'

    # Check models directory
    if model_dir.exists() and model_dir.is_dir():
        for repo_path in model_dir.iterdir():
            if repo_path.name in FILES_TO_IGNORE or not repo_path.is_dir():
                continue
            try:
                repos.add(
                    _scan_cached_repo(repo_path, repo_type=REPO_TYPE_MODEL))
            except CorruptedCacheException as e:
                warnings.append(e)

    # Check datasets directory
    if dataset_dir.exists() and dataset_dir.is_dir():
        for repo_path in dataset_dir.iterdir():
            if repo_path.name in FILES_TO_IGNORE or not repo_path.is_dir():
                continue
            try:
                repos.add(
                    _scan_cached_repo(repo_path, repo_type=REPO_TYPE_DATASET))
            except CorruptedCacheException as e:
                warnings.append(e)

    # Also check for repos directly in cache_dir (older structure)
    for repo_path in cache_dir.iterdir():
        if repo_path.name in ['models', 'datasets', '.locks'
                              ] or repo_path.name in FILES_TO_IGNORE:
            continue

        if not repo_path.is_dir():
            continue

        try:
            # Assume they are all models
            repos.add(_scan_cached_repo(repo_path, repo_type=REPO_TYPE_MODEL))
        except CorruptedCacheException as e:
            warnings.append(e)

    return ModelScopeCacheInfo(
        repos=frozenset(repos),
        size_on_disk=sum(repo.size_on_disk for repo in repos),
        warnings=warnings,
    )


def _scan_cached_repo(repo_path: Path, repo_type: str) -> CachedRepoInfo:
    """Scan a single cache repo and return information about it.

    Any unexpected behavior will raise a [`~CorruptedCacheException`].
    """
    if not repo_path.is_dir():
        raise CorruptedCacheException(
            f'Repo path is not a directory: {repo_path}')

    # ModelScope paths are directly structured as owner/name
    # Extract repo_id from the path based on the parent directories
    if repo_type == REPO_TYPE_MODEL and 'models' in str(repo_path.parent):
        # For models stored in models/owner/name structure
        repo_id = repo_path.name
        if repo_path.parent and repo_path.parent.name not in ['models']:
            repo_id = f'{repo_path.parent.name}/{repo_id}'
    elif repo_type == REPO_TYPE_DATASET and 'datasets' in str(
            repo_path.parent):
        # For datasets stored in datasets/owner/name structure
        repo_id = repo_path.name
        if repo_path.parent and repo_path.parent.name not in ['datasets']:
            repo_id = f'{repo_path.parent.name}/{repo_id}'
    else:
        # Fallback: assume the directory name is the repo ID or contains owner/name format
        if '/' in repo_path.name:
            repo_id = repo_path.name
        else:
            # Try to determine if parent directory might be the owner
            parent_name = repo_path.parent.name
            if parent_name not in ['models', 'datasets', 'hub']:
                repo_id = f'{parent_name}/{repo_path.name}'
            else:
                repo_id = repo_path.name

    # Validate repo type
    if repo_type not in {REPO_TYPE_DATASET, REPO_TYPE_MODEL}:
        raise CorruptedCacheException(
            f'Repo type must be `dataset` or `model`, found `{repo_type}` ({repo_path}).'
        )

    # Use ModelFileSystemCache to get cached files information
    try:
        cache = ModelFileSystemCache(str(repo_path))
        cached_files = cache.cached_files
        cached_model_revision = cache.cached_model_revision
    except Exception as e:
        raise CorruptedCacheException(f'Failed to load cache information: {e}')

    # In ModelScope, snapshots are stored directly in the repo directory
    snapshots_path = repo_path / 'snapshots'
    refs_path = repo_path / 'refs'

    if not snapshots_path.exists() or not snapshots_path.is_dir():
        raise CorruptedCacheException(
            f"Snapshots dir doesn't exist in cached repo: {snapshots_path}")

    # Scan over `refs` directory to get all references
    refs_by_hash: Dict[str, Set[str]] = defaultdict(set)
    if refs_path.exists():
        if refs_path.is_file():
            raise CorruptedCacheException(
                f'Refs directory cannot be a file: {refs_path}')

        for ref_path in refs_path.glob('**/*'):
            # glob("**/*") iterates over all files and directories -> skip directories
            if ref_path.is_dir() or ref_path.name in FILES_TO_IGNORE:
                continue

            ref_name = str(ref_path.relative_to(refs_path))
            try:
                with ref_path.open() as f:
                    commit_hash = f.read().strip()
                    refs_by_hash[commit_hash].add(ref_name)
            except Exception as e:
                raise CorruptedCacheException(
                    f'Failed to read ref file {ref_path}: {e}')

    # Scan snapshots directory to get revision information
    cached_revisions: Set[CachedRevisionInfo] = set()
    blob_stats = {}  # Track blob file stats

    # Process each revision (snapshot)
    for revision_path in snapshots_path.iterdir():
        # Ignore OS-created helper files
        if revision_path.name in FILES_TO_IGNORE:
            continue
        if revision_path.is_file():
            raise CorruptedCacheException(
                f'Snapshots folder corrupted. Found a file: {revision_path}')

        # Collect information about files in this revision
        cached_files_in_revision = set()
        for file_path in revision_path.glob('**/*'):
            # Skip directories and ignored files
            if file_path.is_dir() or file_path.name in FILES_TO_IGNORE:
                continue

            # Get file stats
            blob_path = file_path
            if not blob_path.exists():
                raise CorruptedCacheException(f'File missing: {blob_path}')

            if blob_path not in blob_stats:
                blob_stats[blob_path] = blob_path.stat()

            # Create CachedFileInfo for this file
            cached_files_in_revision.add(
                CachedFileInfo(
                    file_name=file_path.name,
                    file_path=file_path,
                    size_on_disk=blob_stats[blob_path].st_size,
                    blob_path=blob_path,
                    blob_last_accessed=blob_stats[blob_path].st_atime,
                    blob_last_modified=blob_stats[blob_path].st_mtime,
                ))

        # Calculate revision metadata
        if len(cached_files_in_revision) > 0:
            revision_last_modified = max(blob_stats[file.blob_path].st_mtime
                                         for file in cached_files_in_revision)
        else:
            revision_last_modified = revision_path.stat().st_mtime

        # Add revision information
        cached_revisions.add(
            CachedRevisionInfo(
                commit_hash=revision_path.name,
                files=frozenset(cached_files_in_revision),
                refs=frozenset(refs_by_hash.pop(revision_path.name, set())),
                size_on_disk=sum(blob_stats[blob_path].st_size
                                 for blob_path in set(
                                     file.blob_path
                                     for file in cached_files_in_revision)),
                snapshot_path=revision_path,
                last_modified=revision_last_modified,
            ))

    # Check that all refs referred to an existing revision
    if len(refs_by_hash) > 0:
        raise CorruptedCacheException(
            f'Reference(s) refer to missing commit hashes: {dict(refs_by_hash)} ({repo_path}).'
        )

    # Calculate repository-wide statistics
    if len(blob_stats) > 0:
        repo_last_accessed = max(stat.st_atime for stat in blob_stats.values())
        repo_last_modified = max(stat.st_mtime for stat in blob_stats.values())
    else:
        repo_stats = repo_path.stat()
        repo_last_accessed = repo_stats.st_atime
        repo_last_modified = repo_stats.st_mtime

    # Build and return frozen structure
    return CachedRepoInfo(
        nb_files=len(blob_stats),
        repo_id=repo_id,
        repo_path=repo_path,
        repo_type=repo_type,
        revisions=frozenset(cached_revisions),
        size_on_disk=sum(stat.st_size for stat in blob_stats.values()),
        last_accessed=repo_last_accessed,
        last_modified=repo_last_modified,
    )
