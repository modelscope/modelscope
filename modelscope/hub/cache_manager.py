"""Contains utilities to manage the ModelScope cache directory."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Union

from modelscope.hub.errors import CacheNotFound, CorruptedCacheException
from modelscope.hub.utils.caching import ModelFileSystemCache
from modelscope.hub.utils.utils import (convert_readable_size,
                                        format_timesince, tabulate)
from modelscope.utils.constant import REPO_TYPE_DATASET, REPO_TYPE_MODEL
from modelscope.utils.file_utils import get_modelscope_cache_dir
from modelscope.utils.logger import get_logger

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
    file_revision_hash: str
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
        size_on_disk (`int`):
            Sum of the blob file sizes that are symlink-ed by the revision.
        last_modified (`float`):
            Timestamp of the last time the revision has been created/modified.
    """

    commit_hash: str
    snapshot_path: Path
    size_on_disk: int
    files: FrozenSet[CachedFileInfo]

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
        "repo_id", "repo_type", "revision", "size_on_disk", "nb_files", "last_modified", "local_path".

        Example:
        ```py
        >>> from modelscope.hub.cache_manager import scan_cache_dir

        >>> ms_cache_info = scan_cache_dir()
        ModelScopeCacheInfo(...)

        >>> print(ms_cache_info.export_as_table())
        REPO ID                REPO TYPE REVISION   SIZE ON DISK NB FILES LAST_MODIFIED LOCAL PATH
        ---------------------- --------- ---------- ------------ -------- -------------  -------------------------------------------------------------
        damo/bert-base-chinese model     master             2.7M        5 1 week ago     ~/.cache/modelscope/hub/models--damo--bert-base-chinese/...
        damo/structured-bert   model     master             8.8K        1 1 week ago     ~/.cache/modelscope/hub/models--damo--structured-bert/...
        damo/t5-base           model     master           893.8M        4 7 months ago   ~/.cache/modelscope/hub/models--damo--t5-base/...
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
                '{:>12}'.format(repo.size_on_disk_str),
                repo.nb_files,
                repo.last_accessed_str,
                repo.last_modified_str,
                str(repo.repo_path),
            ]

        column_headers = [
            'REPO ID',
            'REPO TYPE',
            'REVISION',
            'SIZE ON DISK',
            'NB FILES',
            'LAST_ACCESSED',
            'LAST_MODIFIED',
            'LOCAL PATH',
        ]

        table_data = [
            format_repo_revision(repo, revision)
            for repo in sorted(self.repos, key=lambda repo: repo.repo_id)
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
                        commit_hash='master',
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
        cache_dir = get_modelscope_cache_dir()

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
        # First level directories are owners/organizations
        model_repos, model_warnings = _scan_dir(
            model_dir, repo_type=REPO_TYPE_MODEL)
        repos.update(model_repos)
        warnings.extend(model_warnings)

    # Check datasets directory
    if dataset_dir.exists() and dataset_dir.is_dir():
        # First level directories are owners/organizations
        dataset_repos, dataset_warnings = _scan_dir(
            dataset_dir, repo_type=REPO_TYPE_DATASET)
        repos.update(dataset_repos)
        warnings.extend(dataset_warnings)

    # Also check for repos directly in cache_dir (older structure)
    # If the repo is not in models/ or datasets/, assume it's a model repo
    other_repos, other_warnings = _scan_dir(
        cache_dir, repo_type=REPO_TYPE_MODEL, inplace=True)
    repos.update(other_repos)
    warnings.extend(other_warnings)

    return ModelScopeCacheInfo(
        repos=frozenset(repos),
        size_on_disk=sum(repo.size_on_disk for repo in repos),
        warnings=warnings,
    )


def _is_valid_dir(dir: Path) -> bool:
    """Check if a directory is valid for scanning."""
    if not dir.exists():
        return False
    if not dir.is_dir():
        return False
    if dir.is_symlink():
        return False
    if dir.name in FILES_TO_IGNORE:
        return False
    return True


def _scan_dir(dir: Path, repo_type: str, inplace: bool = False):
    """Scan a directory for cached repos and return a set of [`~CachedRepoInfo`] and warnings."""
    repos = set()
    warnings = []
    for owner_dir in dir.iterdir():
        # not extend scan the following dirs when scan current dir
        if inplace and owner_dir.name in ['models', 'datasets', 'hub']:
            continue
        if not _is_valid_dir(owner_dir):
            continue
        # Second level directories are repo names
        for name_dir in owner_dir.iterdir():
            if not _is_valid_dir(name_dir):
                continue
            try:
                info = _scan_cached_repo(name_dir, repo_type=repo_type)
                if info is not None:
                    repos.add(info)
            except CorruptedCacheException as e:
                warnings.append(e)
    return repos, warnings


def _scan_cached_repo(repo_path: Path,
                      repo_type: str) -> Optional[CachedRepoInfo]:
    """Scan a single cache repo and return information about it.

    Any unexpected behavior will raise a [`~CorruptedCacheException`].
    """
    if not repo_path.is_dir():
        raise CorruptedCacheException(
            f'Repo path is not a directory: {repo_path}')

    # Use ModelFileSystemCache to get cached files information
    try:
        cache = ModelFileSystemCache(str(repo_path))
        cached_files = cache.cached_files
        cached_model_revision = cache.cached_model_revision
        repo_id = cache.get_model_id().replace('___', '.')

        if repo_id == 'unknown':
            return None  # Skip if repo_id is unknown
    except Exception as e:
        raise CorruptedCacheException(f'Failed to load cache information: {e}')

    # Collect file stats and information
    blob_stats = {}  # Track blob file stats
    cached_files_info = set()

    # Process all cached files
    for cached_file in cached_files:
        file_path = os.path.join(repo_path, cached_file['Path'])
        file_revision_hash = cached_file.get('Revision', '')
        if not os.path.exists(file_path):
            continue

        blob_path = Path(file_path)
        blob_stats[blob_path] = blob_path.stat()

        # Create CachedFileInfo for this file
        cached_files_info.add(
            CachedFileInfo(
                file_name=os.path.basename(cached_file['Path']),
                file_path=blob_path,
                file_revision_hash=file_revision_hash,
                size_on_disk=blob_stats[blob_path].st_size,
                blob_path=blob_path,
                blob_last_accessed=blob_stats[blob_path].st_atime,
                blob_last_modified=blob_stats[blob_path].st_mtime,
            ))

    # Create a single revision from cached files
    revision_hash = 'master'  # Default revision name
    if cached_model_revision:
        # Extract revision hash from cached_model_revision if available
        if 'Revision:' in cached_model_revision:
            revision_hash = cached_model_revision.split('Revision:')[1].split(
                ',')[0]

    # Calculate revision metadata
    if cached_files_info:
        revision_last_modified = max(blob_stats[file.blob_path].st_mtime
                                     for file in cached_files_info)
    else:
        revision_last_modified = repo_path.stat().st_mtime

    # Create a CachedRevisionInfo for the repository
    cached_revision = CachedRevisionInfo(
        commit_hash=revision_hash,
        files=frozenset(cached_files_info),
        size_on_disk=sum(blob_stats[file.blob_path].st_size
                         for file in cached_files_info),
        snapshot_path=repo_path,
        last_modified=revision_last_modified,
    )

    # Calculate repository-wide statistics
    if blob_stats:
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
        revisions=frozenset([cached_revision]),
        size_on_disk=sum(stat.st_size for stat in blob_stats.values()),
        last_accessed=repo_last_accessed,
        last_modified=repo_last_modified,
    )
