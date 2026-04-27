# Copyright (c) Alibaba, Inc. and its affiliates.

import fnmatch
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from tqdm.auto import tqdm

from modelscope.utils.constant import (DEFAULT_DATASET_REVISION,
                                       DEFAULT_MODEL_REVISION,
                                       INTRA_CLOUD_ACCELERATION,
                                       REPO_TYPE_DATASET, REPO_TYPE_MODEL,
                                       REPO_TYPE_SUPPORT)
from modelscope.utils.file_utils import get_modelscope_cache_dir
from modelscope.utils.logger import get_logger
from modelscope.utils.thread_utils import thread_executor
from .api import HubApi, ModelScopeConfig
from .callback import ProgressCallback
from .constants import DEFAULT_MAX_WORKERS
from .errors import FileDownloadError, InvalidParameter
from .file_download import (create_temporary_directory_and_cache,
                            download_file, get_file_download_url)
from .utils.caching import ModelFileSystemCache
from .utils.utils import (extract_root_from_patterns,
                          get_model_masked_directory,
                          model_id_to_group_owner_name, strtobool,
                          weak_file_lock)

logger = get_logger()

DEFAULT_DATASET_PAGE_SIZE = 200


def snapshot_download(
    model_id: str = None,
    revision: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    user_agent: Optional[Union[Dict, str]] = None,
    local_files_only: Optional[bool] = False,
    cookies: Optional[CookieJar] = None,
    ignore_file_pattern: Optional[Union[str, List[str]]] = None,
    allow_file_pattern: Optional[Union[str, List[str]]] = None,
    local_dir: Optional[str] = None,
    allow_patterns: Optional[Union[List[str], str]] = None,
    ignore_patterns: Optional[Union[List[str], str]] = None,
    max_workers: Optional[int] = None,
    repo_id: str = None,
    repo_type: Optional[str] = REPO_TYPE_MODEL,
    enable_file_lock: Optional[bool] = None,
    progress_callbacks: List[Type[ProgressCallback]] = None,
    token: Optional[str] = None,
) -> str:
    """Download all files of a repo.
    Downloads a whole snapshot of a repo's files at the specified revision. This
    is useful when you want all files from a repo, because you don't know which
    ones you will need a priori. All files are nested inside a folder in order
    to keep their actual filename relative to that folder.

    An alternative would be to just clone a repo but this would require that the
    user always has git and git-lfs installed, and properly configured.

    Args:
        repo_id (str): A user or an organization name and a repo name separated by a `/`.
        model_id (str): A user or an organization name and a model name separated by a `/`.
            if `repo_id` is provided, `model_id` will be ignored.
        repo_type (str, optional): The type of the repo, either 'model' or 'dataset'.
        revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
            commit hash. NOTE: currently only branch and tag name is supported
        cache_dir (str, Path, optional): Path to the folder where cached files are stored, model will
            be save as cache_dir/model_id/THE_MODEL_FILES.
        user_agent (str, dict, optional): The user-agent info in the form of a dictionary or a string.
        local_files_only (bool, optional): If `True`, avoid downloading the file and return the path to the
            local cached file if it exists.
        cookies (CookieJar, optional): The cookie of the request, default None.
        ignore_file_pattern (`str` or `List`, *optional*, default to `None`):
            Any file pattern to be ignored in downloading, like exact file names or file extensions.
        allow_file_pattern (`str` or `List`, *optional*, default to `None`):
            Any file pattern to be downloading, like exact file names or file extensions.
        local_dir (str, optional): Specific local directory path to which the file will be downloaded.
        allow_patterns (`str` or `List`, *optional*, default to `None`):
            If provided, only files matching at least one pattern are downloaded, priority over allow_file_pattern.
            For hugging-face compatibility.
        ignore_patterns (`str` or `List`, *optional*, default to `None`):
            If provided, files matching any of the patterns are not downloaded, priority over ignore_file_pattern.
            For hugging-face compatibility.
        max_workers (`int`): The maximum number of workers to download files, default 8.
        enable_file_lock (`bool`): Enable file lock, this is useful in multiprocessing downloading, default `True`.
            If you find something wrong with file lock and have a problem modifying your code,
            change `MODELSCOPE_HUB_FILE_LOCK` env to `false`.
        progress_callbacks (`List[Type[ProgressCallback]]`, **optional**, default to `None`):
            progress callbacks to track the download progress.
        token (str, optional): The user token.
    Raises:
        ValueError: the value details.

    Returns:
        str: Local folder path (string) of repo snapshot

    Note:
        Raises the following errors:
        - [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
        if `use_auth_token=True` and the token cannot be found.
        - [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError) if
        ETag cannot be determined.
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
        if some parameter value is invalid
    """

    repo_id = repo_id or model_id
    if not repo_id:
        raise ValueError('Please provide a valid model_id or repo_id')

    if repo_type not in REPO_TYPE_SUPPORT:
        raise ValueError(
            f'Invalid repo type: {repo_type}, only support: {REPO_TYPE_SUPPORT}'
        )

    max_workers = max_workers or DEFAULT_MAX_WORKERS

    if revision is None:
        revision = DEFAULT_DATASET_REVISION if repo_type == REPO_TYPE_DATASET else DEFAULT_MODEL_REVISION

    if enable_file_lock is None:
        enable_file_lock = strtobool(
            os.environ.get('MODELSCOPE_HUB_FILE_LOCK', 'true'))

    if enable_file_lock:
        system_cache = cache_dir if cache_dir is not None else get_modelscope_cache_dir(
        )
        os.makedirs(os.path.join(system_cache, '.lock'), exist_ok=True)
        lock_file = os.path.join(system_cache, '.lock',
                                 repo_id.replace('/', '___'))
        context = weak_file_lock(lock_file)
    else:
        context = nullcontext()
    with context:
        return _snapshot_download(
            repo_id,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            user_agent=user_agent,
            local_files_only=local_files_only,
            cookies=cookies,
            ignore_file_pattern=ignore_file_pattern,
            allow_file_pattern=allow_file_pattern,
            local_dir=local_dir,
            ignore_patterns=ignore_patterns,
            allow_patterns=allow_patterns,
            max_workers=max_workers,
            progress_callbacks=progress_callbacks,
            token=token)


def dataset_snapshot_download(
    dataset_id: str,
    revision: Optional[str] = DEFAULT_DATASET_REVISION,
    cache_dir: Union[str, Path, None] = None,
    local_dir: Optional[str] = None,
    user_agent: Optional[Union[Dict, str]] = None,
    local_files_only: Optional[bool] = False,
    cookies: Optional[CookieJar] = None,
    ignore_file_pattern: Optional[Union[str, List[str]]] = None,
    allow_file_pattern: Optional[Union[str, List[str]]] = None,
    allow_patterns: Optional[Union[List[str], str]] = None,
    ignore_patterns: Optional[Union[List[str], str]] = None,
    enable_file_lock: Optional[bool] = None,
    max_workers: int = 8,
    token: Optional[str] = None,
) -> str:
    """Download raw files of a dataset.
    Downloads all files at the specified revision. This
    is useful when you want all files from a dataset, because you don't know which
    ones you will need a priori. All files are nested inside a folder in order
    to keep their actual filename relative to that folder.

    An alternative would be to just clone a dataset but this would require that the
    user always has git and git-lfs installed, and properly configured.

    Args:
        dataset_id (str): A user or an organization name and a dataset name separated by a `/`.
        revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
            commit hash. NOTE: currently only branch and tag name is supported
        cache_dir (str, Path, optional): Path to the folder where cached files are stored, dataset will
            be save as cache_dir/dataset_id/THE_DATASET_FILES.
        local_dir (str, optional): Specific local directory path to which the file will be downloaded.
        user_agent (str, dict, optional): The user-agent info in the form of a dictionary or a string.
        local_files_only (bool, optional): If `True`, avoid downloading the file and return the path to the
            local cached file if it exists.
        cookies (CookieJar, optional): The cookie of the request, default None.
        ignore_file_pattern (`str` or `List`, *optional*, default to `None`):
            Any file pattern to be ignored in downloading, like exact file names or file extensions.
            Use regression is deprecated.
        allow_file_pattern (`str` or `List`, *optional*, default to `None`):
            Any file pattern to be downloading, like exact file names or file extensions.
        allow_patterns (`str` or `List`, *optional*, default to `None`):
            If provided, only files matching at least one pattern are downloaded, priority over allow_file_pattern.
            For hugging-face compatibility.
        ignore_patterns (`str` or `List`, *optional*, default to `None`):
            If provided, files matching any of the patterns are not downloaded, priority over ignore_file_pattern.
            For hugging-face compatibility.
        enable_file_lock (`bool`): Enable file lock, this is useful in multiprocessing downloading, default `True`.
            If you find something wrong with file lock and have a problem modifying your code,
            change `MODELSCOPE_HUB_FILE_LOCK` env to `false`.
        max_workers (`int`): The maximum number of workers to download files, default 8.
        token (str, optional): The user token.
    Raises:
        ValueError: the value details.

    Returns:
        str: Local folder path (string) of repo snapshot

    Note:
        Raises the following errors:
        - [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
        if `use_auth_token=True` and the token cannot be found.
        - [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError) if
        ETag cannot be determined.
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
        if some parameter value is invalid
    """
    if enable_file_lock is None:
        enable_file_lock = strtobool(
            os.environ.get('MODELSCOPE_HUB_FILE_LOCK', 'true'))

    if enable_file_lock:
        system_cache = cache_dir if cache_dir is not None else get_modelscope_cache_dir(
        )
        os.makedirs(os.path.join(system_cache, '.lock'), exist_ok=True)
        lock_file = os.path.join(system_cache, '.lock',
                                 dataset_id.replace('/', '___'))
        context = weak_file_lock(lock_file)
    else:
        context = nullcontext()
    with context:
        return _snapshot_download(
            dataset_id,
            repo_type=REPO_TYPE_DATASET,
            revision=revision,
            cache_dir=cache_dir,
            user_agent=user_agent,
            local_files_only=local_files_only,
            cookies=cookies,
            ignore_file_pattern=ignore_file_pattern,
            allow_file_pattern=allow_file_pattern,
            local_dir=local_dir,
            ignore_patterns=ignore_patterns,
            allow_patterns=allow_patterns,
            max_workers=max_workers,
            token=token)


def _snapshot_download(
    repo_id: str,
    *,
    repo_type: Optional[str] = None,
    revision: Optional[str] = DEFAULT_MODEL_REVISION,
    cache_dir: Union[str, Path, None] = None,
    user_agent: Optional[Union[Dict, str]] = None,
    local_files_only: Optional[bool] = False,
    cookies: Optional[CookieJar] = None,
    ignore_file_pattern: Optional[Union[str, List[str]]] = None,
    allow_file_pattern: Optional[Union[str, List[str]]] = None,
    local_dir: Optional[str] = None,
    allow_patterns: Optional[Union[List[str], str]] = None,
    ignore_patterns: Optional[Union[List[str], str]] = None,
    max_workers: int = 8,
    progress_callbacks: List[Type[ProgressCallback]] = None,
    token: Optional[str] = None,
):
    if not repo_type:
        repo_type = REPO_TYPE_MODEL
    if repo_type not in REPO_TYPE_SUPPORT:
        raise InvalidParameter('Invalid repo type: %s, only support: %s' %
                               (repo_type, REPO_TYPE_SUPPORT))

    temporary_cache_dir, cache = create_temporary_directory_and_cache(
        repo_id, local_dir=local_dir, cache_dir=cache_dir, repo_type=repo_type)
    system_cache = cache_dir if cache_dir is not None else get_modelscope_cache_dir(
    )
    if local_files_only:
        if len(cache.cached_files) == 0:
            raise ValueError(
                'Cannot find the requested files in the cached path and outgoing'
                ' traffic has been disabled. To enable look-ups and downloads'
                " online, set 'local_files_only' to False.")
        logger.warning('We can not confirm the cached file is for revision: %s'
                       % revision)
        return cache.get_root_location(
        )  # we can not confirm the cached file is for snapshot 'revision'
    else:
        # make headers
        headers = {
            'user-agent':
            ModelScopeConfig.get_user_agent(user_agent=user_agent, ),
            'snapshot-identifier': str(uuid.uuid4()),
        }

        if INTRA_CLOUD_ACCELERATION == 'true':
            region_id: str = (
                os.getenv('INTRA_CLOUD_ACCELERATION_REGION')
                or HubApi()._get_internal_acceleration_domain())
            if region_id:
                logger.info(
                    f'Intra-cloud acceleration enabled for downloading from {repo_id}'
                )
                headers['x-aliyun-region-id'] = region_id

        _api = HubApi(token=token)
        endpoint = _api.get_endpoint_for_read(
            repo_id=repo_id, repo_type=repo_type, token=token)
        if cookies is None:
            cookies = _api.get_cookies()
        if repo_type == REPO_TYPE_MODEL:
            if local_dir:
                directory = os.path.abspath(local_dir)
            elif cache_dir:
                directory = os.path.join(system_cache, *repo_id.split('/'))
            else:
                directory = os.path.join(system_cache, 'models',
                                         *repo_id.split('/'))
            print(
                f'Downloading Model from {endpoint} to directory: {directory}')
            revision_detail = _api.get_valid_revision_detail(
                repo_id, revision=revision, cookies=cookies, endpoint=endpoint)
            revision = revision_detail['Revision']

            # Add snapshot-ci-test for counting the ci test download
            if 'CI_TEST' in os.environ:
                snapshot_header = {**headers, **{'snapshot-ci-test': 'True'}}
            else:
                snapshot_header = {**headers, **{'Snapshot': 'True'}}

            if cache.cached_model_revision is not None:
                snapshot_header[
                    'cached_model_revision'] = cache.cached_model_revision

            # Extract server-side root filter from include patterns
            extracted_root = extract_root_from_patterns(
                allow_file_pattern=_normalize_patterns(allow_file_pattern),
                allow_patterns=_normalize_patterns(allow_patterns))

            repo_files = _api.get_model_files(
                model_id=repo_id,
                revision=revision,
                root=extracted_root,
                recursive=True,
                use_cookies=False if cookies is None else cookies,
                headers=snapshot_header,
                endpoint=endpoint)

            # Fallback: if root filter yielded no results, retry without it
            if not repo_files and extracted_root is not None:
                logger.warning(
                    f"root='{extracted_root}' returned no model files, "
                    f'falling back to root=None for full listing.')
                repo_files = _api.get_model_files(
                    model_id=repo_id,
                    revision=revision,
                    root=None,
                    recursive=True,
                    use_cookies=False if cookies is None else cookies,
                    headers=snapshot_header,
                    endpoint=endpoint)

            # Apply client-side pattern filtering
            repo_files = filter_files_by_patterns(
                repo_files,
                allow_file_pattern=allow_file_pattern,
                ignore_file_pattern=ignore_file_pattern,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns)

            _download_file_lists(
                repo_files,
                cache,
                temporary_cache_dir,
                repo_id,
                _api,
                None,
                None,
                headers,
                repo_type=repo_type,
                revision=revision,
                cookies=cookies,
                pre_filtered=True,
                max_workers=max_workers,
                endpoint=endpoint,
                progress_callbacks=progress_callbacks,
            )
            if '.' in repo_id:
                masked_directory = get_model_masked_directory(
                    directory, repo_id)
                if os.path.exists(directory):
                    logger.info(
                        'Target directory already exists, skipping creation.')
                else:
                    logger.info(f'Creating symbolic link [{directory}].')
                    try:
                        os.symlink(
                            os.path.abspath(masked_directory),
                            directory,
                            target_is_directory=True)
                    except OSError:
                        logger.warning(
                            f'Failed to create symbolic link {directory} for {os.path.abspath(masked_directory)}.'
                        )

        elif repo_type == REPO_TYPE_DATASET:
            if local_dir:
                directory = os.path.abspath(local_dir)
            elif cache_dir:
                directory = os.path.join(system_cache, *repo_id.split('/'))
            else:
                directory = os.path.join(system_cache, 'datasets',
                                         *repo_id.split('/'))
            print(f'Downloading Dataset to directory: {directory}')
            group_or_owner, name = model_id_to_group_owner_name(repo_id)
            revision_detail = revision or DEFAULT_DATASET_REVISION

            # Extract server-side root filter from include patterns
            extracted_root = extract_root_from_patterns(
                allow_file_pattern=_normalize_patterns(allow_file_pattern),
                allow_patterns=_normalize_patterns(allow_patterns))
            root_path = '/' + extracted_root if extracted_root else '/'

            print(f'Fetching file list (root: {root_path})...')
            file_page_iter = _iter_dataset_file_pages(
                _api,
                repo_id,
                revision_detail,
                endpoint,
                token=token,
                root_path=root_path,
                allow_file_pattern=allow_file_pattern,
                ignore_file_pattern=ignore_file_pattern,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns)

            _pipeline_download_dataset(
                file_page_iter,
                cache=cache,
                temporary_cache_dir=temporary_cache_dir,
                repo_id=repo_id,
                api=_api,
                dataset_name=name,
                namespace=group_or_owner,
                headers=headers,
                revision=revision,
                cookies=cookies,
                max_workers=max_workers,
                endpoint=endpoint,
                progress_callbacks=progress_callbacks)

        cache.save_model_version(revision_info=revision_detail)
        cache_root_path = cache.get_root_location()
        return cache_root_path


def fetch_repo_files(
    _api,
    repo_id,
    revision,
    endpoint,
    token=None,
    root_path='/',
    allow_file_pattern=None,
    ignore_file_pattern=None,
    allow_patterns=None,
    ignore_patterns=None,
    page_size=DEFAULT_DATASET_PAGE_SIZE,
):
    """Fetch and filter dataset repo files with pagination and server-side prefix filtering.

    Applies per-page pattern filtering to minimize memory usage.
    Falls back to root_path='/' if the extracted prefix yields no results.

    Args:
        _api: HubApi instance.
        repo_id: Dataset repo identifier (owner/name).
        revision: Git revision.
        endpoint: API endpoint URL.
        token: Authentication token.
        root_path: Server-side directory prefix filter.
        allow_file_pattern: Include patterns for client-side filtering.
        ignore_file_pattern: Exclude patterns for client-side filtering.
        allow_patterns: Additional include patterns (HF-compatible).
        ignore_patterns: Additional exclude patterns (HF-compatible).
        page_size: Number of files per API page request.

    Returns:
        List of filtered file entry dicts.
    """
    if '/' not in repo_id:
        raise InvalidParameter(
            f"Invalid repo_id: '{repo_id}', expected format 'owner/name'")
    _owner, _dataset_name = repo_id.split('/', 1)
    _hub_id, _ = _api.get_dataset_id_and_type(
        dataset_name=_dataset_name,
        namespace=_owner,
        endpoint=endpoint,
        token=token)

    has_patterns = any([
        allow_file_pattern, ignore_file_pattern, allow_patterns,
        ignore_patterns
    ])

    def _paginate_and_filter(effective_root_path):
        """Fetch all pages with the given root_path, applying per-page filtering."""
        page_number = 1
        repo_files = []

        while True:
            try:
                dataset_files = _api.get_dataset_files(
                    repo_id=repo_id,
                    revision=revision,
                    root_path=effective_root_path,
                    recursive=True,
                    page_number=page_number,
                    page_size=page_size,
                    endpoint=endpoint,
                    token=token,
                    dataset_hub_id=_hub_id)
            except Exception as e:
                logger.error(
                    f'Error fetching dataset files (page {page_number}): {e}')
                break

            if not dataset_files:
                break

            # Per-page filtering: apply patterns immediately to reduce memory
            if has_patterns:
                page_filtered = filter_files_by_patterns(
                    dataset_files,
                    allow_file_pattern=allow_file_pattern,
                    ignore_file_pattern=ignore_file_pattern,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns)
                repo_files.extend(page_filtered)
            else:
                # No patterns: keep all non-tree entries
                repo_files.extend(
                    f for f in dataset_files if f.get('Type') != 'tree')

            if len(dataset_files) < page_size:
                break

            page_number += 1

        return repo_files

    # Primary fetch with optimized root_path
    repo_files = _paginate_and_filter(root_path)

    # Fallback: if optimized root_path yielded nothing and it's not the default
    if not repo_files and root_path != '/':
        logger.warning(f"root_path='{root_path}' returned no results, "
                       f"falling back to root_path='/' for full listing.")
        repo_files = _paginate_and_filter('/')

    return repo_files


def _is_valid_regex(pattern: str):
    try:
        re.compile(pattern)
        return True
    except BaseException:
        return False


def _normalize_patterns(patterns: Union[str, List[str]]):
    if isinstance(patterns, str):
        patterns = [patterns]
    if patterns is not None:
        patterns = [
            item if not item.endswith('/') else item + '*' for item in patterns
        ]
    return patterns


def _get_valid_regex_pattern(patterns: List[str]):
    if patterns is not None:
        regex_patterns = []
        for item in patterns:
            if _is_valid_regex(item):
                regex_patterns.append(item)
        return regex_patterns
    else:
        return None


def filter_files_by_patterns(
    repo_files: List[dict],
    *,
    allow_file_pattern: Optional[List[str]] = None,
    ignore_file_pattern: Optional[List[str]] = None,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
) -> List[dict]:
    """Filter repo file entries by include/exclude patterns.

    Skips 'tree' type entries. Applies fnmatch and regex pattern matching.
    Returns only file entries that pass all filter criteria.

    Args:
        repo_files: List of file entry dicts with 'Type', 'Path', 'Name' keys.
        allow_file_pattern: Include patterns (fnmatch). Files must match at least one.
        ignore_file_pattern: Exclude patterns (fnmatch). Matching files are skipped.
        allow_patterns: Additional include patterns (HF-compatible).
        ignore_patterns: Additional exclude patterns (HF-compatible).

    Returns:
        List of file entries that pass all filters.
    """
    ignore_patterns = _normalize_patterns(ignore_patterns)
    allow_patterns = _normalize_patterns(allow_patterns)
    ignore_file_pattern = _normalize_patterns(ignore_file_pattern)
    allow_file_pattern = _normalize_patterns(allow_file_pattern)
    ignore_regex_pattern = _get_valid_regex_pattern(ignore_file_pattern)

    filtered = []
    for repo_file in repo_files:
        if repo_file['Type'] == 'tree':
            continue
        try:
            if ignore_patterns and any(
                    fnmatch.fnmatch(repo_file['Path'], p)
                    for p in ignore_patterns):
                continue

            if ignore_file_pattern and any(
                    fnmatch.fnmatch(repo_file['Path'], p)
                    for p in ignore_file_pattern):
                continue

            if ignore_regex_pattern and any(
                    re.search(p, repo_file['Name']) is not None
                    for p in ignore_regex_pattern):
                continue

            if allow_patterns and not any(
                    fnmatch.fnmatch(repo_file['Path'], p)
                    for p in allow_patterns):
                continue

            if allow_file_pattern and not any(
                    fnmatch.fnmatch(repo_file['Path'], p)
                    for p in allow_file_pattern):
                continue
        except Exception as e:
            logger.warning('Invalid file pattern: %s' % e)
            continue

        filtered.append(repo_file)

    return filtered


def _iter_dataset_file_pages(
    _api,
    repo_id,
    revision,
    endpoint,
    token=None,
    root_path='/',
    allow_file_pattern=None,
    ignore_file_pattern=None,
    allow_patterns=None,
    ignore_patterns=None,
    page_size=DEFAULT_DATASET_PAGE_SIZE,
):
    """Generator that yields filtered file pages from a dataset repo.

    Each yield is a non-empty list of file-entry dicts for one API page.
    Applies per-page pattern filtering to minimize memory usage.
    Falls back to root_path='/' if the extracted prefix yields no results.

    Args:
        _api: HubApi instance.
        repo_id: Dataset repo identifier (owner/name).
        revision: Git revision.
        endpoint: API endpoint URL.
        token: Authentication token.
        root_path: Server-side directory prefix filter.
        allow_file_pattern: Include patterns (fnmatch).
        ignore_file_pattern: Exclude patterns (fnmatch).
        allow_patterns: Additional include patterns (HF-compatible).
        ignore_patterns: Additional exclude patterns (HF-compatible).
        page_size: Number of files per API page request.

    Yields:
        List[dict]: Non-empty list of filtered file entries per page.
    """
    if '/' not in repo_id:
        raise InvalidParameter(
            f"Invalid repo_id: '{repo_id}', expected format 'owner/name'")

    _owner, _dataset_name = repo_id.split('/', 1)
    _hub_id, _ = _api.get_dataset_id_and_type(
        dataset_name=_dataset_name,
        namespace=_owner,
        endpoint=endpoint,
        token=token)

    has_patterns = any([
        allow_file_pattern, ignore_file_pattern, allow_patterns,
        ignore_patterns
    ])

    def _paginate_pages(effective_root_path):
        """Yield filtered file pages for the given root_path."""
        page_number = 1
        total_found = 0

        while True:
            try:
                dataset_files = _api.get_dataset_files(
                    repo_id=repo_id,
                    revision=revision,
                    root_path=effective_root_path,
                    recursive=True,
                    page_number=page_number,
                    page_size=page_size,
                    endpoint=endpoint,
                    token=token,
                    dataset_hub_id=_hub_id)
            except Exception as e:
                logger.error(
                    f'Error fetching dataset files (page {page_number}): {e}')
                break

            if not dataset_files:
                break

            # Per-page filtering to reduce memory footprint
            if has_patterns:
                page_filtered = filter_files_by_patterns(
                    dataset_files,
                    allow_file_pattern=allow_file_pattern,
                    ignore_file_pattern=ignore_file_pattern,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns)
            else:
                # No patterns: keep all non-tree entries
                page_filtered = [
                    f for f in dataset_files if f.get('Type') != 'tree'
                ]

            total_found += len(page_filtered)
            if page_filtered:
                yield page_filtered

            print(
                f'\r  Fetched {total_found} matching files '
                f'({page_number} pages)...',
                end='',
                flush=True)

            if len(dataset_files) < page_size:
                break

            page_number += 1

    # Primary fetch with optimized root_path
    try:
        yielded_any = False
        for page in _paginate_pages(root_path):
            yielded_any = True
            yield page

        # Fallback: if optimized root_path yielded nothing and it's not the default
        if not yielded_any and root_path != '/':
            print(f"\n  root_path='{root_path}' returned no results, "
                  f"falling back to root_path='/' for full listing.")
            for page in _paginate_pages('/'):
                yield page
    finally:
        # Terminate the \r progress line regardless of how iteration ends
        print()


def _pipeline_download_dataset(
    file_page_iter,
    cache,
    temporary_cache_dir,
    repo_id,
    api,
    dataset_name,
    namespace,
    headers,
    revision,
    cookies,
    max_workers=DEFAULT_MAX_WORKERS,
    endpoint=None,
    progress_callbacks=None,
):
    """Pipeline consumer: download dataset files as pages are yielded.

    Consumes the page iterator from _iter_dataset_file_pages, submitting
    each file to a thread pool for concurrent download. Uses tqdm for
    real-time progress and thread-safe error collection.

    Args:
        file_page_iter: Iterator yielding List[dict] file pages.
        cache: ModelFileSystemCache instance for dedup.
        temporary_cache_dir: Temp staging directory.
        repo_id: Dataset repo identifier.
        api: HubApi instance.
        dataset_name: Dataset name component.
        namespace: Owner/namespace component.
        headers: HTTP request headers.
        revision: Git revision.
        cookies: HTTP cookies.
        max_workers: Thread pool concurrency.
        endpoint: API endpoint URL.
        progress_callbacks: Optional progress callback list.
    """
    total_found = 0
    total_cached = 0
    failed_items = []
    lock = threading.Lock()

    def _on_done(future, repo_file):
        """Done callback: update progress bar and collect failures."""
        try:
            future.result()
        except Exception as exc:
            with lock:
                failed_items.append((repo_file, exc))
            logger.debug(
                f"Download failed for {repo_file.get('Path', '?')}: {exc}")
        finally:
            pbar.update(1)

    # tqdm wraps the executor so all callbacks fire before pbar closes
    with tqdm(total=0, unit=' files', disable=False) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for page_files in file_page_iter:
                for repo_file in page_files:
                    total_found += 1
                    pbar.total = total_found
                    pbar.refresh()

                    # Skip files already in cache
                    if cache.exists(repo_file):
                        total_cached += 1
                        pbar.update(1)
                        continue

                    # Build download URL
                    url = api.get_dataset_file_url(
                        file_name=repo_file['Path'],
                        dataset_name=dataset_name,
                        namespace=namespace,
                        revision=revision,
                        endpoint=endpoint)

                    # Submit download task
                    future = executor.submit(
                        download_file,
                        url,
                        repo_file,
                        temporary_cache_dir,
                        cache,
                        headers,
                        cookies,
                        disable_tqdm=False,
                        progress_callbacks=progress_callbacks,
                    )
                    future.add_done_callback(
                        lambda f, rf=repo_file: _on_done(f, rf))

            # Executor __exit__ waits for all futures to complete

    # Report failures after progress bar closes
    if failed_items:
        failed_paths = [
            item.get('Path', '?') if isinstance(item, dict) else str(item)
            for item, _ in failed_items
        ]
        logger.error(f'{len(failed_items)} file(s) failed to download:\n'
                     + '\n'.join(f'  - {p}' for p in failed_paths))

    # Completion summary (always print, even if raising after)
    downloaded = total_found - total_cached - len(failed_items)
    print(f'Download complete: {total_found} files found, '
          f'{total_cached} cached, {downloaded} downloaded'
          + (f', {len(failed_items)} failed' if failed_items else '') + '.')

    if failed_items:
        raise FileDownloadError(
            f'{len(failed_items)} file(s) failed to download out of '
            f'{total_found}.')


def _download_file_lists(
    repo_files: List[str],
    cache: ModelFileSystemCache,
    temporary_cache_dir: str,
    repo_id: str,
    api: HubApi,
    name: str,
    group_or_owner: str,
    headers,
    repo_type: Optional[str] = None,
    revision: Optional[str] = DEFAULT_MODEL_REVISION,
    cookies: Optional[CookieJar] = None,
    ignore_file_pattern: Optional[Union[str, List[str]]] = None,
    allow_file_pattern: Optional[Union[str, List[str]]] = None,
    allow_patterns: Optional[Union[List[str], str]] = None,
    ignore_patterns: Optional[Union[List[str], str]] = None,
    max_workers: int = 8,
    endpoint: Optional[str] = None,
    progress_callbacks: List[Type[ProgressCallback]] = None,
    pre_filtered: bool = False,
):
    if pre_filtered:
        # Files are already filtered by patterns; only check cache
        filtered_repo_files = []
        for repo_file in repo_files:
            if cache.exists(repo_file):
                file_name = os.path.basename(repo_file['Name'])
                logger.debug(
                    f'File {file_name} already in cache with identical hash, skip downloading!'
                )
                continue
            filtered_repo_files.append(repo_file)
    else:
        # Legacy path: apply pattern filtering + cache check
        ignore_patterns = _normalize_patterns(ignore_patterns)
        allow_patterns = _normalize_patterns(allow_patterns)
        ignore_file_pattern = _normalize_patterns(ignore_file_pattern)
        allow_file_pattern = _normalize_patterns(allow_file_pattern)
        # to compatible regex usage.
        ignore_regex_pattern = _get_valid_regex_pattern(ignore_file_pattern)

        filtered_repo_files = []
        for repo_file in repo_files:
            if repo_file['Type'] == 'tree':
                continue
            try:
                # processing patterns
                if ignore_patterns and any([
                        fnmatch.fnmatch(repo_file['Path'], pattern)
                        for pattern in ignore_patterns
                ]):
                    continue

                if ignore_file_pattern and any([
                        fnmatch.fnmatch(repo_file['Path'], pattern)
                        for pattern in ignore_file_pattern
                ]):
                    continue

                if ignore_regex_pattern and any([
                        re.search(pattern, repo_file['Name']) is not None
                        for pattern in ignore_regex_pattern
                ]):  # noqa E501
                    continue

                if allow_patterns is not None and allow_patterns:
                    if not any(
                            fnmatch.fnmatch(repo_file['Path'], pattern)
                            for pattern in allow_patterns):
                        continue

                if allow_file_pattern is not None and allow_file_pattern:
                    if not any(
                            fnmatch.fnmatch(repo_file['Path'], pattern)
                            for pattern in allow_file_pattern):
                        continue
                # check model_file is exist in cache, if existed, skip download
                if cache.exists(repo_file):
                    file_name = os.path.basename(repo_file['Name'])
                    logger.debug(
                        f'File {file_name} already in cache with identical hash, skip downloading!'
                    )
                    continue
            except Exception as e:
                logger.warning('The file pattern is invalid : %s' % e)
            else:
                filtered_repo_files.append(repo_file)

    @thread_executor(
        max_workers=max_workers, disable_tqdm=False, fault_tolerant=True)
    def _download_single_file(repo_file):
        if repo_type == REPO_TYPE_MODEL:
            url = get_file_download_url(
                model_id=repo_id,
                file_path=repo_file['Path'],
                revision=revision,
                endpoint=endpoint)
        elif repo_type == REPO_TYPE_DATASET:
            url = api.get_dataset_file_url(
                file_name=repo_file['Path'],
                dataset_name=name,
                namespace=group_or_owner,
                revision=revision,
                endpoint=endpoint)
        else:
            raise InvalidParameter(
                f'Invalid repo type: {repo_type}, supported types: {REPO_TYPE_SUPPORT}'
            )

        download_file(
            url,
            repo_file,
            temporary_cache_dir,
            cache,
            headers,
            cookies,
            disable_tqdm=False,
            progress_callbacks=progress_callbacks,
        )

    if len(filtered_repo_files) > 0:
        logger.info(
            f'Got {len(filtered_repo_files)} files, start to download ...')
        download_result = _download_single_file(filtered_repo_files)

        # Handle fault-tolerant results: report failed downloads
        failed_items = []
        if isinstance(download_result, tuple) and len(download_result) == 2:
            _, failed_items = download_result
            if failed_items:
                failed_paths = [
                    item['Path'] if isinstance(item, dict) else str(item)
                    for item, _ in failed_items
                ]
                logger.error(
                    f'{len(failed_items)} file(s) failed to download:\n'
                    + '\n'.join(f'  - {p}' for p in failed_paths))

        logger.info(
            f"Finish downloading {len(filtered_repo_files)} files for repo '{repo_id}'"
        )

        if failed_items:
            raise FileDownloadError(
                f'{len(failed_items)} file(s) failed to download out of '
                f'{len(filtered_repo_files)}.')
