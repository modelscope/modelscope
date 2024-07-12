# Copyright (c) Alibaba, Inc. and its affiliates.

import fnmatch
import os
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Dict, List, Optional, Union

from modelscope.hub.api import HubApi, ModelScopeConfig
from modelscope.hub.errors import NotExistError
from modelscope.utils.constant import DEFAULT_DATASET_REVISION
from modelscope.utils.file_utils import get_dataset_cache_root
from modelscope.utils.logger import get_logger
from .file_download import create_temporary_directory_and_cache, download_file
from .utils.utils import model_id_to_group_owner_name

logger = get_logger()


def dataset_file_download(
    dataset_id: str,
    file_path: str,
    revision: Optional[str] = DEFAULT_DATASET_REVISION,
    cache_dir: Union[str, Path, None] = None,
    local_dir: Optional[str] = None,
    user_agent: Optional[Union[Dict, str]] = None,
    local_files_only: Optional[bool] = False,
    cookies: Optional[CookieJar] = None,
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
        file_path (str): The relative path of the file to download.
        revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
            commit hash. NOTE: currently only branch and tag name is supported
        cache_dir (str, Path, optional): Path to the folder where cached files are stored, dataset file will
            be save as cache_dir/dataset_id/THE_DATASET_FILES.
        local_dir (str, optional): Specific local directory path to which the file will be downloaded.
        user_agent (str, dict, optional): The user-agent info in the form of a dictionary or a string.
        local_files_only (bool, optional): If `True`, avoid downloading the file and return the path to the
            local cached file if it exists.
        cookies (CookieJar, optional): The cookie of the request, default None.
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
    temporary_cache_dir, cache = create_temporary_directory_and_cache(
        dataset_id,
        local_dir=local_dir,
        cache_dir=cache_dir,
        default_cache_root=get_dataset_cache_root())

    if local_files_only:
        cached_file_path = cache.get_file_by_path(file_path)
        if cached_file_path is not None:
            logger.warning(
                "File exists in local cache, but we're not sure it's up to date"
            )
            return cached_file_path
        else:
            raise ValueError(
                'Cannot find the requested files in the cached path and outgoing'
                ' traffic has been disabled. To enable dataset look-ups and downloads'
                " online, set 'local_files_only' to False.")
    else:
        # make headers
        headers = {
            'user-agent':
            ModelScopeConfig.get_user_agent(user_agent=user_agent, )
        }
        _api = HubApi()
        if cookies is None:
            cookies = ModelScopeConfig.get_cookies()
        group_or_owner, name = model_id_to_group_owner_name(dataset_id)
        if not revision:
            revision = DEFAULT_DATASET_REVISION
        files_list_tree = _api.list_repo_tree(
            dataset_name=name,
            namespace=group_or_owner,
            revision=revision,
            root_path='/',
            recursive=True)
        if not ('Code' in files_list_tree and files_list_tree['Code'] == 200):
            print(
                'Get dataset: %s file list failed, request_id: %s, message: %s'
                % (dataset_id, files_list_tree['RequestId'],
                   files_list_tree['Message']))
            return None
        file_meta_to_download = None
        #  find the file to download.
        for file_meta in files_list_tree['Data']['Files']:
            if file_meta['Type'] == 'tree':
                continue

            if fnmatch.fnmatch(file_meta['Path'], file_path):
                # check file is exist in cache, if existed, skip download, otherwise download
                if cache.exists(file_meta):
                    file_name = os.path.basename(file_meta['Name'])
                    logger.debug(
                        f'File {file_name} already in cache, skip downloading!'
                    )
                    return cache.get_file_by_info(file_meta)
                else:
                    file_meta_to_download = file_meta
                break
        if file_meta_to_download is None:
            raise NotExistError('The file path: %s not exist in: %s' %
                                (file_path, dataset_id))

        # start download file.
        # get download url
        url = _api.get_dataset_file_url(
            file_name=file_meta_to_download['Path'],
            dataset_name=name,
            namespace=group_or_owner,
            revision=revision)

        return download_file(url, file_meta_to_download, temporary_cache_dir,
                             cache, headers, cookies)


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
        allow_file_pattern (`str` or `List`, *optional*, default to `None`):
            Any file pattern to be downloading, like exact file names or file extensions.
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
    temporary_cache_dir, cache = create_temporary_directory_and_cache(
        dataset_id,
        local_dir=local_dir,
        cache_dir=cache_dir,
        default_cache_root=get_dataset_cache_root())

    if local_files_only:
        if len(cache.cached_files) == 0:
            raise ValueError(
                'Cannot find the requested files in the cached path and outgoing'
                ' traffic has been disabled. To enable dataset look-ups and downloads'
                " online, set 'local_files_only' to False.")
        logger.warning('We can not confirm the cached file is for revision: %s'
                       % revision)
        return cache.get_root_location(
        )  # we can not confirm the cached file is for snapshot 'revision'
    else:
        # make headers
        headers = {
            'user-agent':
            ModelScopeConfig.get_user_agent(user_agent=user_agent, )
        }
        _api = HubApi()
        if cookies is None:
            cookies = ModelScopeConfig.get_cookies()
        group_or_owner, name = model_id_to_group_owner_name(dataset_id)
        if not revision:
            revision = DEFAULT_DATASET_REVISION
        files_list_tree = _api.list_repo_tree(
            dataset_name=name,
            namespace=group_or_owner,
            revision=revision,
            root_path='/',
            recursive=True)
        if not ('Code' in files_list_tree and files_list_tree['Code'] == 200):
            print(
                'Get dataset: %s file list failed, request_id: %s, message: %s'
                % (dataset_id, files_list_tree['RequestId'],
                   files_list_tree['Message']))
            return None

        if ignore_file_pattern is None:
            ignore_file_pattern = []
        if isinstance(ignore_file_pattern, str):
            ignore_file_pattern = [ignore_file_pattern]
        ignore_file_pattern = [
            item if not item.endswith('/') else item + '*'
            for item in ignore_file_pattern
        ]

        if allow_file_pattern is not None:
            if isinstance(allow_file_pattern, str):
                allow_file_pattern = [allow_file_pattern]
            allow_file_pattern = [
                item if not item.endswith('/') else item + '*'
                for item in allow_file_pattern
            ]

        for file_meta in files_list_tree['Data']['Files']:
            if file_meta['Type'] == 'tree' or \
                    any(fnmatch.fnmatch(file_meta['Path'], pattern) for pattern in ignore_file_pattern):
                continue

            if allow_file_pattern is not None and allow_file_pattern:
                if not any(
                        fnmatch.fnmatch(file_meta['Path'], pattern)
                        for pattern in allow_file_pattern):
                    continue

            # check file is exist in cache, if existed, skip download, otherwise download
            if cache.exists(file_meta):
                file_name = os.path.basename(file_meta['Name'])
                logger.debug(
                    f'File {file_name} already in cache, skip downloading!')
                continue

            # get download url
            url = _api.get_dataset_file_url(
                file_name=file_meta['Path'],
                dataset_name=name,
                namespace=group_or_owner,
                revision=revision)

            download_file(url, file_meta, temporary_cache_dir, cache, headers,
                          cookies)

        cache.save_model_version(revision_info=revision)
        return os.path.join(cache.get_root_location())
