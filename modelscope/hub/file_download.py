# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import hashlib
import io
import os
import shutil
import tempfile
import urllib
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import requests
from requests.adapters import Retry
from tqdm.auto import tqdm

from modelscope.hub.api import HubApi, ModelScopeConfig
from modelscope.hub.constants import (
    API_FILE_DOWNLOAD_CHUNK_SIZE, API_FILE_DOWNLOAD_RETRY_TIMES,
    API_FILE_DOWNLOAD_TIMEOUT, FILE_HASH, MODELSCOPE_DOWNLOAD_PARALLELS,
    MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB, TEMPORARY_FOLDER_NAME)
from modelscope.utils.constant import (DEFAULT_DATASET_REVISION,
                                       DEFAULT_MODEL_REVISION,
                                       INTRA_CLOUD_ACCELERATION,
                                       REPO_TYPE_DATASET, REPO_TYPE_MODEL,
                                       REPO_TYPE_SUPPORT)
from modelscope.utils.file_utils import (get_dataset_cache_root,
                                         get_model_cache_root)
from modelscope.utils.logger import get_logger
from .callback import ProgressCallback, TqdmCallback
from .errors import FileDownloadError, InvalidParameter, NotExistError
from .utils.caching import ModelFileSystemCache
from .utils.utils import (file_integrity_validation, get_endpoint,
                          model_id_to_group_owner_name)

logger = get_logger()


def model_file_download(
    model_id: str,
    file_path: str,
    revision: Optional[str] = DEFAULT_MODEL_REVISION,
    cache_dir: Optional[str] = None,
    user_agent: Union[Dict, str, None] = None,
    local_files_only: Optional[bool] = False,
    cookies: Optional[CookieJar] = None,
    local_dir: Optional[str] = None,
) -> Optional[str]:  # pragma: no cover
    """Download from a given URL and cache it if it's not already present in the local cache.

    Given a URL, this function looks for the corresponding file in the local
    cache. If it's not there, download it. Then return the path to the cached
    file.

    Args:
        model_id (str): The model to whom the file to be downloaded belongs.
        file_path(str): Path of the file to be downloaded, relative to the root of model repo.
        revision(str, optional): revision of the model file to be downloaded.
            Can be any of a branch, tag or commit hash.
        cache_dir (str, Path, optional): Path to the folder where cached files are stored.
        user_agent (dict, str, optional): The user-agent info in the form of a dictionary or a string.
        local_files_only (bool, optional):  If `True`, avoid downloading the file and return the path to the
            local cached file if it exists. if `False`, download the file anyway even it exists.
        cookies (CookieJar, optional): The cookie of download request.
        local_dir (str, optional): Specific local directory path to which the file will be downloaded.

    Returns:
        string: string of local file or if networking is off, last version of
        file cached on disk.

    Raises:
        NotExistError: The file is not exist.
        ValueError: The request parameter error.

    Note:
        Raises the following errors:

            - [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
            if `use_auth_token=True` and the token cannot be found.
            - [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError)
            if ETag cannot be determined.
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            if some parameter value is invalid
    """
    return _repo_file_download(
        model_id,
        file_path,
        repo_type=REPO_TYPE_MODEL,
        revision=revision,
        cache_dir=cache_dir,
        user_agent=user_agent,
        local_files_only=local_files_only,
        cookies=cookies,
        local_dir=local_dir)


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
    return _repo_file_download(
        dataset_id,
        file_path,
        repo_type=REPO_TYPE_DATASET,
        revision=revision,
        cache_dir=cache_dir,
        user_agent=user_agent,
        local_files_only=local_files_only,
        cookies=cookies,
        local_dir=local_dir)


def _repo_file_download(
    repo_id: str,
    file_path: str,
    *,
    repo_type: str = None,
    revision: Optional[str] = DEFAULT_MODEL_REVISION,
    cache_dir: Optional[str] = None,
    user_agent: Union[Dict, str, None] = None,
    local_files_only: Optional[bool] = False,
    cookies: Optional[CookieJar] = None,
    local_dir: Optional[str] = None,
    disable_tqdm: bool = False,
) -> Optional[str]:  # pragma: no cover

    if not repo_type:
        repo_type = REPO_TYPE_MODEL
    if repo_type not in REPO_TYPE_SUPPORT:
        raise InvalidParameter('Invalid repo type: %s, only support: %s' %
                               (repo_type, REPO_TYPE_SUPPORT))

    temporary_cache_dir, cache = create_temporary_directory_and_cache(
        repo_id, local_dir=local_dir, cache_dir=cache_dir, repo_type=repo_type)

    # if local_files_only is `True` and the file already exists in cached_path
    # return the cached path
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
                ' traffic has been disabled. To enable look-ups and downloads'
                " online, set 'local_files_only' to False.")

    _api = HubApi()

    headers = {
        'user-agent': ModelScopeConfig.get_user_agent(user_agent=user_agent, ),
        'snapshot-identifier': str(uuid.uuid4()),
    }

    if INTRA_CLOUD_ACCELERATION == 'true':
        region_id: str = (
            os.getenv('INTRA_CLOUD_ACCELERATION_REGION')
            or _api._get_internal_acceleration_domain())
        if region_id:
            logger.info(
                f'Intra-cloud acceleration enabled for downloading from {repo_id}'
            )
            headers['x-aliyun-region-id'] = region_id

    if cookies is None:
        cookies = ModelScopeConfig.get_cookies()
    repo_files = []
    endpoint = _api.get_endpoint_for_read(repo_id=repo_id, repo_type=repo_type)
    file_to_download_meta = None
    if repo_type == REPO_TYPE_MODEL:
        revision = _api.get_valid_revision(
            repo_id, revision=revision, cookies=cookies, endpoint=endpoint)
        # we need to confirm the version is up-to-date
        # we need to get the file list to check if the latest version is cached, if so return, otherwise download
        repo_files = _api.get_model_files(
            model_id=repo_id,
            revision=revision,
            recursive=True,
            use_cookies=False if cookies is None else cookies,
            endpoint=endpoint)
        for repo_file in repo_files:
            if repo_file['Type'] == 'tree':
                continue

            if repo_file['Path'] == file_path:
                if cache.exists(repo_file):
                    file_name = repo_file['Name']
                    logger.debug(
                        f'File {file_name} already in cache with identical hash, skip downloading!'
                    )
                    return cache.get_file_by_info(repo_file)
                else:
                    file_to_download_meta = repo_file
                break
    elif repo_type == REPO_TYPE_DATASET:
        group_or_owner, name = model_id_to_group_owner_name(repo_id)
        if not revision:
            revision = DEFAULT_DATASET_REVISION
        page_number = 1
        page_size = 100
        while True:
            try:
                dataset_files = _api.get_dataset_files(
                    repo_id=repo_id,
                    revision=revision,
                    root_path='/',
                    recursive=True,
                    page_number=page_number,
                    page_size=page_size,
                    endpoint=endpoint)
            except Exception as e:
                logger.error(
                    f'Get dataset: {repo_id} file list failed, error: {e}')
                break

            is_exist = False
            for repo_file in dataset_files:
                if repo_file['Type'] == 'tree':
                    continue

                if repo_file['Path'] == file_path:
                    if cache.exists(repo_file):
                        file_name = repo_file['Name']
                        logger.debug(
                            f'File {file_name} already in cache with identical hash, skip downloading!'
                        )
                        return cache.get_file_by_info(repo_file)
                    else:
                        file_to_download_meta = repo_file
                        is_exist = True
                    break
            if len(dataset_files) < page_size or is_exist:
                break
            page_number += 1

    if file_to_download_meta is None:
        raise NotExistError('The file path: %s not exist in: %s' %
                            (file_path, repo_id))

    # we need to download again
    if repo_type == REPO_TYPE_MODEL:
        url_to_download = get_file_download_url(repo_id, file_path, revision,
                                                endpoint)
    elif repo_type == REPO_TYPE_DATASET:
        url_to_download = _api.get_dataset_file_url(
            file_name=file_to_download_meta['Path'],
            dataset_name=name,
            namespace=group_or_owner,
            revision=revision,
            endpoint=endpoint)
    else:
        raise ValueError(f'Invalid repo type {repo_type}')

    return download_file(url_to_download, file_to_download_meta,
                         temporary_cache_dir, cache, headers, cookies)


def move_legacy_cache_to_standard_dir(cache_dir: str, model_id: str):
    if cache_dir.endswith(os.path.sep):
        cache_dir = cache_dir.strip(os.path.sep)
    legacy_cache_root = os.path.dirname(cache_dir)
    base_name = os.path.basename(cache_dir)
    if base_name == 'datasets':
        # datasets will not be not affected
        return
    if not legacy_cache_root.endswith('hub'):
        # Two scenarios:
        # We have restructured ModelScope cache directory,
        # Scenery 1:
        #   When MODELSCOPE_CACHE is not set, the default directory remains
        #   the same at  ~/.cache/modelscope/hub
        # Scenery 2:
        #   When MODELSCOPE_CACHE is set, the cache directory is moved from
        #   $MODELSCOPE_CACHE/hub to $MODELSCOPE_CACHE/. In this case,
        #   we will be migrating the hub directory accordingly.
        legacy_cache_root = os.path.join(legacy_cache_root, 'hub')
    group_or_owner, name = model_id_to_group_owner_name(model_id)
    name = name.replace('.', '___')
    temporary_cache_dir = os.path.join(cache_dir, group_or_owner, name)
    legacy_cache_dir = os.path.join(legacy_cache_root, group_or_owner, name)
    if os.path.exists(
            legacy_cache_dir) and not os.path.exists(temporary_cache_dir):
        logger.info(
            f'Legacy cache dir exists: {legacy_cache_dir}, move to {temporary_cache_dir}'
        )
        try:
            shutil.move(legacy_cache_dir, temporary_cache_dir)
        except Exception:  # noqa
            # Failed, skip
            pass


def create_temporary_directory_and_cache(model_id: str,
                                         local_dir: str = None,
                                         cache_dir: str = None,
                                         repo_type: str = REPO_TYPE_MODEL):
    if repo_type == REPO_TYPE_MODEL:
        default_cache_root = get_model_cache_root()
    elif repo_type == REPO_TYPE_DATASET:
        default_cache_root = get_dataset_cache_root()
    else:
        raise ValueError(
            f'repo_type only support model and dataset, but now is : {repo_type}'
        )

    group_or_owner, name = model_id_to_group_owner_name(model_id)
    if local_dir is not None:
        temporary_cache_dir = os.path.join(local_dir, TEMPORARY_FOLDER_NAME)
        cache = ModelFileSystemCache(local_dir)
    else:
        if cache_dir is None:
            cache_dir = default_cache_root
            move_legacy_cache_to_standard_dir(cache_dir, model_id)
        if isinstance(cache_dir, Path):
            cache_dir = str(cache_dir)
        temporary_cache_dir = os.path.join(cache_dir, TEMPORARY_FOLDER_NAME,
                                           group_or_owner, name)
        name = name.replace('.', '___')
        cache = ModelFileSystemCache(cache_dir, group_or_owner, name)

    os.makedirs(temporary_cache_dir, exist_ok=True)
    return temporary_cache_dir, cache


def get_file_download_url(model_id: str,
                          file_path: str,
                          revision: str,
                          endpoint: Optional[str] = None):
    """Format file download url according to `model_id`, `revision` and `file_path`.
    e.g., Given `model_id=john/bert`, `revision=master`, `file_path=README.md`,
    the resulted download url is: https://modelscope.cn/api/v1/models/john/bert/repo?Revision=master&FilePath=README.md

    Args:
        model_id (str): The model_id.
        file_path (str): File path
        revision (str): File revision.
        endpoint (str): The remote endpoint

    Returns:
        str: The file url.
    """
    file_path = urllib.parse.quote_plus(file_path)
    revision = urllib.parse.quote_plus(revision)
    download_url_template = '{endpoint}/api/v1/models/{model_id}/repo?Revision={revision}&FilePath={file_path}'
    if not endpoint:
        endpoint = get_endpoint()
    return download_url_template.format(
        endpoint=endpoint,
        model_id=model_id,
        revision=revision,
        file_path=file_path,
    )


def download_part_with_retry(params):
    # unpack parameters
    model_file_path, progress_callbacks, start, end, url, file_name, cookies, headers = params
    get_headers = {} if headers is None else copy.deepcopy(headers)
    get_headers['X-Request-ID'] = str(uuid.uuid4().hex)
    retry = Retry(
        total=API_FILE_DOWNLOAD_RETRY_TIMES,
        backoff_factor=1,
        allowed_methods=['GET'])
    part_file_name = model_file_path + '_%s_%s' % (start, end)
    while True:
        try:
            partial_length = 0
            if os.path.exists(
                    part_file_name):  # download partial, continue download
                with open(part_file_name, 'rb') as f:
                    partial_length = f.seek(0, io.SEEK_END)
                    for callback in progress_callbacks:
                        callback.update(partial_length)
            download_start = start + partial_length
            if download_start > end:
                break  # this part is download completed.
            get_headers['Range'] = 'bytes=%s-%s' % (download_start, end)
            with open(part_file_name, 'ab+') as f:
                r = requests.get(
                    url,
                    stream=True,
                    headers=get_headers,
                    cookies=cookies,
                    timeout=API_FILE_DOWNLOAD_TIMEOUT)
                for chunk in r.iter_content(
                        chunk_size=API_FILE_DOWNLOAD_CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        for callback in progress_callbacks:
                            callback.update(len(chunk))
            break
        except (Exception) as e:  # no matter what exception, we will retry.
            retry = retry.increment('GET', url, error=e)
            logger.warning('Downloading: %s failed, reason: %s will retry' %
                           (model_file_path, e))
            retry.sleep()


def parallel_download(url: str,
                      local_dir: str,
                      file_name: str,
                      cookies: CookieJar,
                      headers: Optional[Dict[str, str]] = None,
                      file_size: int = None,
                      disable_tqdm: bool = False,
                      progress_callbacks: List[Type[ProgressCallback]] = None,
                      endpoint: str = None):
    progress_callbacks = [] if progress_callbacks is None else progress_callbacks.copy(
    )
    if not disable_tqdm:
        progress_callbacks.append(TqdmCallback)
    progress_callbacks = [
        callback(file_name, file_size) for callback in progress_callbacks
    ]
    # create temp file
    PART_SIZE = 160 * 1024 * 1024  # every part is 160M
    tasks = []
    file_path = os.path.join(local_dir, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    for idx in range(int(file_size / PART_SIZE)):
        start = idx * PART_SIZE
        end = (idx + 1) * PART_SIZE - 1
        tasks.append((file_path, progress_callbacks, start, end, url,
                      file_name, cookies, headers))
    if end + 1 < file_size:
        tasks.append((file_path, progress_callbacks, end + 1, file_size - 1,
                      url, file_name, cookies, headers))
    parallels = min(MODELSCOPE_DOWNLOAD_PARALLELS, 16)
    # download every part
    with ThreadPoolExecutor(
            max_workers=parallels, thread_name_prefix='download') as executor:
        list(executor.map(download_part_with_retry, tasks))
    for callback in progress_callbacks:
        callback.end()
    # merge parts.
    hash_sha256 = hashlib.sha256()
    with open(os.path.join(local_dir, file_name), 'wb') as output_file:
        for task in tasks:
            part_file_name = task[0] + '_%s_%s' % (task[2], task[3])
            with open(part_file_name, 'rb') as part_file:
                while True:
                    chunk = part_file.read(16 * API_FILE_DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    output_file.write(chunk)
                    hash_sha256.update(chunk)
            os.remove(part_file_name)
    return hash_sha256.hexdigest()


def http_get_model_file(
    url: str,
    local_dir: str,
    file_name: str,
    file_size: int,
    cookies: CookieJar,
    headers: Optional[Dict[str, str]] = None,
    disable_tqdm: bool = False,
    progress_callbacks: List[Type[ProgressCallback]] = None,
):
    """Download remote file, will retry 5 times before giving up on errors.

    Args:
        url(str):
            actual download url of the file
        local_dir(str):
            local directory where the downloaded file stores
        file_name(str):
            name of the file stored in `local_dir`
        file_size(int):
            The file size.
        cookies(CookieJar):
            cookies used to authentication the user, which is used for downloading private repos
        headers(Dict[str, str], optional):
            http headers to carry necessary info when requesting the remote file
        disable_tqdm(bool, optional): Disable the progress bar with tqdm.
        progress_callbacks(List[Type[ProgressCallback]], optional):
            progress callbacks to track the download progress.

    Raises:
        FileDownloadError: File download failed.

    """
    progress_callbacks = [] if progress_callbacks is None else progress_callbacks.copy(
    )
    if not disable_tqdm:
        progress_callbacks.append(TqdmCallback)
    progress_callbacks = [
        callback(file_name, file_size) for callback in progress_callbacks
    ]
    get_headers = {} if headers is None else copy.deepcopy(headers)
    get_headers['X-Request-ID'] = str(uuid.uuid4().hex)
    temp_file_path = os.path.join(local_dir, file_name)
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    logger.debug('downloading %s to %s', url, temp_file_path)
    # retry sleep 0.5s, 1s, 2s, 4s
    has_retry = False
    hash_sha256 = hashlib.sha256()
    retry = Retry(
        total=API_FILE_DOWNLOAD_RETRY_TIMES,
        backoff_factor=1,
        allowed_methods=['GET'])

    while True:
        try:
            if file_size == 0:
                # Avoid empty file server request
                with open(temp_file_path, 'w+'):
                    for callback in progress_callbacks:
                        callback.update(1)
                break
            # Determine the length of any existing partial download
            partial_length = 0
            # download partial, continue download
            if os.path.exists(temp_file_path):
                # resuming from interrupted download is also considered as retry
                has_retry = True
                with open(temp_file_path, 'rb') as f:
                    partial_length = f.seek(0, io.SEEK_END)
                    for callback in progress_callbacks:
                        callback.update(partial_length)

            # Check if download is complete
            if partial_length >= file_size:
                break
            # closed range[], from 0.
            get_headers['Range'] = 'bytes=%s-%s' % (partial_length,
                                                    file_size - 1)
            with open(temp_file_path, 'ab+') as f:
                r = requests.get(
                    url,
                    stream=True,
                    headers=get_headers,
                    cookies=cookies,
                    timeout=API_FILE_DOWNLOAD_TIMEOUT)
                r.raise_for_status()
                for chunk in r.iter_content(
                        chunk_size=API_FILE_DOWNLOAD_CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        for callback in progress_callbacks:
                            callback.update(len(chunk))
                        f.write(chunk)
                        # hash would be discarded in retry case anyway
                        if not has_retry:
                            hash_sha256.update(chunk)
            break
        except Exception as e:  # no matter what happen, we will retry.
            has_retry = True
            retry = retry.increment('GET', url, error=e)
            retry.sleep()
    for callback in progress_callbacks:
        callback.end()
    # if anything went wrong, we would discard the real-time computed hash and return None
    return None if has_retry else hash_sha256.hexdigest()


def http_get_file(
    url: str,
    local_dir: str,
    file_name: str,
    cookies: CookieJar,
    headers: Optional[Dict[str, str]] = None,
):
    """Download remote file, will retry 5 times before giving up on errors.

    Args:
        url(str):
            actual download url of the file
        local_dir(str):
            local directory where the downloaded file stores
        file_name(str):
            name of the file stored in `local_dir`
        cookies(CookieJar):
            cookies used to authentication the user, which is used for downloading private repos
        headers(Dict[str, str], optional):
            http headers to carry necessary info when requesting the remote file

    Raises:
        FileDownloadError: File download failed.

    """
    total = -1
    temp_file_manager = partial(
        tempfile.NamedTemporaryFile, mode='wb', dir=local_dir, delete=False)
    get_headers = {} if headers is None else copy.deepcopy(headers)
    get_headers['X-Request-ID'] = str(uuid.uuid4().hex)
    with temp_file_manager() as temp_file:
        logger.debug('downloading %s to %s', url, temp_file.name)
        # retry sleep 0.5s, 1s, 2s, 4s
        retry = Retry(
            total=API_FILE_DOWNLOAD_RETRY_TIMES,
            backoff_factor=1,
            allowed_methods=['GET'])
        while True:
            try:
                downloaded_size = temp_file.tell()
                get_headers['Range'] = 'bytes=%d-' % downloaded_size
                r = requests.get(
                    url,
                    stream=True,
                    headers=get_headers,
                    cookies=cookies,
                    timeout=API_FILE_DOWNLOAD_TIMEOUT)
                r.raise_for_status()
                content_length = r.headers.get('Content-Length')
                total = int(
                    content_length) if content_length is not None else None
                progress = tqdm(
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    total=total,
                    initial=downloaded_size,
                    desc='Downloading [' + file_name + ']',
                )
                for chunk in r.iter_content(
                        chunk_size=API_FILE_DOWNLOAD_CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        progress.update(len(chunk))
                        temp_file.write(chunk)
                progress.close()
                break
            except (Exception) as e:  # no matter what happen, we will retry.
                retry = retry.increment('GET', url, error=e)
                retry.sleep()

    logger.debug('storing %s in cache at %s', url, local_dir)
    downloaded_length = os.path.getsize(temp_file.name)
    if total != downloaded_length:
        os.remove(temp_file.name)
        msg = 'File %s download incomplete, content_length: %s but the \
                    file downloaded length: %s, please download again' % (
            file_name, total, downloaded_length)
        logger.error(msg)
        raise FileDownloadError(msg)
    os.replace(temp_file.name, os.path.join(local_dir, file_name))


def download_file(
    url,
    file_meta,
    temporary_cache_dir,
    cache,
    headers,
    cookies,
    disable_tqdm=False,
    progress_callbacks: List[Type[ProgressCallback]] = None,
):
    if MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB * 1000 * 1000 < file_meta[
            'Size'] and MODELSCOPE_DOWNLOAD_PARALLELS > 1:  # parallel download large file.
        file_digest = parallel_download(
            url,
            temporary_cache_dir,
            file_meta['Path'],
            headers=headers,
            cookies=None if cookies is None else cookies.get_dict(),
            file_size=file_meta['Size'],
            disable_tqdm=disable_tqdm,
            progress_callbacks=progress_callbacks,
        )
    else:
        file_digest = http_get_model_file(
            url,
            temporary_cache_dir,
            file_meta['Path'],
            file_size=file_meta['Size'],
            headers=headers,
            cookies=cookies,
            disable_tqdm=disable_tqdm,
            progress_callbacks=progress_callbacks,
        )

    # check file integrity
    temp_file = os.path.join(temporary_cache_dir, file_meta['Path'])
    if FILE_HASH in file_meta:
        expected_hash = file_meta[FILE_HASH]
        # if a real-time hash has been computed
        if file_digest is not None:
            # if real-time hash mismatched, try to compute it again
            if file_digest != expected_hash:
                print(
                    'Mismatched real-time digest found, falling back to lump-sum hash computation'
                )
                file_integrity_validation(temp_file, expected_hash)
        else:
            file_integrity_validation(temp_file, expected_hash)
    # put file into to cache
    return cache.put_file(file_meta, temp_file)
