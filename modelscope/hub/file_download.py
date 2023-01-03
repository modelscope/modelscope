# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import os
import tempfile
from functools import partial
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Dict, Optional, Union

import requests
from requests.adapters import Retry
from tqdm import tqdm

from modelscope import __version__
from modelscope.hub.api import HubApi, ModelScopeConfig
from modelscope.hub.constants import (API_FILE_DOWNLOAD_CHUNK_SIZE,
                                      API_FILE_DOWNLOAD_RETRY_TIMES,
                                      API_FILE_DOWNLOAD_TIMEOUT, FILE_HASH)
from modelscope.utils.constant import DEFAULT_MODEL_REVISION
from modelscope.utils.logger import get_logger
from .errors import FileDownloadError, NotExistError
from .utils.caching import ModelFileSystemCache
from .utils.utils import (file_integrity_validation, get_cache_dir,
                          get_endpoint, model_id_to_group_owner_name)

logger = get_logger()


def model_file_download(
    model_id: str,
    file_path: str,
    revision: Optional[str] = DEFAULT_MODEL_REVISION,
    cache_dir: Optional[str] = None,
    user_agent: Union[Dict, str, None] = None,
    local_files_only: Optional[bool] = False,
    cookies: Optional[CookieJar] = None,
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
        local_files_only (bool, optional）:  If `True`, avoid downloading the file and return the path to the
            local cached file if it exists. if `False`, download the file anyway even it exists.
        cookies (CookieJar, optional): The cookie of download request.

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
    if cache_dir is None:
        cache_dir = get_cache_dir()
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    temporary_cache_dir = os.path.join(cache_dir, 'temp')
    os.makedirs(temporary_cache_dir, exist_ok=True)

    group_or_owner, name = model_id_to_group_owner_name(model_id)

    cache = ModelFileSystemCache(cache_dir, group_or_owner, name)

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
                ' traffic has been disabled. To enable model look-ups and downloads'
                " online, set 'local_files_only' to False.")

    _api = HubApi()
    headers = {
        'user-agent': ModelScopeConfig.get_user_agent(user_agent=user_agent, )
    }
    if cookies is None:
        cookies = ModelScopeConfig.get_cookies()

    revision = _api.get_valid_revision(
        model_id, revision=revision, cookies=cookies)
    file_to_download_info = None
    # we need to confirm the version is up-to-date
    # we need to get the file list to check if the latest version is cached, if so return, otherwise download
    model_files = _api.get_model_files(
        model_id=model_id,
        revision=revision,
        recursive=True,
        use_cookies=False if cookies is None else cookies)

    for model_file in model_files:
        if model_file['Type'] == 'tree':
            continue

        if model_file['Path'] == file_path:
            if cache.exists(model_file):
                logger.info(
                    f'File {model_file["Name"]} already in cache, skip downloading!'
                )
                return cache.get_file_by_info(model_file)
            else:
                file_to_download_info = model_file
            break

    if file_to_download_info is None:
        raise NotExistError('The file path: %s not exist in: %s' %
                            (file_path, model_id))

    # we need to download again
    url_to_download = get_file_download_url(model_id, file_path, revision)
    file_to_download_info = {
        'Path': file_path,
        'Revision': file_to_download_info['Revision'],
        FILE_HASH: file_to_download_info[FILE_HASH]
    }

    temp_file_name = next(tempfile._get_candidate_names())
    http_get_file(
        url_to_download,
        temporary_cache_dir,
        temp_file_name,
        headers=headers,
        cookies=None if cookies is None else cookies.get_dict())
    temp_file_path = os.path.join(temporary_cache_dir, temp_file_name)
    # for download with commit we can't get Sha256
    if file_to_download_info[FILE_HASH] is not None:
        file_integrity_validation(temp_file_path,
                                  file_to_download_info[FILE_HASH])
    return cache.put_file(file_to_download_info,
                          os.path.join(temporary_cache_dir, temp_file_name))


def get_file_download_url(model_id: str, file_path: str, revision: str):
    """Format file download url according to `model_id`, `revision` and `file_path`.
    e.g., Given `model_id=john/bert`, `revision=master`, `file_path=README.md`,
    the resulted download url is: https://modelscope.co/api/v1/models/john/bert/repo?Revision=master&FilePath=README.md

    Args:
        model_id (str): The model_id.
        file_path (str): File path
        revision (str): File revision.

    Returns:
        str: The file url.
    """
    download_url_template = '{endpoint}/api/v1/models/{model_id}/repo?Revision={revision}&FilePath={file_path}'
    return download_url_template.format(
        endpoint=get_endpoint(),
        model_id=model_id,
        revision=revision,
        file_path=file_path,
    )


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
        FileDownloadError: Failed download failed.

    """
    total = -1
    temp_file_manager = partial(
        tempfile.NamedTemporaryFile, mode='wb', dir=local_dir, delete=False)
    get_headers = {} if headers is None else copy.deepcopy(headers)
    with temp_file_manager() as temp_file:
        logger.info('downloading %s to %s', url, temp_file.name)
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
                    desc='Downloading',
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

    logger.info('storing %s in cache at %s', url, local_dir)
    downloaded_length = os.path.getsize(temp_file.name)
    if total != downloaded_length:
        os.remove(temp_file.name)
        msg = 'File %s download incomplete, content_length: %s but the \
                    file downloaded length: %s, please download again' % (
            file_name, total, downloaded_length)
        logger.error(msg)
        raise FileDownloadError(msg)
    os.replace(temp_file.name, os.path.join(local_dir, file_name))
