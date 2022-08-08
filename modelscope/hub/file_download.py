import copy
import os
import sys
import tempfile
from functools import partial
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Dict, Optional, Union
from uuid import uuid4

import requests
from filelock import FileLock
from tqdm import tqdm

from modelscope import __version__
from modelscope.utils.constant import DEFAULT_MODEL_REVISION
from modelscope.utils.logger import get_logger
from .api import HubApi, ModelScopeConfig
from .errors import NotExistError
from .utils.caching import ModelFileSystemCache
from .utils.utils import (get_cache_dir, get_endpoint,
                          model_id_to_group_owner_name)

SESSION_ID = uuid4().hex
logger = get_logger()


def model_file_download(
    model_id: str,
    file_path: str,
    revision: Optional[str] = DEFAULT_MODEL_REVISION,
    cache_dir: Optional[str] = None,
    user_agent: Union[Dict, str, None] = None,
    local_files_only: Optional[bool] = False,
) -> Optional[str]:  # pragma: no cover
    """
    Download from a given URL and cache it if it's not already present in the
    local cache.

    Given a URL, this function looks for the corresponding file in the local
    cache. If it's not there, download it. Then return the path to the cached
    file.

    Args:
        model_id (`str`):
            The model to whom the file to be downloaded belongs.
        file_path(`str`):
            Path of the file to be downloaded, relative to the root of model repo
        revision(`str`, *optional*):
            revision of the model file to be downloaded.
            Can be any of a branch, tag or commit hash
        cache_dir (`str`, `Path`, *optional*):
            Path to the folder where cached files are stored.
        user_agent (`dict`, `str`, *optional*):
            The user-agent info in the form of a dictionary or a string.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, avoid downloading the file and return the path to the
            local cached file if it exists.
            if `False`, download the file anyway even it exists

    Returns:
        Local path (string) of file or if networking is off, last version of
        file cached on disk.

    <Tip>

    Raises the following errors:

        - [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
          if `use_auth_token=True` and the token cannot be found.
        - [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError)
          if ETag cannot be determined.
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
          if some parameter value is invalid

    </Tip>
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
    headers = {'user-agent': http_user_agent(user_agent=user_agent, )}
    cookies = ModelScopeConfig.get_cookies()
    branches, tags = _api.get_model_branches_and_tags(
        model_id, use_cookies=False if cookies is None else cookies)
    file_to_download_info = None
    is_commit_id = False
    if revision in branches or revision in tags:  # The revision is version or tag,
        # we need to confirm the version is up to date
        # we need to get the file list to check if the lateast version is cached, if so return, otherwise download
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
                    return cache.get_file_by_info(model_file)
                else:
                    file_to_download_info = model_file
                break

        if file_to_download_info is None:
            raise NotExistError('The file path: %s not exist in: %s' %
                                (file_path, model_id))
    else:  # the revision is commit id.
        cached_file_path = cache.get_file_by_path_and_commit_id(
            file_path, revision)
        if cached_file_path is not None:
            file_name = os.path.basename(cached_file_path)
            logger.info(
                f'File {file_name} already in cache, skip downloading!')
            return cached_file_path  # the file is in cache.
        is_commit_id = True
    # we need to download again
    url_to_download = get_file_download_url(model_id, file_path, revision)
    file_to_download_info = {
        'Path': file_path,
        'Revision':
        revision if is_commit_id else file_to_download_info['Revision']
    }
    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache.get_root_location() + '.lock'

    with FileLock(lock_path):
        temp_file_name = next(tempfile._get_candidate_names())
        http_get_file(
            url_to_download,
            temporary_cache_dir,
            temp_file_name,
            headers=headers,
            cookies=None if cookies is None else cookies.get_dict())
        return cache.put_file(
            file_to_download_info,
            os.path.join(temporary_cache_dir, temp_file_name))


def http_user_agent(user_agent: Union[Dict, str, None] = None, ) -> str:
    """Formats a user-agent string with basic info about a request.

    Args:
        user_agent (`str`, `dict`, *optional*):
            The user agent info in the form of a dictionary or a single string.

    Returns:
        The formatted user-agent string.
    """
    ua = f'modelscope/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}'

    if isinstance(user_agent, dict):
        ua = '; '.join(f'{k}/{v}' for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua = user_agent
    return ua


def get_file_download_url(model_id: str, file_path: str, revision: str):
    """
    Format file download url according to `model_id`, `revision` and `file_path`.
    e.g., Given `model_id=john/bert`, `revision=master`, `file_path=README.md`,
    the resulted download url is: https://modelscope.co/api/v1/models/john/bert/repo?Revision=master&FilePath=README.md
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
    """
    Download remote file. Do not gobble up errors.
    This method is only used by snapshot_download, since the behavior is quite different with single file download
    TODO: consolidate with http_get_file() to avoild duplicate code

    Args:
        url(`str`):
            actual download url of the file
        local_dir(`str`):
            local directory where the downloaded file stores
        file_name(`str`):
            name of the file stored in `local_dir`
        cookies(`CookieJar`):
            cookies used to authentication the user, which is used for downloading private repos
        headers(`Optional[Dict[str, str]] = None`):
            http headers to carry necessary info when requesting the remote file

    """
    temp_file_manager = partial(
        tempfile.NamedTemporaryFile, mode='wb', dir=local_dir, delete=False)

    with temp_file_manager() as temp_file:
        logger.info('downloading %s to %s', url, temp_file.name)
        headers = copy.deepcopy(headers)

        r = requests.get(url, stream=True, headers=headers, cookies=cookies)
        r.raise_for_status()

        content_length = r.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None

        progress = tqdm(
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            total=total,
            initial=0,
            desc='Downloading',
        )
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                temp_file.write(chunk)
        progress.close()

    logger.info('storing %s in cache at %s', url, local_dir)
    os.replace(temp_file.name, os.path.join(local_dir, file_name))
