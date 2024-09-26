# noqa: isort:skip_file, yapf: disable
# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.

import json
import os
import re
import copy
import shutil
import time
import warnings
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urljoin, urlparse
import requests
from tqdm import tqdm

from datasets import config
from datasets.utils.file_utils import hash_url_to_filename, \
    get_authentication_headers_for_url, fsspec_head, fsspec_get
from filelock import FileLock

from modelscope.utils.config_ds import MS_DATASETS_CACHE
from modelscope.utils.logger import get_logger
from modelscope.hub.api import ModelScopeConfig

from modelscope import __version__

logger = get_logger()


def get_datasets_user_agent_ms(user_agent: Optional[Union[str, dict]] = None) -> str:
    ua = f'datasets/{__version__}'
    ua += f'; python/{config.PY_VERSION}'
    ua += f'; pyarrow/{config.PYARROW_VERSION}'
    if config.TORCH_AVAILABLE:
        ua += f'; torch/{config.TORCH_VERSION}'
    if config.TF_AVAILABLE:
        ua += f'; tensorflow/{config.TF_VERSION}'
    if config.JAX_AVAILABLE:
        ua += f'; jax/{config.JAX_VERSION}'
    # if config.BEAM_AVAILABLE:
    #     ua += f'; apache_beam/{config.BEAM_VERSION}'
    if isinstance(user_agent, dict):
        ua += f"; {'; '.join(f'{k}/{v}' for k, v in user_agent.items())}"
    elif isinstance(user_agent, str):
        ua += '; ' + user_agent
    return ua


def _request_with_retry_ms(
    method: str,
    url: str,
    max_retries: int = 2,
    base_wait_time: float = 0.5,
    max_wait_time: float = 2,
    timeout: float = 10.0,
    **params,
) -> requests.Response:
    """Wrapper around requests to retry in case it fails with a ConnectTimeout, with exponential backoff.

    Note that if the environment variable HF_DATASETS_OFFLINE is set to 1, then a OfflineModeIsEnabled error is raised.

    Args:
        method (str): HTTP method, such as 'GET' or 'HEAD'.
        url (str): The URL of the resource to fetch.
        max_retries (int): Maximum number of retries, defaults to 0 (no retries).
        base_wait_time (float): Duration (in seconds) to wait before retrying the first time. Wait time between
            retries then grows exponentially, capped by max_wait_time.
        max_wait_time (float): Maximum amount of time between two retries, in seconds.
        **params (additional keyword arguments): Params to pass to :obj:`requests.request`.
    """
    tries, success = 0, False
    response = None
    while not success:
        tries += 1
        try:
            response = requests.request(method=method.upper(), url=url, timeout=timeout, **params)
            success = True
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as err:
            if tries > max_retries:
                raise err
            else:
                logger.info(f'{method} request to {url} timed out, retrying... [{tries/max_retries}]')
                sleep_time = min(max_wait_time, base_wait_time * 2 ** (tries - 1))  # Exponential backoff
                time.sleep(sleep_time)
    return response


def http_head_ms(
    url, proxies=None, headers=None, cookies=None, allow_redirects=True, timeout=10.0, max_retries=0
) -> requests.Response:
    headers = copy.deepcopy(headers) or {}
    headers['user-agent'] = get_datasets_user_agent_ms(user_agent=headers.get('user-agent'))
    response = _request_with_retry_ms(
        method='HEAD',
        url=url,
        proxies=proxies,
        headers=headers,
        cookies=cookies,
        allow_redirects=allow_redirects,
        timeout=timeout,
        max_retries=max_retries,
    )
    return response


def http_get_ms(
    url, temp_file, proxies=None, resume_size=0, headers=None, cookies=None, timeout=100.0, max_retries=0, desc=None
) -> Optional[requests.Response]:
    headers = dict(headers) if headers is not None else {}
    headers['user-agent'] = get_datasets_user_agent_ms(user_agent=headers.get('user-agent'))
    if resume_size > 0:
        headers['Range'] = f'bytes={resume_size:d}-'
    response = _request_with_retry_ms(
        method='GET',
        url=url,
        stream=True,
        proxies=proxies,
        headers=headers,
        cookies=cookies,
        max_retries=max_retries,
        timeout=timeout,
    )
    if temp_file is None:
        return response
    if response.status_code == 416:  # Range not satisfiable
        return
    content_length = response.headers.get('Content-Length')
    total = resume_size + int(content_length) if content_length is not None else None

    progress = tqdm(total=total, initial=resume_size, unit_scale=True, unit='B', desc=desc or 'Downloading')
    for chunk in response.iter_content(chunk_size=1024):
        progress.update(len(chunk))
        temp_file.write(chunk)

    progress.close()


def get_from_cache_ms(
    url,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=100,
    resume_download=False,
    user_agent=None,
    local_files_only=False,
    use_etag=True,
    max_retries=0,
    token=None,
    use_auth_token='deprecated',
    ignore_url_params=False,
    storage_options=None,
    download_desc=None,
    disable_tqdm=None,
) -> str:
    """
    Given a URL, look for the corresponding file in the local cache.
    If it's not there, download it. Then return the path to the cached file.

    Return:
        Local path (string)

    Raises:
        FileNotFoundError: in case of non-recoverable file
            (non-existent or no cache on disk)
        ConnectionError: in case of unreachable url
            and no cache on disk
    """
    if use_auth_token != 'deprecated':
        warnings.warn(
            "'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\n"
            f"You can remove this warning by passing 'token={use_auth_token}' instead.",
            FutureWarning,
        )
        token = use_auth_token
    if cache_dir is None:
        cache_dir = MS_DATASETS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    if ignore_url_params:
        # strip all query parameters and #fragments from the URL
        cached_url = urljoin(url, urlparse(url).path)
    else:
        cached_url = url  # additional parameters may be added to the given URL

    connected = False
    response = None
    cookies = None
    etag = None
    head_error = None
    scheme = None

    # Try a first time to file the file on the local file system without eTag (None)
    # if we don't ask for 'force_download' then we spare a request
    filename = hash_url_to_filename(cached_url, etag=None)
    cache_path = os.path.join(cache_dir, filename)
    if download_desc is None:
        download_desc = 'Downloading [' + filename + ']'

    if os.path.exists(cache_path) and not force_download and not use_etag:
        return cache_path

    # Prepare headers for authentication
    headers = get_authentication_headers_for_url(url, token=token)
    if user_agent is not None:
        headers['user-agent'] = user_agent

    # We don't have the file locally or we need an eTag
    if not local_files_only:
        scheme = urlparse(url).scheme
        if scheme not in ('http', 'https'):
            response = fsspec_head(url, storage_options=storage_options)
            # s3fs uses "ETag", gcsfs uses "etag"
            etag = (response.get('ETag', None) or response.get('etag', None)) if use_etag else None
            connected = True
        try:
            cookies = ModelScopeConfig.get_cookies()
            response = http_head_ms(
                url,
                allow_redirects=True,
                proxies=proxies,
                timeout=etag_timeout,
                max_retries=max_retries,
                headers=headers,
                cookies=cookies,
            )
            if response.status_code == 200:  # ok
                etag = response.headers.get('ETag') if use_etag else None
                for k, v in response.cookies.items():
                    # In some edge cases, we need to get a confirmation token
                    if k.startswith('download_warning') and 'drive.google.com' in url:
                        url += '&confirm=' + v
                        cookies = response.cookies
                connected = True
                # Fix Google Drive URL to avoid Virus scan warning
                if 'drive.google.com' in url and 'confirm=' not in url:
                    url += '&confirm=t'
            # In some edge cases, head request returns 400 but the connection is actually ok
            elif (
                (response.status_code == 400 and 'firebasestorage.googleapis.com' in url)
                or (response.status_code == 405 and 'drive.google.com' in url)
                or (
                    response.status_code == 403
                    and (
                        re.match(r'^https?://github.com/.*?/.*?/releases/download/.*?/.*?$', url)
                        or re.match(r'^https://.*?s3.*?amazonaws.com/.*?$', response.url)
                    )
                )
                or (response.status_code == 403 and 'ndownloader.figstatic.com' in url)
            ):
                connected = True
                logger.info(f"Couldn't get ETag version for url {url}")
            elif response.status_code == 401 and config.HF_ENDPOINT in url and token is None:
                raise ConnectionError(
                    f'Unauthorized for URL {url}. '
                    f'Please use the parameter `token=True` after logging in with `huggingface-cli login`'
                )
        except (OSError, requests.exceptions.Timeout) as e:
            # not connected
            head_error = e
            pass

    # connected == False = we don't have a connection, or url doesn't exist, or is otherwise inaccessible.
    # try to get the last downloaded one
    if not connected:
        if os.path.exists(cache_path) and not force_download:
            return cache_path
        if local_files_only:
            raise FileNotFoundError(
                f'Cannot find the requested files in the cached path at {cache_path} and outgoing traffic has been'
                " disabled. To enable file online look-ups, set 'local_files_only' to False."
            )
        elif response is not None and response.status_code == 404:
            raise FileNotFoundError(f"Couldn't find file at {url}")
        if head_error is not None:
            raise ConnectionError(f"Couldn't reach {url} ({repr(head_error)})")
        elif response is not None:
            raise ConnectionError(f"Couldn't reach {url} (error {response.status_code})")
        else:
            raise ConnectionError(f"Couldn't reach {url}")

    # Try a second time
    filename = hash_url_to_filename(cached_url, etag)
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # From now on, connected is True.
    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + '.lock'
    with FileLock(lock_path):
        # Retry in case previously locked processes just enter after the precedent process releases the lock
        if os.path.exists(cache_path) and not force_download:
            return cache_path

        incomplete_path = cache_path + '.incomplete'

        @contextmanager
        def temp_file_manager(mode='w+b'):
            with open(incomplete_path, mode) as f:
                yield f

        resume_size = 0
        if resume_download:
            temp_file_manager = partial(temp_file_manager, mode='a+b')
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size

        # Download to temporary file, then copy to cache path once finished.
        # Otherwise, you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:

            # GET file object
            if scheme not in ('http', 'https'):
                # fsspec_get_sig = inspect.signature(fsspec_get)
                fsspec_get(url, temp_file, storage_options=storage_options, desc=download_desc)
            else:
                # http_get_sig = inspect.signature(http_get_ms)
                http_get_ms(
                    url,
                    temp_file=temp_file,
                    proxies=proxies,
                    resume_size=resume_size,
                    headers=headers,
                    cookies=cookies,
                    max_retries=max_retries,
                    desc=download_desc,
                )

        logger.info(f'storing {url} in cache at {cache_path}')
        shutil.move(temp_file.name, cache_path)
        umask = os.umask(0o666)
        os.umask(umask)
        os.chmod(cache_path, 0o666 & ~umask)

        logger.info(f'creating metadata file for {cache_path}')
        meta = {'url': url, 'etag': etag}
        meta_path = cache_path + '.json'
        with open(meta_path, 'w', encoding='utf-8') as meta_file:
            json.dump(meta, meta_file)

    return cache_path
