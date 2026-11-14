"""File download — delegates to modelscope_hub for Hub downloads, retains http_get_file for direct HTTP.

Hub file downloads (model_file_download, dataset_file_download) are delegated
to modelscope_hub.compat. Direct HTTP file downloads (http_get_file,
http_get_model_file) are retained as they serve non-Hub use cases.
"""
import copy
import hashlib
import io
import os
import re
import tempfile
import time
import urllib
import uuid
from functools import partial
from http.cookiejar import CookieJar
from typing import Dict, List, Optional, Type

import requests
# --- Hub file downloads (delegated) ---
from modelscope_hub.compat.file_download import \
    dataset_file_download as _compat_dataset_file_download
from modelscope_hub.compat.file_download import \
    model_file_download as _compat_model_file_download
from requests.adapters import Retry
from tqdm.auto import tqdm

from modelscope.hub.constants import (API_FILE_DOWNLOAD_CHUNK_SIZE,
                                      API_FILE_DOWNLOAD_RETRY_TIMES,
                                      API_FILE_DOWNLOAD_TIMEOUT,
                                      MODELSCOPE_SDK_DEBUG)
from modelscope.utils.logger import get_logger
from .callback import ProgressCallback, TqdmCallback
from .errors import FileDownloadError
from .utils.utils import find_reusable_legacy_repo_dir, get_endpoint

logger = get_logger()


def _get_release_timestamp():
    """Compute the release timestamp for revision resolution.

    Returns None (dev-mode) when MODELSCOPE_SDK_DEBUG is set.
    """
    if os.environ.get(MODELSCOPE_SDK_DEBUG):
        return None
    try:
        from modelscope import version
        dt = getattr(version, '__release_datetime__', None)
        if not dt:
            return None
        return int(time.mktime(time.strptime(dt, '%Y-%m-%d %H:%M:%S')))
    except Exception:
        return None


def model_file_download(
    model_id: str,
    file_path: str,
    revision: str = None,
    *,
    cache_dir: str = None,
    local_dir: str = None,
    cookies: dict = None,
    token: str = None,
    endpoint: str = None,
    local_files_only: bool = False,
    user_agent=None,
) -> str:
    """Download a single model file with release-mode revision resolution."""
    if revision is None:
        try:
            from modelscope.hub.api import HubApi
            api = HubApi()
            release_ts = _get_release_timestamp()
            detail = api.get_valid_revision_detail(
                model_id, revision=None, release_timestamp=release_ts)
            revision = detail.get('Revision')
        except Exception:
            pass
    if local_dir is None:
        local_dir = find_reusable_legacy_repo_dir(
            model_id, repo_type='model', cache_dir=cache_dir)
    return _compat_model_file_download(
        model_id,
        file_path,
        revision=revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
        cookies=cookies,
        token=token,
        endpoint=endpoint,
        local_files_only=local_files_only,
        user_agent=user_agent,
    )


def dataset_file_download(
    dataset_id: str,
    file_path: str,
    *,
    cache_dir: str = None,
    local_dir: str = None,
    revision: str = None,
    cookies: dict = None,
    token: str = None,
    endpoint: str = None,
    local_files_only: bool = False,
    user_agent=None,
) -> str:
    """Download a single dataset file, reusing flat/hub legacy caches when present."""
    if local_dir is None:
        local_dir = find_reusable_legacy_repo_dir(
            dataset_id, repo_type='dataset', cache_dir=cache_dir)
    return _compat_dataset_file_download(
        dataset_id,
        file_path,
        cache_dir=cache_dir,
        local_dir=local_dir,
        revision=revision,
        cookies=cookies,
        token=token,
        endpoint=endpoint,
        local_files_only=local_files_only,
        user_agent=user_agent,
    )


# --- Direct HTTP downloads (retained - non-Hub API) ---


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


def _complete_length_from_content_range(content_range: Optional[str],
                                        expected_start: int) -> int:
    """Require a complete suffix with a known representation size."""
    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                         (content_range or '').strip(), re.IGNORECASE)
    if match is not None:
        start, end, complete_length = map(int, match.groups())
        if start == expected_start and start <= end == complete_length - 1:
            return complete_length
    raise FileDownloadError(
        'Cannot verify complete suffix for requested offset %d: %r' %
        (expected_start, content_range))


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
    expected_size = None
    temp_file_manager = partial(
        tempfile.NamedTemporaryFile, mode='wb', dir=local_dir, delete=False)
    get_headers = {} if headers is None else copy.deepcopy(headers)
    get_headers['X-Request-ID'] = str(uuid.uuid4().hex)
    # Resume offsets count decoded bytes on disk. Prefer an unencoded
    # representation, while preserving an explicit caller encoding choice.
    if not any(name.lower() == 'accept-encoding' for name in get_headers):
        get_headers['Accept-Encoding'] = 'identity'
    with temp_file_manager() as temp_file:
        logger.debug('downloading %s to %s', url, temp_file.name)
        # retry sleep 0.5s, 1s, 2s, 4s
        retry = Retry(
            total=API_FILE_DOWNLOAD_RETRY_TIMES,
            backoff_factor=1,
            allowed_methods=['GET'])
        while True:
            r = None
            progress = None
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
                if r.status_code == 200 and downloaded_size:
                    # Range may be ignored: this response starts at byte zero.
                    temp_file.seek(0)
                    temp_file.truncate()
                    downloaded_size = 0
                content_length = r.headers.get('Content-Length')
                content_encoding = r.headers.get('Content-Encoding')
                total = int(
                    content_length) if content_length is not None else None
                # `Content-Length` describes *this* response, which is not
                # always the size of the file we end up writing to disk: it is
                # absent for chunked responses, it counts the encoded bytes
                # when the server applies a `Content-Encoding` such as gzip
                # (requests hands us the decoded stream), and it excludes the
                # bytes already on disk when a resumed request is answered
                # with `206 Partial Content`. Only keep an expected on-disk
                # size when one can actually be established.
                if r.status_code == 206:  # Partial Content
                    # A range is independent of Content-Length (e.g. chunked).
                    # It must cover the complete requested suffix, starting at
                    # exactly the offset already written to the temporary file.
                    expected_size = _complete_length_from_content_range(
                        r.headers.get('Content-Range'), downloaded_size)
                    if (content_encoding or 'identity').lower() != 'identity':
                        raise FileDownloadError(
                            'Cannot verify an encoded partial response')
                elif total is None or (content_encoding
                                       or 'identity').lower() != 'identity':
                    expected_size = None
                else:
                    expected_size = total
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
                break
            except (Exception) as e:  # no matter what happen, we will retry.
                retry = retry.increment('GET', url, error=e)
                retry.sleep()
            finally:
                if progress is not None:
                    progress.close()
                if r is not None:
                    r.close()

    logger.debug('storing %s in cache at %s', url, local_dir)
    downloaded_length = os.path.getsize(temp_file.name)
    # A response that really is truncated already raises while being read,
    # because urllib3 enforces the declared `Content-Length`, so skipping the
    # check when no expected size is available does not lose coverage.
    if expected_size is not None and expected_size != downloaded_length:
        os.remove(temp_file.name)
        msg = 'File %s download incomplete, content_length: %s but the \
                    file downloaded length: %s, please download again' % (
            file_name, expected_size, downloaded_length)
        logger.error(msg)
        raise FileDownloadError(msg)
    os.replace(temp_file.name, os.path.join(local_dir, file_name))


__all__ = [
    'model_file_download',
    'dataset_file_download',
    'http_get_file',
    'http_get_model_file',
    'get_file_download_url',
]
