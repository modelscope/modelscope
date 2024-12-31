# Copyright (c) Alibaba, Inc. and its affiliates.

import hashlib
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, List, Optional, Union

import requests

from modelscope.hub.constants import (DEFAULT_MODELSCOPE_DOMAIN,
                                      DEFAULT_MODELSCOPE_GROUP,
                                      MODEL_ID_SEPARATOR, MODELSCOPE_SDK_DEBUG,
                                      MODELSCOPE_URL_SCHEME)
from modelscope.hub.errors import FileIntegrityError
from modelscope.utils.file_utils import get_default_modelscope_cache_dir
from modelscope.utils.logger import get_logger

logger = get_logger()


def model_id_to_group_owner_name(model_id):
    if MODEL_ID_SEPARATOR in model_id:
        group_or_owner = model_id.split(MODEL_ID_SEPARATOR)[0]
        name = model_id.split(MODEL_ID_SEPARATOR)[1]
    else:
        group_or_owner = DEFAULT_MODELSCOPE_GROUP
        name = model_id
    return group_or_owner, name


# during model download, the '.' would be converted to '___' to produce
# actual physical (masked) directory for storage
def get_model_masked_directory(directory, model_id):
    parts = directory.rsplit('/', 2)
    # this is the actual directory the model files are located.
    masked_directory = os.path.join(parts[0], model_id.replace('.', '___'))
    return masked_directory


def convert_readable_size(size_bytes):
    import math
    if size_bytes == 0:
        return '0B'
    size_name = ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f'{s} {size_name[i]}'


def get_folder_size(folder_path):
    total_size = 0
    for path in Path(folder_path).rglob('*'):
        if path.is_file():
            total_size += path.stat().st_size
    return total_size


# return a readable string that describe size of for a given folder (MB, GB etc.)
def get_readable_folder_size(folder_path) -> str:
    return convert_readable_size(get_folder_size(folder_path=folder_path))


def get_cache_dir(model_id: Optional[str] = None):
    """cache dir precedence:
        function parameter > environment > ~/.cache/modelscope/hub
    Args:
        model_id (str, optional): The model id.
    Returns:
        str: the model_id dir if model_id not None, otherwise cache root dir.
    """
    default_cache_dir = Path.home().joinpath('.cache', 'modelscope')
    base_path = os.getenv('MODELSCOPE_CACHE',
                          os.path.join(default_cache_dir, 'hub'))
    return base_path if model_id is None else os.path.join(
        base_path, model_id + '/')


def get_release_datetime():
    if MODELSCOPE_SDK_DEBUG in os.environ:
        rt = int(round(datetime.now().timestamp()))
    else:
        from modelscope import version
        rt = int(
            round(
                datetime.strptime(version.__release_datetime__,
                                  '%Y-%m-%d %H:%M:%S').timestamp()))
    return rt


def get_endpoint():
    modelscope_domain = os.getenv('MODELSCOPE_DOMAIN',
                                  DEFAULT_MODELSCOPE_DOMAIN)
    return MODELSCOPE_URL_SCHEME + modelscope_domain


def compute_hash(file_path):
    BUFFER_SIZE = 1024 * 64  # 64k buffer size
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(BUFFER_SIZE)
            if not data:
                break
            sha256_hash.update(data)
    return sha256_hash.hexdigest()


def file_integrity_validation(file_path, expected_sha256):
    """Validate the file hash is expected, if not, delete the file

    Args:
        file_path (str): The file to validate
        expected_sha256 (str): The expected sha256 hash

    Raises:
        FileIntegrityError: If file_path hash is not expected.

    """
    file_sha256 = compute_hash(file_path)
    if not file_sha256 == expected_sha256:
        os.remove(file_path)
        msg = 'File %s integrity check failed, expected sha256 signature is %s, actual is %s, the download may be incomplete, please try again.' % (  # noqa E501
            file_path, expected_sha256, file_sha256)
        logger.error(msg)
        raise FileIntegrityError(msg)


def add_patterns_to_file(repo,
                         file_name: str,
                         patterns: List[str],
                         commit_message: Optional[str] = None,
                         ignore_push_error=False) -> None:
    if isinstance(patterns, str):
        patterns = [patterns]
    if commit_message is None:
        commit_message = f'Add `{patterns[0]}` patterns to {file_name}'

    # Get current file content
    repo_dir = repo.model_dir
    file_path = os.path.join(repo_dir, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            current_content = f.read()
    else:
        current_content = ''
    # Add the patterns to file
    content = current_content
    for pattern in patterns:
        if pattern not in content:
            if len(content) > 0 and not content.endswith('\n'):
                content += '\n'
            content += f'{pattern}\n'

    # Write the file if it has changed
    if content != current_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            logger.debug(f'Writing {file_name} file. Content: {content}')
            f.write(content)
    try:
        repo.push(commit_message)
    except Exception as e:
        if ignore_push_error:
            pass
        else:
            raise e


def add_patterns_to_gitignore(repo,
                              patterns: List[str],
                              commit_message: Optional[str] = None) -> None:
    add_patterns_to_file(
        repo, '.gitignore', patterns, commit_message, ignore_push_error=True)


def add_patterns_to_gitattributes(
        repo,
        patterns: List[str],
        commit_message: Optional[str] = None) -> None:
    new_patterns = []
    suffix = 'filter=lfs diff=lfs merge=lfs -text'
    for pattern in patterns:
        if suffix not in pattern:
            pattern = f'{pattern} {suffix}'
        new_patterns.append(pattern)
    file_name = '.gitattributes'
    if commit_message is None:
        commit_message = f'Add `{patterns[0]}` patterns to {file_name}'
    add_patterns_to_file(
        repo, file_name, new_patterns, commit_message, ignore_push_error=True)
