# Copyright (c) Alibaba, Inc. and its affiliates.

import contextlib
import hashlib
import os
import sys
import time
import zoneinfo
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional, Union

from filelock import BaseFileLock, FileLock, SoftFileLock, Timeout

from modelscope.hub.constants import (DEFAULT_MODELSCOPE_DOMAIN,
                                      DEFAULT_MODELSCOPE_GROUP,
                                      DEFAULT_MODELSCOPE_INTL_DOMAIN,
                                      MODEL_ID_SEPARATOR, MODELSCOPE_DOMAIN,
                                      MODELSCOPE_SDK_DEBUG,
                                      MODELSCOPE_URL_SCHEME)
from modelscope.hub.errors import FileIntegrityError
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


def is_env_true(var_name):
    value = os.environ.get(var_name, '').strip().lower()
    return value == 'true'


def get_domain(cn_site=True):
    if MODELSCOPE_DOMAIN in os.environ and os.getenv(MODELSCOPE_DOMAIN):
        return os.getenv(MODELSCOPE_DOMAIN)
    if cn_site:
        return DEFAULT_MODELSCOPE_DOMAIN
    else:
        return DEFAULT_MODELSCOPE_INTL_DOMAIN


def convert_patterns(raw_input: Union[str, List[str]]):
    output = None
    if isinstance(raw_input, str):
        output = list()
        if ',' in raw_input:
            output = [s.strip() for s in raw_input.split(',')]
        else:
            output.append(raw_input.strip())
    elif isinstance(raw_input, list):
        output = list()
        for s in raw_input:
            if isinstance(s, str):
                if ',' in s:
                    output.extend([ss.strip() for ss in s.split(',')])
                else:
                    output.append(s.strip())
    return output


# during model download, the '.' would be converted to '___' to produce
# actual physical (masked) directory for storage
def get_model_masked_directory(directory, model_id):
    if sys.platform.startswith('win'):
        parts = directory.rsplit('\\', 2)
    else:
        parts = directory.rsplit('/', 2)
    # this is the actual directory the model files are located.
    masked_directory = os.path.join(parts[0], model_id.replace('.', '___'))
    return masked_directory


def convert_readable_size(size_bytes: int) -> str:
    import math
    if size_bytes == 0:
        return '0B'
    size_name = ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f'{s} {size_name[i]}'


def get_folder_size(folder_path: str) -> int:
    total_size = 0
    for path in Path(folder_path).rglob('*'):
        if path.is_file():
            total_size += path.stat().st_size
    return total_size


# return a readable string that describe size of for a given folder (MB, GB etc.)
def get_readable_folder_size(folder_path: str) -> str:
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


def get_endpoint(cn_site=True):
    return MODELSCOPE_URL_SCHEME + get_domain(cn_site)


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


def file_integrity_validation(file_path, expected_sha256) -> bool:
    """Validate the file hash is expected, if not, delete the file

    Args:
        file_path (str): The file to validate
        expected_sha256 (str): The expected sha256 hash

    Returns:
        bool: True if the file is valid, False otherwise
    """
    file_sha256 = compute_hash(file_path)
    if not file_sha256 == expected_sha256:
        os.remove(file_path)
        msg = 'File %s integrity check failed, expected sha256 signature is %s, actual is %s, the download may be incomplete, please try again.' % (  # noqa E501
            file_path, expected_sha256, file_sha256)
        logger.error(msg)
        return False

    return True


def add_content_to_file(repo,
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


_TIMESINCE_CHUNKS = (
    # Label, divider, max value
    ('second', 1, 60),
    ('minute', 60, 60),
    ('hour', 60 * 60, 24),
    ('day', 60 * 60 * 24, 6),
    ('week', 60 * 60 * 24 * 7, 6),
    ('month', 60 * 60 * 24 * 30, 11),
    ('year', 60 * 60 * 24 * 365, None),
)


def format_timesince(ts: float) -> str:
    """Format timestamp in seconds into a human-readable string, relative to now.
    """
    delta = time.time() - ts
    if delta < 20:
        return 'a few seconds ago'
    for label, divider, max_value in _TIMESINCE_CHUNKS:  # noqa: B007
        value = round(delta / divider)
        if max_value is not None and value <= max_value:
            break
    return f"{value} {label}{'s' if value > 1 else ''} ago"


def tabulate(rows: List[List[Union[str, int]]], headers: List[str]) -> str:
    """
    Inspired by:

    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    """
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    row_format = ('{{:{}}} ' * len(headers)).format(*col_widths)
    lines = []
    lines.append(row_format.format(*headers))
    lines.append(row_format.format(*['-' * w for w in col_widths]))
    for row in rows:
        lines.append(row_format.format(*row))
    return '\n'.join(lines)


# Part of the code borrowed from the awesome work of huggingface_hub/transformers
def strtobool(val):
    val = val.lower()
    if val in {'y', 'yes', 't', 'true', 'on', '1'}:
        return 1
    if val in {'n', 'no', 'f', 'false', 'off', '0'}:
        return 0
    raise ValueError(f'invalid truth value {val!r}')


@contextlib.contextmanager
def weak_file_lock(lock_file: Union[str, Path],
                   *,
                   timeout: Optional[float] = None
                   ) -> Generator[BaseFileLock, None, None]:
    default_interval = 60
    lock = FileLock(lock_file, timeout=default_interval)
    start_time = time.time()

    while True:
        elapsed_time = time.time() - start_time
        if timeout is not None and elapsed_time >= timeout:
            raise Timeout(str(lock_file))

        try:
            lock.acquire(
                timeout=min(default_interval, timeout - elapsed_time)
                if timeout else default_interval)  # noqa
        except Timeout:
            logger.info(
                f'Still waiting to acquire lock on {lock_file} (elapsed: {time.time() - start_time:.1f} seconds)'
            )
        except NotImplementedError as e:
            if 'use SoftFileLock instead' in str(e):
                logger.warning(
                    'FileSystem does not appear to support flock. Falling back to SoftFileLock for %s',
                    lock_file)
                lock = SoftFileLock(lock_file, timeout=default_interval)
                continue
        else:
            break

    try:
        yield lock
    finally:
        try:
            lock.release()
        except OSError:
            try:
                Path(lock_file).unlink()
            except OSError:
                pass


def convert_timestamp(time_stamp: Union[int, str, datetime],
                      time_zone: str = 'Asia/Shanghai') -> Optional[datetime]:
    """Convert a UNIX/string timestamp to a timezone-aware datetime object.
    Args:
        time_stamp: UNIX timestamp (int), ISO string, or datetime object
        time_zone: Target timezone for non-UTC timestamps (default: 'Asia/Shanghai')
    Returns:
        Timezone-aware datetime object or None if input is None
    """
    if not time_stamp:
        return None

    # Handle datetime objects first
    if isinstance(time_stamp, datetime):
        return time_stamp

    if isinstance(time_stamp, str):
        try:
            if time_stamp.endswith('Z'):
                # Normalize fractional seconds to 6 digits
                if '.' not in time_stamp:
                    # No fractional seconds (e.g., "2024-11-16T00:27:02Z")
                    time_stamp = time_stamp[:-1] + '.000000Z'
                else:
                    # Has fractional seconds (e.g., "2022-08-19T07:19:38.123456789Z")
                    base, fraction = time_stamp[:-1].split('.')
                    # Truncate or pad to 6 digits
                    fraction = fraction[:6].ljust(6, '0')
                    time_stamp = f'{base}.{fraction}Z'

                dt = datetime.strptime(time_stamp,
                                       '%Y-%m-%dT%H:%M:%S.%fZ').replace(
                                           tzinfo=zoneinfo.ZoneInfo('UTC'))
                if time_zone != 'UTC':
                    dt = dt.astimezone(zoneinfo.ZoneInfo(time_zone))
                return dt
            else:
                # Try parsing common ISO formats
                formats = [
                    '%Y-%m-%dT%H:%M:%S.%f',  # With microseconds
                    '%Y-%m-%dT%H:%M:%S',  # Without microseconds
                    '%Y-%m-%d %H:%M:%S.%f',  # Space separator with microseconds
                    '%Y-%m-%d %H:%M:%S',  # Space separator without microseconds
                ]
                for fmt in formats:
                    try:
                        return datetime.strptime(
                            time_stamp,
                            fmt).replace(tzinfo=zoneinfo.ZoneInfo(time_zone))
                    except ValueError:
                        continue

                raise ValueError(
                    f"Unsupported timestamp format: '{time_stamp}'")

        except ValueError as e:
            raise ValueError(
                f"Cannot parse '{time_stamp}' as a datetime. Expected formats: "
                f"'YYYY-MM-DDTHH:MM:SS[.ffffff]Z' (UTC) or 'YYYY-MM-DDTHH:MM:SS[.ffffff]' (local)"
            ) from e

    elif isinstance(time_stamp, int):
        try:
            # UNIX timestamps are always in UTC, then convert to target timezone
            return datetime.fromtimestamp(
                time_stamp, tz=zoneinfo.ZoneInfo('UTC')).astimezone(
                    zoneinfo.ZoneInfo(time_zone))
        except (ValueError, OSError) as e:
            raise ValueError(
                f"Cannot convert '{time_stamp}' to datetime. Ensure it's a valid UNIX timestamp."
            ) from e

    else:
        raise TypeError(
            f"Unsupported type '{type(time_stamp)}'. Expected int, str, or datetime."
        )


def encode_media_to_base64(media_file_path: str) -> str:
    """
    Encode image or video file to base64 string.

    Args:
        media_file_path (str): Path to the image or video file

    Returns:
        str: Base64 encoded string with data URL prefix

    Raises:
        FileNotFoundError: If image/video file doesn't exist
        ValueError: If file is not a valid format
    """
    import base64
    import mimetypes

    # Expand user path
    media_file_path = os.path.expanduser(media_file_path)

    if not os.path.exists(media_file_path):
        raise FileNotFoundError(f'Image file not found: {media_file_path}')

    if not os.path.isfile(media_file_path):
        raise ValueError(f'Path is not a file: {media_file_path}')

    # Get MIME type
    mime_type, _ = mimetypes.guess_type(media_file_path)
    if not mime_type:
        raise ValueError(f'File is not a valid format: {media_file_path}')

    # Read and encode file
    with open(media_file_path, 'rb') as media_file:
        image_data = media_file.read()
        base64_data = base64.b64encode(image_data).decode('utf-8')

    # Return data URL format
    return f'data:{mime_type};base64,{base64_data}'
