# Copyright (c) Alibaba, Inc. and its affiliates.
import hashlib
import inspect
import io
import os
from pathlib import Path
from shutil import Error, copy2, copystat
from typing import BinaryIO, Optional, Union
from urllib.parse import urlparse

from modelscope.utils.logger import get_logger

logger = get_logger()


# TODO: remove this api, unify to flattened args
def func_receive_dict_inputs(func):
    """to decide if a func could receive dict inputs or not

    Args:
        func (class): the target function to be inspected

    Returns:
        bool: if func only has one arg ``input`` or ``inputs``, return True, else return False
    """
    full_args_spec = inspect.getfullargspec(func)
    varargs = full_args_spec.varargs
    varkw = full_args_spec.varkw
    if not (varargs is None and varkw is None):
        return False

    args = [] if not full_args_spec.args else full_args_spec.args
    args.pop(0) if (args and args[0] in ['self', 'cls']) else args

    if len(args) == 1 and args[0] in ['input', 'inputs']:
        return True

    return False


def get_default_modelscope_cache_dir():
    """
    default base dir: '~/.cache/modelscope'
    """
    default_cache_dir = os.path.expanduser(Path.home().joinpath(
        '.cache', 'modelscope', 'hub'))
    return default_cache_dir


def get_modelscope_cache_dir() -> str:
    """Get modelscope cache dir, default location or
       setting with MODELSCOPE_CACHE

    Returns:
        str: the modelscope cache root.
    """
    return os.path.expanduser(
        os.getenv('MODELSCOPE_CACHE', get_default_modelscope_cache_dir()))


def get_model_cache_root() -> str:
    """Get model cache root path.

    Returns:
        str: the modelscope model cache root.
    """
    return os.path.join(get_modelscope_cache_dir(), 'models')


def get_dataset_cache_root() -> str:
    """Get dataset raw file cache root path.
    if `MODELSCOPE_CACHE` is set, return `MODELSCOPE_CACHE/datasets`,
    else return `~/.cache/modelscope/hub/datasets`

    Returns:
        str: the modelscope dataset raw file cache root.
    """
    return os.path.join(get_modelscope_cache_dir(), 'datasets')


def get_dataset_cache_dir(dataset_id: str) -> str:
    """Get the dataset_id's path.
       dataset_cache_root/dataset_id.

    Args:
        dataset_id (str): The dataset id.

    Returns:
        str: The dataset_id's cache root path.
    """
    dataset_root = get_dataset_cache_root()
    return dataset_root if dataset_id is None else os.path.join(
        dataset_root, dataset_id + '/')


def get_model_cache_dir(model_id: str) -> str:
    """cache dir precedence:
        function parameter > environment > ~/.cache/modelscope/hub/model_id

    Args:
        model_id (str, optional): The model id.

    Returns:
        str: the model_id dir if model_id not None, otherwise cache root dir.
    """
    root_path = get_model_cache_root()
    return root_path if model_id is None else os.path.join(
        root_path, model_id + '/')


def read_file(path):

    with open(path, 'r') as f:
        text = f.read()
    return text


def copytree_py37(src,
                  dst,
                  symlinks=False,
                  ignore=None,
                  copy_function=copy2,
                  ignore_dangling_symlinks=False,
                  dirs_exist_ok=False):
    """copy from py37 shutil. add the parameter dirs_exist_ok."""
    names = os.listdir(src)
    if ignore is not None:
        ignored_names = ignore(src, names)
    else:
        ignored_names = set()

    os.makedirs(dst, exist_ok=dirs_exist_ok)
    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if os.path.islink(srcname):
                linkto = os.readlink(srcname)
                if symlinks:
                    # We can't just leave it to `copy_function` because legacy
                    # code with a custom `copy_function` may rely on copytree
                    # doing the right thing.
                    os.symlink(linkto, dstname)
                    copystat(srcname, dstname, follow_symlinks=not symlinks)
                else:
                    # ignore dangling symlink if the flag is on
                    if not os.path.exists(linkto) and ignore_dangling_symlinks:
                        continue
                    # otherwise let the copy occurs. copy2 will raise an error
                    if os.path.isdir(srcname):
                        copytree_py37(
                            srcname,
                            dstname,
                            symlinks,
                            ignore,
                            copy_function,
                            dirs_exist_ok=dirs_exist_ok)
                    else:
                        copy_function(srcname, dstname)
            elif os.path.isdir(srcname):
                copytree_py37(
                    srcname,
                    dstname,
                    symlinks,
                    ignore,
                    copy_function,
                    dirs_exist_ok=dirs_exist_ok)
            else:
                # Will raise a SpecialFileError for unsupported file types
                copy_function(srcname, dstname)
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        except Error as err:
            errors.extend(err.args[0])
        except OSError as why:
            errors.append((srcname, dstname, str(why)))
    try:
        copystat(src, dst)
    except OSError as why:
        # Copying file access times may fail on Windows
        if getattr(why, 'winerror', None) is None:
            errors.append((src, dst, str(why)))
    if errors:
        raise Error(errors)
    return dst


def get_file_size(file_path_or_obj: Union[str, Path, bytes, BinaryIO]) -> int:
    if isinstance(file_path_or_obj, (str, Path)):
        file_path = Path(file_path_or_obj)
        return file_path.stat().st_size
    elif isinstance(file_path_or_obj, bytes):
        return len(file_path_or_obj)
    elif isinstance(file_path_or_obj, io.BufferedIOBase):
        current_position = file_path_or_obj.tell()
        file_path_or_obj.seek(0, os.SEEK_END)
        size = file_path_or_obj.tell()
        file_path_or_obj.seek(current_position)
        return size
    else:
        raise TypeError(
            'Unsupported type: must be string, Path, bytes, or io.BufferedIOBase'
        )


def get_file_hash(
    file_path_or_obj: Union[str, Path, bytes, BinaryIO],
    buffer_size_mb: Optional[int] = 16,
    tqdm_desc: Optional[str] = '[Calculating]',
    disable_tqdm: Optional[bool] = True,
) -> dict:
    """Compute SHA256 hash for a file path, bytes, or file-like object.

    Args:
        file_path_or_obj: File path, bytes, or file-like object.
        buffer_size_mb: Read buffer size in MB. Default 16MB.
        tqdm_desc: Progress bar description.
        disable_tqdm: Whether to disable progress bar.

    Returns:
        dict with keys: file_path_or_obj, file_hash, file_size.
    """
    from tqdm.auto import tqdm

    declared_size = get_file_size(file_path_or_obj)
    if declared_size > 1024 * 1024 * 1024:  # 1GB
        disable_tqdm = False
        name = 'Large File'
        if isinstance(file_path_or_obj, (str, Path)):
            path = file_path_or_obj if isinstance(
                file_path_or_obj, Path) else Path(file_path_or_obj)
            name = path.name
        tqdm_desc = f'[Validating Hash for {name}]'

    buffer_size = buffer_size_mb * 1024 * 1024
    file_hash = hashlib.sha256()

    progress = tqdm(
        total=declared_size,
        initial=0,
        unit_scale=True,
        dynamic_ncols=True,
        unit='B',
        desc=tqdm_desc,
        disable=disable_tqdm,
    )

    if isinstance(file_path_or_obj, (str, Path)):
        bytes_hashed = 0
        with open(file_path_or_obj, 'rb') as f:
            while byte_chunk := f.read(buffer_size):
                file_hash.update(byte_chunk)
                bytes_hashed += len(byte_chunk)
                progress.update(len(byte_chunk))
        file_hash = file_hash.hexdigest()
        if bytes_hashed != declared_size:
            logger.warning(
                f'File size changed during hash computation: '
                f'declared {declared_size} bytes, actually hashed {bytes_hashed} bytes. '
                f'File may have been modified: {file_path_or_obj}')
        file_size = bytes_hashed

    elif isinstance(file_path_or_obj, bytes):
        file_hash.update(file_path_or_obj)
        file_hash = file_hash.hexdigest()
        progress.update(len(file_path_or_obj))
        file_size = len(file_path_or_obj)

    elif isinstance(file_path_or_obj, io.BufferedIOBase):
        bytes_hashed = 0
        file_path_or_obj.seek(0, os.SEEK_SET)
        while byte_chunk := file_path_or_obj.read(buffer_size):
            file_hash.update(byte_chunk)
            bytes_hashed += len(byte_chunk)
            progress.update(len(byte_chunk))
        file_hash = file_hash.hexdigest()
        file_path_or_obj.seek(0, os.SEEK_SET)
        if bytes_hashed != declared_size:
            logger.warning(
                f'File size changed during hash computation: '
                f'declared {declared_size} bytes, actually hashed {bytes_hashed} bytes. '
                f'File may have been modified: {file_path_or_obj}')
        file_size = bytes_hashed

    else:
        progress.close()
        raise ValueError(
            'Input must be str, Path, bytes or a io.BufferedIOBase')

    progress.close()

    return {
        'file_path_or_obj': file_path_or_obj,
        'file_hash': file_hash,
        'file_size': file_size,
    }


def _get_file_hash_async(
    file_path: Union[str, Path],
    buffer_size_mb: int = 16,
    tqdm_desc: Optional[str] = '[Calculating]',
    disable_tqdm: Optional[bool] = True,
) -> dict:
    """Compute SHA256 with async I/O double-buffering for high-latency storage.

    Uses a producer-consumer pattern: a background thread reads file chunks
    into a bounded queue while the main thread computes the hash. This
    overlaps I/O latency with CPU computation.
    """
    import queue
    import threading

    from tqdm.auto import tqdm

    file_path = str(file_path)
    declared_size = os.path.getsize(file_path)
    buffer_size = buffer_size_mb * 1024 * 1024

    chunk_queue = queue.Queue(maxsize=2)  # Bounded to limit memory usage
    read_error = [None]  # Mutable container for thread error propagation

    def _producer():
        try:
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(buffer_size)
                    if not chunk:
                        break
                    chunk_queue.put(chunk)
        except Exception as e:
            read_error[0] = e
        finally:
            chunk_queue.put(None)  # Sentinel to signal EOF

    reader_thread = threading.Thread(target=_producer, daemon=True)
    reader_thread.start()

    file_hash = hashlib.sha256()
    bytes_hashed = 0
    progress = tqdm(
        total=declared_size,
        desc=tqdm_desc,
        disable=disable_tqdm,
        dynamic_ncols=True,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    )

    while True:
        chunk = chunk_queue.get()
        if chunk is None:
            break
        file_hash.update(chunk)
        bytes_hashed += len(chunk)
        progress.update(len(chunk))

    reader_thread.join()
    progress.close()

    if read_error[0] is not None:
        raise read_error[0]

    if bytes_hashed != declared_size:
        logger.warning(
            f'File size changed during hash computation: '
            f'declared {declared_size} bytes, actually hashed {bytes_hashed} bytes. '
            f'File may have been modified: {file_path}')

    return {
        'file_path_or_obj': file_path,
        'file_hash': file_hash.hexdigest(),
        'file_size': bytes_hashed,
    }


def compute_file_hash(
    file_path_or_obj: Union[str, Path, bytes, BinaryIO],
    buffer_size_mb: Optional[int] = 16,
    async_threshold_mb: int = 32,
    tqdm_desc: Optional[str] = '[Calculating]',
    disable_tqdm: Optional[bool] = True,
) -> dict:
    """Compute SHA256 hash with automatic optimization for large files.

    For file paths larger than async_threshold_mb, uses async double-buffered
    I/O to overlap disk reads with hash computation. For smaller files or
    non-path inputs (bytes, BinaryIO), falls back to synchronous hashing.

    Args:
        file_path_or_obj: File path, bytes, or file-like object.
        buffer_size_mb: Read buffer size in MB. Default 16MB for NAS optimization.
        async_threshold_mb: File size threshold for async mode. Default 32MB.
        tqdm_desc: Progress bar description.
        disable_tqdm: Whether to disable progress bar.

    Returns:
        dict with keys: file_path_or_obj, file_hash, file_size.
    """
    # Use async mode for large files accessed by path
    if isinstance(file_path_or_obj, (str, Path)):
        file_size = os.path.getsize(str(file_path_or_obj))
        if file_size >= async_threshold_mb * 1024 * 1024:
            return _get_file_hash_async(
                file_path_or_obj,
                buffer_size_mb=buffer_size_mb,
                tqdm_desc=tqdm_desc,
                disable_tqdm=disable_tqdm,
            )

    # Fallback to synchronous hashing
    return get_file_hash(
        file_path_or_obj,
        buffer_size_mb=buffer_size_mb,
        tqdm_desc=tqdm_desc,
        disable_tqdm=disable_tqdm,
    )


def is_relative_path(url_or_filename: str) -> bool:
    """
    Check if a given string is a relative path.
    """
    return urlparse(
        url_or_filename).scheme == '' and not os.path.isabs(url_or_filename)
