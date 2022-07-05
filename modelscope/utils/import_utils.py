# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import ast
import functools
import importlib.util
import os
import os.path as osp
import sys
import types
from collections import OrderedDict
from functools import wraps
from importlib import import_module
from itertools import chain
from types import ModuleType
from typing import Any

import json
from packaging import version

from modelscope.utils.constant import Fields
from modelscope.utils.error import (PROTOBUF_IMPORT_ERROR,
                                    PYTORCH_IMPORT_ERROR, SCIPY_IMPORT_ERROR,
                                    SENTENCEPIECE_IMPORT_ERROR,
                                    SKLEARN_IMPORT_ERROR,
                                    TENSORFLOW_IMPORT_ERROR, TIMM_IMPORT_ERROR,
                                    TOKENIZERS_IMPORT_ERROR)
from modelscope.utils.logger import get_logger

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

logger = get_logger()


def import_modules_from_file(py_file: str):
    """ Import module from a certrain file

    Args:
        py_file: path to a python file to be imported

    Return:

    """
    dirname, basefile = os.path.split(py_file)
    if dirname == '':
        dirname == './'
    module_name = osp.splitext(basefile)[0]
    sys.path.insert(0, dirname)
    validate_py_syntax(py_file)
    mod = import_module(module_name)
    sys.path.pop(0)
    return module_name, mod


def import_modules(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                logger.warning(f'{imp} failed to import and is ignored.')
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def validate_py_syntax(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        # Setting encoding explicitly to resolve coding issue on windows
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError('There are syntax errors in config '
                          f'file {filename}: {e}')


# following code borrows implementation from huggingface/transformers
ENV_VARS_TRUE_VALUES = {'1', 'ON', 'YES', 'TRUE'}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({'AUTO'})
USE_TF = os.environ.get('USE_TF', 'AUTO').upper()
USE_TORCH = os.environ.get('USE_TORCH', 'AUTO').upper()

_torch_version = 'N/A'
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = importlib.util.find_spec('torch') is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version('torch')
            logger.info(f'PyTorch version {_torch_version} Found.')
        except importlib_metadata.PackageNotFoundError:
            _torch_available = False
else:
    logger.info('Disabling PyTorch because USE_TF is set')
    _torch_available = False

_timm_available = importlib.util.find_spec('timm') is not None
try:
    _timm_version = importlib_metadata.version('timm')
    logger.debug(f'Successfully imported timm version {_timm_version}')
except importlib_metadata.PackageNotFoundError:
    _timm_available = False

_tf_version = 'N/A'
if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
    _tf_available = importlib.util.find_spec('tensorflow') is not None
    if _tf_available:
        candidates = (
            'tensorflow',
            'tensorflow-cpu',
            'tensorflow-gpu',
            'tf-nightly',
            'tf-nightly-cpu',
            'tf-nightly-gpu',
            'intel-tensorflow',
            'intel-tensorflow-avx512',
            'tensorflow-rocm',
            'tensorflow-macos',
        )
        _tf_version = None
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for pkg in candidates:
            try:
                _tf_version = importlib_metadata.version(pkg)
                break
            except importlib_metadata.PackageNotFoundError:
                pass
        _tf_available = _tf_version is not None
    if _tf_available:
        if version.parse(_tf_version) < version.parse('2'):
            pass
        else:
            logger.info(f'TensorFlow version {_tf_version} Found.')
else:
    logger.info('Disabling Tensorflow because USE_TORCH is set')
    _tf_available = False


def is_scipy_available():
    return importlib.util.find_spec('scipy') is not None


def is_sklearn_available():
    if importlib.util.find_spec('sklearn') is None:
        return False
    return is_scipy_available() and importlib.util.find_spec('sklearn.metrics')


def is_sentencepiece_available():
    return importlib.util.find_spec('sentencepiece') is not None


def is_protobuf_available():
    if importlib.util.find_spec('google') is None:
        return False
    return importlib.util.find_spec('google.protobuf') is not None


def is_tokenizers_available():
    return importlib.util.find_spec('tokenizers') is not None


def is_timm_available():
    return _timm_available


def is_torch_available():
    return _torch_available


def is_torch_cuda_available():
    if is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False


def is_tf_available():
    return _tf_available


REQUIREMENTS_MAAPING = OrderedDict([
    ('protobuf', (is_protobuf_available, PROTOBUF_IMPORT_ERROR)),
    ('sentencepiece', (is_sentencepiece_available,
                       SENTENCEPIECE_IMPORT_ERROR)),
    ('sklearn', (is_sklearn_available, SKLEARN_IMPORT_ERROR)),
    ('tf', (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
    ('timm', (is_timm_available, TIMM_IMPORT_ERROR)),
    ('tokenizers', (is_tokenizers_available, TOKENIZERS_IMPORT_ERROR)),
    ('torch', (is_torch_available, PYTORCH_IMPORT_ERROR)),
    ('scipy', (is_scipy_available, SCIPY_IMPORT_ERROR)),
])


def requires(obj, requirements):
    if not isinstance(requirements, (list, tuple)):
        requirements = [requirements]
    if isinstance(obj, str):
        name = obj
    else:
        name = obj.__name__ if hasattr(obj,
                                       '__name__') else obj.__class__.__name__
    checks = (REQUIREMENTS_MAAPING[req] for req in requirements)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError(''.join(failed))


def torch_required(func):
    # Chose a different decorator name than in tests so it's clear they are not the same.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_torch_available():
            return func(*args, **kwargs)
        else:
            raise ImportError(f'Method `{func.__name__}` requires PyTorch.')

    return wrapper


def tf_required(func):
    # Chose a different decorator name than in tests so it's clear they are not the same.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_tf_available():
            return func(*args, **kwargs)
        else:
            raise ImportError(f'Method `{func.__name__}` requires TF.')

    return wrapper
