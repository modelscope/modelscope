# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import ast
import functools
import importlib
import os
import os.path as osp
import sys
from collections import OrderedDict
from importlib import import_module
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import Any

from packaging import version

from modelscope.utils.ast_utils import (INDEX_KEY, MODULE_KEY, REQUIREMENT_KEY,
                                        load_index)
from modelscope.utils.error import *  # noqa
from modelscope.utils.logger import get_logger

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

logger = get_logger()

AST_INDEX = None


def import_modules_from_file(py_file: str):
    """ Import module from a certrain file

    Args:
        py_file: path to a python file to be imported

    Return:

    """
    dirname, basefile = os.path.split(py_file)
    if dirname == '':
        dirname = Path.cwd()
    module_name = osp.splitext(basefile)[0]
    sys.path.insert(0, dirname)
    validate_py_syntax(py_file)
    mod = import_module(module_name)
    sys.path.pop(0)
    return module_name, mod


def is_method_overridden(method, base_class, derived_class):
    """Check if a method of base class is overridden in derived class.

    Args:
        method (str): the method name to check.
        base_class (type): the class of the base class.
        derived_class (type | Any): the class or instance of the derived class.
    """
    assert isinstance(base_class, type), \
        "base_class doesn't accept instance, Please pass class instead."

    if not isinstance(derived_class, type):
        derived_class = derived_class.__class__

    base_method = getattr(base_class, method)
    derived_method = getattr(derived_class, method)
    return derived_method != base_method


def has_method(obj: object, method: str) -> bool:
    """Check whether the object has a method.

    Args:
        method (str): The method name to check.
        obj (object): The object to check.

    Returns:
        bool: True if the object has the method else False.
    """
    return hasattr(obj, method) and callable(getattr(obj, method))


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


def is_wenetruntime_available():
    return importlib.util.find_spec('wenetruntime') is not None


def is_tf_available():
    return _tf_available


def is_opencv_available():
    return importlib.util.find_spec('cv2') is not None


def is_pillow_available():
    return importlib.util.find_spec('PIL.Image') is not None


def _is_package_available_fn(pkg_name):
    return importlib.util.find_spec(pkg_name) is not None


def is_package_available(pkg_name):
    return functools.partial(_is_package_available_fn, pkg_name)


def is_espnet_available(pkg_name):
    return importlib.util.find_spec('espnet2') is not None \
        and importlib.util.find_spec('espnet')


REQUIREMENTS_MAAPING = OrderedDict([
    ('protobuf', (is_protobuf_available, PROTOBUF_IMPORT_ERROR)),
    ('sentencepiece', (is_sentencepiece_available,
                       SENTENCEPIECE_IMPORT_ERROR)),
    ('sklearn', (is_sklearn_available, SKLEARN_IMPORT_ERROR)),
    ('tf', (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
    ('tensorflow', (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
    ('timm', (is_timm_available, TIMM_IMPORT_ERROR)),
    ('tokenizers', (is_tokenizers_available, TOKENIZERS_IMPORT_ERROR)),
    ('torch', (is_torch_available, PYTORCH_IMPORT_ERROR)),
    ('wenetruntime',
     (is_wenetruntime_available,
      WENETRUNTIME_IMPORT_ERROR.replace('TORCH_VER', _torch_version))),
    ('scipy', (is_scipy_available, SCIPY_IMPORT_ERROR)),
    ('cv2', (is_opencv_available, OPENCV_IMPORT_ERROR)),
    ('PIL', (is_pillow_available, PILLOW_IMPORT_ERROR)),
    ('pai-easynlp', (is_package_available('easynlp'), EASYNLP_IMPORT_ERROR)),
    ('espnet2', (is_espnet_available,
                 GENERAL_IMPORT_ERROR.replace('REQ', 'espnet'))),
    ('espnet', (is_espnet_available,
                GENERAL_IMPORT_ERROR.replace('REQ', 'espnet'))),
    ('easyasr', (is_package_available('easyasr'), AUDIO_IMPORT_ERROR)),
    ('funasr', (is_package_available('funasr'), AUDIO_IMPORT_ERROR)),
    ('kwsbp', (is_package_available('kwsbp'), AUDIO_IMPORT_ERROR)),
    ('decord', (is_package_available('decord'), DECORD_IMPORT_ERROR)),
    ('deepspeed', (is_package_available('deepspeed'), DEEPSPEED_IMPORT_ERROR)),
    ('fairseq', (is_package_available('fairseq'), FAIRSEQ_IMPORT_ERROR)),
    ('fasttext', (is_package_available('fasttext'), FASTTEXT_IMPORT_ERROR)),
    ('megatron_util', (is_package_available('megatron_util'),
                       MEGATRON_UTIL_IMPORT_ERROR)),
    ('text2sql_lgesql', (is_package_available('text2sql_lgesql'),
                         TEXT2SQL_LGESQL_IMPORT_ERROR)),
    ('mpi4py', (is_package_available('mpi4py'), MPI4PY_IMPORT_ERROR)),
])

SYSTEM_PACKAGE = set(['os', 'sys', 'typing'])


def requires(obj, requirements):
    if not isinstance(requirements, (list, tuple)):
        requirements = [requirements]
    if isinstance(obj, str):
        name = obj
    else:
        name = obj.__name__ if hasattr(obj,
                                       '__name__') else obj.__class__.__name__
    checks = []
    for req in requirements:
        if req == '' or req in SYSTEM_PACKAGE:
            continue
        if req in REQUIREMENTS_MAAPING:
            check = REQUIREMENTS_MAAPING[req]
        else:
            check_fn = is_package_available(req)
            err_msg = GENERAL_IMPORT_ERROR.replace('REQ', req)
            check = (check_fn, err_msg)
        checks.append(check)

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


class LazyImportModule(ModuleType):
    AST_INDEX = None
    if AST_INDEX is None:
        AST_INDEX = load_index()

    def __init__(self,
                 name,
                 module_file,
                 import_structure,
                 module_spec=None,
                 extra_objects=None,
                 try_to_pre_import=False):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(
            chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure
        if try_to_pre_import:
            self._try_to_import()

    def _try_to_import(self):
        for sub_module in self._class_to_module.keys():
            try:
                getattr(self, sub_module)
            except Exception as e:
                logger.warning(
                    f'pre load module {sub_module} error, please check {e}')

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(
                f'module {self.__name__} has no attribute {name}')

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            # check requirements before module import
            module_name_full = self.__name__ + '.' + module_name
            if module_name_full in LazyImportModule.AST_INDEX[REQUIREMENT_KEY]:
                requirements = LazyImportModule.AST_INDEX[REQUIREMENT_KEY][
                    module_name_full]
                requires(module_name_full, requirements)
            return importlib.import_module('.' + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f'Failed to import {self.__name__}.{module_name} because of the following error '
                f'(look up to see its traceback):\n{e}') from e

    def __reduce__(self):
        return self.__class__, (self._name, self.__file__,
                                self._import_structure)

    @staticmethod
    def import_module(signature):
        """ import a lazy import module using signature

        Args:
            signature (tuple): a tuple of str, (registry_name, registry_group_name, module_name)
        """
        if signature in LazyImportModule.AST_INDEX[INDEX_KEY]:
            mod_index = LazyImportModule.AST_INDEX[INDEX_KEY][signature]
            module_name = mod_index[MODULE_KEY]
            if module_name in LazyImportModule.AST_INDEX[REQUIREMENT_KEY]:
                requirements = LazyImportModule.AST_INDEX[REQUIREMENT_KEY][
                    module_name]
                requires(module_name, requirements)
            importlib.import_module(module_name)
        else:
            logger.warning(f'{signature} not found in ast index file')
