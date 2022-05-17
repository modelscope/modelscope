# Copyright (c) Alibaba, Inc. and its affiliates.

import ast
import os
import os.path as osp
import sys
import types
from importlib import import_module

from maas_lib.utils.logger import get_logger

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
