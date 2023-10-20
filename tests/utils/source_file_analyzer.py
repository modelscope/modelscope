# Copyright (c) Alibaba, Inc. and its affiliates.

from __future__ import print_function
import ast
import importlib.util
import os
import pkgutil
import site
import sys

import json

from modelscope.utils.logger import get_logger

logger = get_logger()


class AnalysisSourceFileDefines(ast.NodeVisitor):
    """Analysis source file function, class, global variable defines.
    """

    def __init__(self, source_file_path) -> None:
        super().__init__()
        self.global_variables = []
        self.functions = []
        self.classes = []
        self.async_functions = []
        self.symbols = []

        self.source_file_path = source_file_path
        rel_file_path = source_file_path
        if os.path.isabs(source_file_path):
            rel_file_path = os.path.relpath(source_file_path, os.getcwd())

        if rel_file_path.endswith('__init__.py'):  # processing package
            self.base_module_name = os.path.dirname(rel_file_path).replace(
                '/', '.')
        else:  # import x.y.z  z is the filename
            self.base_module_name = rel_file_path.replace('/', '.').replace(
                '.py', '')
        self.symbols.append(self.base_module_name)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.symbols.append(self.base_module_name + '.' + node.name)
        self.classes.append(node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.symbols.append(self.base_module_name + '.' + node.name)
        self.functions.append(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.symbols.append(self.base_module_name + '.' + node.name)
        self.async_functions.append(node.name)

    def visit_Assign(self, node: ast.Assign):
        for tg in node.targets:
            if isinstance(tg, ast.Name):
                self.symbols.append(self.base_module_name + '.' + tg.id)
                self.global_variables.append(tg.id)


def is_relative_import(path):
    # from .x import y or from ..x import y
    return path.startswith('.')


def convert_to_path(name):
    if name.startswith('.'):
        remainder = name.lstrip('.')
        dot_count = (len(name) - len(remainder))
        prefix = '../' * (dot_count - 1)
    else:
        remainder = name
        dot_count = 0
        prefix = ''
    filename = prefix + os.path.join(*remainder.split('.'))
    return filename


def resolve_relative_import(source_file_path, module_name, all_symbols):
    current_package = os.path.dirname(source_file_path).replace('/', '.')
    absolute_name = importlib.util.resolve_name(module_name,
                                                current_package)  # get
    return resolve_absolute_import(absolute_name, all_symbols)


def resolve_absolute_import(module_name, all_symbols):
    # direct imports
    if module_name in all_symbols:
        return all_symbols[module_name]

    # some symble import by package __init__.py, we need find the real file which define the symbel.
    parent, sub = module_name.rsplit('.', 1)

    # case module_name is a python Definition
    for symbol, symbol_path in all_symbols.items():
        if symbol.startswith(parent) and symbol.endswith(sub):
            return all_symbols[symbol]

    return None


class IndirectDefines(ast.NodeVisitor):
    """Analysis source file function, class, global variable defines.
    """

    def __init__(self, source_file_path, all_symbols,
                 file_symbols_map) -> None:
        super().__init__()
        self.symbols_map = {
        }  # key symbol name in current file, value the real file path.
        self.all_symbols = all_symbols
        self.file_symbols_map = file_symbols_map
        self.source_file_path = source_file_path

        rel_file_path = source_file_path
        if os.path.isabs(source_file_path):
            rel_file_path = os.path.relpath(source_file_path, os.getcwd())

        if rel_file_path.endswith('__init__.py'):  # processing package
            self.base_module_name = os.path.dirname(rel_file_path).replace(
                '/', '.')
        else:  # import x.y.z  z is the filename
            self.base_module_name = rel_file_path.replace('/', '.').replace(
                '.py', '')

    # import from will get the symbol in current file.
    # from a import b, will get b in current file.
    def visit_ImportFrom(self, node):
        # level 0 absolute import such as from os.path import join
        # level 1 from .x import y
        # level 2 from ..x import y
        module_name = '.' * node.level + (node.module or '')
        for alias in node.names:
            file_path = None
            if alias.name == '*':  # from x import *
                if is_relative_import(module_name):
                    # resolve model path.
                    file_path = resolve_relative_import(
                        self.source_file_path, module_name, self.all_symbols)
                elif module_name.startswith('modelscope'):
                    file_path = resolve_absolute_import(
                        module_name, self.all_symbols)
                else:
                    file_path = None  # ignore other package.
                if file_path is not None:
                    for symbol in self.file_symbols_map[file_path][1:]:
                        symbol_name = symbol.split('.')[-1]
                        self.symbols_map[self.base_module_name
                                         + symbol_name] = file_path
            else:
                if not module_name.endswith('.'):
                    module_name = module_name + '.'
                name = module_name + alias.name
                if alias.asname is not None:
                    current_module_name = self.base_module_name + '.' + alias.asname
                else:
                    current_module_name = self.base_module_name + '.' + alias.name
                if is_relative_import(name):
                    # resolve model path.
                    file_path = resolve_relative_import(
                        self.source_file_path, name, self.all_symbols)
                elif name.startswith('modelscope'):
                    file_path = resolve_absolute_import(name, self.all_symbols)
                if file_path is not None:
                    self.symbols_map[current_module_name] = file_path


class AnalysisSourceFileImports(ast.NodeVisitor):
    """Analysis source file imports
        List imports of the modelscope.
    """

    def __init__(self, source_file_path, all_symbols) -> None:
        super().__init__()
        self.imports = []
        self.source_file_path = source_file_path
        self.all_symbols = all_symbols

    def visit_Import(self, node):
        """Processing import x,y,z or import os.path as osp"""
        for alias in node.names:
            if alias.name.startswith('modelscope'):
                file_path = resolve_absolute_import(alias.name,
                                                    self.all_symbols)
                self.imports.append(os.path.relpath(file_path, os.getcwd()))

    def visit_ImportFrom(self, node):
        # level 0 absolute import such as from os.path import join
        # level 1 from .x import y
        # level 2 from ..x import y
        module_name = '.' * node.level + (node.module or '')
        for alias in node.names:
            if alias.name == '*':  # from x import *
                if is_relative_import(module_name):
                    # resolve model path.
                    file_path = resolve_relative_import(
                        self.source_file_path, module_name, self.all_symbols)
                elif module_name.startswith('modelscope'):
                    file_path = resolve_absolute_import(
                        module_name, self.all_symbols)
                else:
                    file_path = None  # ignore other package.
            else:
                if not module_name.endswith('.'):
                    module_name = module_name + '.'
                name = module_name + alias.name
                if is_relative_import(name):
                    # resolve model path.
                    file_path = resolve_relative_import(
                        self.source_file_path, name, self.all_symbols)
                    if file_path is None:
                        logger.warning(
                            'File: %s, import %s%s not exist!' %
                            (self.source_file_path, module_name, alias.name))
                elif name.startswith('modelscope'):
                    file_path = resolve_absolute_import(name, self.all_symbols)
                    if file_path is None:
                        logger.warning(
                            'File: %s, import %s%s not exist!' %
                            (self.source_file_path, module_name, alias.name))
                else:
                    file_path = None  # ignore other package.

            if file_path is not None:
                if file_path.startswith(site.getsitepackages()[0]):
                    self.imports.append(
                        os.path.relpath(file_path,
                                        site.getsitepackages()[0]))
                else:
                    self.imports.append(
                        os.path.relpath(file_path, os.getcwd()))
            elif module_name.startswith('modelscope'):
                logger.warning(
                    'File: %s, import %s%s not exist!' %
                    (self.source_file_path, module_name, alias.name))


class AnalysisSourceFileRegisterModules(ast.NodeVisitor):
    """Get register_module call of the python source file.


    Args:
        ast (NodeVisitor): The ast node.

    Examples:
        >>> with open(source_file_path, "rb") as f:
        >>>     src = f.read()
        >>> analyzer = AnalysisSourceFileRegisterModules(source_file_path)
        >>> analyzer.visit(ast.parse(src, filename=source_file_path))
    """

    def __init__(self, source_file_path) -> None:
        super().__init__()
        self.source_file_path = source_file_path
        self.register_modules = []

    def visit_ClassDef(self, node: ast.ClassDef):
        if len(node.decorator_list) > 0:
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call):
                    target_name = ''
                    module_name_param = ''
                    task_param = ''
                    if isinstance(dec.func, ast.Attribute
                                  ) and dec.func.attr == 'register_module':
                        target_name = dec.func.value.id  # MODELS
                        if len(dec.args) > 0:
                            if isinstance(dec.args[0], ast.Attribute):
                                task_param = dec.args[0].attr
                            elif isinstance(dec.args[0], ast.Constant):
                                task_param = dec.args[0].value
                        if len(dec.keywords) > 0:
                            for kw in dec.keywords:
                                if kw.arg == 'module_name':
                                    if isinstance(kw.value, ast.Str):
                                        module_name_param = kw.value.s
                                    else:
                                        module_name_param = kw.value.attr
                                elif kw.arg == 'group_key':
                                    if isinstance(kw.value, ast.Str):
                                        task_param = kw.value.s
                                    elif isinstance(kw.value, ast.Name):
                                        task_param = kw.value.id
                                    else:
                                        task_param = kw.value.attr
                        if task_param == '' and module_name_param == '':
                            logger.warn(
                                'File %s %s.register_module has no parameters'
                                % (self.source_file_path, target_name))
                            continue
                        if target_name == 'PIPELINES' and task_param == '':
                            logger.warn(
                                'File %s %s.register_module has no task_param'
                                % (self.source_file_path, target_name))
                        self.register_modules.append(
                            (target_name, task_param, module_name_param,
                             node.name))  # PIPELINES, task, module, class_name


def get_imported_files(file_path, all_symbols):
    """Get file dependencies.
    """
    if os.path.isabs(file_path):
        file_path = os.path.relpath(file_path, os.getcwd())
    with open(file_path, 'rb') as f:
        src = f.read()
    analyzer = AnalysisSourceFileImports(file_path, all_symbols)
    analyzer.visit(ast.parse(src, filename=file_path))
    return list(set(analyzer.imports))


def path_to_module_name(file_path):
    if os.path.isabs(file_path):
        file_path = os.path.relpath(file_path, os.getcwd())
    module_name = os.path.dirname(file_path).replace('/', '.')
    return module_name


def get_file_register_modules(file_path):
    with open(file_path, 'rb') as f:
        src = f.read()
    analyzer = AnalysisSourceFileRegisterModules(file_path)
    analyzer.visit(ast.parse(src, filename=file_path))
    return analyzer.register_modules


def get_file_defined_symbols(file_path):
    if os.path.isabs(file_path):
        file_path = os.path.relpath(file_path, os.getcwd())
    with open(file_path, 'rb') as f:
        src = f.read()
    analyzer = AnalysisSourceFileDefines(file_path)
    analyzer.visit(ast.parse(src, filename=file_path))
    return analyzer.symbols


def get_indirect_symbols(file_path, symbols, file_symbols_map):
    if os.path.isabs(file_path):
        file_path = os.path.relpath(file_path, os.getcwd())
    with open(file_path, 'rb') as f:
        src = f.read()
    analyzer = IndirectDefines(file_path, symbols, file_symbols_map)
    analyzer.visit(ast.parse(src, filename=file_path))
    return analyzer.symbols_map


def get_import_map():
    all_files = [
        os.path.join(dp, f) for dp, dn, filenames in os.walk(
            os.path.join(os.getcwd(), 'modelscope')) for f in filenames
        if os.path.splitext(f)[1] == '.py'
    ]
    all_symbols = {}
    file_symbols_map = {}
    for f in all_files:
        file_path = os.path.relpath(f, os.getcwd())
        file_symbols_map[file_path] = get_file_defined_symbols(f)
        for s in file_symbols_map[file_path]:
            all_symbols[s] = file_path

    # get indirect(imported) symbols, refer to origin define.
    for f in all_files:
        for name, real_path in get_indirect_symbols(f, all_symbols,
                                                    file_symbols_map).items():
            all_symbols[name] = os.path.relpath(real_path, os.getcwd())

    with open('symbols.json', 'w') as f:
        json.dump(all_symbols, f)
    import_map = {}
    for f in all_files:
        files = get_imported_files(f, all_symbols)
        import_map[os.path.relpath(f, os.getcwd())] = files

    return import_map


def get_reverse_import_map():
    all_files = [
        os.path.join(dp, f) for dp, dn, filenames in os.walk(
            os.path.join(os.getcwd(), 'modelscope')) for f in filenames
        if os.path.splitext(f)[1] == '.py'
    ]
    import_map = get_import_map()

    reverse_depend_map = {}
    for f in all_files:
        depend_by = []
        for k, v in import_map.items():
            if f in v and f != k:
                depend_by.append(k)
        reverse_depend_map[f] = depend_by

    return reverse_depend_map, import_map


def get_all_register_modules():
    all_files = [
        os.path.join(dp, f) for dp, dn, filenames in os.walk(
            os.path.join(os.getcwd(), 'modelscope')) for f in filenames
        if os.path.splitext(f)[1] == '.py'
    ]
    all_register_modules = []
    for f in all_files:
        all_register_modules.extend(get_file_register_modules(f))
    return all_register_modules


if __name__ == '__main__':
    pass
