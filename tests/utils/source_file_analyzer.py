# Copyright (c) Alibaba, Inc. and its affiliates.

from __future__ import print_function
import ast
import importlib.util
import os
import pkgutil
import site
import sys

from modelscope.utils.logger import get_logger

logger = get_logger()


def is_relative_import(path):
    # from .x import y or from ..x import y
    return path.startswith('.')


def resolve_import(module_name):
    try:
        spec = importlib.util.find_spec(module_name)
        return spec and spec.origin
    except Exception:
        return None


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


def resolve_relative_import(source_file_path, module_name):
    current_package = os.path.dirname(source_file_path).replace('/', '.')
    absolute_name = importlib.util.resolve_name(module_name,
                                                current_package)  # get
    return resolve_absolute_import(absolute_name)


def onerror(name):
    logger.error('Importing module %s error!' % name)


def resolve_absolute_import(module_name):
    module_file_path = resolve_import(module_name)
    if module_file_path is None:
        # find from base module.
        parent_module, sub_module = module_name.rsplit('.', 1)
        if parent_module in sys.modules:
            if hasattr(sys.modules[parent_module], '_import_structure'):
                import_structure = sys.modules[parent_module]._import_structure
                for k, v in import_structure.items():
                    if sub_module in v:
                        parent_module = parent_module + '.' + k
                        break
            module_file_path = resolve_absolute_import(parent_module)
            # the parent_module is a package, we need find the module_name's file
            if os.path.basename(module_file_path) == '__init__.py' and \
                (os.path.relpath(module_file_path, site.getsitepackages()[0]) != 'modelscope/__init__.py'
                 or os.path.relpath(module_file_path, os.getcwd()) != 'modelscope/__init__.py'):
                for _, sub_module_name, _ in pkgutil.walk_packages(
                    [os.path.dirname(module_file_path)],
                        parent_module + '.',
                        onerror=onerror):
                    try:
                        module_ = importlib.import_module(sub_module_name)
                        for k, v in module_.__dict__.items():
                            if k == sub_module and v.__module__ == module_.__name__:
                                module_file_path = module_.__file__
                                break
                    except ModuleNotFoundError as e:
                        logger.warn(
                            'Import error in %s, ModuleNotFoundError: %s' %
                            (sub_module_name, e))
                        continue
                    except Exception as e:
                        logger.warn('Import error in %s, Exception: %s' %
                                    (sub_module_name, e))
                        continue
            else:
                return module_file_path
        else:
            module_file_path = resolve_absolute_import(parent_module)
    return module_file_path


class AnalysisSourceFileImports(ast.NodeVisitor):
    """Analysis source file imports
        List imports of the modelscope.
    """

    def __init__(self, source_file_path) -> None:
        super().__init__()
        self.imports = []
        self.source_file_path = source_file_path

    def visit_Import(self, node):
        """Processing import x,y,z or import os.path as osp"""
        for alias in node.names:
            if alias.name.startswith('modelscope'):
                file_path = resolve_absolute_import(alias.name)
                if file_path.startswith(site.getsitepackages()[0]):
                    self.imports.append(
                        os.path.relpath(file_path,
                                        site.getsitepackages()[0]))
                else:
                    self.imports.append(
                        os.path.relpath(file_path, os.getcwd()))

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
                        self.source_file_path, module_name)
                elif module_name.startswith('modelscope'):
                    file_path = resolve_absolute_import(module_name)
                else:
                    file_path = None  # ignore other package.
            else:
                if not module_name.endswith('.'):
                    module_name = module_name + '.'
                name = module_name + alias.name
                if is_relative_import(name):
                    # resolve model path.
                    file_path = resolve_relative_import(
                        self.source_file_path, name)
                elif name.startswith('modelscope'):
                    file_path = resolve_absolute_import(name)
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


def get_imported_files(file_path):
    """Get file dependencies.
    """
    if os.path.isabs(file_path):
        file_path = os.path.relpath(file_path, os.getcwd())
    with open(file_path, 'rb') as f:
        src = f.read()
    analyzer = AnalysisSourceFileImports(file_path)
    analyzer.visit(ast.parse(src, filename=file_path))
    return list(set(analyzer.imports))


def path_to_module_name(file_path):
    if os.path.isabs(file_path):
        file_path = os.path.relpath(file_path, os.getcwd())
    module_name = os.path.dirname(file_path).replace('/', '.')
    return module_name


def get_file_register_modules(file_path):
    logger.info('Get file: %s register_module' % file_path)
    with open(file_path, 'rb') as f:
        src = f.read()
    analyzer = AnalysisSourceFileRegisterModules(file_path)
    analyzer.visit(ast.parse(src, filename=file_path))
    return analyzer.register_modules


def get_import_map():
    all_files = [
        os.path.join(dp, f) for dp, dn, filenames in os.walk(
            os.path.join(os.getcwd(), 'modelscope')) for f in filenames
        if os.path.splitext(f)[1] == '.py'
    ]
    import_map = {}
    for f in all_files:
        files = get_imported_files(f)
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
