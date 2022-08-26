import ast
import contextlib
import hashlib
import importlib
import os
import os.path as osp
import time
import traceback
from functools import reduce
from pathlib import Path
from typing import Generator, Union

import gast
import json

from modelscope import __version__
from modelscope.fileio.file import LocalStorage
from modelscope.metainfo import (Datasets, Heads, Hooks, LR_Schedulers,
                                 Metrics, Models, Optimizers, Pipelines,
                                 Preprocessors, TaskModels, Trainers)
from modelscope.utils.constant import Fields, Tasks
from modelscope.utils.file_utils import get_default_cache_dir
from modelscope.utils.logger import get_logger
from modelscope.utils.registry import default_group

logger = get_logger()
storage = LocalStorage()
p = Path(__file__)

# get the path of package 'modelscope'
MODELSCOPE_PATH = p.resolve().parents[1]
REGISTER_MODULE = 'register_module'
IGNORED_PACKAGES = ['modelscope', '.']
SCAN_SUB_FOLDERS = [
    'models', 'metrics', 'pipelines', 'preprocessors', 'trainers', 'msdatasets'
]
INDEXER_FILE = 'ast_indexer'
DECORATOR_KEY = 'decorators'
FROM_IMPORT_KEY = 'from_imports'
IMPORT_KEY = 'imports'
FILE_NAME_KEY = 'filepath'
VERSION_KEY = 'version'
MD5_KEY = 'md5'
INDEX_KEY = 'index'
REQUIREMENT_KEY = 'requirements'
MODULE_KEY = 'module'
CLASS_NAME = 'class_name'


class AstScaning(object):

    def __init__(self) -> None:
        self.result_import = dict()
        self.result_from_import = dict()
        self.result_decorator = []

    def _is_sub_node(self, node: object) -> bool:
        return isinstance(node,
                          ast.AST) and not isinstance(node, ast.expr_context)

    def _is_leaf(self, node: ast.AST) -> bool:
        for field in node._fields:
            attr = getattr(node, field)
            if self._is_sub_node(attr):
                return False
            elif isinstance(attr, (list, tuple)):
                for val in attr:
                    if self._is_sub_node(val):
                        return False
        else:
            return True

    def _fields(self, n: ast.AST, show_offsets: bool = True) -> tuple:
        if show_offsets:
            return n._attributes + n._fields
        else:
            return n._fields

    def _leaf(self, node: ast.AST, show_offsets: bool = True) -> str:
        output = dict()
        local_print = list()
        if isinstance(node, ast.AST):
            local_dict = dict()
            for field in self._fields(node, show_offsets=show_offsets):
                field_output, field_prints = self._leaf(
                    getattr(node, field), show_offsets=show_offsets)
                local_dict[field] = field_output
                local_print.append('{}={}'.format(field, field_prints))

            prints = '{}({})'.format(
                type(node).__name__,
                ', '.join(local_print),
            )
            output[type(node).__name__] = local_dict
            return output, prints
        elif isinstance(node, list):
            if '_fields' not in node:
                return node, repr(node)
            for item in node:
                item_output, item_prints = self._leaf(
                    getattr(node, item), show_offsets=show_offsets)
                local_print.append(item_prints)
            return node, '[{}]'.format(', '.join(local_print), )
        else:
            return node, repr(node)

    def _refresh(self):
        self.result_import = dict()
        self.result_from_import = dict()
        self.result_decorator = []

    def scan_ast(self, node: Union[ast.AST, None, str]):
        self._setup_global()
        self.scan_import(node, indent='  ', show_offsets=False)

    def scan_import(
        self,
        node: Union[ast.AST, None, str],
        indent: Union[str, int] = '    ',
        show_offsets: bool = True,
        _indent: int = 0,
        parent_node_name: str = '',
    ) -> tuple:
        if node is None:
            return node, repr(node)
        elif self._is_leaf(node):
            return self._leaf(node, show_offsets=show_offsets)
        else:
            if isinstance(indent, int):
                indent_s = indent * ' '
            else:
                indent_s = indent

            class state:
                indent = _indent

            @contextlib.contextmanager
            def indented() -> Generator[None, None, None]:
                state.indent += 1
                yield
                state.indent -= 1

            def indentstr() -> str:
                return state.indent * indent_s

            def _scan_import(el: Union[ast.AST, None, str],
                             _indent: int = 0,
                             parent_node_name: str = '') -> str:
                return self.scan_import(
                    el,
                    indent=indent,
                    show_offsets=show_offsets,
                    _indent=_indent,
                    parent_node_name=parent_node_name)

            out = type(node).__name__ + '(\n'
            outputs = dict()
            # add relative path expression
            if type(node).__name__ == 'ImportFrom':
                level = getattr(node, 'level')
                if level >= 1:
                    path_level = ''.join(['.'] * level)
                    setattr(node, 'level', 0)
                    module_name = getattr(node, 'module')
                    if module_name is None:
                        setattr(node, 'module', path_level)
                    else:
                        setattr(node, 'module', path_level + module_name)
            with indented():
                for field in self._fields(node, show_offsets=show_offsets):
                    attr = getattr(node, field)
                    if attr == []:
                        representation = '[]'
                        outputs[field] = []
                    elif (isinstance(attr, list) and len(attr) == 1
                          and isinstance(attr[0], ast.AST)
                          and self._is_leaf(attr[0])):
                        local_out, local_print = _scan_import(attr[0])
                        representation = f'[{local_print}]'
                        outputs[field] = local_out

                    elif isinstance(attr, list):
                        representation = '[\n'
                        el_dict = dict()
                        with indented():
                            for el in attr:
                                local_out, local_print = _scan_import(
                                    el, state.indent,
                                    type(el).__name__)
                                representation += '{}{},\n'.format(
                                    indentstr(),
                                    local_print,
                                )
                                name = type(el).__name__
                                if (name == 'Import' or name == 'ImportFrom'
                                        or parent_node_name == 'ImportFrom'
                                        or parent_node_name == 'Import'):
                                    if name not in el_dict:
                                        el_dict[name] = []
                                    el_dict[name].append(local_out)
                        representation += indentstr() + ']'
                        outputs[field] = el_dict
                    elif isinstance(attr, ast.AST):
                        output, representation = _scan_import(
                            attr, state.indent)
                        outputs[field] = output
                    else:
                        representation = repr(attr)
                        outputs[field] = attr

                    if (type(node).__name__ == 'Import'
                            or type(node).__name__ == 'ImportFrom'):
                        if type(node).__name__ == 'ImportFrom':
                            if field == 'module':
                                self.result_from_import[
                                    outputs[field]] = dict()
                            if field == 'names':
                                if isinstance(outputs[field]['alias'], list):
                                    item_name = []
                                    for item in outputs[field]['alias']:
                                        local_name = item['alias']['name']
                                        item_name.append(local_name)
                                    self.result_from_import[
                                        outputs['module']] = item_name
                                else:
                                    local_name = outputs[field]['alias'][
                                        'name']
                                    self.result_from_import[
                                        outputs['module']] = [local_name]

                        if type(node).__name__ == 'Import':
                            final_dict = outputs[field]['alias']
                            if isinstance(final_dict, list):
                                for item in final_dict:
                                    self.result_import[
                                        item['alias']['name']] = item['alias']
                            else:
                                self.result_import[outputs[field]['alias']
                                                   ['name']] = final_dict

                    if 'decorator_list' == field and attr != []:
                        for item in attr:
                            setattr(item, CLASS_NAME, node.name)
                        self.result_decorator.extend(attr)

                    out += f'{indentstr()}{field}={representation},\n'

            out += indentstr() + ')'
            return {
                IMPORT_KEY: self.result_import,
                FROM_IMPORT_KEY: self.result_from_import,
                DECORATOR_KEY: self.result_decorator
            }, out

    def _parse_decorator(self, node: ast.AST) -> tuple:

        def _get_attribute_item(node: ast.AST) -> tuple:
            value, id, attr = None, None, None
            if type(node).__name__ == 'Attribute':
                value = getattr(node, 'value')
                id = getattr(value, 'id')
                attr = getattr(node, 'attr')
            if type(node).__name__ == 'Name':
                id = getattr(node, 'id')
            return id, attr

        def _get_args_name(nodes: list) -> list:
            result = []
            for node in nodes:
                result.append(_get_attribute_item(node))
            return result

        def _get_keyword_name(nodes: ast.AST) -> list:
            result = []
            for node in nodes:
                if type(node).__name__ == 'keyword':
                    attribute_node = getattr(node, 'value')
                    if type(attribute_node).__name__ == 'Str':
                        result.append((attribute_node.s, None))
                    else:
                        result.append(_get_attribute_item(attribute_node))
            return result

        functions = _get_attribute_item(node.func)
        args_list = _get_args_name(node.args)
        keyword_list = _get_keyword_name(node.keywords)
        return functions, args_list, keyword_list

    def _get_registry_value(self, key_item):
        if key_item is None:
            return None
        if key_item == 'default_group':
            return default_group
        split_list = key_item.split('.')
        # in the case, the key_item is raw data, not registred
        if len(split_list) == 1:
            return key_item
        else:
            return getattr(eval(split_list[0]), split_list[1])

    def _registry_indexer(self, parsed_input: tuple, class_name: str) -> tuple:
        """format registry information to a tuple indexer

        Return:
            tuple: (MODELS, Tasks.text-classification, Models.structbert)
        """
        functions, args_list, keyword_list = parsed_input

        # ignore decocators other than register_module
        if REGISTER_MODULE != functions[1]:
            return None
        output = [functions[0]]

        if len(args_list) == 0 and len(keyword_list) == 0:
            args_list.append(default_group)
        if len(keyword_list) == 0 and len(args_list) == 1:
            args_list.append(class_name)
        if len(keyword_list) == 1 and len(args_list) == 0:
            args_list.append(default_group)

        args_list.extend(keyword_list)

        for item in args_list:
            # the case empty input
            if item is None:
                output.append(None)
            # the case (default_group)
            elif item[1] is None:
                output.append(item[0])
            elif isinstance(item, str):
                output.append(item)
            else:
                output.append('.'.join(item))
        return (output[0], self._get_registry_value(output[1]),
                self._get_registry_value(output[2]))

    def parse_decorators(self, nodes: list) -> list:
        """parse the AST nodes of decorators object to registry indexer

        Args:
            nodes (list): list of AST decorator nodes

        Returns:
            list: list of registry indexer
        """
        results = []
        for node in nodes:
            if type(node).__name__ != 'Call':
                continue
            parse_output = self._parse_decorator(node)
            index = self._registry_indexer(parse_output,
                                           getattr(node, CLASS_NAME))
            if None is not index:
                results.append(index)
        return results

    def generate_ast(self, file):
        self._refresh()
        with open(file, 'r') as code:
            data = code.readlines()
        data = ''.join(data)

        node = gast.parse(data)
        output, _ = self.scan_import(node, indent='  ', show_offsets=False)
        output[DECORATOR_KEY] = self.parse_decorators(output[DECORATOR_KEY])
        return output


class FilesAstScaning(object):

    def __init__(self) -> None:
        self.astScaner = AstScaning()
        self.file_dirs = []

    def _parse_import_path(self,
                           import_package: str,
                           current_path: str = None) -> str:
        """
        Args:
            import_package (str): relative import or abs import
            current_path (str): path/to/current/file
        """
        if import_package.startswith(IGNORED_PACKAGES[0]):
            return MODELSCOPE_PATH + '/' + '/'.join(
                import_package.split('.')[1:]) + '.py'
        elif import_package.startswith(IGNORED_PACKAGES[1]):
            current_path_list = current_path.split('/')
            import_package_list = import_package.split('.')
            level = 0
            for index, item in enumerate(import_package_list):
                if item != '':
                    level = index
                    break

            abs_path_list = current_path_list[0:-level]
            abs_path_list.extend(import_package_list[index:])
            return '/' + '/'.join(abs_path_list) + '.py'
        else:
            return current_path

    def _traversal_import(
        self,
        import_abs_path,
    ):
        pass

    def parse_import(self, scan_result: dict) -> list:
        """parse import and from import dicts to a third party package list

        Args:
            scan_result (dict): including the import and from import result

        Returns:
            list: a list of package ignored 'modelscope' and relative path import
        """
        output = []
        output.extend(list(scan_result[IMPORT_KEY].keys()))
        output.extend(list(scan_result[FROM_IMPORT_KEY].keys()))

        # get the package name
        for index, item in enumerate(output):
            if '' == item.split('.')[0]:
                output[index] = '.'
            else:
                output[index] = item.split('.')[0]

        ignored = set()
        for item in output:
            for ignored_package in IGNORED_PACKAGES:
                if item.startswith(ignored_package):
                    ignored.add(item)
        return list(set(output) - set(ignored))

    def traversal_files(self, path, check_sub_dir):
        self.file_dirs = []
        if check_sub_dir is None or len(check_sub_dir) == 0:
            self._traversal_files(path)

        for item in check_sub_dir:
            sub_dir = os.path.join(path, item)
            if os.path.isdir(sub_dir):
                self._traversal_files(sub_dir)

    def _traversal_files(self, path):
        dir_list = os.scandir(path)
        for item in dir_list:
            if item.name.startswith('__'):
                continue
            if item.is_dir():
                self._traversal_files(item.path)
            elif item.is_file() and item.name.endswith('.py'):
                self.file_dirs.append(item.path)

    def _get_single_file_scan_result(self, file):
        try:
            output = self.astScaner.generate_ast(file)
        except Exception as e:
            detail = traceback.extract_tb(e.__traceback__)
            raise Exception(
                f'During ast indexing, error is in the file {detail[-1].filename}'
                f' line: {detail[-1].lineno}: "{detail[-1].line}" with error msg: '
                f'"{type(e).__name__}: {e}"')

        import_list = self.parse_import(output)
        return output[DECORATOR_KEY], import_list

    def _inverted_index(self, forward_index):
        inverted_index = dict()
        for index in forward_index:
            for item in forward_index[index][DECORATOR_KEY]:
                inverted_index[item] = {
                    FILE_NAME_KEY: index,
                    IMPORT_KEY: forward_index[index][IMPORT_KEY],
                    MODULE_KEY: forward_index[index][MODULE_KEY],
                }
        return inverted_index

    def _module_import(self, forward_index):
        module_import = dict()
        for index, value_dict in forward_index.items():
            module_import[value_dict[MODULE_KEY]] = value_dict[IMPORT_KEY]
        return module_import

    def get_files_scan_results(self,
                               target_dir=MODELSCOPE_PATH,
                               target_folders=SCAN_SUB_FOLDERS):
        """the entry method of the ast scan method

        Args:
            target_dir (str, optional): the absolute path of the target directory to be scaned. Defaults to None.
            target_folder (list, optional): the list of
            sub-folders to be scaned in the target folder.
            Defaults to SCAN_SUB_FOLDERS.

        Returns:
            dict: indexer of registry
        """

        self.traversal_files(target_dir, target_folders)
        start = time.time()
        logger.info(
            f'AST-Scaning the path "{target_dir}" with the following sub folders {target_folders}'
        )

        result = dict()
        for file in self.file_dirs:
            filepath = file[file.rfind('modelscope'):]
            module_name = filepath.replace(osp.sep, '.').replace('.py', '')
            decorator_list, import_list = self._get_single_file_scan_result(
                file)
            result[file] = {
                DECORATOR_KEY: decorator_list,
                IMPORT_KEY: import_list,
                MODULE_KEY: module_name
            }
        inverted_index_with_results = self._inverted_index(result)
        module_import = self._module_import(result)
        index = {
            INDEX_KEY: inverted_index_with_results,
            REQUIREMENT_KEY: module_import
        }
        logger.info(
            f'Scaning done! A number of {len(inverted_index_with_results)}'
            f' files indexed! Time consumed {time.time()-start}s')
        return index

    def files_mtime_md5(self,
                        target_path=MODELSCOPE_PATH,
                        target_subfolder=SCAN_SUB_FOLDERS):
        self.file_dirs = []
        self.traversal_files(target_path, target_subfolder)
        files_mtime = []
        for item in self.file_dirs:
            files_mtime.append(os.path.getmtime(item))
        result_str = reduce(lambda x, y: str(x) + str(y), files_mtime, '')
        md5 = hashlib.md5(result_str.encode())
        return md5.hexdigest()


file_scanner = FilesAstScaning()


def _save_index(index, file_path):
    # convert tuple key to str key
    index[INDEX_KEY] = {str(k): v for k, v in index[INDEX_KEY].items()}
    index[VERSION_KEY] = __version__
    index[MD5_KEY] = file_scanner.files_mtime_md5()
    json_index = json.dumps(index)
    storage.write(json_index.encode(), file_path)
    index[INDEX_KEY] = {
        ast.literal_eval(k): v
        for k, v in index[INDEX_KEY].items()
    }


def _load_index(file_path):
    bytes_index = storage.read(file_path)
    wrapped_index = json.loads(bytes_index)
    # convert str key to tuple key
    wrapped_index[INDEX_KEY] = {
        ast.literal_eval(k): v
        for k, v in wrapped_index[INDEX_KEY].items()
    }
    return wrapped_index


def load_index(force_rebuild=False):
    """get the index from scan results or cache

    Args:
        force_rebuild: If set true, rebuild and load index
    Returns:
        dict: the index information for all registred modules, including key:
        index, requirments, version and md5, the detail is shown below example:
        {
            'index': {
                ('MODELS', 'nlp', 'bert'):{
                    'filepath' : 'path/to/the/registered/model', 'imports':
                    ['os', 'torch', 'typeing'] 'module':
                    'modelscope.models.nlp.bert'
                },
                ...
            }, 'requirments': {
                'modelscope.models.nlp.bert': ['os', 'torch', 'typeing'],
                'modelscope.models.nlp.structbert': ['os', 'torch', 'typeing'],
                ...
            }, 'version': '0.2.3', 'md5': '8616924970fe6bc119d1562832625612',
        }
    """
    cache_dir = os.getenv('MODELSCOPE_CACHE', get_default_cache_dir())
    file_path = os.path.join(cache_dir, INDEXER_FILE)
    logger.info(f'Loading ast index from {file_path}')
    index = None
    if not force_rebuild and os.path.exists(file_path):
        wrapped_index = _load_index(file_path)
        md5 = file_scanner.files_mtime_md5()
        if (wrapped_index[VERSION_KEY] == __version__
                and wrapped_index[MD5_KEY] == md5):
            index = wrapped_index

    if index is None:
        if force_rebuild:
            logger.info('Force rebuilding ast index')
        else:
            logger.info(
                f'No valid ast index found from {file_path}, rebuilding ast index!'
            )
        index = file_scanner.get_files_scan_results()
        _save_index(index, file_path)
    logger.info(
        f'Loading done! Current index file version is {index[VERSION_KEY]}, '
        f'with md5 {index[MD5_KEY]}')
    return index


def check_import_module_avaliable(module_dicts: dict) -> list:
    missed_module = []
    for module in module_dicts.keys():
        loader = importlib.find_loader(module)
        if loader is None:
            missed_module.append(module)
    return missed_module


if __name__ == '__main__':
    index = load_index()
    print(index)
