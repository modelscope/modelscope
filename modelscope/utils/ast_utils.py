# Copyright (c) Alibaba, Inc. and its affiliates.

import ast
import hashlib
import logging
import os
import os.path as osp
import time
import traceback
from functools import reduce
from pathlib import Path
from typing import Union

import json

# do not delete
from modelscope.metainfo import (CustomDatasets, Heads, Hooks, LR_Schedulers,
                                 Metrics, Models, Optimizers, Pipelines,
                                 Preprocessors, TaskModels, Trainers)
from modelscope.utils.constant import Fields, Tasks
from modelscope.utils.file_utils import get_modelscope_cache_dir
from modelscope.utils.logger import get_logger
from modelscope.utils.registry import default_group

logger = get_logger(log_level=logging.WARNING)
p = Path(__file__)

# get the path of package 'modelscope'
SKIP_FUNCTION_SCANNING = True
MODELSCOPE_PATH = p.resolve().parents[1]
INDEXER_FILE_DIR = get_modelscope_cache_dir()
REGISTER_MODULE = 'register_module'
IGNORED_PACKAGES = ['modelscope', '.']
SCAN_SUB_FOLDERS = [
    'models', 'metrics', 'pipelines', 'preprocessors', 'trainers',
    'msdatasets', 'exporters'
]
INDEXER_FILE = 'ast_indexer'
DECORATOR_KEY = 'decorators'
EXPRESS_KEY = 'express'
FROM_IMPORT_KEY = 'from_imports'
IMPORT_KEY = 'imports'
FILE_NAME_KEY = 'filepath'
MODELSCOPE_PATH_KEY = 'modelscope_path'
VERSION_KEY = 'version'
MD5_KEY = 'md5'
INDEX_KEY = 'index'
FILES_MTIME_KEY = 'files_mtime'
REQUIREMENT_KEY = 'requirements'
MODULE_KEY = 'module'
CLASS_NAME = 'class_name'
GROUP_KEY = 'group_key'
MODULE_NAME = 'module_name'
MODULE_CLS = 'module_cls'
TEMPLATE_PATH = 'TEMPLATE_PATH'
TEMPLATE_FILE = 'ast_index_file.py'


class AstScanning(object):

    def __init__(self) -> None:
        self.result_import = dict()
        self.result_from_import = dict()
        self.result_decorator = []
        self.express = []

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

    def _skip_function(self, node: Union[ast.AST, 'str']) -> bool:
        if SKIP_FUNCTION_SCANNING:
            if type(node).__name__ == 'FunctionDef' or node == 'FunctionDef':
                return True
        return False

    def _fields(self, n: ast.AST, show_offsets: bool = True) -> tuple:
        if show_offsets:
            return n._attributes + n._fields
        else:
            return n._fields

    def _leaf(self, node: ast.AST, show_offsets: bool = True) -> str:
        output = dict()
        if isinstance(node, ast.AST):
            local_dict = dict()
            for field in self._fields(node, show_offsets=show_offsets):
                field_output = self._leaf(
                    getattr(node, field), show_offsets=show_offsets)
                local_dict[field] = field_output
            output[type(node).__name__] = local_dict
            return output
        else:
            return node

    def _refresh(self):
        self.result_import = dict()
        self.result_from_import = dict()
        self.result_decorator = []
        self.result_express = []

    def scan_ast(self, node: Union[ast.AST, None, str]):
        self._setup_global()
        self.scan_import(node, indent='  ', show_offsets=False)

    def scan_import(
        self,
        node: Union[ast.AST, None, str],
        show_offsets: bool = True,
        parent_node_name: str = '',
    ) -> tuple:
        if node is None:
            return node
        elif self._is_leaf(node):
            return self._leaf(node, show_offsets=show_offsets)
        else:

            def _scan_import(el: Union[ast.AST, None, str],
                             parent_node_name: str = '') -> str:
                return self.scan_import(
                    el,
                    show_offsets=show_offsets,
                    parent_node_name=parent_node_name)

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
            for field in self._fields(node, show_offsets=show_offsets):
                attr = getattr(node, field)
                if attr == []:
                    outputs[field] = []
                elif self._skip_function(parent_node_name):
                    continue
                elif (isinstance(attr, list) and len(attr) == 1
                      and isinstance(attr[0], ast.AST)
                      and self._is_leaf(attr[0])):
                    local_out = _scan_import(attr[0])
                    outputs[field] = local_out
                elif isinstance(attr, list):
                    el_dict = dict()
                    for el in attr:
                        local_out = _scan_import(el, type(el).__name__)
                        name = type(el).__name__
                        if (name == 'Import' or name == 'ImportFrom'
                                or parent_node_name == 'ImportFrom'
                                or parent_node_name == 'Import'):
                            if name not in el_dict:
                                el_dict[name] = []
                            el_dict[name].append(local_out)
                    outputs[field] = el_dict
                elif isinstance(attr, ast.AST):
                    output = _scan_import(attr)
                    outputs[field] = output
                else:
                    outputs[field] = attr

                if (type(node).__name__ == 'Import'
                        or type(node).__name__ == 'ImportFrom'):
                    if type(node).__name__ == 'ImportFrom':
                        if field == 'module':
                            self.result_from_import[outputs[field]] = dict()
                        if field == 'names':
                            if isinstance(outputs[field]['alias'], list):
                                item_name = []
                                for item in outputs[field]['alias']:
                                    local_name = item['alias']['name']
                                    item_name.append(local_name)
                                self.result_from_import[
                                    outputs['module']] = item_name
                            else:
                                local_name = outputs[field]['alias']['name']
                                self.result_from_import[outputs['module']] = [
                                    local_name
                                ]

                    if type(node).__name__ == 'Import':
                        final_dict = outputs[field]['alias']
                        if isinstance(final_dict, list):
                            for item in final_dict:
                                self.result_import[item['alias']
                                                   ['name']] = item['alias']
                        else:
                            self.result_import[outputs[field]['alias']
                                               ['name']] = final_dict

                if 'decorator_list' == field and attr != []:
                    for item in attr:
                        setattr(item, CLASS_NAME, node.name)
                    self.result_decorator.extend(attr)

                if attr != [] and type(
                        attr
                ).__name__ == 'Call' and parent_node_name == 'Expr':
                    self.result_express.append(attr)

            return {
                IMPORT_KEY: self.result_import,
                FROM_IMPORT_KEY: self.result_from_import,
                DECORATOR_KEY: self.result_decorator,
                EXPRESS_KEY: self.result_express
            }

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
                if type(node).__name__ == 'Str':
                    result.append((node.s, None))
                elif type(node).__name__ == 'Constant':
                    result.append((node.value, None))
                else:
                    result.append(_get_attribute_item(node))
            return result

        def _get_keyword_name(nodes: ast.AST) -> list:
            result = []
            for node in nodes:
                if type(node).__name__ == 'keyword':
                    attribute_node = getattr(node, 'value')
                    if type(attribute_node).__name__ == 'Str':
                        result.append((getattr(node,
                                               'arg'), attribute_node.s, None))
                    elif type(attribute_node).__name__ == 'Constant':
                        result.append(
                            (getattr(node, 'arg'), attribute_node.value, None))
                    else:
                        result.append((getattr(node, 'arg'), )
                                      + _get_attribute_item(attribute_node))
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
        # in the case, the key_item is raw data, not registered
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

        # ignore decorators other than register_module
        if REGISTER_MODULE != functions[1]:
            return None
        output = [functions[0]]

        if len(args_list) == 0 and len(keyword_list) == 0:
            args_list.append(default_group)
        if len(keyword_list) == 0 and len(args_list) == 1:
            args_list.append(class_name)

        if len(keyword_list) > 0 and len(args_list) == 0:
            remove_group_item = None
            for item in keyword_list:
                key, name, attr = item
                if key == GROUP_KEY:
                    args_list.append((name, attr))
                    remove_group_item = item
            if remove_group_item is not None:
                keyword_list.remove(remove_group_item)

        if len(args_list) == 0:
            args_list.append(default_group)

        for item in keyword_list:
            key, name, attr = item
            if key == MODULE_CLS:
                class_name = name
            else:
                args_list.append((name, attr))

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
            class_name = getattr(node, CLASS_NAME, None)
            func = getattr(node, 'func')

            if getattr(func, 'attr', None) != REGISTER_MODULE:
                continue

            parse_output = self._parse_decorator(node)
            index = self._registry_indexer(parse_output, class_name)
            if None is not index:
                results.append(index)
        return results

    def generate_ast(self, file):
        self._refresh()
        with open(file, 'r', encoding='utf8') as code:
            data = code.readlines()
        data = ''.join(data)
        node = ast.parse(data)
        output = self.scan_import(node, show_offsets=False)
        output[DECORATOR_KEY] = self.parse_decorators(output[DECORATOR_KEY])
        output[EXPRESS_KEY] = self.parse_decorators(output[EXPRESS_KEY])
        output[DECORATOR_KEY].extend(output[EXPRESS_KEY])
        return output


class FilesAstScanning(object):

    def __init__(self) -> None:
        self.astScaner = AstScanning()
        self.file_dirs = []
        self.requirement_dirs = []

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

    def traversal_files(self, path, check_sub_dir=None, include_init=False):
        self.file_dirs = []
        if check_sub_dir is None or len(check_sub_dir) == 0:
            self._traversal_files(path, include_init=include_init)
        else:
            for item in check_sub_dir:
                sub_dir = os.path.join(path, item)
                if os.path.isdir(sub_dir):
                    self._traversal_files(sub_dir, include_init=include_init)

    def _traversal_files(self, path, include_init=False):
        dir_list = os.scandir(path)
        for item in dir_list:
            if item.name == '__init__.py' and not include_init:
                continue
            elif (item.name.startswith('__')
                  and item.name != '__init__.py') or item.name.endswith(
                      '.json') or item.name.endswith('.md'):
                continue
            if item.is_dir():
                self._traversal_files(item.path, include_init=include_init)
            elif item.is_file() and item.name.endswith('.py'):
                self.file_dirs.append(item.path)
            elif item.is_file() and 'requirement' in item.name:
                self.requirement_dirs.append(item.path)

    def _get_single_file_scan_result(self, file):
        try:
            output = self.astScaner.generate_ast(file)
        except Exception as e:
            detail = traceback.extract_tb(e.__traceback__)
            raise Exception(
                f'During ast indexing the file {file}, a related error excepted '
                f'in the file {detail[-1].filename} at line: '
                f'{detail[-1].lineno}: "{detail[-1].line}" with error msg: '
                f'"{type(e).__name__}: {e}", please double check the origin file {file} '
                f'to see whether the file is correctly edited.')

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

    def _ignore_useless_keys(self, inverted_index):
        if ('OPTIMIZERS', 'default', 'name') in inverted_index:
            del inverted_index[('OPTIMIZERS', 'default', 'name')]
        if ('LR_SCHEDULER', 'default', 'name') in inverted_index:
            del inverted_index[('LR_SCHEDULER', 'default', 'name')]
        return inverted_index

    def get_files_scan_results(self,
                               target_file_list=None,
                               target_dir=MODELSCOPE_PATH,
                               target_folders=SCAN_SUB_FOLDERS):
        """the entry method of the ast scan method

        Args:
            target_file_list can override the dir and folders combine
            target_dir (str, optional): the absolute path of the target directory to be scanned. Defaults to None.
            target_folder (list, optional): the list of
            sub-folders to be scanned in the target folder.
            Defaults to SCAN_SUB_FOLDERS.

        Returns:
            dict: indexer of registry
        """
        start = time.time()
        if target_file_list is not None:
            self.file_dirs = target_file_list
        else:
            self.traversal_files(target_dir, target_folders)
        logger.info(
            f'AST-Scanning the path "{target_dir}" with the following sub folders {target_folders}'
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
        inverted_index_with_results = self._ignore_useless_keys(
            inverted_index_with_results)
        module_import = self._module_import(result)
        index = {
            INDEX_KEY: inverted_index_with_results,
            REQUIREMENT_KEY: module_import
        }
        logger.info(
            f'Scanning done! A number of {len(inverted_index_with_results)} '
            f'components indexed or updated! Time consumed {time.time()-start}s'
        )
        return index

    def files_mtime_md5(self,
                        target_path=MODELSCOPE_PATH,
                        target_subfolder=SCAN_SUB_FOLDERS,
                        file_list=None):
        self.file_dirs = []
        if file_list and isinstance(file_list, list):
            self.file_dirs = file_list
        else:
            self.traversal_files(target_path, target_subfolder)
        files_mtime = []
        files_mtime_dict = dict()
        for item in self.file_dirs:
            mtime = os.path.getmtime(item)
            files_mtime.append(mtime)
            files_mtime_dict[item] = mtime
        result_str = reduce(lambda x, y: str(x) + str(y), files_mtime, '')
        md5 = hashlib.md5(result_str.encode())
        return md5.hexdigest(), files_mtime_dict


file_scanner = FilesAstScanning()


def ensure_write(obj: bytes, filepath: Union[str, Path]) -> None:
    """Write data to a given ``filepath`` with 'wb' mode.

    Note:
        ``write`` will create a directory if the directory of ``filepath``
        does not exist.

    Args:
        obj (bytes): Data to be written.
        filepath (str or Path): Path to write data.
    """
    dirname = os.path.dirname(filepath)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    with open(filepath, 'wb') as f:
        f.write(obj)


def _save_index(index, file_path, file_list=None, with_template=False):
    # convert tuple key to str key
    index[INDEX_KEY] = {str(k): v for k, v in index[INDEX_KEY].items()}
    from modelscope.version import __version__
    index[VERSION_KEY] = __version__
    index[MD5_KEY], index[FILES_MTIME_KEY] = file_scanner.files_mtime_md5(
        file_list=file_list)
    index[MODELSCOPE_PATH_KEY] = MODELSCOPE_PATH.as_posix()
    json_index = json.dumps(index)
    if with_template:
        json_index = json_index.replace(MODELSCOPE_PATH.as_posix(),
                                        TEMPLATE_PATH)
    ensure_write(json_index.encode(), file_path)
    index[INDEX_KEY] = {
        ast.literal_eval(k): v
        for k, v in index[INDEX_KEY].items()
    }


def _load_index(file_path, with_template=False):
    with open(file_path, 'rb') as f:
        bytes_index = f.read()
    if with_template:
        bytes_index = bytes_index.decode().replace(TEMPLATE_PATH,
                                                   MODELSCOPE_PATH.as_posix())
    wrapped_index = json.loads(bytes_index)
    # convert str key to tuple key
    wrapped_index[INDEX_KEY] = {
        ast.literal_eval(k): v
        for k, v in wrapped_index[INDEX_KEY].items()
    }
    return wrapped_index


def _update_index(index, files_mtime):
    # inplace update index
    origin_files_mtime = index[FILES_MTIME_KEY]
    new_files = list(set(files_mtime) - set(origin_files_mtime))
    removed_files = list(set(origin_files_mtime) - set(files_mtime))
    updated_files = []
    for file in origin_files_mtime:
        if file not in removed_files and \
                (origin_files_mtime[file] != files_mtime[file]):
            updated_files.append(file)
    removed_files.extend(updated_files)
    updated_files.extend(new_files)

    # remove deleted index
    if len(removed_files) > 0:
        remove_index_keys = []
        remove_requirement_keys = []
        for key in index[INDEX_KEY]:
            if index[INDEX_KEY][key][FILE_NAME_KEY] in removed_files:
                remove_index_keys.append(key)
                remove_requirement_keys.append(
                    index[INDEX_KEY][key][MODULE_KEY])
        for key in remove_index_keys:
            del index[INDEX_KEY][key]
        for key in remove_requirement_keys:
            if key in index[REQUIREMENT_KEY]:
                del index[REQUIREMENT_KEY][key]

    # add new index
    updated_index = file_scanner.get_files_scan_results(updated_files)
    index[INDEX_KEY].update(updated_index[INDEX_KEY])
    index[REQUIREMENT_KEY].update(updated_index[REQUIREMENT_KEY])


def load_index(
    file_list=None,
    force_rebuild=False,
    indexer_file_dir=INDEXER_FILE_DIR,
    indexer_file=INDEXER_FILE,
):
    """get the index from scan results or cache

    Args:
        file_list: load indexer only from the file lists if provided, default as None
        force_rebuild: If set true, rebuild and load index, default as False,
        indexer_file_dir: The dir where the indexer file saved, default as INDEXER_FILE_DIR
        indexer_file: The indexer file name, default as INDEXER_FILE
    Returns:
        dict: the index information for all registered modules, including key:
        index, requirements, files last modified time, modelscope home path,
        version and md5, the detail is shown below example: {
            'index': {
                ('MODELS', 'nlp', 'bert'):{
                    'filepath' : 'path/to/the/registered/model', 'imports':
                    ['os', 'torch', 'typing'] 'module':
                    'modelscope.models.nlp.bert'
                },
                ...
            }, 'requirements': {
                'modelscope.models.nlp.bert': ['os', 'torch', 'typing'],
                'modelscope.models.nlp.structbert': ['os', 'torch', 'typing'],
                ...
            }, 'files_mtime' : {
                '/User/Path/To/Your/Modelscope/modelscope/preprocessors/nlp/text_generation_preprocessor.py':
                16554565445, ...
            },'version': '0.2.3', 'md5': '8616924970fe6bc119d1562832625612',
            'modelscope_path': '/User/Path/To/Your/Modelscope'
        }
    """
    # env variable override
    cache_dir = os.getenv('MODELSCOPE_CACHE', indexer_file_dir)
    index_file = os.getenv('MODELSCOPE_INDEX_FILE', indexer_file)
    file_path = os.path.join(cache_dir, index_file)
    logger.info(f'Loading ast index from {file_path}')
    index = None
    local_changed = False
    if not force_rebuild and os.path.exists(file_path):
        wrapped_index = _load_index(file_path)
        md5, files_mtime = file_scanner.files_mtime_md5(file_list=file_list)
        from modelscope.version import __version__
        if (wrapped_index[VERSION_KEY] == __version__):
            index = wrapped_index
            if (wrapped_index[MD5_KEY] != md5):
                local_changed = True
    full_index_flag = False

    if index is None:
        full_index_flag = True
    elif index and local_changed and FILES_MTIME_KEY not in index:
        full_index_flag = True
    elif index and local_changed and MODELSCOPE_PATH_KEY not in index:
        full_index_flag = True
    elif index and local_changed and index[
            MODELSCOPE_PATH_KEY] != MODELSCOPE_PATH.as_posix():
        full_index_flag = True

    if full_index_flag:
        if force_rebuild:
            logger.info('Force rebuilding ast index from scanning every file!')
            index = file_scanner.get_files_scan_results(file_list)
        else:
            logger.info(
                f'No valid ast index found from {file_path}, generating ast index from prebuilt!'
            )
            index = load_from_prebuilt()
            if index is None:
                index = file_scanner.get_files_scan_results(file_list)
        _save_index(index, file_path, file_list)
    elif local_changed and not full_index_flag:
        logger.info(
            'Updating the files for the changes of local files, '
            'first time updating will take longer time! Please wait till updating done!'
        )
        _update_index(index, files_mtime)
        _save_index(index, file_path, file_list)

    logger.info(
        f'Loading done! Current index file version is {index[VERSION_KEY]}, '
        f'with md5 {index[MD5_KEY]} and a total number of '
        f'{len(index[INDEX_KEY])} components indexed')
    return index


def load_from_prebuilt(file_path=None):
    if file_path is None:
        local_path = p.resolve().parents[0]
        file_path = os.path.join(local_path, TEMPLATE_FILE)
    if os.path.exists(file_path):
        index = _load_index(file_path, with_template=True)
    else:
        index = None
    return index


def generate_ast_template(file_path=None, force_rebuild=True):
    index = load_index(force_rebuild=force_rebuild)
    if file_path is None:
        local_path = p.resolve().parents[0]
        file_path = os.path.join(local_path, TEMPLATE_FILE)
    _save_index(index, file_path, with_template=True)
    if not os.path.exists(file_path):
        raise Exception(
            'The index file is not create correctly, please double check')
    return index


if __name__ == '__main__':
    index = load_index(force_rebuild=True)
    print(index)
