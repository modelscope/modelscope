# Copyright (c) OpenMMLab. All rights reserved.
# Major implementation is borrowed and modified from
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py

import copy
import os
import os.path as osp
import platform
import shutil
import sys
import tempfile
import types
from pathlib import Path
from types import FunctionType
from typing import Dict, Union

import addict
import json

from modelscope.utils.constant import ConfigFields, ModelFile
from modelscope.utils.logger import get_logger

logger = get_logger()

BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
DEPRECATION_KEY = '_deprecation_'
RESERVED_KEYS = ['filename', 'text', 'pretty_text']


class ConfigDict(addict.Dict):
    """ Dict which support get value through getattr

    Examples:
        >>> cdict = ConfigDict({'a':1232})
        >>> print(cdict.a)
        >>> # 1232
    """

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no "
                                f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


class Config:
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.

    Example:
        >>> cfg = Config(dict(a=1, b=dict(c=[1,2,3], d='dd')))
        >>> cfg.a
        1
        >>> cfg.b
        {'c': [1, 2, 3], 'd': 'dd'}
        >>> cfg.b.d
        'dd'
        >>> cfg = Config.from_file('configs/examples/configuration.json')
        >>> cfg.filename
       'configs/examples/configuration.json'
        >>> cfg.b
        {'c': [1, 2, 3], 'd': 'dd'}
        >>> cfg = Config.from_file('configs/examples/configuration.py')
        >>> cfg.filename
        "configs/examples/configuration.py"
        >>> cfg = Config.from_file('configs/examples/configuration.yaml')
        >>> cfg.filename
        "configs/examples/configuration.yaml"
    """

    @staticmethod
    def _file2dict(filename):
        filename = osp.abspath(osp.expanduser(filename))
        if not osp.exists(filename):
            raise ValueError(f'File does not exists {filename}')
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in ['.py', '.json', '.yaml', '.yml']:
            raise IOError('Only py/yml/yaml/json type are supported now!')

        with tempfile.TemporaryDirectory() as tmp_cfg_dir:
            tmp_cfg_file = tempfile.NamedTemporaryFile(
                dir=tmp_cfg_dir, suffix=fileExtname)
            if platform.system() == 'Windows':
                tmp_cfg_file.close()
            tmp_cfg_name = osp.basename(tmp_cfg_file.name)
            shutil.copyfile(filename, tmp_cfg_file.name)

            if filename.endswith('.py'):
                # import as needed.
                from modelscope.utils.import_utils import import_modules_from_file
                module_nanme, mod = import_modules_from_file(
                    osp.join(tmp_cfg_dir, tmp_cfg_name))
                cfg_dict = {}
                for name, value in mod.__dict__.items():
                    if not name.startswith('__') and \
                       not isinstance(value, types.ModuleType) and \
                       not isinstance(value, types.FunctionType):
                        cfg_dict[name] = value

                # delete imported module
                del sys.modules[module_nanme]
            elif filename.endswith(('.yml', '.yaml', '.json')):
                from modelscope.fileio import load
                cfg_dict = load(tmp_cfg_file.name)
            # close temp file
            tmp_cfg_file.close()

        cfg_text = filename + '\n'
        with open(filename, 'r', encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        return cfg_dict, cfg_text

    @staticmethod
    def from_file(filename):
        if isinstance(filename, Path):
            filename = str(filename)
        cfg_dict, cfg_text = Config._file2dict(filename)
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def from_string(cfg_str, file_format):
        """Generate config from config str.

        Args:
            cfg_str (str): Config str.
            file_format (str): Config file format corresponding to the
               config str. Only py/yml/yaml/json type are supported now!

        Returns:
            :obj:`Config`: Config obj.
        """
        if file_format not in ['.py', '.json', '.yaml', '.yml']:
            raise IOError('Only py/yml/yaml/json type are supported now!')
        if file_format != '.py' and 'dict(' in cfg_str:
            # check if users specify a wrong suffix for python
            logger.warning(
                'Please check "file_format", the file format may be .py')
        with tempfile.NamedTemporaryFile(
                'w', encoding='utf-8', suffix=file_format,
                delete=False) as temp_file:
            temp_file.write(cfg_str)
            # on windows, previous implementation cause error
            # see PR 1077 for details
        cfg = Config.from_file(temp_file.name)
        os.remove(temp_file.name)
        return cfg

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')

        if isinstance(filename, Path):
            filename = str(filename)

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = ''
        super(Config, self).__setattr__('_text', text)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    @property
    def pretty_text(self):

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = '[\n'
                v_str += '\n'.join(
                    f'dict({_indent(_format_dict(v_), indent)}),'
                    for v_ in v).rstrip(',')
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f'{k_str}: {v_str}'
                else:
                    attr_str = f'{str(k)}={v_str}'
                attr_str = _indent(attr_str, indent) + ']'
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= \
                    (not str(key_name).isidentifier())
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ''
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += '{'
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = '' if outest_level or is_last else ','
                if isinstance(v, dict):
                    v_str = '\n' + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f'{k_str}: dict({v_str}'
                    else:
                        attr_str = f'{str(k)}=dict({v_str}'
                    attr_str = _indent(attr_str, indent) + ')' + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += '\n'.join(s)
            if use_mapping:
                r += '}'
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # copied from setup.cfg
        yapf_style = dict(
            based_on_style='pep8',
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True)
        from yapf.yapflib.yapf_api import FormatCode
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)

        return text

    def __repr__(self):
        return f'Config (path: {self.filename}): {self._cfg_dict.__repr__()}'

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self):
        return (self._cfg_dict, self._filename, self._text)

    def __copy__(self):
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)

        return other

    def __deepcopy__(self, memo):
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            super(Config, other).__setattr__(key, copy.deepcopy(value, memo))

        return other

    def __setstate__(self, state):
        _cfg_dict, _filename, _text = state
        super(Config, self).__setattr__('_cfg_dict', _cfg_dict)
        super(Config, self).__setattr__('_filename', _filename)
        super(Config, self).__setattr__('_text', _text)

    def safe_get(self, key_chain: str, default=None, type_field='type'):
        """Get a value with a key-chain in str format, if key does not exist, the default value will be returned.

        This method is safe to call, and will not edit any value.

        Args:
            key_chain: The input key chain, for example: 'train.hooks[0].type'
            default: The default value returned when any key does not exist, default None.
            type_field: Get an object from a list or tuple for example by 'train.hooks.CheckPointHook', in which
                'hooks' is a list, and 'CheckPointHook' is a value of the content of key `type_field`.
                If there are multiple matched objects, the first element will be returned.
        Returns:
            The value, or the default value.
        """
        try:
            keys = key_chain.split('.')
            _cfg_dict = self._cfg_dict
            for key in keys:
                val = None
                if '[' in key:
                    key, val = key.split('[')
                    val, _ = val.split(']')

                if isinstance(_cfg_dict, (list, tuple)):
                    assert type_field is not None, 'Getting object without an index from a list or tuple ' \
                                                   'needs an valid `type_field` param.'
                    _sub_cfg_dict = list(
                        filter(lambda sub: sub[type_field] == key, _cfg_dict))
                    _cfg_dict = _sub_cfg_dict[0]
                else:
                    _cfg_dict = _cfg_dict[key]
                if val is not None:
                    _cfg_dict = _cfg_dict[int(val)]
            return _cfg_dict
        except Exception as e:
            logger.debug(
                f'Key not valid in Config: {key_chain}, return the default value: {default}'
            )
            logger.debug(e)
            return default

    def dump(self, file: str = None):
        """Dumps config into a file or returns a string representation of the
        config.

        If a file argument is given, saves the config to that file using the
        format defined by the file argument extension.

        Otherwise, returns a string representing the config. The formatting of
        this returned string is defined by the extension of `self.filename`. If
        `self.filename` is not defined, returns a string representation of a
         dict (lowercased and using ' for strings).

        Examples:
            >>> cfg_dict = dict(item1=[1, 2], item2=dict(a=0),
            ...     item3=True, item4='test')
            >>> cfg = Config(cfg_dict=cfg_dict)
            >>> dump_file = "a.py"
            >>> cfg.dump(dump_file)

        Args:
            file (str, optional): Path of the output file where the config
                will be dumped. Defaults to None.
        """
        from modelscope.fileio import dump
        cfg_dict = super(Config, self).__getattribute__('_cfg_dict').to_dict()
        if file is None:
            if self.filename is None or self.filename.endswith('.py'):
                return self.pretty_text
            else:
                file_format = self.filename.split('.')[-1]
                return dump(cfg_dict, file_format=file_format)
        elif file.endswith('.py'):
            with open(file, 'w', encoding='utf-8') as f:
                f.write(self.pretty_text)
        else:
            file_format = file.split('.')[-1]
            return dump(cfg_dict, file=file, file_format=file_format)

    def merge_from_dict(self, options, allow_list_keys=True, force=True):
        """Merge dict into cfg_dict.

        Merge the dict parsed by MultipleKVAction into this cfg.

        Examples:
            >>> options = {'model.backbone.depth': 50,
            ...            'model.backbone.with_cp':True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...     model=dict(backbone=dict(type='ResNet', depth=50, with_cp=True)))

            >>> # Merge list element for replace target index
            >>> cfg = Config(dict(pipeline=[
            ...     dict(type='Resize'), dict(type='RandomDistortion')]))
            >>> options = dict(pipeline={'0': dict(type='MyResize')})
            >>> cfg.merge_from_dict(options, allow_list_keys=True)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(pipeline=[
            ...     dict(type='MyResize'), dict(type='RandomDistortion')])

            >>> # Merge list element for replace args and add to list, only support list of type dict with key ``type``,
            >>> # if you add new list element, the list does not guarantee the order,
            >>> # it is only suitable for the case where the order of the list is not concerned.
            >>> cfg = Config(dict(pipeline=[
            ...     dict(type='Resize', size=224), dict(type='RandomDistortion')]))
            >>> options = dict(pipeline=[dict(type='Resize', size=256), dict(type='RandomFlip')])
            >>> cfg.merge_from_dict(options, allow_list_keys=True)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(pipeline=[
            ...     dict(type='Resize', size=256), dict(type='RandomDistortion'), dict(type='RandomFlip')])

            >>> # force usage
            >>> options = {'model.backbone.depth': 18,
            ...            'model.backbone.with_cp':True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet', depth=50))))
            >>> cfg.merge_from_dict(options, force=False)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...     model=dict(backbone=dict(type='ResNet', depth=50, with_cp=True)))

        Args:
            options (dict): dict of configs to merge from.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in ``options`` and will replace the element of the
              corresponding index in the config if the config is a list.
              Or you can directly replace args for list or add new list element,
              only support list of type dict with key ``type``,
              but if you add new list element, the list does not guarantee the order,
              It is only suitable for the case where the order of the list is not concerned.
              Default: True.
            force (bool): If True, existing key-value will be replaced by new given.
                If False, existing key-value will not be updated.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
        super(Config, self).__setattr__(
            '_cfg_dict',
            Config._merge_a_into_b(
                option_cfg_dict,
                cfg_dict,
                allow_list_keys=allow_list_keys,
                force=force))

    @staticmethod
    def _merge_a_into_b(a, b, allow_list_keys=False, force=True):
        """merge dict ``a`` into dict ``b`` (non-inplace).

        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.

        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in source ``a`` and will replace the element of the
              corresponding index in b if b is a list. Default: False.
            force (bool): If True, existing key-value will be replaced by new given.
                If False, existing key-value will not be updated.

        Returns:
            dict: The modified dict of ``b`` using ``a``.

        Examples:
            # Normally merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # Delete b first and merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # b is a list
            >>> Config._merge_a_into_b(
            ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)
            [{'a': 2}, {'b': 2}]

            # value of a and b are both list, only support list of type dict with key ``type``,
            # You can directly replace args for list or add new list element,
            # but if you add new list element, the list does not guarantee the order,
            # it is only suitable for the case where the order of the list is not concerned.
            >>> Config._merge_a_into_b(
            ...     {'k': [dict(a=2), dict(c=3)]}, {'k': [dict(a=1), dict(b=2)]}, True)
            {'k': [dict(a=2), dict(b=2), dict(c=3)]}

            # force is False
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2, b=2)), dict(obj=dict(a=1))), True, force=False)
            {'obj': {'a': 1, b=2}}
        """
        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f'Index {k} exceeds the length of list {b}')
                b[k] = Config._merge_a_into_b(
                    v, b[k], allow_list_keys, force=force)
            elif allow_list_keys and isinstance(v, list) and k in b:
                if not isinstance(b[k], list):
                    raise ValueError(
                        f'type mismatch {type(v)} and {type(b[k])} between a and b for key {k}'
                    )
                _is_dict_with_type = True
                for list_i in b[k] + v:
                    if not isinstance(list_i, dict) or 'type' not in list_i:
                        if k not in b or force:
                            b[k] = v
                        _is_dict_with_type = False
                if _is_dict_with_type:
                    res_list = []
                    added_index_bk, added_index_v = [], []
                    for i, b_li in enumerate(b[k]):
                        for j, a_lj in enumerate(v):
                            if a_lj['type'] == b_li['type']:
                                res_list.append(
                                    Config._merge_a_into_b(
                                        a_lj,
                                        b_li,
                                        allow_list_keys,
                                        force=force))
                                added_index_v.append(j)
                                added_index_bk.append(i)
                                break
                    rest_bk = [
                        b[k][i] for i in range(len(b[k]))
                        if i not in added_index_bk
                    ]
                    rest_v = [
                        v[i] for i in range(len(v)) if i not in added_index_v
                    ]
                    rest = rest_bk + rest_v
                    res_list += [
                        Config._merge_a_into_b(
                            rest[i], {}, allow_list_keys, force=force)
                        for i in range(len(rest))
                    ]
                    b[k] = res_list
            elif isinstance(v,
                            dict) and k in b and not v.pop(DELETE_KEY, False):
                allowed_types = (dict, list) if allow_list_keys else dict
                if not isinstance(b[k], allowed_types):
                    raise TypeError(
                        f'{k}={v} in child config cannot inherit from base '
                        f'because {k} is a dict in the child config but is of '
                        f'type {type(b[k])} in base config. You may set '
                        f'`{DELETE_KEY}=True` to ignore the base config')
                b[k] = Config._merge_a_into_b(
                    v, b[k], allow_list_keys, force=force)
            else:
                if k not in b or force:
                    b[k] = v
        return b

    def to_dict(self) -> Dict:
        """ Convert Config object to python dict
        """
        return self._cfg_dict.to_dict()

    def to_args(self, parse_fn, use_hyphen=True):
        """ Convert config obj to args using parse_fn

        Args:
            parse_fn: a function object, which takes args as input,
                such as ['--foo', 'FOO'] and return parsed args, an
                example is given as follows
                including literal blocks::
                    def parse_fn(args):
                        parser = argparse.ArgumentParser(prog='PROG')
                        parser.add_argument('-x')
                        parser.add_argument('--foo')
                        return parser.parse_args(args)
            use_hyphen (bool, optional): if set true, hyphen in keyname
                will be converted to underscore
        Return:
            args: arg object parsed by argparse.ArgumentParser
        """
        args = []
        for k, v in self._cfg_dict.items():
            arg_name = f'--{k}'
            if use_hyphen:
                arg_name = arg_name.replace('_', '-')
            if isinstance(v, bool) and v:
                args.append(arg_name)
            elif isinstance(v, (int, str, float)):
                args.append(arg_name)
                args.append(str(v))
            elif isinstance(v, list):
                args.append(arg_name)
                assert isinstance(v, (int, str, float, bool)), 'Element type in list ' \
                    f'is expected to be either int,str,float, but got type {v[0]}'
                args.append(str(v))
            else:
                raise ValueError(
                    'type in config file which supported to be '
                    'converted to args should be either bool, '
                    f'int, str, float or list of them but got type {v}')

        return parse_fn(args)


def check_config(cfg: Union[str, ConfigDict], is_training=False):
    """ Check whether configuration file is valid, If anything wrong, exception will be raised.

    Args:
        cfg (str or ConfigDict): Config file path or config object.
        is_training: indicate if checking training related elements
    """

    if isinstance(cfg, str):
        cfg = Config.from_file(cfg)

    def check_attr(attr_name, msg=''):
        assert hasattr(cfg, attr_name), f'Attribute {attr_name} is missing from ' \
            f'{ModelFile.CONFIGURATION}. {msg}'

    check_attr(ConfigFields.framework)
    check_attr(ConfigFields.task)
    check_attr(ConfigFields.pipeline)

    if is_training:
        check_attr(ConfigFields.model)
        check_attr(ConfigFields.train)
        check_attr(ConfigFields.preprocessor)
        check_attr(ConfigFields.evaluation)


class JSONIteratorEncoder(json.JSONEncoder):
    """Implement this method in order that supporting arbitrary iterators, it returns
        a serializable object for ``obj``, or calls the base implementation
        (to raise a ``TypeError``).

    """

    def default(self, obj):
        if isinstance(obj, FunctionType):
            return None
        try:
            iterable = iter(obj)
        except TypeError:
            pass
        else:
            return list(iterable)
        return json.JSONEncoder.default(self, obj)
