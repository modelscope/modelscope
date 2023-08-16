# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

# Registry class & build_from_config function partially modified from
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py
# Copyright 2018-2020 Open-MMLab. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import inspect
import warnings


def build_from_config(cfg, registry, **kwargs):
    """ Default builder function.

    Args:
        cfg (dict): A dict which contains parameters passes to target class or function.
            Must contains key 'type', indicates the target class or function name.
        registry (Registry): An registry to search target class or function.
        kwargs (dict, optional): Other params not in config dict.

    Returns:
        Target class object or object returned by invoking function.

    Raises:
        TypeError:
        KeyError:
        Exception:
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'config must be type dict, got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(f'config must contain key type, got {cfg}')
    if not isinstance(registry, Registry):
        raise TypeError(
            f'registry must be type Registry, got {type(registry)}')

    cfg = copy.deepcopy(cfg)

    req_type = cfg.pop('type')
    req_type_entry = req_type
    if isinstance(req_type, str):
        req_type_entry = registry.get(req_type)
        if req_type_entry is None:
            raise KeyError(f'{req_type} not found in {registry.name} registry')

    if kwargs is not None:
        cfg.update(kwargs)

    if inspect.isclass(req_type_entry):
        try:
            return req_type_entry(**cfg)
        except Exception as e:
            raise Exception(f'Failed to init class {req_type_entry}, with {e}')
    elif inspect.isfunction(req_type_entry):
        try:
            return req_type_entry(**cfg)
        except Exception as e:
            raise Exception(
                f'Failed to invoke function {req_type_entry}, with {e}')
    else:
        raise TypeError(
            f'type must be str or class, got {type(req_type_entry)}')


class Registry(object):
    """ A registry maps key to classes or functions.

    Example:
         >>> MODELS = Registry('MODELS')
         >>> @MODELS.register_class()
         >>> class ResNet(object):
         >>>     pass
         >>> resnet = MODELS.build(dict(type="ResNet"))
         >>>
         >>> import torchvision
         >>> @MODELS.register_function("InceptionV3")
         >>> def get_inception_v3(pretrained=False, progress=True):
         >>>     return torchvision.models.inception_v3(pretrained=pretrained, progress=progress)
         >>> inception_v3 = MODELS.build(dict(type='InceptionV3', pretrained=True))

    Args:
        name (str): Registry name.
        build_func (func, None): Instance construct function. Default is build_from_config.
        allow_types (tuple): Indicates how to construct the instance, by constructing class or invoking function.
    """

    def __init__(self,
                 name,
                 build_func=None,
                 allow_types=('class', 'function')):
        self.name = name
        self.allow_types = allow_types
        self.class_map = {}
        self.func_map = {}
        self.build_func = build_func or build_from_config

    def get(self, req_type):
        return self.class_map.get(req_type) or self.func_map.get(req_type)

    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)

    def register_class(self, name=None):

        def _register(cls):
            if not inspect.isclass(cls):
                raise TypeError(f'Module must be type class, got {type(cls)}')
            if 'class' not in self.allow_types:
                raise TypeError(
                    f'Register {self.name} only allows type {self.allow_types}, got class'
                )
            module_name = name or cls.__name__
            if module_name in self.class_map:
                warnings.warn(
                    f'Class {module_name} already registered by {self.class_map[module_name]}, '
                    f'will be replaced by {cls}')
            self.class_map[module_name] = cls
            return cls

        return _register

    def register_function(self, name=None):

        def _register(func):
            if not inspect.isfunction(func):
                raise TypeError(
                    f'Registry must be type function, got {type(func)}')
            if 'function' not in self.allow_types:
                raise TypeError(
                    f'Registry {self.name} only allows type {self.allow_types}, got function'
                )
            func_name = name or func.__name__
            if func_name in self.class_map:
                warnings.warn(
                    f'Function {func_name} already registered by {self.func_map[func_name]}, '
                    f'will be replaced by {func}')
            self.func_map[func_name] = func
            return func

        return _register

    def _list(self):
        keys = sorted(list(self.class_map.keys()) + list(self.func_map.keys()))
        descriptions = []
        for key in keys:
            if key in self.class_map:
                descriptions.append(f'{key}: {self.class_map[key]}')
            else:
                descriptions.append(
                    f"{key}: <function '{self.func_map[key].__module__}.{self.func_map[key].__name__}'>"
                )
        return '\n'.join(descriptions)

    def __repr__(self):
        description = self._list()
        description = '\n'.join(['\t' + s for s in description.split('\n')])
        return f'{self.__class__.__name__} [{self.name}], \n' + description
