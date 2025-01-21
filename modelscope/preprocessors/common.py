# Copyright (c) Alibaba, Inc. and its affiliates.

import time
from collections.abc import Sequence
from typing import Mapping

import numpy as np
import torch

from modelscope.utils.registry import default_group
from .builder import PREPROCESSORS, build_preprocessor


@PREPROCESSORS.register_module()
class Compose(object):
    """Compose a data pipeline with a sequence of transforms.
    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
        profiling (bool, optional): If set True, will profile and
            print preprocess time for each step.
    """

    def __init__(self, transforms, field_name=None, profiling=False):
        assert isinstance(transforms, Sequence)
        self.profiling = profiling
        self.transforms = []
        self.field_name = field_name
        for transform in transforms:
            if isinstance(transform, dict):
                if self.field_name is None:
                    transform = build_preprocessor(transform, default_group)
                else:
                    # if not found key in field_name, try field_name=None(default_group)
                    try:
                        transform = build_preprocessor(transform, field_name)
                    except KeyError:
                        transform = build_preprocessor(transform,
                                                       default_group)
            elif callable(transform):
                pass
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')
            self.transforms.append(transform)

    def __call__(self, data):
        for t in self.transforms:
            if self.profiling:
                start = time.time()

            data = t(data)

            if self.profiling:
                print(f'{t} time {time.time()-start}')

            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PREPROCESSORS.register_module()
class ToTensor(object):
    """Convert target object to tensor.

    Args:
        keys (Sequence[str]): Key of data to be converted to Tensor.
            Only valid when data is type of `Mapping`. If `keys` is None,
            all values of keys ​​will be converted to tensor by default.
    """

    def __init__(self, keys=None):
        self.keys = keys

    def __call__(self, data):
        if isinstance(data, Mapping):
            if self.keys is None:
                self.keys = list(data.keys())

            for key in self.keys:
                if key in data:
                    data[key] = to_tensor(data[key])
        else:
            data = to_tensor(data)

        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PREPROCESSORS.register_module()
class Filter(object):
    """This is usually the last stage of the dataloader transform.
    Only data of reserved keys will be kept and passed directly to the model, others will be removed.

    Args:
        keys (Sequence[str]): Keys of data to be reserved, others will be removed.
    """

    def __init__(self, reserved_keys):
        self.reserved_keys = reserved_keys

    def __call__(self, data):
        assert isinstance(data, Mapping)

        reserved_data = {}
        for key in self.reserved_keys:
            if key in data:
                reserved_data[key] = data[key]

        return reserved_data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.reserved_keys})'


def to_numpy(data):
    """Convert objects of various python types to `numpy.ndarray`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data.numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return np.asarray(data)
    elif isinstance(data, int):
        return np.asarray(data, dtype=np.int64)
    elif isinstance(data, float):
        return np.asarray(data, dtype=np.float64)
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PREPROCESSORS.register_module()
class ToNumpy(object):
    """Convert target object to numpy.ndarray.

    Args:
        keys (Sequence[str]): Key of data to be converted to numpy.ndarray.
            Only valid when data is type of `Mapping`. If `keys` is None,
            all values of keys ​​will be converted to numpy.ndarray by default.
    """

    def __init__(self, keys=None):
        self.keys = keys

    def __call__(self, data):
        if isinstance(data, Mapping):
            if self.keys is None:
                self.keys = list(data.keys())

            for key in self.keys:
                if key in data:
                    data[key] = to_numpy(data[key])
        else:
            data = to_numpy(data)

        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PREPROCESSORS.register_module()
class Rename(object):
    """Change the name of the input keys to output keys, respectively.
    """

    def __init__(self, input_keys=[], output_keys=[]):
        self.input_keys = input_keys
        self.output_keys = output_keys

    def __call__(self, data):
        if isinstance(data, Mapping):
            for in_key, out_key in zip(self.input_keys, self.output_keys):
                if in_key in data and out_key not in data:
                    data[out_key] = data[in_key]
                    data.pop(in_key)
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PREPROCESSORS.register_module()
class Identity(object):

    def __init__(self):
        pass

    def __call__(self, item):
        return item
