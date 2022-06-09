# Copyright (c) Alibaba, Inc. and its affiliates.

import time
from collections.abc import Sequence

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
                    transform = build_preprocessor(transform, field_name)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

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
