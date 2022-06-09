# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod


class FormatHandler(metaclass=ABCMeta):
    # if `text_format` is True, file
    # should use text mode otherwise binary mode
    text_mode = True

    @abstractmethod
    def load(self, file, **kwargs):
        pass

    @abstractmethod
    def dump(self, obj, file, **kwargs):
        pass

    @abstractmethod
    def dumps(self, obj, **kwargs):
        pass
