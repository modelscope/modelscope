# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Any, Dict

from modelscope.utils.constant import ModeKeys


class Preprocessor(ABC):

    def __init__(self, *args, **kwargs):
        self._mode = ModeKeys.INFERENCE
        pass

    @abstractmethod
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value
