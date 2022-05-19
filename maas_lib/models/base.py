# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

Tensor = Union['torch.Tensor', 'tf.Tensor']


class Model(ABC):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.post_process(self.forward(input))

    @abstractmethod
    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pass

    def post_process(self, input: Dict[str, Tensor],
                     **kwargs) -> Dict[str, Tensor]:
        # model specific postprocess, implementation is optional
        # will be called in Pipeline and evaluation loop(in the future)
        return input

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *model_args, **kwargs):
        raise NotImplementedError('from_preatrained has not been implemented')
