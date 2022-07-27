# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import Dict, Union

from modelscope.models.base.base_model import Model
from modelscope.utils.config import ConfigDict
from modelscope.utils.logger import get_logger

logger = get_logger()

Tensor = Union['torch.Tensor', 'tf.Tensor']
Input = Union[Dict[str, Tensor], Model]


class Head(ABC):
    """
    The head base class is for the tasks head method definition

    """

    def __init__(self, **kwargs):
        self.config = ConfigDict(kwargs)

    @abstractmethod
    def forward(self, input: Input) -> Dict[str, Tensor]:
        """
        This method will use the output from backbone model to do any
        downstream tasks
        Args:
            input: The tensor output or a model from backbone model
            (text generation need a model as input)
        Returns: The output from downstream taks
        """
        pass

    @abstractmethod
    def compute_loss(self, outputs: Dict[str, Tensor],
                     labels) -> Dict[str, Tensor]:
        """
        compute loss for head during the finetuning

        Args:
            outputs (Dict[str, Tensor]):  the output from the model forward
        Returns:  the loss(Dict[str, Tensor]):
        """
        pass
