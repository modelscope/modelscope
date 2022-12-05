# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

import torch
from torch import nn

from modelscope.utils.file_utils import func_receive_dict_inputs
from modelscope.utils.hub import parse_label_mapping
from modelscope.utils.logger import get_logger
from .base_model import Model

logger = get_logger()


class TorchModel(Model, torch.nn.Module):
    """ Base model interface for pytorch

    """

    def __init__(self, model_dir=None, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        torch.nn.Module.__init__(self)

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        # Adapting a model with only one dict arg, and the arg name must be input or inputs
        if func_receive_dict_inputs(self.forward):
            return self.postprocess(self.forward(args[0], **kwargs))
        else:
            return self.postprocess(self.forward(*args, **kwargs))

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def post_init(self):
        """
        A method executed at the end of each model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()

    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
