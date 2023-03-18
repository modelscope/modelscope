# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import Any, Dict

import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks
from .vision_efficient_tuning import VisionEfficientTuning


@MODELS.register_module(
    Tasks.vision_efficient_tuning, module_name=Models.vision_efficient_tuning)
class VisionEfficientTuningModel(TorchModel):
    """ The implementation of vision efficient tuning model based on TorchModel.

    This model is constructed with the following parts:
        - 'backbone': pre-trained backbone model with parameters.
        - 'head': classification head with fine-tuning.
    """

    def __init__(self, model_dir: str, **kwargs):
        """ Initialize a vision efficient tuning model.

        Args:
          model_dir: model id or path, where model_dir/pytorch_model.pt contains:
                    - 'backbone_weight': parameters of backbone.
                    - 'head_weight': parameters of head.
        """
        super().__init__(model_dir)

        self.model = VisionEfficientTuning(model_dir=model_dir, **kwargs)
        self.CLASSES = self.model.CLASSES

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """ Dynamic forward function of vision efficient tuning model.

        Args:
            input: the input data dict contanis:
                - imgs: (B, 3, H, W).
                - labels: (B), when training stage.
        """
        output = self.model(**input)
        return output
