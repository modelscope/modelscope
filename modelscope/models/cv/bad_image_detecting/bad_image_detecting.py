# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Union

import numpy as np
import torch.cuda
from torchvision import models

from modelscope.metainfo import Models
from modelscope.models.base import Tensor
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['BadImageDetecting']


@MODELS.register_module(
    Tasks.bad_image_detecting, module_name=Models.bad_image_detecting)
class BadImageDetecting(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the image_quality_assessment_mos model from the `model_dir` path.

        Args:
            model_dir (str): the model path.

        """
        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))
        model_path = os.path.join(model_dir, ModelFile.TORCH_MODEL_FILE)

        self.model = models.mobilenet_v2(
            pretrained=False, width_mult=0.35, num_classes=3)
        self.model = self._load_pretrained(self.model, model_path)
        self.model.eval()

    def _train_forward(self, input: Tensor,
                       target: Tensor) -> Dict[str, Tensor]:
        losses = dict()
        return losses

    def _inference_forward(self, input: Tensor) -> Dict[str, Tensor]:

        ret = self.model(input)

        return {'output': ret}

    def _evaluate_postprocess(self, input: Tensor,
                              target: Tensor) -> Dict[str, list]:
        torch.cuda.empty_cache()
        with torch.no_grad():
            preds = self.model(input)
            _, pred_ = torch.max(preds, dim=1)
        del input
        torch.cuda.empty_cache()
        return {'pred': pred_, 'target': target}

    def forward(self, inputs: Dict[str,
                                   Tensor]) -> Dict[str, Union[list, Tensor]]:
        """return the result by the model

        Args:
            inputs (Tensor): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
        """
        if self.training:
            return self._train_forward(**inputs)
        elif 'target' in inputs:
            return self._evaluate_postprocess(**inputs)
        else:
            return self._inference_forward(**inputs)
