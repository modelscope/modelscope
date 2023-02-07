# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Union

import torch.cuda
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models.base import Tensor
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.image_quality_assessment_mos.censeo_ivqa_model import \
    CenseoIVQAModel
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['ImageQualityAssessmentMos']


@MODELS.register_module(
    Tasks.image_quality_assessment_mos,
    module_name=Models.image_quality_assessment_mos)
class ImageQualityAssessmentMos(TorchModel):

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

        self.model = CenseoIVQAModel(pretrained=False)
        self.model = self._load_pretrained(self.model, model_path)
        self.model.eval()

    def _train_forward(self, input: Tensor,
                       target: Tensor) -> Dict[str, Tensor]:
        losses = dict()
        return losses

    def _inference_forward(self, input: Tensor) -> Dict[str, Tensor]:
        return {'output': self.model(input).clamp(0, 1)}

    def _evaluate_postprocess(self, input: Tensor,
                              target: Tensor) -> Dict[str, list]:

        torch.cuda.empty_cache()
        with torch.no_grad():
            preds = self.model(input)
            preds = preds.clamp(0, 1).cpu()
        del input
        target = target.cpu()
        torch.cuda.empty_cache()
        return {'pred': preds, 'target': target}

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
