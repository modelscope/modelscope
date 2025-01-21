# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Union

import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models.base import Tensor
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.image_quality_assessment_degradation.degradation_model import \
    DegradationIQA
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['ImageQualityAssessmentDegradation']


@MODELS.register_module(
    Tasks.image_quality_assessment_degradation,
    module_name=Models.image_quality_assessment_degradation)
class ImageQualityAssessmentDegradation(TorchModel):
    """
    Its architecture is based on the modified resnet50, output with blur degree, noise degree, compression degree.
    Reference: Rich features for perceptual quality assessment of UGC videos.
    """

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the image_quality_assessment_degradation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.

        """
        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir
        self.sigmoid_layer = nn.Sigmoid()
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))
        model_path = os.path.join(model_dir, ModelFile.TORCH_MODEL_FILE)

        self.model = DegradationIQA()
        self.model = self._load_pretrained(self.model, model_path)
        self.model.eval()

    def _train_forward(self, input: Tensor,
                       target: Tensor) -> Dict[str, Tensor]:
        losses = dict()
        return losses

    def _inference_forward(self, input: Tensor) -> Dict[str, Tensor]:
        # Todo
        # if img_tensor.shape[2]*img_tensor.shape[3] > 720*1280:
        #     img_tensor = torchvision.transforms.functional.resize(img_tensor, 720)
        preds = self.model(input, require_map=False)
        noise_degree, blur_degree, comp_degree = preds[1][:3]
        noise_degree, blur_degree, comp_degree = self.sigmoid_layer(
            noise_degree), self.sigmoid_layer(blur_degree), self.sigmoid_layer(
                comp_degree)
        if noise_degree > 0.3:
            noise_degree = noise_degree + 0.1
        if noise_degree >= 0.2 and noise_degree <= 0.3:
            noise_degree = (noise_degree - 0.2) * 2 + 0.2
        blur_degree = blur_degree + comp_degree / 2

        return {
            'noise_degree': noise_degree,
            'blur_degree': blur_degree,
            'comp_degree': comp_degree
        }

    def _evaluate_postprocess(self, input: Tensor, item_id: Tensor,
                              distortion_type: Tensor, target: Tensor,
                              **kwargs) -> Dict[str, list]:
        torch.cuda.empty_cache()
        with torch.no_grad():
            preds = self.model(input, require_map=False)
            noise_degree, blur_degree, comp_degree = preds[1][:3]
            noise_degree, blur_degree, comp_degree = self.sigmoid_layer(
                noise_degree), self.sigmoid_layer(
                    blur_degree), self.sigmoid_layer(comp_degree)
            noise_degree, blur_degree, comp_degree = noise_degree.cpu(
            ), blur_degree.cpu(), comp_degree.cpu()
            if noise_degree > 0.3:
                noise_degree = noise_degree + 0.1
            if noise_degree >= 0.2 and noise_degree <= 0.3:
                noise_degree = (noise_degree - 0.2) * 2 + 0.2
            blur_degree = blur_degree + comp_degree / 2
        del input
        target = target.cpu()
        torch.cuda.empty_cache()
        return {
            'item_id': item_id,
            'distortion_type': distortion_type,
            'noise_degree': noise_degree,
            'blur_degree': blur_degree,
            'comp_degree': comp_degree,
            'target': target
        }

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
