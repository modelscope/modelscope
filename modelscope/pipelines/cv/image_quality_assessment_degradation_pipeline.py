# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import tempfile
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
from torchvision import transforms

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_quality_assessment_degradation import \
    ImageQualityAssessmentDegradation
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_quality_assessment_degradation,
    module_name=Pipelines.image_quality_assessment_degradation)
class ImageQualityAssessmentDegradationPipeline(Pipeline):
    """ Image Quality Assessment Degradation Pipeline which will return mean option score for the input image.

        Example:

        ```python
        >>> from modelscope.pipelines import pipeline
        >>> from modelscope.outputs import OutputKeys
        >>> from modelscope.utils.constant import Tasks

        >>> test_image = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/dogs.jpg'
        >>> assessment_predictor = pipeline(Tasks.image_quality_assessment_degradation, \
            model='damo/cv_resnet50_image-quality-assessment_degradation')
        >>> out_res = assessment_predictor(test_image)[OutputKeys.SCORES]
        >>> print('Pipeline: the output noise degree is {}, the output blur degree is {}, \
                the output compression degree is {}'.format(out_res[0], out_res[1], out_res[2]))

        ```
        """

    def __init__(self, model: Union[ImageQualityAssessmentDegradation, str],
                 **kwargs):
        """
        use `model` to create image quality assessment degradation pipeline for prediction
        Args:
            model: model id on modelscope hub or `ImageQualityAssessmentDegradation` Model.
            preprocessor: preprocessor for input image

        """
        super().__init__(model=model, **kwargs)

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        logger.info('load vqa-degradation model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_img(input)
        w, h = img.size
        if h * w < 1280 * 720:
            img = transforms.functional.resize(img, 720)
        test_transforms = transforms.Compose([transforms.ToTensor()])
        img = test_transforms(img).unsqueeze(0)
        result = {'src': img.to(self._device)}
        return result

    @torch.no_grad()
    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        inference for image quality assessment degradation prediction
        Args:
            input: dict including torch tensor.

        """
        outputs = self.model._inference_forward(input['src'])
        noise_degree, blur_degree, comp_degree = outputs['noise_degree'].cpu(
        ), outputs['blur_degree'].cpu(), outputs['comp_degree'].cpu()
        return {
            OutputKeys.SCORES:
            [noise_degree.item(),
             blur_degree.item(),
             comp_degree.item()],
            OutputKeys.LABELS: ['噪声强度', '模糊程度', '压缩强度']
        }

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
