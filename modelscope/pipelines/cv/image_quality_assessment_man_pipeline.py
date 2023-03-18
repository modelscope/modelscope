# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import tempfile
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
from torchvision import transforms

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_quality_assessment_man import \
    ImageQualityAssessmentMAN
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.preprocessors.cv import ImageQualityAssessmentMANPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_quality_assessment_mos,
    module_name=Pipelines.image_quality_assessment_man)
class ImageQualityAssessmentMANPipeline(Pipeline):
    """ Image Quality Assessment MAN Pipeline which will use Multi-dimension Attention Network
        to return Mean Opinion Score (MOS) for the input image.

        Example:

        ```python
        >>> from modelscope.pipelines import pipeline
        >>> from modelscope.outputs import OutputKeys
        >>> from modelscope.utils.constant import Tasks

        >>> test_image = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/dogs.jpg'
        >>> assessment_predictor = pipeline(Tasks.image_quality_assessment_man, \
            model='damo/cv_man_image-quality-assessment')
        >>> out_mos = assessment_predictor(test_image)[OutputKeys.SCORE]
        >>> print('Pipeline: the output mos is {}'.format(out_mos))

        ```
        """

    def __init__(self,
                 model: Union[ImageQualityAssessmentMAN, str],
                 preprocessor=ImageQualityAssessmentMANPreprocessor(),
                 **kwargs):
        """
        use `model` to create image quality assessment man pipeline for prediction
        Args:
            model: model id on modelscope hub or `ImageQualityAssessmentMAN` Model.
            preprocessor: preprocessor for input image

        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        logger.info('load MANIQA model done')

    @torch.no_grad()
    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        inference for image quality assessment prediction
        Args:
            input: dict including torch tensor.

        """
        outputs = self.model.forward({'input': input['input']})['output'].cpu()
        return {OutputKeys.SCORE: outputs.item()}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
