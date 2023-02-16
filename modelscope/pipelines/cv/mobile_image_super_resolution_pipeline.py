# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import numpy as np
import skimage.color as sc
import torch
from torchvision import transforms

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.cv.super_resolution import ECBSRModel
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['MobileImageSuperResolutionPipeline']


@PIPELINES.register_module(
    Tasks.image_super_resolution,
    module_name=Pipelines.mobile_image_super_resolution)
class MobileImageSuperResolutionPipeline(Pipeline):

    def __init__(self,
                 model: Union[ECBSRModel, str],
                 preprocessor=None,
                 **kwargs):
        """The inference pipeline for all the image super-resolution tasks.

        Args:
            model (`str` or `Model` or module instance): A model instance or a model local dir
                or a model id in the model hub.
            preprocessor (`Preprocessor`, `optional`): A Preprocessor instance.
            kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.

        Example:
            >>> from modelscope.pipelines import pipeline
            >>> import cv2
            >>> from modelscope.outputs import OutputKeys
            >>> from modelscope.pipelines import pipeline
            >>> from modelscope.utils.constant import Tasks
            >>> sr = pipeline(Tasks.image_super_resolution, model='damo/cv_ecbsr_image-super-resolution_mobile')
            >>> result = sr('data/test/images/butterfly_lrx2_y.png')
            >>> cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model.eval()
        self.config = self.model.config

        self.y_input = self.model.config.model.y_input
        self.tensor_max_value = self.model.config.model.tensor_max_value

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        logger.info('load image mobile sr model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_img(input)

        if self.y_input:
            img = sc.rgb2ycbcr(img)[:, :, 0:1]

        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img).to(self._device)

        img = img.float()
        if self.tensor_max_value == 1.0:
            img /= 255.0

        result = {'input': img.unsqueeze(0)}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:

        def set_phase(model, is_train):
            if is_train:
                model.train()
            else:
                model.eval()

        is_train = False
        set_phase(self.model, is_train)
        with torch.no_grad():
            output = self.model(input)  # output Tensor

        return {'output_tensor': output['outputs']}

    def postprocess(self, input: Dict[str, Any]) -> Dict[str, Any]:
        output = input['output_tensor'].squeeze(0)
        if self.tensor_max_value == 1.0:
            output *= 255.0

        output = output.clamp(0, 255).to(torch.uint8)
        output = output.permute(1, 2, 0).contiguous().cpu().numpy()

        return {OutputKeys.OUTPUT_IMG: output}
