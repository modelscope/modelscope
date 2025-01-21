# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import albumentations as A
import cv2
import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import depth_to_color
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_depth_estimation,
    module_name=Pipelines.image_bts_depth_estimation)
class ImageBTSDepthEstimationPipeline(Pipeline):
    r""" Image depth estimation pipeline of BTS model.

        Examples:

        >>> import cv2
        >>> from modelscope.outputs import OutputKeys
        >>> from modelscope.pipelines import pipeline
        >>> from modelscope.utils.constant import Tasks

        >>> estimator = pipeline(Tasks.image_depth_estimation, 'damo/cv_densenet161_image-depth-estimation_bts')
        >>> result = estimator(
            "https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_depth_estimation_kitti_007517.png")
        >>> cv2.imwrite('result_depth_color.jpg', result[OutputKeys.DEPTHS_COLOR])
        >>> cv2.imwrite('result_depth.jpg', result[OutputKeys.DEPTHS])
        >>>
        """

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image depth estimation pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        self.transform = A.Compose([A.Normalize(always_apply=True)])

        logger.info('BTS depth estimation model, pipeline init')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)

        h, w, _ = img.shape
        top, left = int(h - 352), int((w - 1216) / 2)
        img = img[top:top + 352, left:left + 1216]

        img = self.transform(image=img)['image']
        img = torch.tensor(img).float().transpose(0, 2).transpose(1, 2)

        imgs = img[None, ...]
        data = {'imgs': imgs}

        return data

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.inference(input)
        return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.postprocess(inputs)
        depths = results[OutputKeys.DEPTHS].detach().cpu()
        depths = np.asarray(
            np.squeeze(
                (255 - torch.clamp_max(depths * 4, 250)).byte().numpy()),
            np.uint8)
        depths_color = depth_to_color(depths)

        outputs = {
            OutputKeys.DEPTHS: depths,
            OutputKeys.DEPTHS_COLOR: depths_color
        }

        return outputs
