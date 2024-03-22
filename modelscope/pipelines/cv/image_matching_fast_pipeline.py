# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Union

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_matching, module_name=Pipelines.image_matching_fast)
class ImageMatchingFastPipeline(Pipeline):
    """ Image Matching Pipeline.

    Examples:

    >>> from modelscope.outputs import OutputKeys
    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks

    >>> task = 'image-matching'
    >>> model_id = 'Damo_XR_Lab/cv_transformer_image-matching_fast'

    >>> input_location =  [[
    >>>        'data/test/images/image_matching1.jpg',
    >>>         'data/test/images/image_matching1.jpg',
    >>>    ]]
    >>> estimator = pipeline(task, model=model_id)
    >>> result = estimator(input_location)
    >>> kpts0, kpts1, confidence = result[0][OutputKeys.MATCHES]
    >>> print(f'Found {len(kpts0)} matches')
    """

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image matching pipeline fast for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)

        # check if cuda is available
        if not torch.cuda.is_available():
            raise RuntimeError(
                'Cuda is not available. Image matching model only supports cuda.'
            )

        logger.info('image matching model, pipeline init')

    def load_image(self, img_name):
        image_loader = LoadImage(backend='cv2')
        img = image_loader(img_name)['img']
        return img

    def preprocess(self, input: Input):
        assert len(input) == 2, 'input should be a list of two images'
        img1 = self.load_image(input[0])
        img2 = self.load_image(input[1])

        return {'image0': img1, 'image1': img2}

    def forward(self, input: Dict[str, Any]) -> list:
        results = self.model.inference(input)
        return results

    def postprocess(self, inputs: list) -> Dict[str, Any]:
        results = self.model.postprocess(inputs)
        matches = results[OutputKeys.MATCHES]

        kpts0 = matches['kpts0'].detach().cpu().numpy()
        kpts1 = matches['kpts1'].detach().cpu().numpy()
        confidence = matches['confidence'].detach().cpu().numpy()

        outputs = {
            OutputKeys.MATCHES: [kpts0, kpts1, confidence],
        }

        return outputs

    def __call__(self, input, **kwargs):
        """
        Match two images and return the matched keypoints and confidence.

        Args:
            input (`List[List[str]]`): A list of two image paths.

        Return:
            A list of result.
            The list contain the following values:

            - kpts0 -- Matched keypoints in the first image
            - kpts1 -- Matched keypoints in the second image
            - confidence -- Confidence of the match
        """
        return super().__call__(input, **kwargs)
