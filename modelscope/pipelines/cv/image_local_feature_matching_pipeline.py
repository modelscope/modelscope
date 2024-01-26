# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Union

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
    Tasks.image_local_feature_matching,
    module_name=Pipelines.image_local_feature_matching)
class ImageLocalFeatureMatchingPipeline(Pipeline):
    r""" Image Local Feature Matching Pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline

    >>> matcher = pipeline(Tasks.image_local_feature_matching,
    >>>    model='Damo_XR_Lab/cv_resnet-transformer_local-feature-matching_outdoor-data')
    >>> matcher([['https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_matching1.jpg',
    >>>          'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_matching2.jpg']])
    >>>   [{
    >>>    'matches': [array([[720.5    , 187.8    ],
    >>>        [707.4    , 198.23334],
    >>>        ...,
    >>>        [746.7    , 594.7    ],
    >>>        [759.8    , 594.7    ]], dtype=float32),
    >>>        array([[652.49744 ,  29.599142],
    >>>        [639.25287 ,  45.90798 ],
    >>>        [653.041   ,  43.399014],
    >>>        ...,
    >>>        [670.8787  , 547.8298  ],
    >>>        [608.5573  , 548.97815 ],
    >>>        [617.82574 , 548.601   ]], dtype=float32),
    >>>        array([0.25541496, 0.2781789 , 0.20776041, ..., 0.39656195, 0.7202848 ,
    >>>        0.37208357], dtype=float32)],
    >>>    'output_img': array([[[255, 255, 255],
    >>>         [255, 255, 255],
    >>>         [255, 255, 255],
    >>>         ...,
    >>>         [255, 255, 255],
    >>>         [255, 255, 255],
    >>>         [255, 255, 255]],
    >>>         ...,
    >>>        [[255, 255, 255],
    >>>         [255, 255, 255],
    >>>         [255, 255, 255],
    >>>         ...,
    >>>         [255, 255, 255],
    >>>         [255, 255, 255],
    >>>         [255, 255, 255]]], dtype=uint8)}]
    """

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image local feature matching pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)

    def load_image(self, img_name):
        img = LoadImage.convert_to_ndarray(img_name).astype(np.float32)
        img = img / 255.
        # convert rgb to gray
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        H, W = 480, 640
        h_scale, w_scale = H / img.shape[0], W / img.shape[1]
        img = cv2.resize(img, (W, H))
        return img, h_scale, w_scale

    def preprocess(self, input: Input):
        assert len(input) == 2, 'input should be a list of two images'

        img1, h_scale1, w_scale1 = self.load_image(input[0])

        img2, h_scale2, w_scale2 = self.load_image(input[1])

        img1 = torch.from_numpy(img1)[None][None].cuda().float()
        img2 = torch.from_numpy(img2)[None][None].cuda().float()
        return {
            'image0': img1,
            'image1': img2,
            'scale_info': [h_scale1, w_scale1, h_scale2, w_scale2]
        }

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.inference(input)
        return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.postprocess(inputs)
        matches = results[OutputKeys.MATCHES]

        kpts0 = matches['kpts0'].cpu().numpy()
        kpts1 = matches['kpts1'].cpu().numpy()
        conf = matches['conf'].cpu().numpy()
        scale_info = [v.cpu().numpy() for v in inputs['scale_info']]
        kpts0[:, 0] = kpts0[:, 0] / scale_info[1]
        kpts0[:, 1] = kpts0[:, 1] / scale_info[0]
        kpts1[:, 0] = kpts1[:, 0] / scale_info[3]
        kpts1[:, 1] = kpts1[:, 1] / scale_info[2]

        outputs = {
            OutputKeys.MATCHES: [kpts0, kpts1, conf],
            OutputKeys.OUTPUT_IMG: results[OutputKeys.OUTPUT_IMG]
        }

        return outputs
