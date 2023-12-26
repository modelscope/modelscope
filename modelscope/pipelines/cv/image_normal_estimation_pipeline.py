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
    Tasks.image_normal_estimation,
    module_name=Pipelines.image_normal_estimation)
class ImageNormalEstimationPipeline(Pipeline):
    r""" Image Normal Estimation Pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline

    >>> estimator = pipeline(
    >>>        Tasks.image_normal_estimation, model='Damo_XR_Lab/cv_omnidata_image-normal-estimation_normal')
    >>> estimator("https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_normal_estimation.jpg")
    >>>   {
    >>>    "normals": array([[[0.09233217, 0.07563387, 0.08025375, ..., 0.06992684,
    >>>                       0.07490329, 0.14308228],
    >>>                       [0.07833742, 0.06736029, 0.07296766, ..., 0.09184352,
    >>>                       0.0800755 , 0.09726034],
    >>>                       [0.07676302, 0.06631223, 0.07067154, ..., 0.09527256,
    >>>                       0.09292313, 0.08056315],
    >>>                       ...,
    >>>                       [0.26432115, 0.29100573, 0.2956126 , ..., 0.2913087 ,
    >>>                       0.29201347, 0.29539976],
    >>>                       [0.24557455, 0.26430887, 0.28548756, ..., 0.2877307 ,
    >>>                       0.28856137, 0.2937242 ],
    >>>                       [0.26316068, 0.2718169 , 0.28436714, ..., 0.29435217,
    >>>                       0.29842147, 0.2943223 ]],
    >>>                      [[0.59257126, 0.6459297 , 0.66572756, ..., 0.68350476,
    >>>                       0.6882835 , 0.66579086],
    >>>                       [0.7054596 , 0.6592535 , 0.6728153 , ..., 0.6589912 ,
    >>>                       0.64541686, 0.63954735],
    >>>                       [0.6912665 , 0.6638877 , 0.67816293, ..., 0.6607329 ,
    >>>                       0.6472897 , 0.64633334],
    >>>                       ...,
    >>>                       [0.04231769, 0.04427819, 0.04816979, ..., 0.04485315,
    >>>                       0.04652229, 0.04869233],
    >>>                       [0.04601872, 0.03706329, 0.04397734, ..., 0.04522909,
    >>>                       0.04745695, 0.04823782],
    >>>                       [0.06671816, 0.0520605 , 0.0563788 , ..., 0.04913886,
    >>>                       0.04974678, 0.04954173]],
    >>>                      [[0.4338835 , 0.43240184, 0.43519282, ..., 0.36894026,
    >>>                       0.35207224, 0.33153164],
    >>>                       [0.4786287 , 0.4399531 , 0.4350407 , ..., 0.34690523,
    >>>                       0.3179497 , 0.26544768],
    >>>                       [0.47692937, 0.4416514 , 0.437603  , ..., 0.34660107,
    >>>                       0.3102659 , 0.27787644],
    >>>                       ...,
    >>>                       [0.49566334, 0.48355937, 0.48710674, ..., 0.4964854 ,
    >>>                       0.48945957, 0.49413157],
    >>>                       [0.490632  , 0.4706958 , 0.48100013, ..., 0.48724395,
    >>>                       0.4799561 , 0.48129278],
    >>>                       [0.49428058, 0.47433382, 0.4823783 , ..., 0.48930234,
    >>>                       0.48616886, 0.47176325]]], dtype=float32),
    >>>    'normals_color': array([[[ 23, 151, 110],
    >>>                             [ 19, 164, 110],
    >>>                             [ 20, 169, 110],
    >>>                             ...,
    >>>                             [ 17, 174,  94],
    >>>                             [ 19, 175,  89],
    >>>                             [ 36, 169,  84]],
    >>>                            [[ 19, 179, 122],
    >>>                             [ 17, 168, 112],
    >>>                             [ 18, 171, 110],
    >>>                             ...,
    >>>                             [ 23, 168,  88],
    >>>                             [ 20, 164,  81],
    >>>                             [ 24, 163,  67]],
    >>>                            [[ 19, 176, 121],
    >>>                             [ 16, 169, 112],
    >>>                             [ 18, 172, 111],
    >>>                             ...,
    >>>                             [ 24, 168,  88],
    >>>                             [ 23, 165,  79],
    >>>                             [ 20, 164,  70]],
    >>>                             ...,
    >>>                            [[ 67,  10, 126],
    >>>                             [ 74,  11, 123],
    >>>                             [ 75,  12, 124],
    >>>                             ...,
    >>>                             [ 74,  11, 126],
    >>>                             [ 74,  11, 124],
    >>>                             [ 75,  12, 126]],
    >>>                            [[ 62,  11, 125],
    >>>                             [ 67,   9, 120],
    >>>                             [ 72,  11, 122],
    >>>                             ...,
    >>>                             [ 73,  11, 124],
    >>>                             [ 73,  12, 122],
    >>>                             [ 74,  12, 122]],
    >>>                            [[ 67,  17, 126],
    >>>                             [ 69,  13, 120],
    >>>                             [ 72,  14, 123],
    >>>                             ...,
    >>>                             [ 75,  12, 124],
    >>>                             [ 76,  12, 123],
    >>>                             [ 75,  12, 120]]], dtype=uint8)}
    """

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image normal estimation pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)

        logger.info('normal estimation model, pipeline init')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input).astype(np.float32)
        H, W = 384, 384
        img = cv2.resize(img, [W, H])
        img = img.transpose(2, 0, 1) / 255.0
        imgs = img[None, ...]
        data = {'imgs': imgs}

        return data

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.inference(input)
        return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.postprocess(inputs)
        normals = results[OutputKeys.NORMALS]
        if isinstance(normals, torch.Tensor):
            normals = normals.detach().cpu().squeeze().numpy()
        normals_color = (np.transpose(normals,
                                      (1, 2, 0)) * 255).astype(np.uint8)
        outputs = {
            OutputKeys.NORMALS: normals,
            OutputKeys.NORMALS_COLOR: normals_color
        }

        return outputs
