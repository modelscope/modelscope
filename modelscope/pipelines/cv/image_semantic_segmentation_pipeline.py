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
from modelscope.preprocessors import load_image
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_segmentation,
    module_name=Pipelines.image_semantic_segmentation)
class ImageSemanticSegmentationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image semantic segmentation pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)

        logger.info('semantic segmentation model, pipeline init')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        from mmdet.datasets.pipelines import Compose
        from mmcv.parallel import collate, scatter
        from mmdet.datasets import replace_ImageToTensor

        cfg = self.model.cfg
        # build the data pipeline

        if isinstance(input, str):
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
            img = np.array(load_image(input))
            img = img[:, :, ::-1]  # convert to bgr
        elif isinstance(input, PIL.Image.Image):  # BGR
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
            img = np.array(input)[:, :, ::-1]
        elif isinstance(input, np.ndarray):
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
            if len(input.shape) == 2:
                img = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
            else:
                img = input
        else:
            raise TypeError(f'input should be either str, PIL.Image,'
                            f' np.array, but got {type(input)}')

        # collect data
        data = dict(img=img)
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        test_pipeline = Compose(cfg.data.test.pipeline)

        data = test_pipeline(data)
        # copy from mmdet_model collect data
        data = collate([data], samples_per_gpu=1)
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
        data['img'] = [img.data[0] for img in data['img']]
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [next(self.model.parameters()).device])[0]

        return data

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.inference(input)
        return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.postprocess(inputs)
        outputs = {
            OutputKeys.MASKS: results[OutputKeys.MASKS],
            OutputKeys.LABELS: results[OutputKeys.LABELS],
            OutputKeys.SCORES: results[OutputKeys.SCORES]
        }

        return outputs
