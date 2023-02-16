# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from modelscope.metainfo import Pipelines
from modelscope.models.base.base_model import Model
from modelscope.models.cv.image_instance_segmentation import (
    CascadeMaskRCNNSwinModel, get_img_ins_seg_result)
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import (ImageInstanceSegmentationPreprocessor,
                                      build_preprocessor, load_image)
from modelscope.utils.config import Config
from modelscope.utils.constant import Fields, ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_segmentation,
    module_name=Pipelines.image_instance_segmentation)
class ImageInstanceSegmentationPipeline(Pipeline):

    def __init__(self,
                 model: Union[CascadeMaskRCNNSwinModel, str],
                 preprocessor: Optional[
                     ImageInstanceSegmentationPreprocessor] = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a image instance segmentation pipeline for prediction

        Args:
            model (CascadeMaskRCNNSwinModel | str): a model instance
            preprocessor (CascadeMaskRCNNSwinPreprocessor | None): a preprocessor instance
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

        if preprocessor is None:
            assert isinstance(self.model, Model), \
                f'please check whether model config exists in {ModelFile.CONFIGURATION}'
            config_path = os.path.join(self.model.model_dir,
                                       ModelFile.CONFIGURATION)
            cfg = Config.from_file(config_path)
            self.preprocessor = build_preprocessor(cfg.preprocessor, Fields.cv)
        else:
            self.preprocessor = preprocessor

        self.preprocessor.eval()
        self.model.eval()

    def _collate_fn(self, data):
        # don't require collating
        return data

    def preprocess(self, input: Input, **preprocess_params) -> Dict[str, Any]:
        filename = None
        img = None
        if isinstance(input, str):
            filename = input
            img = np.array(load_image(input))
            img = img[:, :, ::-1]  # convert to bgr
        elif isinstance(input, Image.Image):
            img = np.array(input.convert('RGB'))
            img = img[:, :, ::-1]  # convert to bgr
        elif isinstance(input, np.ndarray):
            if len(input.shape) == 2:
                img = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError(f'input should be either str, PIL.Image,'
                            f' np.array, but got {type(input)}')

        result = {
            'img': img,
            'img_shape': img.shape,
            'ori_shape': img.shape,
            'img_fields': ['img'],
            'img_prefix': '',
            'img_info': {
                'filename': filename,
                'ann_file': None,
                'classes': None
            },
        }
        result = self.preprocessor(result)

        # stacked as a batch
        result['img'] = torch.stack([result['img']], dim=0)
        result['img_metas'] = [result['img_metas'].data]

        return result

    def forward(self, input: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            output = self.model(input)
        return output

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = get_img_ins_seg_result(
            img_seg_result=inputs['eval_result'][0],
            class_names=self.model.model.classes)
        return result
