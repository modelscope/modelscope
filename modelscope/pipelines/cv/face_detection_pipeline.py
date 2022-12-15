# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict, List, Union

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.base.base_model import Model
from modelscope.models.cv.face_detection import ScrfdDetect, SCRFDPreprocessor
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.typing import Image

logger = get_logger()


@PIPELINES.register_module(
    Tasks.face_detection, module_name=Pipelines.face_detection)
class FaceDetectionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a face detection pipeline for prediction
        Args:
            model (`str` or `Model`): model_id or `ScrfdDetect` or `TinyMogDetect` model.
            preprocessor(`Preprocessor`, *optional*,  defaults to None): `SCRFDPreprocessor`.
        """
        super().__init__(model=model, **kwargs)
        config_path = osp.join(model, ModelFile.CONFIGURATION)
        cfg = Config.from_file(config_path)
        cfg_model = getattr(cfg, 'model', None)
        if cfg_model is None:
            # backward compatibility
            detector = ScrfdDetect(model_dir=model, **kwargs)
        else:
            assert isinstance(self.model,
                              Model), 'model object is not initialized.'
            detector = self.model.to(self.device)

        # backward compatibility
        if self.preprocessor is None:
            self.preprocessor = SCRFDPreprocessor()

        self.detector = detector

    def __call__(self, input: Union[Image, List[Image]], **kwargs):
        """
        Detect objects (bounding boxes or keypoints) in the image(s) passed as inputs.

        Args:
            input (`Image` or `List[Image]`):
                The pipeline handles three types of images:

                - A string containing an HTTP(S) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the
                same format.

        Return:
            A dictionary of result or a list of dictionary of result. If the input is an image, a dictionary
            is returned. If input is a list of image, a list of dictionary is returned.

            The dictionary contain the following keys:

            - **scores** (`List[float]`) -- The detection score for each card in the image.
            - **boxes** (`List[float]) -- The bounding boxe [x1, y1, x2, y2] of detected objects in in image's
                original size.
            - **keypoints** (`List[Dict[str, int]]`, optional) -- The corner kepoint [x1, y1, x2, y2, x3, y3, x4, y4]
                of detected object in image's original size.
        """
        return super().__call__(input, **kwargs)

    def preprocess(self, input: Image) -> Dict[str, Any]:
        result = self.preprocessor(input)

        # openmmlab model compatibility
        if 'img_metas' in result:
            from mmcv.parallel import collate, scatter
            result = collate([result], samples_per_gpu=1)
            if next(self.model.parameters()).is_cuda:
                # scatter to specified GPU
                result = scatter(result,
                                 [next(self.model.parameters()).device])[0]
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.detector(**input)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
