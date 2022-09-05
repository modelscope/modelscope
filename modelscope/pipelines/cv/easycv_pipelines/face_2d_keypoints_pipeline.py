# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from .base import EasyCVPipeline


@PIPELINES.register_module(
    Tasks.face_2d_keypoints, module_name=Pipelines.face_2d_keypoints)
class Face2DKeypointsPipeline(EasyCVPipeline):
    """Pipeline for face 2d keypoints detection."""

    def __init__(self,
                 model: str,
                 model_file_pattern=ModelFile.TORCH_MODEL_FILE,
                 *args,
                 **kwargs):
        """
            model (str): model id on modelscope hub or local model path.
            model_file_pattern (str): model file pattern.
        """

        super(Face2DKeypointsPipeline, self).__init__(
            model=model,
            model_file_pattern=model_file_pattern,
            *args,
            **kwargs)

    def show_result(self, img, points, scale=2, save_path=None):
        return self.predict_op.show_result(img, points, scale, save_path)

    def __call__(self, inputs) -> Any:
        output = self.predict_op(inputs)[0][0]
        points = output['point']
        poses = output['pose']

        return {OutputKeys.KEYPOINTS: points, OutputKeys.POSES: poses}
