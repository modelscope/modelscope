# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np
from PIL import Image

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.human_normal_estimation,
    module_name=Pipelines.human_normal_estimation)
class HumanNormalEstimationPipeline(Pipeline):
    r""" Human Normal Estimation Pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline

    >>> estimator = pipeline(
    >>>        Tasks.human_normal_estimation, model='Damo_XR_Lab/cv_human_monocular-normal-estimation')
    >>> estimator(f"{model_dir}/tests/image_normal_estimation.jpg")
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
        """

        Args:
            input: string or ndarray or Image.Image

        Returns:
            data: dict including inference inputs
        """
        if isinstance(input, str):
            img = np.array(Image.open(input))
        if isinstance(input, Image.Image):
            img = np.array(input)

        img_h, img_w, img_ch = img.shape[0:3]

        if img_ch == 3:
            msk = np.full((img_h, img_w, 1), 255, dtype=np.uint8)
            img = np.concatenate((img, msk), axis=-1)

        H, W = 1024, 1024
        scale_factor = min(W / img_w, H / img_h)
        img = Image.fromarray(img)
        img = img.resize((int(img_w * scale_factor), int(img_h * scale_factor)), Image.LANCZOS)

        new_img = Image.new('RGBA', (W, H), color=(0, 0, 0, 0))
        paste_pos_w = (W - img.width) // 2
        paste_pos_h = (H - img.height) // 2
        new_img.paste(img, (paste_pos_w, paste_pos_h))

        bbox = (paste_pos_w, paste_pos_h, paste_pos_w + img.width, paste_pos_h + img.height)
        img = np.array(new_img)

        data = {'img': img[:, :, 0:3],
                'msk': img[:, :, -1],
                'bbox': bbox}

        return data

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.inference(input)
        return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.postprocess(inputs)
        normals = results[OutputKeys.NORMALS]

        normals_vis = (((normals + 1) * 0.5) * 255).astype(np.uint8)
        normals_vis = normals_vis[...,[2,1,0]]
        outputs = {
            OutputKeys.NORMALS: normals,
            OutputKeys.NORMALS_COLOR: normals_vis
        }
        return outputs
