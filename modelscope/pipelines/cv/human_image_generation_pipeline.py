# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

from typing import Any, Dict

import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.human_image_generation import \
    human_image_generation_infer
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.human_image_generation, module_name=Pipelines.human_image_generation)
class FreqHPTForHumanImageGenerationPipeline(Pipeline):
    """ Human Image Generation Pipeline.
    Examples:
    >>> human_image_generation = pipeline(Tasks.human_image_generation, model='damo/cv_FreqHPT_human-image-generation')
    >>> input_images = {'source_img_path': '/your_path/source_img.jpg',
    >>>                 'target_pose_path': '/your_path/target_pose.txt'}
    >>> result = human_image_generation(input_images)
    >>> result[OutputKeys.OUTPUT_IMG]
    """

    def __init__(self, model: str, **kwargs):
        """
            use `model` to create human image generation pipeline for prediction
            Args:
                model: model id on modelscope hub.
            """

        super().__init__(model=model, **kwargs)
        self.model_path = model
        logger.info('load model done')
        if torch.cuda.is_available():
            self.device = 'cuda'
            logger.info('Use GPU')
        else:
            self.device = 'cpu'
            logger.info('Use CPU')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        return input

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        human_image_generation = human_image_generation_infer.infer(
            self.model, input['source_img_path'], input['target_pose_path'],
            self.device)
        return {OutputKeys.OUTPUT_IMG: human_image_generation}
