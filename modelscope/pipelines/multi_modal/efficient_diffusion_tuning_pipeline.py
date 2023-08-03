# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.efficient_diffusion_tuning,
    module_name=Pipelines.efficient_diffusion_tuning)
class EfficientDiffusionTuningPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a diffusion efficient tuning pipeline for prediction
        Args:
            model: model id on modelscope hub.
        Example:
            >>> from modelscope.pipelines import pipeline
            >>> petl_pipeline = pipeline('efficient-diffusion-tuning',
                'damo/cv_vitb16_classification_vision-efficient-tuning-adapter')
            >>> result = petl_pipeline(
                'data/test/images/vision_efficient_tuning_test_1.png')
            >>> print(f'Output: {result}.')
        """
        super().__init__(model=model, **kwargs)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.eval()
        self.preprocessor = transforms.Compose([
            transforms.Resize(
                512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def preprocess(self, inputs: Input, **preprocess_params) -> Dict[str, Any]:
        """ Preprocess method build from transforms or Preprocessor """
        assert isinstance(inputs, dict)
        result = {}
        if 'cond' in inputs:
            img = LoadImage.convert_to_img(inputs['cond'])
            data = self.preprocessor(img)
            result['cond'] = data.unsqueeze(0).to(self.device)
        if 'prompt' in inputs:
            result['prompt'] = inputs['prompt']
        return result

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            results = self.model(**inputs, **forward_params)
            return results

    def postprocess(self, inputs: Dict[str, Any],
                    **post_params) -> Dict[str, Any]:
        images = []
        for idx, img in enumerate(inputs):
            if isinstance(img, Image.Image):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                images.append(img)
                cv2.imwrite(f'{self.model.tuner_name}_{idx}.jpg', img)
        return {OutputKeys.OUTPUT_IMGS: images}

    def _sanitize_parameters(self, **pipeline_parameters):
        return pipeline_parameters, pipeline_parameters, pipeline_parameters
