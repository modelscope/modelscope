# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.vision_efficient_tuning,
    module_name=Pipelines.vision_efficient_tuning)
class VisionEfficientTuningPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a vision efficient tuning pipeline for prediction
        Args:
            model: model id on modelscope hub.
        Example:
            >>> from modelscope.pipelines import pipeline
            >>> petl_pipeline = pipeline('vision-efficient-tuning',
                'damo/cv_vitb16_classification_vision-efficient-tuning-adapter')
            >>> result = petl_pipeline(
                'data/test/images/vision_efficient_tuning_test_1.png')
            >>> print(f'Output: {result}.')
        """
        super().__init__(model=model, **kwargs)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_img(input)
        data = self.transform(img).unsqueeze(0).to(self.device)
        return data

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            results = self.model(input)
            return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        scores = F.softmax(inputs, dim=1).cpu().numpy()
        pred_scores = np.sort(scores, axis=1)[0][::-1][:5]
        pred_labels = np.argsort(scores, axis=1)[0][::-1][:5]

        result = {
            'pred_score': [score for score in pred_scores],
            'pred_class': [self.model.CLASSES[label] for label in pred_labels]
        }

        outputs = {
            OutputKeys.SCORES: result['pred_score'],
            OutputKeys.LABELS: result['pred_class']
        }
        return outputs
