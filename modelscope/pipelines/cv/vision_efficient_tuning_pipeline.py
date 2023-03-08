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
from modelscope.preprocessors import LoadImage, Preprocessor
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

        self.preprocessor = Preprocessor.from_pretrained(
            self.model.model_dir, **kwargs)

        if self.preprocessor is None:
            self.preprocessor = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def preprocess(self, inputs: Input, **preprocess_params) -> Dict[str, Any]:
        """ Preprocess method build from transforms or Preprocessor """
        in_key = 'img_path:FILE'
        other_in_keys = ['image']
        out_key = 'imgs'
        if isinstance(self.preprocessor, Preprocessor):
            if not isinstance(inputs, dict):
                inputs = {in_key: inputs}
            elif in_key not in inputs:
                for ik in other_in_keys:
                    if ik in inputs and isinstance(inputs[ik], str):
                        inputs = {in_key: inputs[ik]}
                        break
            data = self.preprocessor(inputs)
            result = {out_key: data[out_key].unsqueeze(0).to(self.device)}
        else:
            if isinstance(inputs, dict):
                for ik in [in_key] + other_in_keys:
                    if ik in inputs:
                        inputs = inputs[ik]
                        break
            img = LoadImage.convert_to_img(inputs)
            data = self.preprocessor(img)
            result = {out_key: data.unsqueeze(0).to(self.device)}
        return result

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            results = self.model(inputs)
            return results

    def postprocess(self, inputs: Dict[str, Any],
                    **post_params) -> Dict[str, Any]:
        """ Postprocess for classification """
        scores = inputs[OutputKeys.SCORES].cpu().numpy()
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
