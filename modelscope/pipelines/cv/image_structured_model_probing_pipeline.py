# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import math
import os
import os.path as osp
from typing import Any, Dict

import numpy as np
import torch
import torchvision.transforms as transforms
from mmcv.parallel import collate, scatter

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_classification,
    module_name=Pipelines.image_structured_model_probing)
class ImageStructuredModelProbingPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a vision middleware pipeline for prediction
        Args:
            model: model id on modelscope hub.
        Example:
            >>> from modelscope.pipelines import pipeline
            >>> recognition_pipeline = pipeline(self.task, self.model_id)
            >>> file_name = 'data/test/images/\
                image_structured_model_probing_test_image.jpg'
            >>> result = recognition_pipeline(file_name)
            >>> print(f'recognition output: {result}.')
        """
        super().__init__(model=model, **kwargs)
        self.model.eval()
        model_dir = os.path.join(model, 'food101-clip-vitl14-full.pt')
        model_file = torch.load(model_dir)
        self.label_map = model_file['meta_info']['label_map']
        logger.info('load model done')

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def preprocess(self, input: Input) -> Dict[str, Any]:

        img = LoadImage.convert_to_img(input)

        data = self.transform(img)
        data = collate([data], samples_per_gpu=1)
        if next(self.model.parameters()).is_cuda:
            data = scatter(data, [next(self.model.parameters()).device])[0]

        return data

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            results = self.model(input)
            return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        scores = torch.softmax(inputs, dim=1).cpu()
        labels = torch.argmax(scores, dim=1).cpu().tolist()
        label_names = [self.label_map[label] for label in labels]

        return {OutputKeys.LABELS: label_names, OutputKeys.SCORES: scores}
