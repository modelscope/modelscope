# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict

import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.base.base_model import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks


@PIPELINES.register_module(
    Tasks.image_fewshot_detection,
    module_name=Pipelines.image_fewshot_detection)
class ImageDefrcnDetectionPipeline(Pipeline):
    r"""
    Image DeFRCN few-shot detection Pipeline. Given a image, pipeline will return the detection results on the image.

    Examples:

        >>> from modelscope.pipelines import pipeline
        >>> detector = pipeline('image-fewshot-detection', 'damo/cv_resnet101_detection_fewshot-defrcn')
        >>> detector('/Path/Image')
        >>> {'scores': [0.8307567834854126, 0.1606406420469284],
        >>>  'labels': ['person', 'dog'],
        >>>  'boxes': [[27.391937255859375, 0.0, 353.0, 500.0],
        >>>            [64.22428131103516, 229.2884521484375, 213.90573120117188, 370.0657958984375]]}
    """

    def __init__(self, model: str, **kwargs):
        """
            model: model id on modelscope hub.
        """
        super().__init__(model=model, auto_collate=False, **kwargs)

        assert isinstance(
            self.model, Model
        ), f'please check whether model config exists in {ModelFile.CONFIGURATION}'

        model_path = os.path.join(self.model.model_dir,
                                  ModelFile.TORCH_MODEL_FILE)
        self.model.model = self._load_pretrained(
            self.model.model, model_path, self.model.model_cfg.MODEL.DEVICE)

    def _load_pretrained(self, net, load_path, device='cuda', strict=True):

        load_net = torch.load(load_path, map_location=device)
        if 'scheduler' in load_net:
            del load_net['scheduler']
        if 'optimizer' in load_net:
            del load_net['optimizer']
        if 'iteration' in load_net:
            del load_net['iteration']
        net.load_state_dict(load_net['model'], strict=strict)

        return net

    def preprocess(self, input: Input) -> Dict[str, Any]:

        img = LoadImage.convert_to_ndarray(input)

        image = img[..., ::-1].copy()  # rgb to bgr
        tim = torch.Tensor(image).permute(2, 0, 1)  # hwc to chw

        result = {'image': tim, 'image_numpy': image}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:

        outputs = self.model.inference(input)
        result = {'data': outputs}
        return result

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if inputs['data'] is None:
            outputs = {
                OutputKeys.SCORES: [],
                OutputKeys.LABELS: [],
                OutputKeys.BOXES: []
            }
            return outputs

        objects = inputs['data']['instances'].get_fields()
        labels, bboxes = [], []
        for label, box in zip(objects['pred_classes'], objects['pred_boxes']):
            labels.append(self.model.config.model.classes[label])
            bboxes.append(box.tolist())

        scores = objects['scores'].tolist()

        outputs = {
            OutputKeys.SCORES: scores,
            OutputKeys.LABELS: labels,
            OutputKeys.BOXES: bboxes
        }
        return outputs
