# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torchvision.transforms as T

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_instance_segmentation import FastInst
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_segmentation, module_name=Pipelines.fast_instance_segmentation)
class FastInstanceSegmentationPipeline(Pipeline):

    def __init__(self,
                 model: Union[FastInst, str],
                 preprocessor: Optional = None,
                 **kwargs):
        r"""The inference pipeline for fastinst models.

        The model outputs a dict with keys of `scores`, `labels`, and `masks`.

        Args:
            model (`str` or `Model` or module instance): A model instance or a model local dir
                or a model id in the model hub.
            preprocessor (`Preprocessor`, `optional`): A Preprocessor instance.
            kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.

        Examples:
            >>> from modelscope.outputs import OutputKeys
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline('image-segmentation',
                model='damo/cv_resnet50_fast-instance-segmentation_coco')
            >>> input_img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_instance_segmentation.jpg'
            >>> print(pipeline_ins(input_img)[OutputKeys.LABELS])
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model.eval()

    def _get_preprocess_shape(self, oldh, oldw, short_edge_length, max_size):
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def preprocess(self,
                   input: Input,
                   min_size=640,
                   max_size=1333) -> Dict[str, Any]:
        image = LoadImage.convert_to_img(input)
        w, h = image.size[:2]
        dataset_dict = {'width': w, 'height': h}
        new_h, new_w = self._get_preprocess_shape(h, w, min_size, max_size)
        test_transforms = T.Compose([
            T.Resize((new_h, new_w)),
            T.ToTensor(),
        ])
        image = test_transforms(image)
        dataset_dict['image'] = image * 255.
        result = {'batched_inputs': [dataset_dict]}
        return result

    def forward(self, input: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            output = self.model(**input)
        return output

    def postprocess(self,
                    inputs: Dict[str, Any],
                    score_thr=0.5) -> Dict[str, Any]:
        predictions = inputs['eval_result'][0]['instances']
        scores = predictions['scores'].detach().cpu().numpy()
        pred_masks = predictions['pred_masks'].detach().cpu().numpy()
        pred_classes = predictions['pred_classes'].detach().cpu().numpy()

        thresholded_idxs = np.array(scores) >= score_thr
        scores = scores[thresholded_idxs]
        pred_classes = pred_classes[thresholded_idxs]
        pred_masks = pred_masks[thresholded_idxs]

        results_dict = {
            OutputKeys.MASKS: [],
            OutputKeys.LABELS: [],
            OutputKeys.SCORES: []
        }
        for score, cls, mask in zip(scores, pred_classes, pred_masks):
            score = np.float64(score)
            label = self.model.classes[int(cls)]
            mask = np.array(mask, dtype=np.float64)

            results_dict[OutputKeys.SCORES].append(score)
            results_dict[OutputKeys.LABELS].append(label)
            results_dict[OutputKeys.MASKS].append(mask)

        return results_dict
