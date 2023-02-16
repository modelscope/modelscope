# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torchvision.transforms as T

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_human_parsing import (
    M2FP, center_to_target_size_test)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_segmentation, module_name=Pipelines.image_human_parsing)
class ImageHumanParsingPipeline(Pipeline):

    def __init__(self,
                 model: Union[M2FP, str],
                 preprocessor: Optional = None,
                 **kwargs):
        """use `model` and `preprocessor` to create an image human parsing
        pipeline for prediction

        Args:
            model (M2FPModel | str): a model instance
            preprocessor (None): a preprocessor instance
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
        if self.model.single_human:
            image = np.asarray(image)
            image, crop_box = center_to_target_size_test(
                image, self.model.input_single_human['sizes'][0])
            dataset_dict['image'] = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)))
            dataset_dict['crop_box'] = crop_box
        else:
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
            output = self.model(input)
        return output

    def postprocess(self,
                    inputs: Dict[str, Any],
                    score_thr=0.0) -> Dict[str, Any]:
        predictions = inputs['eval_result'][0]
        class_names = self.model.classes
        results_dict = {
            OutputKeys.MASKS: [],
            OutputKeys.LABELS: [],
            OutputKeys.SCORES: []
        }
        if 'sem_seg' in predictions:
            semantic_pred = predictions['sem_seg']
            semantic_seg = semantic_pred.argmax(dim=0).detach().cpu().numpy()
            semantic_pred = semantic_pred.sigmoid().detach().cpu().numpy()
            class_ids = np.unique(semantic_seg)
            for class_id in class_ids:
                label = class_names[class_id]
                mask = np.array(semantic_seg == class_id, dtype=np.float64)
                score = (mask * semantic_pred[class_id]).sum() / (
                    mask.sum() + 1)
                results_dict[OutputKeys.SCORES].append(score)
                results_dict[OutputKeys.LABELS].append(label)
                results_dict[OutputKeys.MASKS].append(mask)
        elif 'parsing' in predictions:
            parsing_res = predictions['parsing']
            part_outputs = parsing_res['part_outputs']
            human_outputs = parsing_res['human_outputs']

            # process semantic_outputs
            for output in part_outputs + human_outputs:
                score = output['score']
                label = class_names[output['category_id']]
                mask = (output['mask'] > 0).float().detach().cpu().numpy()
                if score > score_thr:
                    results_dict[OutputKeys.SCORES].append(score)
                    results_dict[OutputKeys.LABELS].append(label)
                    results_dict[OutputKeys.MASKS].append(mask)
        else:
            raise NotImplementedError

        return results_dict
