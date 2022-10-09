# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any

import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from .base import EasyCVPipeline


@PIPELINES.register_module(
    Tasks.image_segmentation, module_name=Pipelines.easycv_segmentation)
class EasyCVSegmentationPipeline(EasyCVPipeline):
    """Pipeline for easycv segmentation task."""

    def __init__(self, model: str, model_file_pattern='*.pt', *args, **kwargs):
        """
            model (str): model id on modelscope hub or local model path.
            model_file_pattern (str): model file pattern.
        """

        super(EasyCVSegmentationPipeline, self).__init__(
            model=model,
            model_file_pattern=model_file_pattern,
            *args,
            **kwargs)

    def __call__(self, inputs) -> Any:
        outputs = self.predict_op(inputs)

        semantic_result = outputs[0]['seg_pred']

        ids = np.unique(semantic_result)[::-1]
        legal_indices = ids != len(self.predict_op.CLASSES)  # for VOID label
        ids = ids[legal_indices]
        segms = (semantic_result[None] == ids[:, None, None])
        masks = [it.astype(np.int) for it in segms]
        labels_txt = np.array(self.predict_op.CLASSES)[ids].tolist()

        results = {
            OutputKeys.MASKS: masks,
            OutputKeys.LABELS: labels_txt,
            OutputKeys.SCORES: [0.999 for _ in range(len(labels_txt))]
        }
        return results
