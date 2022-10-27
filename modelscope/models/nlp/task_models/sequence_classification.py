# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np

from modelscope.metainfo import TaskModels
from modelscope.models.builder import MODELS
from modelscope.models.nlp.task_models.task_model import \
    SingleBackboneTaskModelBase
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks
from modelscope.utils.hub import parse_label_mapping

__all__ = ['SequenceClassificationModel']


@MODELS.register_module(
    Tasks.text_classification, module_name=TaskModels.text_classification)
class SequenceClassificationModel(SingleBackboneTaskModelBase):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the sequence classification model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        if 'base_model_prefix' in kwargs:
            self._base_model_prefix = kwargs['base_model_prefix']

        # get the num_labels from label_mapping.json
        self.id2label = {}
        # get the num_labels
        num_labels = kwargs.get('num_labels')
        if num_labels is None:
            label2id = parse_label_mapping(model_dir)
            if label2id is not None and len(label2id) > 0:
                num_labels = len(label2id)
            self.id2label = {id: label for label, id in label2id.items()}
        self.head_cfg['num_labels'] = num_labels

        self.build_backbone(self.backbone_cfg)
        self.build_head(self.head_cfg)

    def forward(self, **input: Dict[str, Any]) -> Dict[str, np.ndarray]:
        # backbone do not need labels, only head need for loss compute
        labels = input.pop(OutputKeys.LABELS, None)

        outputs = super().forward(input)
        pooled_output = outputs.pooler_output
        outputs = self.head.forward(pooled_output)
        if labels is not None:
            input[OutputKeys.LABELS] = labels
            loss = self.compute_loss(outputs, labels)
            outputs.update(loss)
        return outputs
