# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np
import torch

from modelscope.metainfo import TaskModels
from modelscope.models.builder import MODELS
from modelscope.models.nlp.task_models.task_model import \
    SingleBackboneTaskModelBase
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks
from modelscope.utils.hub import parse_label_mapping
from modelscope.utils.tensor_utils import (torch_nested_detach,
                                           torch_nested_numpify)

__all__ = ['TokenClassificationModel']


@MODELS.register_module(
    Tasks.token_classification, module_name=TaskModels.token_classification)
class TokenClassificationModel(SingleBackboneTaskModelBase):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the token classification model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        if 'base_model_prefix' in kwargs:
            self._base_model_prefix = kwargs['base_model_prefix']

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
        labels = None
        if OutputKeys.LABEL in input:
            labels = input.pop(OutputKeys.LABEL)
        elif OutputKeys.LABELS in input:
            labels = input.pop(OutputKeys.LABELS)

        outputs = super().forward(input)
        sequence_output, pooled_output = self.extract_backbone_outputs(outputs)
        outputs = self.head.forward(sequence_output)
        if labels in input:
            loss = self.compute_loss(outputs, labels)
            outputs.update(loss)
        return outputs

    def extract_logits(self, outputs):
        return outputs[OutputKeys.LOGITS].cpu().detach()

    def extract_backbone_outputs(self, outputs):
        sequence_output = None
        pooled_output = None
        if hasattr(self.backbone, 'extract_sequence_outputs'):
            sequence_output = self.backbone.extract_sequence_outputs(outputs)
        return sequence_output, pooled_output

    def postprocess(self, input, **kwargs):
        logits = self.extract_logits(input)
        pred = torch.argmax(logits[0], dim=-1)
        pred = torch_nested_numpify(torch_nested_detach(pred))
        logits = torch_nested_numpify(torch_nested_detach(logits))
        res = {OutputKeys.PREDICTIONS: pred, OutputKeys.LOGITS: logits}
        return res
