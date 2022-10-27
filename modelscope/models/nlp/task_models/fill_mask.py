# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np

from modelscope.metainfo import TaskModels
from modelscope.models.builder import MODELS
from modelscope.models.nlp.bert import BertConfig
from modelscope.models.nlp.task_models.task_model import \
    SingleBackboneTaskModelBase
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks
from modelscope.utils.hub import parse_label_mapping

__all__ = ['FillMaskModel']


@MODELS.register_module(Tasks.fill_mask, module_name=TaskModels.fill_mask)
class FillMaskModel(SingleBackboneTaskModelBase):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the fill mask model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        if 'base_model_prefix' in kwargs:
            self._base_model_prefix = kwargs['base_model_prefix']

        self.build_backbone(self.backbone_cfg)
        self.build_head(self.head_cfg)

    def forward(self, **input: Dict[str, Any]) -> Dict[str, np.ndarray]:

        # backbone do not need labels, only head need for loss compute
        labels = input.pop(OutputKeys.LABELS, None)

        outputs = super().forward(input)
        sequence_output = outputs.last_hidden_state
        outputs = self.head.forward(sequence_output)

        if labels is not None:
            input[OutputKeys.LABELS] = labels
            loss = self.compute_loss(outputs, labels)
            outputs.update(loss)
        outputs[OutputKeys.INPUT_IDS] = input[OutputKeys.INPUT_IDS]
        return outputs
