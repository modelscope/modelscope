# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np

from modelscope.metainfo import TaskModels
from modelscope.models.builder import MODELS
from modelscope.models.nlp.task_models.task_model import \
    SingleBackboneTaskModelBase
from modelscope.outputs import InformationExtractionOutput, OutputKeys
from modelscope.utils.constant import Tasks

__all__ = ['InformationExtractionModel']


@MODELS.register_module(
    Tasks.information_extraction,
    module_name=TaskModels.information_extraction)
@MODELS.register_module(
    Tasks.relation_extraction, module_name=TaskModels.information_extraction)
class InformationExtractionModel(SingleBackboneTaskModelBase):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the information extraction model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)

        self.build_backbone(self.backbone_cfg)
        self.build_head(self.head_cfg)

    def forward(self, **input: Dict[str, Any]) -> InformationExtractionOutput:
        outputs = super().forward(input)
        sequence_output = outputs.last_hidden_state
        outputs = self.head.forward(sequence_output, input['text'],
                                    input['offsets'])
        return InformationExtractionOutput(spo_list=outputs)
