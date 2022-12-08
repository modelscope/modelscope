# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np

from modelscope.metainfo import TaskModels
from modelscope.models.builder import MODELS
from modelscope.models.nlp.task_models.task_model import \
    SingleBackboneTaskModelBase
from modelscope.outputs import FeatureExtractionOutput, OutputKeys
from modelscope.utils.constant import Tasks

__all__ = ['FeatureExtractionModel']


@MODELS.register_module(
    Tasks.feature_extraction, module_name=TaskModels.feature_extraction)
class FeatureExtractionModel(SingleBackboneTaskModelBase):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the fill mask model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        if 'base_model_prefix' in kwargs:
            self._base_model_prefix = kwargs['base_model_prefix']

        self.build_backbone(self.backbone_cfg)

    def forward(self, **input: Dict[str, Any]) -> FeatureExtractionOutput:
        # backbone do not need labels, only head need for loss compute
        input.pop(OutputKeys.LABELS, None)
        outputs = super().forward(input)
        sequence_output = outputs.last_hidden_state
        return FeatureExtractionOutput(text_embedding=sequence_output)
