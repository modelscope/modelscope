# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np

from modelscope.metainfo import TaskModels
from modelscope.models.builder import MODELS
from modelscope.models.nlp.task_models.task_model import \
    SingleBackboneTaskModelBase
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks

__all__ = ['InformationExtractionModel']


@MODELS.register_module(
    Tasks.information_extraction,
    module_name=TaskModels.information_extraction)
class InformationExtractionModel(SingleBackboneTaskModelBase):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the information extraction model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)

        backbone_cfg = self.cfg.backbone
        head_cfg = self.cfg.head
        self.build_backbone(backbone_cfg)
        self.build_head(head_cfg)

    def forward(self, input: Dict[str, Any]) -> Dict[str, np.ndarray]:
        outputs = super().forward(input)
        sequence_output, pooled_output = self.extract_backbone_outputs(outputs)
        outputs = self.head.forward(sequence_output, input['text'],
                                    input['offsets'])
        return {OutputKeys.SPO_LIST: outputs}

    def extract_backbone_outputs(self, outputs):
        sequence_output = None
        pooled_output = None
        if hasattr(self.backbone, 'extract_sequence_outputs'):
            sequence_output = self.backbone.extract_sequence_outputs(outputs)
        return sequence_output, pooled_output
