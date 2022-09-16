import os
from typing import Any, Dict

import json
import numpy as np

from modelscope.metainfo import TaskModels
from modelscope.models.builder import MODELS
from modelscope.models.nlp.task_models.task_model import \
    SingleBackboneTaskModelBase
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks

__all__ = ['SequenceClassificationModel']


@MODELS.register_module(
    Tasks.sentiment_classification, module_name=TaskModels.text_classification)
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

        backbone_cfg = self.cfg.backbone
        head_cfg = self.cfg.head

        # get the num_labels from label_mapping.json
        self.id2label = {}
        self.label_path = os.path.join(model_dir, 'label_mapping.json')
        if os.path.exists(self.label_path):
            with open(self.label_path) as f:
                self.label_mapping = json.load(f)
            self.id2label = {
                idx: name
                for name, idx in self.label_mapping.items()
            }
        head_cfg['num_labels'] = len(self.label_mapping)

        self.build_backbone(backbone_cfg)
        self.build_head(head_cfg)

    def forward(self, **input: Dict[str, Any]) -> Dict[str, np.ndarray]:
        outputs = super().forward(input)
        sequence_output, pooled_output = self.extract_backbone_outputs(outputs)
        outputs = self.head.forward(pooled_output)
        if 'labels' in input:
            loss = self.compute_loss(outputs, input['labels'])
            outputs.update(loss)
        return outputs

    def extract_logits(self, outputs):
        return outputs[OutputKeys.LOGITS].cpu().detach()

    def extract_backbone_outputs(self, outputs):
        sequence_output = None
        pooled_output = None
        if hasattr(self.backbone, 'extract_sequence_outputs'):
            sequence_output = self.backbone.extract_sequence_outputs(outputs)
        if hasattr(self.backbone, 'extract_pooled_outputs'):
            pooled_output = self.backbone.extract_pooled_outputs(outputs)
        return sequence_output, pooled_output

    def compute_loss(self, outputs, labels):
        loss = self.head.compute_loss(outputs, labels)
        return loss

    def postprocess(self, input, **kwargs):
        logits = self.extract_logits(input)
        probs = logits.softmax(-1).numpy()
        pred = logits.argmax(-1).numpy()
        logits = logits.numpy()
        res = {
            OutputKeys.PREDICTIONS: pred,
            OutputKeys.PROBABILITIES: probs,
            OutputKeys.LOGITS: logits
        }
        return res
