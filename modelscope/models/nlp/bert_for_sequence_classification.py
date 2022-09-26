# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict

import json
import numpy as np

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks

__all__ = ['BertForSequenceClassification']


@MODELS.register_module(Tasks.text_classification, module_name=Models.bert)
class BertForSequenceClassification(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        # Model.__init__(self, model_dir, model_cls, first_sequence, *args, **kwargs)
        # Predictor.__init__(self, *args, **kwargs)
        """initialize the sequence classification model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """

        super().__init__(model_dir, *args, **kwargs)
        import torch
        from easynlp.appzoo import SequenceClassification
        from easynlp.core.predictor import get_model_predictor
        self.model = get_model_predictor(
            model_dir=self.model_dir,
            model_cls=SequenceClassification,
            input_keys=[('input_ids', torch.LongTensor),
                        ('attention_mask', torch.LongTensor),
                        ('token_type_ids', torch.LongTensor)],
            output_keys=['predictions', 'probabilities', 'logits'])

        self.label_path = os.path.join(self.model_dir, 'label_mapping.json')
        with open(self.label_path) as f:
            self.label_mapping = json.load(f)
        self.id2label = {idx: name for name, idx in self.label_mapping.items()}

    def forward(self, input: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """return the result by the model

        Args:
            input (Dict[str, Any]): the preprocessed data

        Returns:
            Dict[str, np.ndarray]: results
                Example:
                    {
                        'predictions': array([1]), # lable 0-negative 1-positive
                        'probabilities': array([[0.11491239, 0.8850876 ]], dtype=float32),
                        'logits': array([[-0.53860897,  1.5029076 ]], dtype=float32) # true value
                    }
        """
        return self.model.predict(input)

    def postprocess(self, inputs: Dict[str, np.ndarray],
                    **kwargs) -> Dict[str, np.ndarray]:
        # N x num_classes
        probs = inputs['probabilities']
        result = {
            'probs': probs,
        }

        return result
