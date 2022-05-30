from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from maas_lib.utils.constant import Tasks
from ..base import Model
from ..builder import MODELS

__all__ = ['SequenceClassificationModel']


@MODELS.register_module(
    Tasks.text_classification, module_name=r'bert-sentiment-analysis')
class SequenceClassificationModel(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        # Model.__init__(self, model_dir, model_cls, first_sequence, *args, **kwargs)
        # Predictor.__init__(self, *args, **kwargs)
        """initialize the sequence classification model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """

        super().__init__(model_dir, *args, **kwargs)
        from easynlp.appzoo import SequenceClassification
        from easynlp.core.predictor import get_model_predictor
        self.model = get_model_predictor(
            model_dir=self.model_dir,
            model_cls=SequenceClassification,
            input_keys=[('input_ids', torch.LongTensor),
                        ('attention_mask', torch.LongTensor),
                        ('token_type_ids', torch.LongTensor)],
            output_keys=['predictions', 'probabilities', 'logits'])

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
