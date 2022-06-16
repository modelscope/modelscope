from typing import Any, Dict

import numpy as np
import torch

from modelscope.utils.constant import Tasks
from ..base import Model
from ..builder import MODELS

__all__ = ['BertForZeroShotClassification']


@MODELS.register_module(
    Tasks.zero_shot_classification,
    module_name=r'bert-zero-shot-classification')
class BertForZeroShotClassification(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the zero shot classification model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """

        super().__init__(model_dir, *args, **kwargs)
        from sofa import SbertForSequenceClassification
        self.model = SbertForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def forward(self, input: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """return the result by the model

        Args:
            input (Dict[str, Any]): the preprocessed data

        Returns:
            Dict[str, np.ndarray]: results
                Example:
                    {
                        'logits': array([[-0.53860897,  1.5029076 ]], dtype=float32) # true value
                    }
        """
        with torch.no_grad():
            outputs = self.model(**input)
        logits = outputs['logits'].numpy()
        res = {'logits': logits}
        return res
