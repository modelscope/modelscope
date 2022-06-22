from typing import Any, Dict

import numpy as np

from modelscope.utils.constant import Tasks
from ..base import Model
from ..builder import MODELS
from ...metainfo import Models

__all__ = ['SbertForZeroShotClassification']


@MODELS.register_module(
    Tasks.zero_shot_classification,
    module_name=Models.structbert)
class SbertForZeroShotClassification(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the zero shot classification model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """

        super().__init__(model_dir, *args, **kwargs)
        from sofa import SbertForSequenceClassification
        self.model = SbertForSequenceClassification.from_pretrained(model_dir)

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
        outputs = self.model(**input)
        logits = outputs['logits'].numpy()
        res = {'logits': logits}
        return res
