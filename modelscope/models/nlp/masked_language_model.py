from typing import Any, Dict, Optional, Union

import numpy as np
from ..base import Model, Tensor
from ..builder import MODELS
from ...utils.constant import Tasks

__all__ = ['MaskedLanguageModel']


@MODELS.register_module(Tasks.fill_mask, module_name=r'sbert')
class MaskedLanguageModel(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        from sofa.utils.backend import AutoConfig, AutoModelForMaskedLM
        self.model_dir = model_dir
        super().__init__(model_dir, *args, **kwargs)

        self.config = AutoConfig.from_pretrained(model_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(model_dir, config=self.config)


    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, np.ndarray]:
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
        rst =  self.model(
                          input_ids=inputs["input_ids"],
                          attention_mask=inputs['attention_mask'],
                          token_type_ids=inputs["token_type_ids"]
                         )
        return {'logits': rst['logits'], 'input_ids': inputs['input_ids']}
