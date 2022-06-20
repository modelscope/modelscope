from typing import Any, Dict, Optional, Union

import numpy as np

from ...utils.constant import Tasks
from ..base import Model, Tensor
from ..builder import MODELS

__all__ = [
    'StructBertForMaskedLM', 'VecoForMaskedLM', 'AliceMindBaseForMaskedLM'
]


class AliceMindBaseForMaskedLM(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        from sofa.utils.backend import AutoConfig, AutoModelForMaskedLM
        self.model_dir = model_dir
        super().__init__(model_dir, *args, **kwargs)

        self.config = AutoConfig.from_pretrained(model_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_dir, config=self.config)

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, np.ndarray]:
        """return the result by the model

        Args:
            input (Dict[str, Any]): the preprocessed data

        Returns:
            Dict[str, np.ndarray]: results
        """
        rst = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids'])
        return {'logits': rst['logits'], 'input_ids': inputs['input_ids']}


@MODELS.register_module(Tasks.fill_mask, module_name=r'sbert')
class StructBertForMaskedLM(AliceMindBaseForMaskedLM):
    pass


@MODELS.register_module(Tasks.fill_mask, module_name=r'veco')
class VecoForMaskedLM(AliceMindBaseForMaskedLM):
    pass
