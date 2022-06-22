from typing import Any, Dict, Optional, Union

import numpy as np

from ...utils.constant import Tasks
from ..base import Model, Tensor
from ..builder import MODELS
from ...metainfo import Models

__all__ = ['StructBertForMaskedLM', 'VecoForMaskedLM', 'MaskedLMModelBase']


class MaskedLMModelBase(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.model = self.build_model()

    def build_model(self):
        raise NotImplementedError()

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, np.ndarray]:
        """return the result by the model

        Args:
            inputs (Dict[str, Any]): the preprocessed data

        Returns:
            Dict[str, np.ndarray]: results
        """
        rst = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids'])
        return {'logits': rst['logits'], 'input_ids': inputs['input_ids']}


@MODELS.register_module(Tasks.fill_mask, module_name=Models.structbert)
class StructBertForMaskedLM(MaskedLMModelBase):

    def build_model(self):
        from sofa import SbertForMaskedLM
        return SbertForMaskedLM.from_pretrained(self.model_dir)


@MODELS.register_module(Tasks.fill_mask, module_name=Models.veco)
class VecoForMaskedLM(MaskedLMModelBase):

    def build_model(self):
        from sofa import VecoForMaskedLM
        return VecoForMaskedLM.from_pretrained(self.model_dir)
