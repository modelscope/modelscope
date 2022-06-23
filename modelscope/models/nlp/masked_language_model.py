from typing import Any, Dict, Optional, Union

import numpy as np

from ...metainfo import Models
from ...utils.constant import Tasks
from ..base import Model, Tensor
from ..builder import MODELS

__all__ = ['StructBertForMaskedLM', 'VecoForMaskedLM', 'MaskedLanguageModelBase']


class MaskedLanguageModelBase(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.model = self.build_model()

    def build_model(self):
        raise NotImplementedError()

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()

    @property
    def config(self):
        if hasattr(self.model, 'config'):
            return self.model.config
        return None

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, np.ndarray]:
        """return the result by the model

        Args:
            input (Dict[str, Any]): the preprocessed data

        Returns:
            Dict[str, np.ndarray]: results
        """
        rst = self.model(
            input_ids=input['input_ids'],
            attention_mask=input['attention_mask'],
            token_type_ids=input['token_type_ids'])
        return {'logits': rst['logits'], 'input_ids': input['input_ids']}


@MODELS.register_module(Tasks.fill_mask, module_name=Models.structbert)
class StructBertForMaskedLM(MaskedLanguageModelBase):

    def build_model(self):
        from sofa import SbertForMaskedLM
        return SbertForMaskedLM.from_pretrained(self.model_dir)


@MODELS.register_module(Tasks.fill_mask, module_name=Models.veco)
class VecoForMaskedLM(MaskedLanguageModelBase):

    def build_model(self):
        from sofa import VecoForMaskedLM
        return VecoForMaskedLM.from_pretrained(self.model_dir)
