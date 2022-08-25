from typing import Dict, Any

import torch
from ...builder import MODELS
from ....metainfo import Models
from ....utils.constant import Tasks
from modelscope.models.nlp.structbert import SbertTokenizer
from modelscope.models.nlp.utils.distributed import DistributedTorchModel
from . import DistributedPlug
from ...base import Tensor
from modelscope.utils.hub import read_config

__all__ = ['PlugForTextGeneration']


@MODELS.register_module(Tasks.text_generation, module_name=Models.plug)
class PlugForTextGeneration(DistributedTorchModel):
   
    def __init__(self, model_dir, cls_token_id, **kwargs):
        assert cls_token_id is not None
        super().__init__(model_dir, **kwargs)
        self.cls_token_id = cls_token_id

    def _forward_one(self, input: Dict[str, Any]) -> Dict[str, Tensor]:
        return self.__class__.model(input)

    def generate(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch_size = input['input_ids'].shape[0]
        dec_input_ids = torch.full([batch_size, 1], self.cls_token_id, dtype=torch.long)
        input["dec_input_ids"] = dec_input_ids
        res = self.forward(input)
        return res

    def _instantiate_one(self, rank, model_dir, **kwargs):
        cfg = read_config(model_dir)
        self.__class__.model = DistributedPlug(model_dir, rank, **cfg.model, **kwargs)

        


