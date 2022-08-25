from typing import Dict, Any

import torch
from ...builder import MODELS
from ....metainfo import Models
from ....utils.constant import Tasks
from modelscope.models.nlp.structbert import SbertTokenizer
from modelscope.models.nlp.utils.distributed import DistributedTorchModel
from . import DistributedPlug
from ...base import Tensor

__all__ = ['PlugForTextGeneration']


@MODELS.register_module(Tasks.text_generation, module_name=Models.plug)
class PlugForTextGeneration(DistributedTorchModel):

    def _forward_one(self, input: Dict[str, Any]) -> Dict[str, Tensor]:
        return self.model(**input)

    def generate(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch_size = input['input_ids'].shape[0]
        dec_input_ids = torch.full([batch_size, 1], self.cls_token_id, dtype=torch.long)
        input["dec_input_ids"] = dec_input_ids
        res = self.model_pool.map(DistributedPlug.forward, [input]*self.world_size)
        return res[0]

    def _instantiate_one(self, model_dir, rank):
        tokenizer = SbertTokenizer.from_pretrained(model_dir)
        self.cls_token_id = tokenizer.cls_token_id
        self.model = DistributedPlug.instantiate(model_dir, rank)

        


