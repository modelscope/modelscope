import torch
from typing import Dict
from functools import partial

from . import DistributedPlug
from ...base import Tensor, TorchModel
from ...builder import MODELS
from ....metainfo import Models
from ....outputs import OutputKeys
from ....utils.constant import Tasks

__all__ = ['PLUGForTextGeneration']

@MODELS.register_module(Tasks.text_generation, module_name=Models.plug)
class PlugForTextGeneration(TorchModel):
    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        import torch

        from transformers import BertTokenizer
        from multiprocessing import Pool
        from .arguments import get_args
        from . import PlugNLGConfig
        import torch
        torch.multiprocessing.set_start_method("spawn")

        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        model_config = PlugNLGConfig.from_pretrained(model_dir)
        
        # TODO(suluyan): Arguments
        args = get_args()
        args.world_size = 8
        args.model_parallel_size = 8
        args.pre_load = True
        args.distributed_backend = 'nccl'
        args.fp16 = True
        args.fp32_layernorm = True
        args.checkpoint_activations = True 
        args.batch_size = 1
        args.top_k = 20
        args.top_p = 0.0
        args.temperature = 0.9
        self.args = args

        self.world_size = args.world_size
        ranks = list(range(self.world_size))
        self.model_pool = Pool(self.world_size)
        self.model_pool.map(partial(DistributedPlug.init, model_dir=model_dir, model_config=model_config, args=args), ranks)

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.model(**input)

    def generate(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        dec_input_ids = torch.full([self.args.batch_size, 1], self.tokenizer.cls_token_id, dtype=torch.long)
        input["dec_input_ids"] = dec_input_ids
        res = self.model_pool.map(DistributedPlug.forward, [input]*self.world_size)
        return res[0]


