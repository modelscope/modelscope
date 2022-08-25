from modelscope.models import Model
from multiprocessing import Pool
from functools import partial
from typing import Dict, Any
from modelscope.utils.hub import read_config
from modelscope.utils.torch_utils import init_dist, _is_free_port, _find_free_port
import torch
import os
import math


class DistributedTorchModel(Model):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.model_pool = None
        self.world_size = None
    
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['model_pool']
        return self_dict

    @classmethod
    def _instantiate(cls, model_dir, **kwargs):
        model = cls(model_dir=model_dir, **kwargs)
        torch.multiprocessing.set_start_method("spawn")
        cfg = read_config(model_dir)
        model.world_size = cfg.model.world_size
        ranks = list(range(model.world_size))
        model.model_pool = Pool(model.world_size)
        model.model_pool.map(partial(model._instantiate_one, model_dir=model_dir), ranks)
        return model

    def _instantiate_one(self, rank, model_dir):
        pass

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        res = self.model_pool.map(self._forward_one, [input]*self.world_size)
        return res[0]

    def _forward_one(self, input):
        pass


def initialize_distributed(rank, mpu, world_size, model_parallel_size):
    """Initialize torch.distributed."""
    # Manually set the device ids.
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', '29500')
    # if not _is_free_port(int(master_port)):
    #    master_port = str(_find_free_port())
    init_method += master_ip + ':' + master_port
    init_dist('pytorch', world_size=world_size, rank=rank, init_method=init_method)
    # Set the model-parallel communicators.
    mpu.initialize_model_parallel(model_parallel_size)


def normal_init_method(mean, std):
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=mean, std=std)

    return init_


def scaled_init_method(mean, std, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = std / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=mean, std=std)

    return init_
