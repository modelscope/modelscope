# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Dict

import torch
import torch.nn.functional as F
from megatron import mpu
from megatron.fp16 import FP16_Module
from megatron.utils import print_rank_0

from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.utils.logger import get_logger
from modelscope.utils.nlp.distributed import initialize_distributed
from modelscope.utils.nlp.load_checkpoint import pre_load
from modelscope.utils.torch_utils import set_random_seed_mpu
from . import PlugModel
from .configuration import PlugNLGConfig

logger = get_logger()


class DistributedPlug(TorchModel):
    """
    The wapper class of PLUG Model to initialize parallel environment, load model weights, generate sentences.
    Parameters:
        model_dir (`str`, *required*):
            Path to model damo/nlp_plug_text-generation_27B.
        The model structure in model_dir should be like this:
        model_dir
            |_ config.json
            |_ configuration.json
            |_ ds_zero-offload_10B_config.json
            |_ vocab.txt
            |_ model <-- an empty directory

        Model binaries shall be downloaded separately to populate the model directory, so that
        the model directory would contain the following binaries:
            |_ model
                |_ mp_rank_00_model_states.pt
                |_ mp_rank_01_model_states.pt
                |_ mp_rank_02_model_states.pt
                |_ mp_rank_03_model_states.pt
                |_ mp_rank_04_model_states.pt
                |_ mp_rank_05_model_states.pt
                |_ mp_rank_06_model_states.pt
                |_ mp_rank_07_model_states.pt
        rank (`int`, *required*):
            Used to identify different GPUs in a tensor parallel environment. eg. The rank of GPU #0 is 0, and the
            model file `mp_rank_00_model_states.pt` will be loaded on this GPU.
        world_size (`int`, *required*, defaults to 8):
            The parallel size in total.
        model_parallel_size (`int`, *required*, defaults to 8):
            The parallel size of model(tensor parallel).
        master_ip (`str`, *required*):
            The master IP, can usually be set to `"127.0.0.1"`, used as part of
            [`~torch.distributed.init_process_group`] method parameter `init_method`.
            `init_method` = `"tcp://{master_ip}:{master_port}"`
        master_port (`str`, *required*):
            The master port, can usually be set to `"29500"`, used as part of
            [`~torch.distributed.init_process_group`] method parameter `init_method`.
            `init_method` = `"tcp://{master_ip}:{master_port}"`
        seed (`int`, *optional*, defaults to 42):
            Random seed to control sampling.
    """

    def __init__(self, model_dir, rank, **kwargs):
        super().__init__(model_dir, **kwargs)
        self.rank = rank
        self.model_cfg = kwargs
        self.config = PlugNLGConfig.from_pretrained(model_dir)
        initialize_distributed(rank, mpu, kwargs['world_size'],
                               kwargs['model_parallel_size'],
                               kwargs['master_ip'], kwargs['master_port'])
        seed = 42 if 'seed' not in kwargs else kwargs['seed']
        set_random_seed_mpu(seed)
        self.iteration = 0
        self.model = self.initialize_model(path_load_tag='model')

    def initialize_model(self, path_load_tag='model'):
        """Build the model."""
        print_rank_0('Building Plug model. It will take a few minutes ...')
        model = PlugModel(self.config)

        if mpu.get_data_parallel_rank() == 0:
            logger.info(
                ' > number of parameters on model parallel rank {}: {}'.format(
                    mpu.get_model_parallel_rank(),
                    sum([p.nelement() for p in model.parameters()])))

        if self.config.deepspeed and self.config.fp16:
            model.half()

        # GPU allocation.
        model.cuda(torch.cuda.current_device())

        # Fp16 conversion.
        if self.config.fp16:
            model = FP16_Module(model)
            if self.config.fp32_embedding:
                model.module.model.bert.embeddings.word_embeddings.float()
                model.module.model.bert.embeddings.position_embeddings.float()
                model.module.model.bert.embeddings.token_type_embeddings.float(
                )
            if self.config.fp32_tokentypes:
                model.module.model.bert.embeddings.token_type_embeddings.float(
                )
            if self.config.fp32_layernorm:
                for name, _module in model.named_modules():
                    if 'LayerNorm' in name:
                        _module.float()

        load_model = pre_load(
            mpu.get_model_parallel_rank(), self.model_dir, tag=path_load_tag)
        model_dict = model.module.model.state_dict()
        for key in load_model:
            if key not in model_dict.keys():
                print_rank_0('Skip key: ' + key)
            else:
                print_rank_0('Loading key: ' + key)
        model.module.model.load_state_dict(load_model, strict=False)
        return model

    def forward(self,
                input_tokens,
                token_type_ids=None,
                attention_mask=None,
                target_tokens=None,
                position_ids=None,
                decode_attention_mask=None,
                checkpoint_activations=False,
                is_infer=False,
                sequence_output=None,
                parallel_output=True):
        return self.model(
            input_tokens,
            token_type_ids,
            attention_mask,
            target_tokens,
            position_ids,
            decode_attention_mask,
            checkpoint_activations=checkpoint_activations,
            is_infer=is_infer,
            sequence_output=sequence_output,
            parallel_output=parallel_output)

    def generate(self, input: Dict[str, Tensor], out_length=128, *kwargs):
        return self.model.generate(input, out_length, self.model_cfg, *kwargs)
