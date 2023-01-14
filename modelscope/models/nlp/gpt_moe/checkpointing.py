# Copyright 2021-2022 The Alibaba PAI Team Authors.
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from megatron_util import mpu
from megatron_util.model import Float16Module
from megatron_util.utils import unwrap_model
from torch.nn.parallel import DistributedDataParallel as torchDDP

from .configuration import logger
from .moe.layer import MoE


def get_checkpoint_names(checkpoints_path,
                         path_load_tag,
                         num_experts,
                         tensor_rank=None,
                         expp_rank=None):
    """Determine the directory name for this rank's checkpoint."""
    if tensor_rank is None:
        tensor_rank = mpu.get_tensor_model_parallel_rank()

    common_path = os.path.join(checkpoints_path, path_load_tag,
                               f'mp_rank_{tensor_rank:02d}')

    if num_experts[0] > 0:
        model_name = common_path + '_model_states.pt'
        optim_name = os.path.join(
            checkpoints_path, path_load_tag,
            f'expp_rank_{expp_rank}_mp_rank_{tensor_rank:02d}_optim_states.pt')
    else:
        model_name = optim_name = os.path.join(common_path,
                                               'model_optim_rng.pt')

    return model_name, optim_name


def _get_expert_ckpt_name(checkpoints_path, layer_id, expert_id):
    mp_rank = mpu.get_tensor_model_parallel_rank()
    ckpt_name = os.path.join(
        os.path.join(checkpoints_path, 'model'),
        f'layer_{layer_id}_expert_{expert_id}_mp_rank_{mp_rank:02d}_model_states.pt'
    )
    return ckpt_name


def _load_base_checkpoint(load_dir, path_load_tag=None, num_experts=None):
    """ Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """
    largest_group_name = mpu.get_max_expert_size_name()
    expp_rank = mpu.get_expert_parallel_rank(largest_group_name)
    checkpoint_names = get_checkpoint_names(
        load_dir,
        path_load_tag=path_load_tag,
        num_experts=num_experts,
        expp_rank=expp_rank)
    model_checkpoint_name, optim_checkpoint_name = checkpoint_names

    logger.info(f'Loading model checkpoint from {model_checkpoint_name}')
    model_state_dict = torch.load(model_checkpoint_name, map_location='cpu')

    return model_state_dict


def load_checkpoint(model,
                    load_dir,
                    num_experts=None,
                    strict=True,
                    path_load_tag='model',
                    load_ds_ckpts=True):
    model = unwrap_model(model, (torchDDP, Float16Module))

    model_state_dict = _load_base_checkpoint(
        load_dir, path_load_tag=path_load_tag, num_experts=num_experts)

    assert model_state_dict is not None

    if load_ds_ckpts:
        load_moe_checkpoint(model, model_state_dict['module'], load_dir)
    else:
        load_moe_checkpoint(model, model_state_dict['model'], load_dir)

    if load_ds_ckpts:
        model.load_state_dict(model_state_dict['module'], strict=strict)
    else:
        model.load_state_dict(model_state_dict['model'], strict=strict)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def load_moe_checkpoint(model, state_dict, load_dir):
    moe_layer_id = 0
    for n_module, module in model.named_modules():
        if isinstance(module, MoE):  # and torch.distributed.get_rank() == 0:
            group_name = module.expert_group_name
            num_local_experts = module.num_local_experts
            expp_rank = mpu.get_expert_parallel_rank(group_name)
            # loop all local_experts
            for local_expert_id in range(num_local_experts):
                global_expert_id = expp_rank * num_local_experts + local_expert_id
                moe_load_path = _get_expert_ckpt_name(load_dir, moe_layer_id,
                                                      global_expert_id)
                logger.info(f'Loading expert states from {moe_load_path}')
                expert_state_dict = torch.load(
                    moe_load_path, map_location=torch.device('cpu'))
                # Updating global -> local expert ids
                moe_str_prefix = '.deepspeed_moe.experts.deepspeed_experts.'
                for key in list(expert_state_dict.keys()):
                    local_key = key.replace(
                        f'{moe_str_prefix}{global_expert_id}',
                        f'{moe_str_prefix}{local_expert_id}')
                    expert_state_dict[local_key] = expert_state_dict.pop(key)
                state_dict.update(expert_state_dict)
            moe_layer_id += 1
