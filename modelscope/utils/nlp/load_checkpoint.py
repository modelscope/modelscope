# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


def load_checkpoint(model,
                    load_dir,
                    tag,
                    load_module_strict=True,
                    load_optimizer_states=True,
                    load_lr_scheduler_states=True):
    r"""Load training checkpoint

    Arguments:
        load_dir: Required. Directory to load the checkpoint from
        tag: Required. Checkpoint tag used as a unique identifier for the checkpoint. Ex. Global Step.
        load_module_strict: Optional. Boolean to strictly enforce that the keys in state_dict of module and
         checkpoint match.
        load_optimizer_states: Optional. Boolean to load the training optimizer states from Checkpoint.
         Ex. ADAM's momentum and variance
        load_lr_scheduler_states: Optional. Boolean to add the learning rate scheduler states from Checkpoint.
    Return:
        load_path: Path of the loaded checkpoint. None if loading the checkpoint failed
        client_state: State dictionary used for loading required training states in the client code.
    """

    load_path, client_states = _load_checkpoint(
        model,
        load_dir,
        tag,
        load_module_strict=load_module_strict,
        load_optimizer_states=load_optimizer_states,
        load_lr_scheduler_states=load_lr_scheduler_states)

    if load_optimizer_states:
        if model.zero_optimization() and load_path is not None:
            model._load_zero_checkpoint(
                load_dir, tag, load_optimizer_states=load_optimizer_states)

    return load_path, client_states


def _get_ckpt_name(mp_rank, checkpoints_path, tag):
    ckpt_name = os.path.join(
        checkpoints_path, str(tag),
        'mp_rank_{:02d}'.format(mp_rank) + '_model_states.pt')
    return ckpt_name


def pre_load(mp_rank, load_dir, tag=''):
    load_path = _get_ckpt_name(mp_rank, load_dir, tag)
    checkpoint = torch.load(
        load_path, map_location=lambda storage, loc: storage)
    return checkpoint['module']


def _load_checkpoint(model,
                     load_dir,
                     tag,
                     load_module_strict=True,
                     load_optimizer_states=True,
                     load_lr_scheduler_states=True):

    load_path = model._get_ckpt_name(load_dir, tag)

    if not os.path.exists(load_path):
        return None, None

    checkpoint = torch.load(
        load_path, map_location=lambda storage, loc: storage)

    model.load_module_state_dict(
        state_dict=checkpoint['module'], strict=load_module_strict)
    if not model.zero_optimization() and load_optimizer_states:
        if model.fp16_enabled():
            model.optimizer.load_state_dict(
                checkpoint['optimizer'],
                load_optimizer_states=load_optimizer_states)
        elif load_optimizer_states:
            model.optimizer.load_state_dict(checkpoint['optimizer'])

    if load_lr_scheduler_states and model.lr_scheduler is not None:
        model.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    model.csr_tensor_module_names = checkpoint['csr_tensor_module_names']
    model.global_steps = checkpoint['global_steps']
    model.global_samples = checkpoint.get(
        'global_samples', model.global_steps * model.train_batch_size())
    model.skipped_steps = checkpoint['skipped_steps']
    model.loaded_checkpoint_mp_world_size = checkpoint['mp_world_size']
    model.loaded_checkpoint_dp_world_size = checkpoint['dp_world_size']
    deepspeed_states = [
        'module', 'optimizer', 'lr_scheduler', 'csr_tensor_module_names',
        'skipped_steps', 'global_steps', 'dp_world_size', 'mp_world_size'
    ]
    client_state = {
        key: value
        for key, value in checkpoint.items() if key not in deepspeed_states
    }

    return load_path, client_state
