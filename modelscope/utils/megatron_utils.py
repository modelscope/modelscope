# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from typing import Dict, List, Union

import torch
from torch import nn

from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import is_master

logger = get_logger()

_DEFAULT_CFG_WITH_MODEL_TYPE = {
    'gpt-moe': {
        'version': 'moe',
        'world_size': 8
    },
    'plug': {
        'version': 'v1',
        'world_size': 8,
        'tensor_model_parallel_size': 8,
        'seed': 1234
    },
    'mglm-text-summarization': {
        'version': 'v1',
        'seed': 1234
    },
}

_CHECKPOINT_FORMAT = 'mp_rank_XX_model_states.pt'

_IS_MEGATRON_INITIALIZED = False


def init_megatron_util(megatron_cfg=None, model_dir=None, **kwargs):
    """Initialize megatron_util environment for megatron_based model.

    If argument `megatron_cfg` is not specified, then the megatorn_cfg will be load
    from configuration.json file in the model_dir.

    Args:
        megatron_cfg (Dict, optional): Megatron Config will be send to megatron_util.
        model_dir (str, optional): The model path for configuration. Defaults to None.
    """
    from modelscope.utils.hub import read_config
    from megatron_util import initialize_megatron

    assert not (megatron_cfg is None and model_dir is None), \
        'cfg and model_dir cannot both be None when initializing megatron_util'
    if megatron_cfg is None:
        cfg = read_config(model_dir)
        try:
            megatron_cfg = cfg.megatron
        except AttributeError:
            try:
                model_type = cfg.model.type
            except AttributeError:
                # Fit models without model type, such as mglm
                model_type = cfg.pipeline.type
            megatron_cfg = _DEFAULT_CFG_WITH_MODEL_TYPE[model_type] \
                if model_type in _DEFAULT_CFG_WITH_MODEL_TYPE else {}
    megatron_cfg.update(kwargs)
    initialize_megatron(megatron_cfg)
    global _IS_MEGATRON_INITIALIZED
    _IS_MEGATRON_INITIALIZED = True


def is_megatron_initialized() -> bool:
    return _IS_MEGATRON_INITIALIZED


def convert_megatron_checkpoint(
        model: nn.Module, checkpoint_dir: Union[str, bytes, os.PathLike],
        target_dir: Union[str, bytes, os.PathLike]) -> None:
    """Split or Merge checkpoint for megatron_based model.

    Args:
        model (nn.Module): Any megatron_based model.
        checkpoint_dir (Union[str, bytes, os.PathLike]): The save path of origin checkpoint.
        target_dir (Union[str, bytes, os.PathLike]): The target path of new checkpoint.
    """

    def log_master(information: str):
        if is_master():
            logger.info(information)

    if os.path.exists(os.path.join(checkpoint_dir, 'model')):
        checkpoint_dir = os.path.join(checkpoint_dir, 'model')

    origin_num_partitions = len(os.listdir(checkpoint_dir))
    target_num_partitions = int(os.getenv('WORLD_SIZE'))

    _check_origin_dir(checkpoint_dir)
    _check_target_num_partitions(target_num_partitions)
    log_master(
        f'origin_num_partitions: {origin_num_partitions}, target_num_partitions: {target_num_partitions}'
    )

    if origin_num_partitions < target_num_partitions:
        os.makedirs(target_dir, exist_ok=True)
        state_dict = _split_checkpoint(
            model, checkpoint_dir,
            target_num_partitions // origin_num_partitions)
        _save_converted_checkpoint(state_dict, target_dir)
        log_master('Split checkpoints succeeded.')
    elif origin_num_partitions > target_num_partitions:
        os.makedirs(target_dir, exist_ok=True)
        state_dict = _merge_checkpoint(
            model, checkpoint_dir,
            origin_num_partitions // target_num_partitions)
        _save_converted_checkpoint(state_dict, target_dir)
        log_master('Merge checkpoints succeeded.')
    else:
        shutil.copytree(checkpoint_dir, target_dir)
        log_master('Copy checkpoints succeeded.')


def _check_origin_dir(origin_dir: Union[str, bytes, os.PathLike]) -> None:
    filenames = os.listdir(origin_dir)
    assert len(filenames) & (
        len(filenames) - 1) == 0, 'The number of files must be a power of 2!'
    for i in range(len(filenames)):
        checkpoint_name = _CHECKPOINT_FORMAT.replace('XX', f'{i:02d}')
        assert checkpoint_name in filenames, \
            f'Can not find {checkpoint_name} file!'


def _check_target_num_partitions(num_partitions: int) -> None:
    assert num_partitions & (num_partitions - 1) == 0, \
        'The number of target partitions must be a power of 2!'


def _split_checkpoint(model: nn.Module, checkpoint_dir: Union[str, bytes,
                                                              os.PathLike],
                      num_partitions: int) -> Dict[str, torch.Tensor]:
    target_rank = int(os.getenv('RANK'))
    origin_rank = target_rank // num_partitions
    state_dict = _load_by_rank(checkpoint_dir, origin_rank)

    target_state_dict = {}
    for name, parameter in model.named_parameters():
        dim = _get_diff_dim(parameter, state_dict[name])
        if dim == -1:
            target_state_dict[name] = state_dict[name]
            continue
        partitions_list = _split_tensor(state_dict[name], num_partitions, dim)
        target_state_dict[name] = partitions_list[target_rank
                                                  % num_partitions].clone()
    return target_state_dict


def _merge_checkpoint(model: nn.Module, checkpoint_dir: Union[str, bytes,
                                                              os.PathLike],
                      num_partitions: int) -> Dict[str, torch.Tensor]:
    target_rank = int(os.getenv('RANK'))
    origin_rank_list = [
        target_rank * num_partitions + i for i in range(num_partitions)
    ]
    state_dict_list = [
        _load_by_rank(checkpoint_dir, i) for i in origin_rank_list
    ]

    target_state_dict = {}
    for name, parameter in model.named_parameters():
        dim = _get_diff_dim(parameter, state_dict_list[0][name])
        if dim == -1:
            target_state_dict[name] = state_dict_list[0][name]
            continue
        target_state_dict[name] = torch.cat(
            [state_dict[name] for state_dict in state_dict_list],
            dim=dim).clone()
    return target_state_dict


def _save_converted_checkpoint(
        state_dict: Dict[str, torch.Tensor],
        target_dir: Union[str, bytes, os.PathLike]) -> None:
    target_rank = int(os.getenv('RANK'))
    target_name = _CHECKPOINT_FORMAT.replace('XX', f'{target_rank:02d}')
    torch.save(state_dict, os.path.join(target_dir, target_name))


def _get_diff_dim(tensor1: torch.Tensor, tensor2: torch.Tensor) -> int:
    for i, (s1, s2) in enumerate(zip(tensor1.shape, tensor2.shape)):
        if s1 != s2:
            return i
    return -1


def _load_by_rank(checkpoint_dir: Union[str, bytes, os.PathLike],
                  rank: int) -> Dict[str, torch.Tensor]:
    checkpoint_name = _CHECKPOINT_FORMAT.replace('XX', f'{rank:02d}')
    state_dict = torch.load(
        os.path.join(checkpoint_dir, checkpoint_name),
        map_location=lambda storage, loc: storage)
    return state_dict['module'] if 'module' in state_dict else state_dict


def _split_tensor(tensor: torch.Tensor, num_partitions: int,
                  partition_dim: int) -> List[torch.Tensor]:
    from megatron_util import mpu
    per_partition_size = mpu.utils.divide(
        tensor.size(partition_dim), num_partitions)
    partitions_list = torch.split(
        tensor, per_partition_size, dim=partition_dim)
    return partitions_list
