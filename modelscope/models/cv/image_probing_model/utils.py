# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import re

import torch
import torch.nn as nn


def load_pretrained(model: torch.nn.Module,
                    state_dict,
                    local_path: str,
                    map_location='cpu',
                    logger=None,
                    sub_level=None):
    return load_pretrained_dict(model, state_dict, logger, sub_level=sub_level)


def load_pretrained_dict(model: torch.nn.Module,
                         state_dict: dict,
                         logger=None,
                         sub_level=None):
    """
    Load parameters to model with
    1. Sub name by revise_keys For DataParallelModel or DistributeParallelModel.
    2. Load 'state_dict' again if possible by key 'state_dict' or 'model_state'.
    3. Take sub level keys from source, e.g. load 'backbone' part from a classifier into a backbone model.
    4. Auto remove invalid parameters from source.
    5. Log or warning if unexpected key exists or key misses.

    Args:
        model (torch.nn.Module):
        state_dict (dict): dict of parameters
        logger (logging.Logger, None):
        sub_level (str, optional): If not None, parameters with key startswith sub_level will remove the prefix
            to fit actual model keys. This action happens if user want to load sub module parameters
            into a sub module model.
    """
    revise_keys = [(r'^module\.', '')]

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']

    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}

    if sub_level:
        sub_level = sub_level if sub_level.endswith('.') else (sub_level + '.')
        sub_level_len = len(sub_level)
        state_dict = {
            key[sub_level_len:]: value
            for key, value in state_dict.items() if key.startswith(sub_level)
        }

    state_dict = _auto_drop_invalid(model, state_dict, logger=logger)

    load_status = model.load_state_dict(state_dict, strict=False)
    unexpected_keys = load_status.unexpected_keys
    missing_keys = load_status.missing_keys
    err_msgs = []
    if unexpected_keys:
        err_msgs.append('unexpected key in source '
                        f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msgs.append('missing key in source '
                        f'state_dict: {", ".join(missing_keys)}\n')
    err_msgs = '\n'.join(err_msgs)

    if len(err_msgs) > 0:
        if logger:
            logger.warning(err_msgs)
        else:
            import warnings
            warnings.warn(err_msgs)


def convert_weights(model: nn.Module):
    """
    Convert applicable model parameters to fp16.
    """

    def _convert_weights_to_fp16(layer):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            layer.weight.data = layer.weight.data.half()
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.half()

        if isinstance(layer, nn.MultiheadAttention):
            for attr in [
                    *[f'{s}_proj_weight' for s in ['in', 'q', 'k', 'v']],
                    'in_proj_bias', 'bias_k', 'bias_v'
            ]:
                tensor = getattr(layer, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ['text_projection', 'proj']:
            if hasattr(layer, name):
                attr = getattr(layer, name)
                if attr is not None:
                    attr.data = attr.data.half()

        for name in ['prompt_embeddings']:
            if hasattr(layer, name):
                attr = getattr(layer, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def _auto_drop_invalid(model: torch.nn.Module, state_dict: dict, logger=None):
    """
    Strip unmatched parameters in state_dict, e.g. shape not matched, type not matched.

    Args:
        model (torch.nn.Module):
        state_dict (dict):
        logger (logging.Logger, None):

    Returns:
        A new state dict.
    """
    ret_dict = state_dict.copy()
    invalid_msgs = []
    for key, value in model.state_dict().items():
        if key in state_dict:
            # Check shape
            new_value = state_dict[key]
            if value.shape != new_value.shape:
                invalid_msgs.append(
                    f'{key}: invalid shape, dst {value.shape} vs. src {new_value.shape}'
                )
                ret_dict.pop(key)
            elif value.dtype != new_value.dtype:
                invalid_msgs.append(
                    f'{key}: invalid dtype, dst {value.dtype} vs. src {new_value.dtype}'
                )
                ret_dict.pop(key)
    if len(invalid_msgs) > 0:
        warning_msg = 'ignore keys from source: \n' + '\n'.join(invalid_msgs)
        if logger:
            logger.warning(warning_msg)
        else:
            import warnings
            warnings.warn(warning_msg)
    return ret_dict
