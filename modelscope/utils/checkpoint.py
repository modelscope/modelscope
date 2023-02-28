# Copyright (c) Alibaba, Inc. and its affiliates.

import io
import os
import re
import time
from collections import OrderedDict
from functools import partial
from shutil import copytree, ignore_patterns, rmtree
from typing import Callable, Dict, Optional, Union

import json
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from modelscope import __version__
from modelscope.fileio import File, LocalStorage
from modelscope.utils.config import Config, JSONIteratorEncoder
from modelscope.utils.constant import ConfigFields, ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import is_master

logger = get_logger()

storage = LocalStorage()


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(state_dict, '_metadata', OrderedDict())
    return state_dict_cpu


def save_checkpoint(model: torch.nn.Module,
                    filename: str,
                    optimizer: Optional[Optimizer] = None,
                    lr_scheduler: Optional[_LRScheduler] = None,
                    meta: Optional[dict] = None,
                    with_meta: bool = True,
                    with_model: bool = True) -> None:
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default, ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        lr_scheduler(:obj:`_LRScheduler`, optional): LRScheduler to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
        with_meta (bool, optional): Save meta info.
        with_model(bool, optional): Save model states.
    """
    checkpoint = {}
    if not with_meta and not with_model:
        raise ValueError(
            'Save meta by "with_meta=True" or model by "with_model=True"')

    if with_meta:
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta must be a dict or None, but got {type(meta)}')
        meta.update(modelscope=__version__, time=time.asctime())

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        if hasattr(model, 'CLASSES') and model.CLASSES is not None:
            # save class name to the meta
            meta.update(CLASSES=model.CLASSES)

        checkpoint['meta'] = meta

        # save optimizer state dict in the checkpoint
        if isinstance(optimizer, Optimizer):
            checkpoint['optimizer'] = optimizer.state_dict()
        elif isinstance(optimizer, dict):
            checkpoint['optimizer'] = {}
            for name, optim in optimizer.items():
                checkpoint['optimizer'][name] = optim.state_dict()

        # save lr_scheduler state dict in the checkpoint
        if lr_scheduler is not None and hasattr(lr_scheduler, 'state_dict'):
            checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

    if with_model:
        _weights = weights_to_cpu(model.state_dict())
        if not with_meta:
            checkpoint = _weights
        else:
            checkpoint['state_dict'] = _weights

    with io.BytesIO() as f:
        torch.save(checkpoint, f)
        File.write(f.getvalue(), filename)


def load_checkpoint(filename,
                    model,
                    optimizer: Optimizer = None,
                    lr_scheduler: _LRScheduler = None):
    if not os.path.exists(filename):
        raise ValueError(f'Checkpoint file {filename} does not exist!')
    checkpoint = torch.load(filename, map_location='cpu')

    if optimizer is not None:
        if 'optimizer' in checkpoint:
            if isinstance(optimizer, Optimizer):
                optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(optimizer, dict):
                optimizer_dict = checkpoint['optimizer']
                for key, optimizer_ins in optimizer.items():
                    if key in optimizer_dict:
                        optimizer_ins.load_state_dict(optimizer_dict[key])
                    else:
                        logger.warning(
                            f'The state dict of optimizer {key} cannot be found in checkpoint file: {filename}'
                        )
        else:
            logger.warning(
                f'The state dict of optimizer cannot be found in checkpoint file: {filename}'
            )

    if lr_scheduler is not None:
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            logger.warning(
                f'The state dict of lr_scheduler cannot be found in checkpoint file: {filename}'
            )

    if model is not None:
        state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint[
            'state_dict']
        model.load_state_dict(state_dict)
    return checkpoint.get('meta', {})


def load_task_model_checkpoint(model_to_load,
                               model_local_dir,
                               default_dtype=None,
                               load_state_fn=None,
                               **kwargs):
    """
    Load model checkpoint file and feed the parameters into the model.
    Args:
        model_to_load: The model to be load
        model_local_dir: The actual checkpoint dir on local disk.
        default_dtype: Set the default float type by 'torch.set_default_dtype'
        load_state_fn: An optional load_state_fn used to load state_dict into the model.

    Returns:

    """

    def _add_head_prefix_to_state_dict(state_dicts, head_prefix,
                                       expected_keys_without_head_prefix,
                                       missing_keys):
        new_state_dict = OrderedDict()
        for name, module in state_dicts.items():
            if name in expected_keys_without_head_prefix:
                name_with_head = '.'.join([head_prefix, name])
                new_state_dict[name_with_head] = module
                expected_keys_without_head_prefix.remove(name)
                missing_keys = list(set(missing_keys) - set([name_with_head]))
            else:
                new_state_dict[name] = module

        missing_head_keys = []
        if len(expected_keys_without_head_prefix) > 0:
            missing_head_keys = expected_keys_without_head_prefix.copy()
        return new_state_dict, missing_head_keys, missing_keys

    def _find_mismatched_keys(
        state_dicts,
        model_state_dict,
        loaded_keys,
        prefix,
        add_prefix_to_model,
        remove_prefix_from_model,
        ignore_mismatched_sizes,
    ):
        mismatched_key = []
        if ignore_mismatched_sizes:
            for checkpoint_key in loaded_keys:
                model_key = checkpoint_key
                if remove_prefix_from_model:
                    # The model key starts with `prefix` but `checkpoint_key` doesn't, so we add it.
                    model_key = f'{prefix}.{checkpoint_key}'
                elif add_prefix_to_model:
                    # The model key doesn't start with `prefix` but `checkpoint_key` does, so we remove it.
                    model_key = '.'.join(checkpoint_key.split('.')[1:])

                if model_key in model_state_dict:
                    model_shape = model_state_dict[model_key].shape
                    checkpoint_shape = state_dicts[checkpoint_key].shape
                    if checkpoint_shape != model_shape:
                        mismatched_key.append(
                            (checkpoint_key, state_dicts[checkpoint_key].shape,
                             model_state_dict[model_key].shape))
                        del state_dicts[checkpoint_key]
        return mismatched_key

    def _load_state_dict_into_model(
        model,
        state_dict,
        start_prefix,
        head_prefix_keys,
        load_state_fn=None,
    ):
        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        error_msgs = []

        if load_state_fn is not None:
            load_state_fn(
                model,
                state_dict,
                prefix=start_prefix,
                head_prefix_keys=head_prefix_keys,
                local_metadata=None,
                error_msgs=error_msgs)
        else:

            def load(module: nn.Module, prefix=''):
                local_metadata = {} if metadata is None else metadata.get(
                    prefix[:-1], {})
                args = (state_dict, prefix, local_metadata, True, [], [],
                        error_msgs)
                module._load_from_state_dict(*args)
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + '.')

            load(model, prefix=start_prefix)

        return error_msgs

    def _load_checkpoint(
        model,
        state_dict,
        load_state_fn,
        ignore_mismatched_sizes,
        _fast_init,
    ):
        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        keys_from_pretrained = list(state_dict.keys())

        prefix = model.base_model_prefix

        # during loading stage, base model prefix is complicated, should consider remove or add
        if len(prefix) > 0:
            # nlp: encoder, decoder
            pretrained_has_prefix_module = any(
                s.startswith(prefix) for s in keys_from_pretrained)
            model_expects_prefix_module = any(
                s.startswith(prefix) for s in expected_keys)
        else:
            # nlp:encoder-decoder, cv:backbone-head,
            pretrained_has_prefix_module = False
            model_expects_prefix_module = False

        remove_prefix_from_model = not pretrained_has_prefix_module and model_expects_prefix_module
        add_prefix_to_model = pretrained_has_prefix_module and not model_expects_prefix_module

        if remove_prefix_from_model:
            expected_keys_not_base_model_prefixed = [
                s for s in expected_keys if not s.startswith(prefix)
            ]
            expected_keys = [
                '.'.join(s.split('.')[1:]) if s.startswith(prefix) else s
                for s in expected_keys
            ]
        elif add_prefix_to_model:
            # backbone only
            expected_keys = ['.'.join([prefix, s]) for s in expected_keys]
            expected_keys_not_base_model_prefixed = []

        missing_keys = list(set(expected_keys) - set(keys_from_pretrained))
        unexpected_keys = list(set(keys_from_pretrained) - set(expected_keys))

        # during loading stage head prefix is simple, add or not add
        prefix_heads = model.head_prefix
        expected_head_keys_without_head_prefix = []
        missing_head_keys = []
        unexpected_head_keys = []
        pretrained_has_prefix_head = dict()
        head_prefix_keys = dict()

        # only for case of head mismatched with state-dict
        if len(prefix_heads) > 0 and len(unexpected_keys) > 0:
            if isinstance(prefix_heads, str):
                prefix_heads = [prefix_heads]

            # to double-check if head matched with state-dict
            for prefix_head in prefix_heads:
                pretrained_has_prefix_head[prefix_head] = any(
                    s.startswith(prefix_head) for s in keys_from_pretrained)

            for prefix_head in prefix_heads:
                expected_keys_without_head_prefix = [
                    '.'.join(s.split('.')[1:]) for s in expected_keys
                    if s.startswith(prefix_head)
                ]
                expected_head_keys_without_head_prefix.extend(
                    expected_keys_without_head_prefix)
                head_prefix_keys[
                    prefix_head] = expected_keys_without_head_prefix
            unexpected_head_keys = list(
                set(unexpected_keys)
                - set(expected_head_keys_without_head_prefix))
            unexpected_keys = list(
                set(unexpected_keys)
                - set(expected_head_keys_without_head_prefix))

        _keys_to_ignore_on_load_missing = kwargs.pop(
            '_keys_to_ignore_on_load_missing', None)
        _keys_to_ignore_on_load_unexpected = kwargs.pop(
            '_keys_to_ignore_on_load_unexpected', None)
        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if _keys_to_ignore_on_load_missing is not None:
            for pat in _keys_to_ignore_on_load_missing:
                missing_keys = [
                    k for k in missing_keys if re.search(pat, k) is None
                ]

        if _keys_to_ignore_on_load_unexpected is not None:
            for pat in _keys_to_ignore_on_load_unexpected:
                unexpected_keys = [
                    k for k in unexpected_keys if re.search(pat, k) is None
                ]

        # retrieve uninitialized modules and initialize before maybe overriding that with the pretrained weights.
        if _fast_init:
            uninitialized_modules = retrieve_modules_from_names(
                model,
                missing_keys,
                prefix=prefix,
                add_prefix=add_prefix_to_model,
                remove_prefix=remove_prefix_from_model)
            for module in uninitialized_modules:
                model._init_weights(module)

        # Make sure we are able to load head correctly by revise state-dict
        missing_head_keys_by_head = dict()
        if len(head_prefix_keys) > 0:
            for head_prefix in head_prefix_keys:
                if not pretrained_has_prefix_head[head_prefix]:
                    state_dict, missing_head_keys, missing_keys = _add_head_prefix_to_state_dict(
                        state_dict, head_prefix, head_prefix_keys[head_prefix],
                        missing_keys)
                    missing_head_keys_by_head[head_prefix] = missing_head_keys

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ''
        model_to_load = model
        heads_to_load = dict()
        if len(model.base_model_prefix) > 0 and not hasattr(
                model,
                model.base_model_prefix) and pretrained_has_prefix_module:
            start_prefix = model.base_model_prefix + '.'
        if len(model.base_model_prefix) > 0 and hasattr(
                model,
                model.base_model_prefix) and not pretrained_has_prefix_module:
            model_to_load = getattr(model, model.base_model_prefix)
            for head_prefix in prefix_heads:
                heads_to_load[head_prefix] = getattr(model, head_prefix)
            if any(key in expected_keys_not_base_model_prefixed
                   for key in keys_from_pretrained):
                raise ValueError(
                    'The state dictionary of the model you are trying to load is corrupted. Are you sure it was '
                    'properly saved?')

        # Whole checkpoint
        mismatched_keys = _find_mismatched_keys(
            state_dict,
            model_state_dict,
            keys_from_pretrained,
            prefix,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        )
        error_msgs = _load_state_dict_into_model(model_to_load, state_dict,
                                                 start_prefix, load_state_fn)

        if len(heads_to_load) > 0:
            for head in heads_to_load:
                local_error_msgs = _load_state_dict_into_model(
                    heads_to_load[head], state_dict, head + '.', load_state_fn)
                error_msgs.extend(local_error_msgs)

        if len(error_msgs) > 0:
            error_msg = '\n\t'.join(error_msgs)
            raise RuntimeError(
                f'Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}'
            )

        if len(unexpected_keys) > 0:
            logger.warning(
                f'Some weights of the model checkpoint were not used when'
                f' initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are'
                f' initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or'
                ' with another architecture (e.g. initializing a BertForTokenClassification model from a'
                ' BertForPreTraining model).\n- This IS NOT expected if you are initializing'
                f' {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical'
                ' (initializing a BertForTokenClassification model from a BertForTokenClassification model).'
            )
        elif len(unexpected_head_keys) > 0:
            logger.warning(
                f'Some weights of the model checkpoint were not used when'
                f' initializing {model.__class__.__name__}: {unexpected_head_keys}\n- This IS Not expected if you are'
                f' initializing {model.__class__.__name__} from the checkpoint of a model with a same task while the'
                ' structure is different (e.g. initializing a BertForTokenClassification model from a'
                ' BertForTokenClassification model).')
        else:
            logger.info(
                f'All model checkpoint weights were used when initializing {model.__class__.__name__}.\n'
            )
        if len(missing_keys) > 0:
            logger.warning(
                f'Some weights of {model.__class__.__name__} were not initialized from the model checkpoint'
                f' and are newly initialized: {missing_keys}\nYou should probably'
                ' TRAIN this model on a down-stream task to be able to use it for predictions and inference.'
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f'All the weights of {model.__class__.__name__} were initialized from the model checkpoint '
                f'If your task is similar to the task the model of the checkpoint'
                f' was trained on, you can already use {model.__class__.__name__} for predictions without further'
                ' training.')
        if len(mismatched_keys) > 0:
            mismatched_warning = '\n'.join([
                f'- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated'
                for key, shape1, shape2 in mismatched_keys
            ])
            logger.warning(
                f'Some weights of {model.__class__.__name__} were not initialized from the model checkpoint'
                f' and are newly initialized because the shapes did not'
                f' match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able'
                ' to use it for predictions and inference.')

        return missing_keys, unexpected_keys, mismatched_keys, error_msgs

    def retrieve_modules_from_names(model,
                                    names,
                                    prefix=None,
                                    add_prefix=False,
                                    remove_prefix=False):
        module_keys = set(['.'.join(key.split('.')[:-1]) for key in names])

        # torch.nn.ParameterList is a special case where two parameter keywords
        # are appended to the module name, *e.g.* bert.special_embeddings.0
        module_keys = module_keys.union(
            set([
                '.'.join(key.split('.')[:-2]) for key in names
                if key[-1].isdigit()
            ]))

        retrieved_modules = []
        # retrieve all modules that has at least one missing weight name
        for name, module in model.named_modules():
            if remove_prefix:
                name = '.'.join(
                    name.split('.')[1:]) if name.startswith(prefix) else name
            elif add_prefix:
                name = '.'.join([prefix, name]) if len(name) > 0 else prefix

            if name in module_keys:
                retrieved_modules.append(module)

        return retrieved_modules

    # TODO Sharded ckpt
    ckpt_file = os.path.join(model_local_dir, ModelFile.TORCH_MODEL_BIN_FILE)
    state_dict = torch.load(ckpt_file, map_location='cpu')
    if default_dtype is not None:
        torch.set_default_dtype(default_dtype)

    missing_keys, unexpected_keys, mismatched_keys, error_msgs = _load_checkpoint(
        model_to_load,
        state_dict,
        load_state_fn=load_state_fn,
        ignore_mismatched_sizes=True,
        _fast_init=True,
    )

    return {
        'model': model_to_load,
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys,
        'mismatched_keys': mismatched_keys,
        'error_msgs': error_msgs,
    }


def save_configuration(target_folder, config: Dict):
    if isinstance(config, Config):
        config = config.to_dict()
    if ConfigFields.pipeline not in config:
        config[ConfigFields.pipeline] = {'type': config[ConfigFields.task]}
    cfg_str = json.dumps(config, indent=4, cls=JSONIteratorEncoder)
    config_file = os.path.join(target_folder, ModelFile.CONFIGURATION)
    storage.write(cfg_str.encode(), config_file)


def save_pretrained(model,
                    target_folder: Union[str, os.PathLike],
                    save_checkpoint_name: str = None,
                    save_function: Callable = None,
                    **kwargs):
    """save the pretrained model, its configuration and other related files to a directory, so that it can be re-loaded

    Args:
        model (Model): Model whose params are to be saved.

        target_folder (Union[str, os.PathLike]):
        Directory to which to save. Will be created if it doesn't exist.

        save_checkpoint_name (str):
        The checkpoint name to be saved in the target_folder

        save_function (Callable):
        The function to use to save the state dictionary.
    """

    if save_function is None or not isinstance(save_function, Callable):
        raise Exception('A valid save function must be passed in')

    if target_folder is None or os.path.isfile(target_folder):
        raise ValueError(
            f'Provided path ({target_folder}) should be a directory, not a file'
        )

    if save_checkpoint_name is None:
        raise Exception(
            'At least pass in one checkpoint name for saving method')

    # Clean the folder from a previous save
    if os.path.exists(target_folder):
        rmtree(target_folder)

    # Single ckpt path, sharded ckpt logic will be added later
    output_ckpt_path = os.path.join(target_folder, save_checkpoint_name)

    # Save the files to be copied to the save directory, ignore the original ckpts and configuration
    origin_file_to_be_ignored = [save_checkpoint_name]
    ignore_file_set = set(origin_file_to_be_ignored)
    ignore_file_set.add(ModelFile.CONFIGURATION)
    ignore_file_set.add('.*')
    if hasattr(model,
               'model_dir') and model.model_dir is not None and is_master():
        copytree(
            model.model_dir,
            target_folder,
            ignore=ignore_patterns(*ignore_file_set))

    # Save the ckpt to the save directory
    try:
        save_function(model, output_ckpt_path, **kwargs)
    except Exception as e:
        raise Exception(
            f'During saving checkpoints, the error of "{type(e).__name__} '
            f'with msg {e} thrown')
