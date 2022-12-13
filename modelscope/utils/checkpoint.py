# Copyright (c) Alibaba, Inc. and its affiliates.

import io
import os
import time
from collections import OrderedDict
from shutil import copytree, ignore_patterns, rmtree
from typing import Callable, Dict, Optional, Union

import json
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from modelscope import __version__
from modelscope.fileio import File, LocalStorage
from modelscope.utils.config import JSONIteratorEncoder
from modelscope.utils.constant import ConfigFields, ModelFile
from modelscope.utils.logger import get_logger

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
                    with_meta: bool = True) -> None:
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default, ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        lr_scheduler(:obj:`_LRScheduler`, optional): LRScheduler to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
        with_meta (bool, optional):
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(modelscope=__version__, time=time.asctime())

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES=model.CLASSES)

    if with_meta:
        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(model.state_dict())
        }

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
    else:
        checkpoint = weights_to_cpu(model.state_dict())

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

    state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint[
        'state_dict']
    model.load_state_dict(state_dict)
    return checkpoint.get('meta', {})


def save_configuration(target_folder, config: Dict):
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
    if hasattr(model, 'model_dir') and model.model_dir is not None:
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
            f'with msg {e} throwed')
