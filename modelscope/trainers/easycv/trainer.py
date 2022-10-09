# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset

from modelscope.metainfo import Trainers
from modelscope.models.base import TorchModel
from modelscope.msdatasets import MsDataset
from modelscope.preprocessors import Preprocessor
from modelscope.trainers import EpochBasedTrainer
from modelscope.trainers.base import TRAINERS
from modelscope.trainers.easycv.utils import register_util
from modelscope.trainers.hooks import HOOKS
from modelscope.trainers.parallel.builder import build_parallel
from modelscope.trainers.parallel.utils import is_parallel
from modelscope.utils.config import Config
from modelscope.utils.constant import DEFAULT_MODEL_REVISION
from modelscope.utils.import_utils import LazyImportModule
from modelscope.utils.registry import default_group


@TRAINERS.register_module(module_name=Trainers.easycv)
class EasyCVEpochBasedTrainer(EpochBasedTrainer):
    """Epoch based Trainer for EasyCV.

    Args:
        cfg_file(str): The config file of EasyCV.
        model (:obj:`torch.nn.Module` or :obj:`TorchModel` or `str`): The model to be run, or a valid model dir
            or a model id. If model is None, build_model method will be called.
        train_dataset (`MsDataset` or `torch.utils.data.Dataset`, *optional*):
            The dataset to use for training.
            Note that if it's a `torch.utils.data.IterableDataset` with some randomization and you are training in a
            distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
            `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will
            manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
            sets the seed of the RNGs used.
        eval_dataset (`MsDataset` or `torch.utils.data.Dataset`, *optional*): The dataset to use for evaluation.
        preprocessor (:obj:`Preprocessor`, *optional*): The optional preprocessor.
            NOTE: If the preprocessor has been called before the dataset fed into this trainer by user's custom code,
            this parameter should be None, meanwhile remove the 'preprocessor' key from the cfg_file.
            Else the preprocessor will be instantiated from the cfg_file or assigned from this parameter and
            this preprocessing action will be executed every time the dataset's __getitem__ is called.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]`, *optional*): A tuple
            containing the optimizer and the scheduler to use.
        max_epochs: (int, optional): Total training epochs.
    """

    def __init__(
            self,
            cfg_file: Optional[str] = None,
            model: Optional[Union[TorchModel, nn.Module, str]] = None,
            arg_parse_fn: Optional[Callable] = None,
            train_dataset: Optional[Union[MsDataset, Dataset]] = None,
            eval_dataset: Optional[Union[MsDataset, Dataset]] = None,
            preprocessor: Optional[Preprocessor] = None,
            optimizers: Tuple[torch.optim.Optimizer,
                              torch.optim.lr_scheduler._LRScheduler] = (None,
                                                                        None),
            model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
            **kwargs):

        register_util.register_parallel()
        register_util.register_part_mmcv_hooks_to_ms()

        super(EasyCVEpochBasedTrainer, self).__init__(
            model=model,
            cfg_file=cfg_file,
            arg_parse_fn=arg_parse_fn,
            preprocessor=preprocessor,
            optimizers=optimizers,
            model_revision=model_revision,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs)

        # reset data_collator
        from mmcv.parallel import collate

        self.train_data_collator = partial(
            collate,
            samples_per_gpu=self.cfg.train.dataloader.batch_size_per_gpu)
        self.eval_data_collator = partial(
            collate,
            samples_per_gpu=self.cfg.evaluation.dataloader.batch_size_per_gpu)

        # Register easycv hooks dynamicly. If the hook already exists in modelscope,
        # the hook in modelscope will be used, otherwise register easycv hook into ms.
        # We must manually trigger lazy import to detect whether the hook is in modelscope.
        # TODO: use ast index to detect whether the hook is in modelscope
        for h_i in self.cfg.train.get('hooks', []):
            sig = ('HOOKS', default_group, h_i['type'])
            LazyImportModule.import_module(sig)
            if h_i['type'] not in HOOKS._modules[default_group]:
                if h_i['type'] in [
                        'TensorboardLoggerHookV2', 'WandbLoggerHookV2'
                ]:
                    raise ValueError(
                        'Not support hook %s now, we will support it in the future!'
                        % h_i['type'])
                register_util.register_hook_to_ms(h_i['type'], self.logger)

        # reset parallel
        if not self._dist:
            assert not is_parallel(
                self.model
            ), 'Not support model wrapped by custom parallel if not in distributed mode!'
            dp_cfg = dict(
                type='MMDataParallel',
                module=self.model,
                device_ids=[torch.cuda.current_device()])
            self.model = build_parallel(dp_cfg)

    def create_optimizer_and_scheduler(self):
        """ Create optimizer and lr scheduler
        """
        optimizer, lr_scheduler = self.optimizers
        if optimizer is None:
            optimizer_cfg = self.cfg.train.get('optimizer', None)
        else:
            optimizer_cfg = None

        optim_options = {}
        if optimizer_cfg is not None:
            optim_options = optimizer_cfg.pop('options', {})
            from easycv.apis.train import build_optimizer
            optimizer = build_optimizer(self.model, optimizer_cfg)

        if lr_scheduler is None:
            lr_scheduler_cfg = self.cfg.train.get('lr_scheduler', None)
        else:
            lr_scheduler_cfg = None

        lr_options = {}
        # Adapt to mmcv lr scheduler hook.
        # Please refer to: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py
        if lr_scheduler_cfg is not None:
            assert optimizer is not None
            lr_options = lr_scheduler_cfg.pop('options', {})
            assert 'policy' in lr_scheduler_cfg
            policy_type = lr_scheduler_cfg.pop('policy')
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'LrUpdaterHook'
            lr_scheduler_cfg['type'] = hook_type

            self.cfg.train.lr_scheduler_hook = lr_scheduler_cfg

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        return self.optimizer, self.lr_scheduler, optim_options, lr_options

    def to_parallel(self, model) -> Union[nn.Module, TorchModel]:
        if self.cfg.get('parallel', None) is not None:
            self.cfg.parallel.update(
                dict(module=model, device_ids=[torch.cuda.current_device()]))
            return build_parallel(self.cfg.parallel)

        dp_cfg = dict(
            type='MMDistributedDataParallel',
            module=model,
            device_ids=[torch.cuda.current_device()])

        return build_parallel(dp_cfg)
