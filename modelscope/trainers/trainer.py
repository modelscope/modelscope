# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
import time
from collections.abc import Mapping
from distutils.version import LooseVersion
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import json
import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.metrics import build_metric, task_default_metrics
from modelscope.models.base import Model, TorchModel
from modelscope.msdatasets.ms_dataset import MsDataset
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import build_preprocessor
from modelscope.preprocessors.common import Compose
from modelscope.task_datasets.builder import build_task_dataset
from modelscope.task_datasets.torch_base_dataset import TorchTaskDataset
from modelscope.trainers.hooks.builder import HOOKS
from modelscope.trainers.hooks.priority import Priority, get_priority
from modelscope.trainers.lrscheduler.builder import build_lr_scheduler
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import (DEFAULT_MODEL_REVISION, ConfigFields,
                                       ConfigKeys, Hubs, ModeKeys, ModelFile,
                                       Tasks, TrainerStages)
from modelscope.utils.data_utils import to_device
from modelscope.utils.file_utils import func_receive_dict_inputs
from modelscope.utils.logger import get_logger
from modelscope.utils.registry import build_from_cfg
from modelscope.utils.torch_utils import (create_device, get_dist_info,
                                          init_dist)
from .base import BaseTrainer
from .builder import TRAINERS
from .default_config import DEFAULT_CONFIG
from .hooks.hook import Hook
from .parallel.builder import build_parallel
from .parallel.utils import is_parallel


@TRAINERS.register_module(module_name=Trainers.default)
class EpochBasedTrainer(BaseTrainer):
    """Epoch based Trainer, a training helper for PyTorch.

    Args:
        cfg_file(str): The local config file.
        model (:obj:`torch.nn.Module` or :obj:`TorchModel` or `str`): The model to be run, or a valid model dir
            or a model id. If model is None, build_model method will be called.
        data_collator (`Callable`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`.
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
            model: Optional[Union[TorchModel, nn.Module, str]] = None,
            cfg_file: Optional[str] = None,
            arg_parse_fn: Optional[Callable] = None,
            data_collator: Optional[Callable] = None,
            train_dataset: Optional[Union[MsDataset, Dataset]] = None,
            eval_dataset: Optional[Union[MsDataset, Dataset]] = None,
            preprocessor: Optional[Union[Preprocessor,
                                         Dict[str, Preprocessor]]] = None,
            optimizers: Tuple[torch.optim.Optimizer,
                              torch.optim.lr_scheduler._LRScheduler] = (None,
                                                                        None),
            model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
            **kwargs):

        if isinstance(model, str):
            if os.path.exists(model):
                self.model_dir = model if os.path.isdir(
                    model) else os.path.dirname(model)
            else:
                self.model_dir = snapshot_download(
                    model, revision=model_revision)
            if cfg_file is None:
                cfg_file = os.path.join(self.model_dir,
                                        ModelFile.CONFIGURATION)
            self.model = self.build_model()
        else:
            assert cfg_file is not None, 'Config file should not be None if model is an nn.Module class'
            assert isinstance(
                model,
                (TorchModel, nn.Module
                 )), 'model should be either str, TorchMode or nn.Module.'
            self.model_dir = os.path.dirname(cfg_file)
            self.model = model

        super().__init__(cfg_file, arg_parse_fn)
        # add default config
        self.cfg.merge_from_dict(self._get_default_config(), force=False)
        self.cfg = self.rebuild_config(self.cfg)

        if 'work_dir' in kwargs:
            self.work_dir = kwargs['work_dir']
        else:
            self.work_dir = self.cfg.train.get('work_dir', './work_dir')

        self.train_preprocessor, self.eval_preprocessor = None, None
        if isinstance(preprocessor, Preprocessor):
            self.train_preprocessor = preprocessor
            self.eval_preprocessor = preprocessor
        elif isinstance(preprocessor, Mapping):
            if not (ConfigKeys.train in preprocessor
                    or ConfigKeys.val in preprocessor):
                raise ValueError(
                    f'Preprocessor must split with `{ConfigKeys.train}` and `{ConfigKeys.val}` keys!'
                )
            if ConfigKeys.train in preprocessor:
                assert isinstance(preprocessor[ConfigKeys.train], Preprocessor)
                self.train_preprocessor = preprocessor[ConfigKeys.train]
            if ConfigKeys.val in preprocessor:
                assert isinstance(preprocessor[ConfigKeys.val], Preprocessor)
                self.eval_preprocessor = preprocessor[ConfigKeys.val]
        elif hasattr(self.cfg, ConfigFields.preprocessor):
            self.train_preprocessor, self.eval_preprocessor = self.build_preprocessor(
            )

        if self.train_preprocessor is not None:
            self.train_preprocessor.mode = ModeKeys.TRAIN
        if self.eval_preprocessor is not None:
            self.eval_preprocessor.mode = ModeKeys.EVAL

        device_name = kwargs.get('device', 'gpu')
        assert device_name in ['gpu',
                               'cpu'], 'device should be either cpu or gpu.'
        self.device = create_device(device_name == 'cpu')

        self.train_dataset = self.to_task_dataset(
            train_dataset,
            mode=ModeKeys.TRAIN,
            preprocessor=self.train_preprocessor)
        self.eval_dataset = self.to_task_dataset(
            eval_dataset,
            mode=ModeKeys.EVAL,
            preprocessor=self.eval_preprocessor)

        self.data_collator = data_collator if data_collator is not None else default_collate
        self.metrics = self.get_metrics()
        self._metric_values = None
        self.optimizers = optimizers
        self.logger = get_logger(log_level=self.cfg.get('log_level', 'INFO'))
        self._mode = ModeKeys.TRAIN
        self._hooks: List[Hook] = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        if 'max_epochs' not in kwargs:
            assert hasattr(
                self.cfg.train,
                'max_epochs'), 'max_epochs is missing in configuration file'
            self._max_epochs = self.cfg.train.max_epochs
        else:
            self._max_epochs = kwargs['max_epochs']

        self._train_iters_per_epoch = kwargs.get('train_iters_per_epoch', None)
        self._eval_iters_per_epoch = kwargs.get('val_iters_per_epoch', None)
        if self._train_iters_per_epoch is None and hasattr(
                self.cfg.train, 'train_iters_per_epoch'):
            self._train_iters_per_epoch = self.cfg.train.train_iters_per_epoch
        if self._eval_iters_per_epoch is None and hasattr(
                self.cfg, 'evaluation') and hasattr(self.cfg.evaluation,
                                                    'val_iters_per_epoch'):
            self._eval_iters_per_epoch = self.cfg.evaluation.val_iters_per_epoch

        self.use_fp16 = kwargs.get('use_fp16', False)

        # TODO @wenmeng.zwm add seed init fn
        self._seed = 0

        if kwargs.get('launcher', None) is not None:
            init_dist(kwargs['launcher'])

        self._dist = get_dist_info()[1] > 1

        # model placement
        if self.device.type == 'cuda':
            self.model.to(self.device)
            if not is_parallel(self.model) and self._dist:
                self.model = self.to_parallel(self.model)

    def rebuild_config(self, cfg: Config):
        """A method used to rebuild the config, any subclass can override this method.

        Returns: The rebuilt config

        """
        return cfg

    @property
    def mode(self):
        return self._mode

    @property
    def hooks(self) -> List[Hook]:
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self) -> int:
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self) -> int:
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self) -> int:
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_epochs * self.iters_per_epoch

    @property
    def iters_per_epoch(self):
        """int: Total iterations of one epoch"""

        def _get_data_len(data_loader):
            try:
                return len(data_loader)
            except Exception as e:
                self.logger.error(e)
                raise ValueError(
                    'Please implement ``__len__`` method for your dataset, '
                    'or add `train_iters_per_epoch` and `train_iters_per_epoch` '
                    'to your configuration file or kwargs')

        if self.mode == ModeKeys.TRAIN:
            if self._train_iters_per_epoch is not None:
                return self._train_iters_per_epoch
            else:
                return _get_data_len(self.train_dataloader)
        elif self.mode == ModeKeys.EVAL:
            if self._eval_iters_per_epoch is not None:
                return self._eval_iters_per_epoch
            else:
                return _get_data_len(self.eval_dataloader)

    def to_task_dataset(self,
                        datasets: Union[Dataset, List[Dataset]],
                        mode: str,
                        preprocessor: Optional[Preprocessor] = None):
        """Build the task specific dataset processor for this trainer.

        Returns: The task dataset processor for the task. If no result for the very model-type and task,
        the default TaskDataset will be returned.
        """
        try:
            if not datasets:
                return datasets
            if isinstance(datasets, TorchTaskDataset):
                return datasets
            elif isinstance(datasets, MsDataset):
                datasets = datasets.to_torch_dataset(
                    preprocessors=preprocessor)
                return datasets
            elif isinstance(datasets, List) and isinstance(
                    datasets[0], MsDataset):
                datasets = [
                    d.to_torch_dataset(preprocessor=preprocessor)
                    for d in datasets
                ]
                cfg = ConfigDict(
                    type=self.cfg.task, mode=mode, datasets=datasets)
                return build_task_dataset(cfg, self.cfg.task)
            else:
                cfg = ConfigDict(
                    type=self.cfg.model.type,
                    mode=mode,
                    datasets=datasets,
                    preprocessor=preprocessor)
                return build_task_dataset(cfg, self.cfg.task)
        except Exception:
            if isinstance(datasets, (List, Tuple)) or preprocessor is not None:
                return TorchTaskDataset(
                    datasets,
                    mode=mode,
                    preprocessor=preprocessor,
                    **(dict(type=self.cfg.model.type) if hasattr(
                        self.cfg, 'model') else {}))
            else:
                return datasets

    def build_preprocessor(self) -> Tuple[Preprocessor, Preprocessor]:
        """Build train and eval preprocessor.

        User can override this method to implement custom logits.

        Returns: The train preprocessor and eval preprocessor instance.

        """
        field_name = Tasks.find_field_by_task(self.cfg.task)
        train_preprocessor, eval_preprocessor = None, None
        _train_cfg, _eval_cfg = {}, {}
        _dafault_args = {'model_dir': self.model_dir}

        if 'type' not in self.cfg.preprocessor and (
                'train' in self.cfg.preprocessor
                or 'val' in self.cfg.preprocessor):
            if 'train' in self.cfg.preprocessor:
                _train_cfg = self.cfg.preprocessor.train
            if 'val' in self.cfg.preprocessor:
                _eval_cfg = self.cfg.preprocessor.val
        else:
            _train_cfg = self.cfg.preprocessor
            _eval_cfg = self.cfg.preprocessor

        if len(_train_cfg):
            if isinstance(_train_cfg, Sequence):
                # TODO: for Sequence, need adapt to `mode` and `mode_dir` args,
                # and add mode for Compose or other plans
                raise NotImplementedError('Not supported yet!')
            _train_cfg.update(_dafault_args)
            train_preprocessor = build_preprocessor(_train_cfg, field_name)
        if len(_eval_cfg):
            if isinstance(_eval_cfg, Sequence):
                raise NotImplementedError('Not supported yet!')
            _eval_cfg.update(_dafault_args)
            eval_preprocessor = build_preprocessor(_eval_cfg, field_name)

        return train_preprocessor, eval_preprocessor

    def get_metrics(self) -> List[str]:
        """Get the metric class types.

        The first choice will be the metrics configured in the config file, if not found, the default metrics will be
        used.
        If no metrics is found and the eval dataset exists, the method will raise an error.

        Returns: The metric types.

        """
        metrics = self.cfg.evaluation.metrics if hasattr(
            self.cfg, 'evaluation') and hasattr(self.cfg.evaluation,
                                                'metrics') else None
        metrics = metrics if metrics is not None else task_default_metrics.get(
            self.cfg.task)
        if metrics is None and self.eval_dataset is not None:
            raise ValueError(
                f'Metrics are needed in evaluation, please try to either '
                f'add metrics in configuration.json or add the default metric for {self.cfg.task}.'
            )
        if isinstance(metrics, str):
            metrics = [metrics]
        return metrics

    def train(self, *args, **kwargs):
        self.model.train()
        self._mode = ModeKeys.TRAIN

        if self.train_dataset is None:
            self.train_dataloader = self.get_train_dataloader()
        else:
            self.train_dataloader = self._build_dataloader_with_dataset(
                self.train_dataset,
                dist=self._dist,
                seed=self._seed,
                **self.cfg.train.get('dataloader', {}))
        self.data_loader = self.train_dataloader

        self.register_optimizers_hook()
        self.register_hook_from_cfg(self.cfg.train.hooks)

        self.train_loop(self.train_dataloader)

    def evaluate(self, checkpoint_path=None):
        self.model.eval()
        self._mode = ModeKeys.EVAL

        if self.eval_dataset is None:
            self.eval_dataloader = self.get_eval_data_loader()
        else:
            self.eval_dataloader = self._build_dataloader_with_dataset(
                self.eval_dataset,
                dist=self._dist,
                seed=self._seed,
                **self.cfg.evaluation.get('dataloader', {}))
        self.data_loader = self.eval_dataloader
        metric_classes = [build_metric(metric) for metric in self.metrics]
        for m in metric_classes:
            m.trainer = self
        metric_values = self.evaluation_loop(self.eval_dataloader,
                                             checkpoint_path, metric_classes)

        self._metric_values = metric_values
        return metric_values

    @property
    def metric_values(self):
        return self._metric_values

    def build_model(self) -> Union[nn.Module, TorchModel]:
        """ Instantiate a pytorch model and return.

        By default, we will create a model using config from configuration file. You can
        override this method in a subclass.

        """
        model = Model.from_pretrained(self.model_dir)
        if not isinstance(model, nn.Module) and hasattr(model, 'model'):
            return model.model
        elif isinstance(model, nn.Module):
            return model

    def to_parallel(self, model) -> Union[nn.Module, TorchModel]:
        # config format to reserve custom ddp
        if self.cfg.get('parallel', None) is not None:
            self.cfg.parallel.update(
                dict(module=model, device_ids=[torch.cuda.current_device()]))
            return build_parallel(self.cfg.parallel)

        dp_cfg = dict(
            type='DistributedDataParallel',
            module=model,
            device_ids=[torch.cuda.current_device()])

        return build_parallel(dp_cfg)

    def train_step(self, model, inputs):
        """ Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`TorchModel`): The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # EvaluationHook will do evaluate and change mode to val, return to train mode
        # TODO: find more pretty way to change mode
        model.train()
        self._mode = ModeKeys.TRAIN
        # call model forward but not __call__ to skip postprocess
        if isinstance(inputs,
                      Mapping) and not func_receive_dict_inputs(model.forward):
            train_outputs = model.forward(**inputs)
        else:
            train_outputs = model.forward(inputs)

        if not isinstance(train_outputs, dict):
            raise TypeError('"model.forward()" must return a dict')

        # add model output info to log
        if 'log_vars' not in train_outputs:
            default_keys_pattern = ['loss']
            match_keys = set([])
            for key_p in default_keys_pattern:
                match_keys.update(
                    [key for key in train_outputs.keys() if key_p in key])

            log_vars = {}
            for key in match_keys:
                value = train_outputs.get(key, None)
                if value is not None:
                    if dist.is_available() and dist.is_initialized():
                        value = value.data.clone()
                        dist.all_reduce(value.div_(dist.get_world_size()))
                    log_vars.update({key: value.item()})
            self.log_buffer.update(log_vars)
        else:
            self.log_buffer.update(train_outputs['log_vars'])

        self.train_outputs = train_outputs

    def prediction_step(self, model, inputs):
        """ Perform forward step by `model` using `inputs`.

        Args:
            model (`TorchModel`): The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        raise NotImplementedError

    def get_train_dataloader(self):
        """ Builder torch dataloader for training.

        We provide a reasonable default that works well. If you want to use something else, you can change
        the config for data.train in configuration file, or subclass and override this method
        (or `get_train_dataloader` in a subclass.
        """
        if self.train_dataset is None:
            train_data = self.cfg.dataset.train
            self.train_dataset = self.build_dataset(
                train_data,
                mode=ModeKeys.TRAIN,
                preprocessor=self.train_preprocessor)

        data_loader = self._build_dataloader_with_dataset(
            self.train_dataset,
            dist=self._dist,
            seed=self._seed,
            **self.cfg.train.get('dataloader', {}))
        return data_loader

    def get_eval_data_loader(self):
        """ Builder torch dataloader for evaluation.

        We provide a reasonable default that works well. If you want to use something else, you can change
        the config for dataset.eval in configuration file, or subclass and override this method in a subclass.
        pass
        """
        if self.eval_dataset is None:
            val_data = self.cfg.dataset.val
            self.eval_dataset = self.build_dataset(
                val_data,
                mode=ModeKeys.EVAL,
                preprocessor=self.eval_preprocessor)

        batch_size = self.cfg.evaluation.batch_size
        workers = self.cfg.evaluation.workers
        shuffle = self.cfg.evaluation.get('shuffle', False)
        data_loader = self._build_dataloader_with_dataset(
            self.eval_dataset,
            batch_size_per_gpu=batch_size,
            workers_per_gpu=workers,
            shuffle=shuffle,
            dist=self._dist,
            seed=self._seed,
            persistent_workers=True,
        )
        return data_loader

    def build_dataset(self, data_cfg, mode, preprocessor=None):
        """ Build torch dataset object using data config
        """
        dataset = MsDataset.load(
            dataset_name=data_cfg.name,
            split=data_cfg.split,
            subset_name=data_cfg.subset_name if hasattr(
                data_cfg, 'subset_name') else None,
            hub=data_cfg.hub if hasattr(data_cfg, 'hub') else Hubs.modelscope,
        )
        torch_dataset = dataset.to_torch_dataset(preprocessors=preprocessor)
        dataset = self.to_task_dataset(torch_dataset, mode)
        return dataset

    def create_optimizer_and_scheduler(self):
        """ Create optimizer and lr scheduler

        We provide a default implementation, if you want to customize your own optimizer
        and lr scheduler, you can either pass a tuple through trainer init function or
        subclass this class and override this method.
        """
        optimizer, lr_scheduler = self.optimizers
        if optimizer is None:
            optimizer_cfg = self.cfg.train.get('optimizer', None)
        else:
            optimizer_cfg = None

        optim_options = {}
        if optimizer_cfg is not None:
            optim_options = optimizer_cfg.pop('options', {})
            optimizer = build_optimizer(self.model, cfg=optimizer_cfg)

        if lr_scheduler is None:
            lr_scheduler_cfg = self.cfg.train.get('lr_scheduler', None)
        else:
            lr_scheduler_cfg = None

        lr_options = {}
        if lr_scheduler_cfg is not None:
            assert optimizer is not None
            lr_options = lr_scheduler_cfg.pop('options', {})
            lr_scheduler = build_lr_scheduler(
                cfg=lr_scheduler_cfg, default_args={'optimizer': optimizer})

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        return self.optimizer, self.lr_scheduler, optim_options, lr_options

    def register_optimizers_hook(self):
        """ Register optimizer hook and lr scheduler hook.
        """
        _, lr_scheduler, optim_options, lr_options = self.create_optimizer_and_scheduler(
        )

        optim_hook = self.cfg.train.get('optimizer_hook', None)
        lr_hook = self.cfg.train.get('lr_scheduler_hook', None)

        # adapt to `ReduceLROnPlateau`
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        if isinstance(lr_scheduler, ReduceLROnPlateau) and lr_hook is None:
            plateau_cfg = {
                'train': {
                    'lr_scheduler_hook': {
                        'type': 'PlateauLrSchedulerHook',
                        'metric_key':
                        'Metric Key used for PlateauLrSchedulerHook'
                    }
                }
            }
            plateau_cfg = json.dumps(
                plateau_cfg, sort_keys=False, indent=4, separators=(',', ':'))
            raise ValueError(
                'Must add `lr_scheduler_hook` to configuration for `ReduceLROnPlateau` lr scheduler as follows:'
                + '\n' + plateau_cfg)

        if lr_hook is None:
            lr_hook = dict(type='LrSchedulerHook', **lr_options)
        if optim_hook is None:
            if self.use_fp16:
                optim_hook = dict(
                    type='TorchAMPOptimizerHook', **optim_options)
            else:
                optim_hook = dict(type='OptimizerHook', **optim_options)

        self.register_hook_from_cfg([lr_hook, optim_hook])

    def _build_dataloader_with_dataset(self,
                                       dataset: Dataset,
                                       batch_size_per_gpu: int,
                                       workers_per_gpu: int,
                                       dist: bool = False,
                                       shuffle: bool = True,
                                       seed: int = 0,
                                       persistent_workers=False,
                                       **kwargs) -> DataLoader:
        """Build dataloader using input dataset and cfg. Used by `EpochBasedTrainer.train()`
        and `EpochBasedTrainer.evaluate()`.

        In distributed training, each GPU/process has a dataloader.
        In non-distributed training, there is only one dataloader for all GPUs.

        Args:
            dataset (Dataset): A PyTorch dataset.
            batch_size_per_gpu (int): Number of training samples on each GPU, i.e.,
                batch size of each GPU.
            workers_per_gpu (int): How many subprocesses to use for data loading
                for each GPU.
            dist (bool): Distributed training/test or not. Default: True.
            shuffle (bool): Whether to shuffle the data at every epoch.
                Default: True.
            seed (int, Optional): Seed to be used. Default: 0.
            runner_type (str): Type of runner. Default: `EpochBasedRunner`
            persistent_workers (bool): If True, the data loader will not shutdown
                the worker processes after a dataset has been consumed once.
                This allows to maintain the workers `Dataset` instances alive.
                This argument is only valid when PyTorch>=1.7.0. Default: False.
            kwargs: any keyword argument to be used to initialize DataLoader

        Returns:
            DataLoader: A PyTorch dataloader.
        """
        rank, world_size = get_dist_info()

        if dist:
            # When model is :obj:`DistributedDataParallel`,
            # `batch_size` of :obj:`dataloader` is the
            # number of training samples on each GPU.
            batch_size = batch_size_per_gpu
            num_workers = workers_per_gpu
        else:
            batch_size = batch_size_per_gpu
            num_workers = workers_per_gpu

        if dist:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        else:
            sampler = None

        batch_sampler = None

        init_fn = partial(
            worker_init_fn, num_workers=num_workers, rank=rank,
            seed=seed) if seed is not None else None

        if LooseVersion(torch.__version__) >= LooseVersion('1.7.0'):
            kwargs['persistent_workers'] = persistent_workers
        elif persistent_workers is True:
            self.logger.warning(
                'persistent_workers is invalid because your pytorch '
                'version is lower than 1.7.0')

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            pin_memory=kwargs.pop('pin_memory', False),
            worker_init_fn=init_fn,
            **kwargs)

        return data_loader

    def train_loop(self, data_loader):
        """ Training loop used by `EpochBasedTrainer.train()`
        """
        self.invoke_hook(TrainerStages.before_run)
        self._epoch = 0
        kwargs = {}
        self.model.train()
        for _ in range(self._epoch, self._max_epochs):
            self.invoke_hook(TrainerStages.before_train_epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            for i, data_batch in enumerate(data_loader):
                data_batch = to_device(data_batch, self.device)
                self.data_batch = data_batch
                self._inner_iter = i
                self.invoke_hook(TrainerStages.before_train_iter)
                self.train_step(self.model, data_batch, **kwargs)
                self.invoke_hook(TrainerStages.after_train_iter)
                del self.data_batch
                self._iter += 1

                if i + 1 >= self.iters_per_epoch:
                    break

            self.invoke_hook(TrainerStages.after_train_epoch)
            self._epoch += 1

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.invoke_hook(TrainerStages.after_run)

    def evaluation_loop(self, data_loader, checkpoint_path, metric_classes):
        """ Evaluation loop used by `EpochBasedTrainer.evaluate()`.

        """
        if self._dist:
            from modelscope.trainers.utils.inference import multi_gpu_test
            metric_values = multi_gpu_test(
                self.model,
                data_loader,
                device=self.device,
                tmpdir=None,
                gpu_collect=False,
                metric_classes=metric_classes,
                data_loader_iters_per_gpu=self.iters_per_epoch)
        else:
            from modelscope.trainers.utils.inference import single_gpu_test
            metric_values = single_gpu_test(
                self.model,
                data_loader,
                device=self.device,
                metric_classes=metric_classes,
                data_loader_iters=self.iters_per_epoch)

        self._inner_iter = self.iters_per_epoch - 1  # start from index 0

        return metric_values

    def register_hook(self, hook: Hook) -> None:
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
        """
        assert isinstance(hook, Hook)
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if get_priority(hook.PRIORITY) > get_priority(
                    self._hooks[i].PRIORITY):
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def register_hook_from_cfg(self, hook_cfg: Dict) -> None:
        """Register a hook from its cfg.

        Args:
            hook_cfg (dict): Hook config. It should have at least keys 'type'
              and 'priority' indicating its type and priority.

        Note:
            The specific hook class to register should not use 'type' and
            'priority' arguments during initialization.
        """
        hook_cfg = hook_cfg.copy()
        assert isinstance(hook_cfg, list)
        for cfg_i in hook_cfg:
            hook = build_from_cfg(cfg_i, HOOKS)
            self.register_hook(hook)

    def invoke_hook(self, fn_name: str) -> None:
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def get_hook_info(self) -> str:
        # Get hooks info in each stage
        stage_hook_map: Dict[str, list] = {stage: [] for stage in Hook.stages}
        for hook in self.hooks:
            try:
                priority = Priority(hook.priority).name  # type: ignore
            except ValueError:
                priority = hook.priority  # type: ignore
            classname = hook.__class__.__name__
            hook_info = f'({priority:<12}) {classname:<35}'
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f'{stage}:\n'
                info += '\n'.join(hook_infos)
                info += '\n -------------------- '
                stage_hook_infos.append(info)
        return '\n'.join(stage_hook_infos)

    def _get_default_config(self):
        return DEFAULT_CONFIG


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
