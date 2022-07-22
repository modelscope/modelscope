# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
import random
import time
from collections.abc import Mapping
from distutils.version import LooseVersion
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from addict import Dict
from torch import distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metrics import build_metric, task_default_metrics
from modelscope.models.base import Model, TorchModel
from modelscope.msdatasets.ms_dataset import MsDataset
from modelscope.preprocessors import build_preprocessor
from modelscope.preprocessors.base import Preprocessor
from modelscope.trainers.hooks.builder import HOOKS
from modelscope.trainers.hooks.priority import Priority, get_priority
from modelscope.trainers.lrscheduler.builder import build_lr_scheduler
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import (DEFAULT_MODEL_REVISION, Hubs, ModeKeys,
                                       ModelFile, Tasks, TrainerStages)
from modelscope.utils.logger import get_logger
from modelscope.utils.registry import build_from_cfg
from modelscope.utils.tensor_utils import torch_default_data_collator
from modelscope.utils.torch_utils import get_dist_info
from modelscope.utils.utils import if_func_recieve_dict_inputs
from .base import BaseTrainer
from .builder import TRAINERS
from .default_config import DEFAULT_CONFIG
from .hooks.hook import Hook


@TRAINERS.register_module()
class EpochBasedTrainer(BaseTrainer):
    """Epoch based Trainer, a training helper for PyTorch.

    Args:
        cfg_file(str): The local config file.
        model (:obj:`torch.nn.Module` or :obj:`TorchModel` or `str`): The model to be run, or a valid model dir
            or a model id. If model is None, build_model method will be called.
        data_collator (`Callable`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`.
        train_dataset (`MsDataset`, *optional*):
            The dataset to use for training.

            Note that if it's a `torch.utils.data.IterableDataset` with some randomization and you are training in a
            distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
            `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will
            manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
            sets the seed of the RNGs used.
        eval_dataset (`torch.utils.data.Dataset`, *optional*): The dataset to use for evaluation.
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
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            preprocessor: Optional[Preprocessor] = None,
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
            cfg_file = os.path.join(self.model_dir, ModelFile.CONFIGURATION)
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

        if 'work_dir' in kwargs:
            self.work_dir = kwargs['work_dir']
        else:
            self.work_dir = self.cfg.train.get('work_dir', './work_dir')

        self.preprocessor = None
        if isinstance(preprocessor, Preprocessor):
            self.preprocessor = preprocessor
        elif hasattr(self.cfg, 'preprocessor'):
            self.preprocessor = self.build_preprocessor()
        if self.preprocessor is not None:
            self.preprocessor.mode = ModeKeys.TRAIN
        # TODO @wenmeng.zwm add data collator option
        # TODO how to fill device option?
        self.device = int(
            os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else None
        self.train_dataset = train_dataset.to_torch_dataset(
            preprocessors=self.preprocessor) if train_dataset else None
        self.eval_dataset = eval_dataset.to_torch_dataset(
            preprocessors=self.preprocessor) if eval_dataset else None
        self.data_collator = data_collator if data_collator is not None else torch_default_data_collator
        self.metrics = self.get_metrics()
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

        self.use_fp16 = kwargs.get('use_fp16', False)

        # TODO @wenmeng.zwm add seed init fn
        self._seed = 0

        self._dist = get_dist_info()[1] > 1

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
        return self._max_epochs * len(self.data_loader)

    def build_preprocessor(self) -> Preprocessor:
        """Build the preprocessor.

        User can override this method to implement custom logits.

        Returns: The preprocessor instance.

        """
        # TODO @wenmeng.zwm @jiangnana.jnn add support for different preprocessor
        # when they are different ones in training and evaluation
        cfg = ConfigDict({
            **getattr(self.cfg, 'preprocessor'), 'model_dir':
            self.model_dir
        })
        return build_preprocessor(cfg, Tasks.find_field_by_task(self.cfg.task))

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
                self.train_dataset, **self.cfg.train.get('dataloader', {}))
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
                self.eval_dataset, **self.cfg.evaluation.get('dataloader', {}))
        self.data_loader = self.eval_dataloader
        metric_classes = [build_metric(metric) for metric in self.metrics]
        self.evaluation_loop(self.eval_dataloader, checkpoint_path,
                             metric_classes)

        metric_values = {}
        for metric_cls in metric_classes:
            metric_values.update(metric_cls.evaluate())
        return metric_values

    def build_model(self) -> Union[nn.Module, TorchModel]:
        """ Instantiate a pytorch model and return.

        By default, we will create a model using config from configuration file. You can
        subclass and override this method in a subclass.

        """
        # TODO temp implementation, waiting for @zhangzhicheng
        model = Model.from_pretrained(self.model_dir)
        if not isinstance(model, nn.Module) and hasattr(model, 'model'):
            return model.model
        elif isinstance(model, nn.Module):
            return model

    def collate_fn(self, data):
        """Prepare the input just before the forward function.
        This method will move the tensors to the right device.
        Usually this method does not need to be overridden.

        Args:
            data: The data out of the dataloader.

        Returns: The processed data.

        """
        if isinstance(data, dict):
            return type(data)({k: self.collate_fn(v) for k, v in data.items()})
        elif isinstance(data, (tuple, np.ndarray, list)):
            return type(data)(self.collate_fn(v) for v in data)
        elif isinstance(data, torch.Tensor) and self.device is not None:
            kwargs = dict(device=self.device)
            return data.to(**kwargs)
        return data

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
        inputs = self.collate_fn(inputs)
        if isinstance(inputs, Mapping) and not if_func_recieve_dict_inputs(
                model.forward, inputs):
            train_outputs = model.forward(**inputs)
        else:
            train_outputs = model.forward(inputs)

        if not isinstance(train_outputs, dict):
            raise TypeError(
                '"model.train_step()" and "model.val_step()" must return a dict'
            )

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
        train_data = self.cfg.dataset.train
        if self.train_dataset is None:
            self.train_dataset = self.build_dataset(
                train_data, mode=ModeKeys.TRAIN)

        data_loader = self._build_dataloader_with_dataset(
            self.train_dataset, **self.cfg.train.get('dataloader', {}))
        return data_loader

    def get_eval_data_loader(self):
        """ Builder torch dataloader for evaluation.

        We provide a reasonable default that works well. If you want to use something else, you can change
        the config for dataset.eval in configuration file, or subclass and override this method in a subclass.
        pass
        """
        val_data = self.cfg.dataset.val
        if self.eval_dataset is None:
            self.eval_dataset = self.build_dataset(
                val_data, mode=ModeKeys.TRAIN)

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

    def build_dataset(self, data_cfg, mode):
        """ Build torch dataset object using data config
        """
        dataset = MsDataset.load(
            dataset_name=data_cfg.name,
            split=data_cfg.split,
            subset_name=data_cfg.subset_name if hasattr(
                data_cfg, 'subset_name') else None,
            hub=data_cfg.hub if hasattr(data_cfg, 'hub') else Hubs.modelscope,
        )
        torch_dataset = dataset.to_torch_dataset(
            preprocessors=self.preprocessor, )
        return torch_dataset

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
        optimizer, lr_scheduler = self.optimizers
        opti_error_msg = 'optimizers should be a tuple of `torch.optim.Optimizer`'\
                         ' and `torch.optim.lr_scheduler._LRScheduler`'
        if optimizer is not None:
            assert isinstance(optimizer, torch.optim.Optimizer), opti_error_msg
        if lr_scheduler is not None:
            assert isinstance(
                lr_scheduler,
                torch.optim.lr_scheduler._LRScheduler), opti_error_msg

        _, _, optim_options, lr_options = self.create_optimizer_and_scheduler()
        lr_hook = dict(type='LrSchedulerHook', **lr_options)
        if self.use_fp16:
            optim_hook = dict(type='TorchAMPOptimizerHook', **optim_options)
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
                dataset, world_size, rank, shuffle=shuffle, seed=seed)
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
        for _ in range(self._epoch, self._max_epochs):
            self.invoke_hook(TrainerStages.before_train_epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            for i, data_batch in enumerate(data_loader):
                self.data_batch = data_batch
                self._inner_iter = i
                self.invoke_hook(TrainerStages.before_train_iter)
                self.train_step(self.model, data_batch, **kwargs)
                self.invoke_hook(TrainerStages.after_train_iter)
                del self.data_batch
                self._iter += 1

            self.invoke_hook(TrainerStages.after_train_epoch)
            self._epoch += 1

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.invoke_hook(TrainerStages.after_run)

    def evaluation_loop(self, data_loader, checkpoint_path, metric_classes):
        """ Evaluation loop used by `EpochBasedTrainer.evaluate()`.

        """
        if self._dist:
            from modelscope.trainers.utils.inference import multi_gpu_test
            multi_gpu_test(
                self.model,
                data_loader,
                tmpdir=None,
                gpu_collect=False,
                data_collate_fn=self.collate_fn,
                metric_classes=metric_classes)
        else:
            from modelscope.trainers.utils.inference import single_gpu_test
            single_gpu_test(
                self.model,
                data_loader,
                data_collate_fn=self.collate_fn,
                metric_classes=metric_classes)

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
