# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import os
from collections.abc import Mapping
from copy import deepcopy
from distutils.version import LooseVersion
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import json
import torch
from torch import distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from modelscope.metainfo import Trainers
from modelscope.metrics import build_metric, task_default_metrics
from modelscope.metrics.prediction_saving_wrapper import \
    PredictionSavingWrapper
from modelscope.models.base import Model, TorchModel
from modelscope.msdatasets.ms_dataset import MsDataset
from modelscope.msdatasets.task_datasets.builder import build_task_dataset
from modelscope.msdatasets.task_datasets.torch_base_dataset import \
    TorchTaskDataset
from modelscope.outputs import ModelOutputBase
from modelscope.preprocessors.base import Preprocessor
from modelscope.trainers.hooks.builder import HOOKS
from modelscope.trainers.hooks.priority import Priority, get_priority
from modelscope.trainers.lrscheduler.builder import build_lr_scheduler
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.utils.config import Config, ConfigDict, JSONIteratorEncoder
from modelscope.utils.constant import (DEFAULT_MODEL_REVISION, ConfigFields,
                                       ConfigKeys, ModeKeys, ModelFile,
                                       ThirdParty, TrainerStages)
from modelscope.utils.data_utils import to_device
from modelscope.utils.device import create_device
from modelscope.utils.file_utils import func_receive_dict_inputs
from modelscope.utils.logger import get_logger
from modelscope.utils.megatron_utils import is_megatron_initialized
from modelscope.utils.registry import build_from_cfg
from modelscope.utils.torch_utils import (get_dist_info, get_local_rank,
                                          init_dist, is_dist, is_master,
                                          set_random_seed)
from .base import BaseTrainer
from .builder import TRAINERS
from .default_config import merge_cfg, merge_hooks
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
        seed (int): The optional random seed for torch, cuda, numpy and random.
        max_epochs: (int, optional): Total training epochs.
        cfg_modify_fn: An input fn which is used to modify the cfg read out of the file.
        remove_unused_data: Automatically remove unused data keys in mini-batches.
            The remove action based on the `inspect` on the model's forward method, the removed columns will be
            moved to the mini-batch's attributes.

        Examples of cfg_modify_fn:
            >>> def cfg_modify_fn(cfg):
            >>>     cfg.preprocessor.first_sequence= 'text1'
            >>>     cfg.preprocessor.second_sequence='text2'
            >>>     return cfg
    """

    def __init__(
            self,
            model: Optional[Union[TorchModel, nn.Module, str]] = None,
            cfg_file: Optional[str] = None,
            cfg_modify_fn: Optional[Callable] = None,
            arg_parse_fn: Optional[Callable] = None,
            data_collator: Optional[Union[Callable, Dict[str,
                                                         Callable]]] = None,
            train_dataset: Optional[Union[MsDataset, Dataset]] = None,
            eval_dataset: Optional[Union[MsDataset, Dataset]] = None,
            preprocessor: Optional[Union[Preprocessor,
                                         Dict[str, Preprocessor]]] = None,
            optimizers: Tuple[torch.optim.Optimizer,
                              torch.optim.lr_scheduler._LRScheduler] = (None,
                                                                        None),
            model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
            seed: int = 42,
            **kwargs):

        self._seed = seed
        set_random_seed(self._seed)
        self._metric_values = None
        self.optimizers = optimizers
        self._mode = ModeKeys.TRAIN
        self._hooks: List[Hook] = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._stop_training = False

        if isinstance(model, str):
            third_party = kwargs.get(ThirdParty.KEY)
            if third_party is not None:
                kwargs.pop(ThirdParty.KEY)

            self.model_dir = self.get_or_download_model_dir(
                model, model_revision, third_party)
            if cfg_file is None:
                cfg_file = os.path.join(self.model_dir,
                                        ModelFile.CONFIGURATION)
        else:
            assert cfg_file is not None, 'Config file should not be None if model is not from pretrained!'
            self.model_dir = os.path.dirname(cfg_file)

        super().__init__(cfg_file, arg_parse_fn)
        self.cfg_modify_fn = cfg_modify_fn
        # add default config
        merge_cfg(self.cfg)
        self.cfg = self.rebuild_config(self.cfg)
        if 'cfg_options' in kwargs:
            self.cfg.merge_from_dict(kwargs['cfg_options'])

        if isinstance(model, (TorchModel, nn.Module)):
            self.model = model
        else:
            self.model = self.build_model()

        if 'work_dir' in kwargs:
            self.work_dir = kwargs['work_dir']
        else:
            self.work_dir = self.cfg.train.get('work_dir', './work_dir')

        self.train_preprocessor, self.eval_preprocessor = self.get_preprocessors(
            preprocessor)

        self._dist = self.init_dist(kwargs.get('launcher'))

        if is_master() and not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        self.device = self.get_device(kwargs.get('device'))

        # init logger after distribution init
        log_file = os.path.join(self.work_dir, '{}.log'.format(self.timestamp))
        self.logger = get_logger(
            log_file=log_file, log_level=self.cfg.get('log_level', 'INFO'))

        if is_master():
            self.logger.info(
                '==========================Training Config Start=========================='
            )
            self.logger.info(
                json.dumps(
                    self.cfg._cfg_dict, indent=4, cls=JSONIteratorEncoder))
            self.logger.info(
                '===========================Training Config End==========================='
            )

        self.train_dataset = self.to_task_dataset(
            train_dataset,
            mode=ModeKeys.TRAIN,
            task_data_config=self.cfg.safe_get('dataset.train'),
            preprocessor=self.train_preprocessor,
            **kwargs)
        self.eval_dataset = self.to_task_dataset(
            eval_dataset,
            mode=ModeKeys.EVAL,
            task_data_config=self.cfg.safe_get('dataset.val'),
            preprocessor=self.eval_preprocessor,
            **kwargs)

        self.train_data_collator, self.eval_data_collator = self.get_data_collator(
            data_collator,
            remove_unused_data=kwargs.get('remove_unused_data', False))
        self.metrics = self.get_metrics()
        self._max_epochs = kwargs.get('max_epochs',
                                      self.cfg.safe_get('train.max_epochs'))
        assert self._max_epochs is not None, 'max_epochs should be provided by the init arguments or configured ' \
                                             'in the `train.max_epochs` key in the configuration file.'
        self._train_iters_per_epoch = kwargs.get(
            'train_iters_per_epoch',
            self.cfg.safe_get('train.train_iters_per_epoch'))
        self._eval_iters_per_epoch = kwargs.get(
            'val_iters_per_epoch',
            self.cfg.safe_get('evaluation.val_iters_per_epoch'))
        self.use_fp16 = kwargs.get('use_fp16', False)
        # model placement
        self.place_model()

        Hook.clear_strategies()

    def place_model(self):
        """Place model to device, or to DDP
        """
        if self.device.type == 'cuda':
            self.model.to(self.device)
            if not is_parallel(self.model) and self._dist:
                self.model = self.to_parallel(self.model)

    def get_data_collator(self, data_collator, remove_unused_data=False):
        """Get the data collator for both training and evaluating.

        Args:
            data_collator: The input data_collator param.
            remove_unused_data: Remove the unused data with 'RemoveColumnsCollator'.
        Returns:
            The train_data_collator and eval_data_collator, can be None.
        """

        train_data_collator, eval_data_collator = None, None
        if isinstance(data_collator, Mapping):
            if ConfigKeys.train in data_collator:
                assert isinstance(data_collator[ConfigKeys.train], Callable)
                train_data_collator = data_collator[ConfigKeys.train]
            if ConfigKeys.val in data_collator:
                assert isinstance(data_collator[ConfigKeys.val], Callable)
                eval_data_collator = data_collator[ConfigKeys.val]
        else:
            collate_fn = default_collate if data_collator is None else data_collator
            train_data_collator = collate_fn
            eval_data_collator = collate_fn

        if remove_unused_data:
            from modelscope.utils.data_collators import RemoveColumnsCollator

            def _set_signature_columns_if_needed():
                signature = inspect.signature(self.model.forward)
                return list(signature.parameters.keys())

            model_inputs = _set_signature_columns_if_needed()
            train_data_collator = RemoveColumnsCollator(
                train_data_collator, model_inputs)
            eval_data_collator = RemoveColumnsCollator(eval_data_collator,
                                                       model_inputs)
        return train_data_collator, eval_data_collator

    def init_dist(self, launcher=None):
        """Init dist and returns the dist information.

        Args:
            launcher: The launcher info.

        Returns:
            _dist: If world_size is greater than 1.
        """
        if launcher is not None:
            init_dist(launcher)

        _, world_size = get_dist_info()
        _dist = world_size > 1
        return _dist

    def get_device(self, device=None):
        """Get the device information.

        Args:
            device: The input device info.

        Returns:
            device_name: The final device name.
        """
        device_name = device if device is not None else 'gpu'
        if is_dist():
            local_rank = get_local_rank()
            device_name = f'cuda:{local_rank}'

        return create_device(device_name)

    def get_preprocessors(self, preprocessor):
        """Get the preprocessors information.

        Args:
            preprocessor: The input preprocessor info.

        Returns:
            The train_preprocessor and eval_preprocessor, can be None.
        """
        train_preprocessor = None
        eval_preprocessor = None
        if isinstance(preprocessor, Preprocessor):
            train_preprocessor = preprocessor
            eval_preprocessor = preprocessor
        elif isinstance(preprocessor, Mapping):
            if ConfigKeys.train in preprocessor:
                assert isinstance(preprocessor[ConfigKeys.train], Callable)
                train_preprocessor = preprocessor[ConfigKeys.train]
            if ConfigKeys.val in preprocessor:
                assert isinstance(preprocessor[ConfigKeys.val], Callable)
                eval_preprocessor = preprocessor[ConfigKeys.val]
        elif hasattr(self.cfg, ConfigFields.preprocessor
                     ) and self.cfg.preprocessor is not None:
            train_preprocessor, eval_preprocessor = self.build_preprocessor()

        if train_preprocessor is not None:
            train_preprocessor.mode = ModeKeys.TRAIN
        if eval_preprocessor is not None:
            eval_preprocessor.mode = ModeKeys.EVAL
        return train_preprocessor, eval_preprocessor

    def rebuild_config(self, cfg: Config):
        """A method used to rebuild the config, any subclass can override this method.

        Returns: The rebuilt config

        """
        if hasattr(self, 'cfg_modify_fn') and self.cfg_modify_fn is not None:
            cfg = self.cfg_modify_fn(cfg)
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
                        task_data_config: Config = None,
                        preprocessor: Optional[Preprocessor] = None,
                        **kwargs):
        """Build the task specific dataset processor for this trainer.

        Returns: The task dataset processor for the task. If no result for the very model-type and task,
        the default TaskDataset will be returned.
        """
        try:
            to_tensor = kwargs.get('to_tensor', True)
            if not datasets:
                return datasets
            if isinstance(datasets, TorchTaskDataset):
                return datasets
            elif isinstance(datasets, MsDataset):
                if task_data_config is None:
                    # adapt to some special models
                    task_data_config = ConfigDict(
                        type=self.cfg.model.type) if hasattr(
                            self.cfg, ConfigFields.model) else ConfigDict(
                                type=None)
                task_data_config.update(dict(mode=mode))
                return datasets.to_torch_dataset(
                    task_data_config=task_data_config,
                    task_name=self.cfg.task,
                    preprocessors=preprocessor,
                    to_tensor=to_tensor)
            elif isinstance(datasets, List) and isinstance(
                    datasets[0], MsDataset):
                if task_data_config is None:
                    # adapt to some special models
                    task_data_config = ConfigDict(
                        type=self.cfg.model.type) if hasattr(
                            self.cfg, ConfigFields.model) else ConfigDict(
                                type=None)
                task_data_config.update(dict(mode=mode))
                datasets = [
                    d.to_torch_dataset(
                        task_data_config=task_data_config,
                        task_name=self.cfg.task,
                        preprocessors=preprocessor,
                        to_tensor=to_tensor) for d in datasets
                ]
                cfg = ConfigDict(
                    type=self.cfg.model.type, mode=mode, datasets=datasets)
                task_dataset = build_task_dataset(cfg, self.cfg.task)
                task_dataset.trainer = self
                return task_dataset
            else:
                if task_data_config is None:
                    # adapt to some special models
                    task_data_config = {}
                # avoid add no str value datasets, preprocessors in cfg
                task_data_build_config = ConfigDict(
                    type=self.cfg.model.type,
                    mode=mode,
                    datasets=datasets,
                    preprocessor=preprocessor)
                task_data_build_config.update(task_data_config)
                task_dataset = build_task_dataset(task_data_build_config,
                                                  self.cfg.task)
                task_dataset.trainer = self
                return task_dataset
        except Exception:
            if isinstance(datasets, (List, Tuple)) or preprocessor is not None:
                task_dataset = TorchTaskDataset(
                    datasets,
                    mode=mode,
                    preprocessor=preprocessor,
                    **(dict(type=self.cfg.model.type) if hasattr(
                        self.cfg, 'model') else {}))
                task_dataset.trainer = self
                return task_dataset
            else:
                return datasets

    def build_preprocessor(self) -> Tuple[Preprocessor, Preprocessor]:
        """Build train and eval preprocessor.

        User can override this method to implement custom logits.

        Returns: The train preprocessor and eval preprocessor instance.

        """
        train_preprocessor = Preprocessor.from_pretrained(
            self.model_dir,
            cfg_dict=self.cfg,
            preprocessor_mode=ModeKeys.TRAIN)
        eval_preprocessor = Preprocessor.from_pretrained(
            self.model_dir, cfg_dict=self.cfg, preprocessor_mode=ModeKeys.EVAL)
        return train_preprocessor, eval_preprocessor

    def get_metrics(self) -> List[Union[str, Dict]]:
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
        if isinstance(metrics, (str, Mapping)):
            metrics = [metrics]
        return metrics

    def set_checkpoint_file_to_hook(self, checkpoint_path, load_all_state,
                                    strict):
        if checkpoint_path is not None:
            from modelscope.trainers.hooks import LoadCheckpointHook
            load_ckpt_hooks = list(
                filter(lambda hook: isinstance(hook, LoadCheckpointHook),
                       self.hooks))
            if len(load_ckpt_hooks) == 0:
                load_ckpt_hook = LoadCheckpointHook()
                self.register_hook(load_ckpt_hook)
                load_ckpt_hooks.append(load_ckpt_hook)
            load_ckpt_hooks[0].checkpoint_file = checkpoint_path
            load_ckpt_hooks[0].load_all_state = load_all_state
            load_ckpt_hooks[0].strict = strict

    def train(self,
              checkpoint_path=None,
              load_all_state=True,
              *args,
              **kwargs):
        """Start training.

        Args:
            checkpoint_path(`str`, `optional`): The previous saving checkpoint to read,
                usually it's a `some-file-name.pth` file generated by this trainer.
            load_all_state(`bool`: `optional`): Load all state out of the `checkpoint_path` file, including the
                state dict of model, optimizer, lr_scheduler, the random state and epoch/iter number. If False, only
                the model's state dict will be read, and model will be trained again.
            kwargs:
                strict(`boolean`): If strict, any unmatched keys will cause an error.
        """

        self._mode = ModeKeys.TRAIN
        self.train_dataloader = self.get_train_dataloader()
        self.data_loader = self.train_dataloader
        self.register_optimizers_hook()
        hooks = merge_hooks(self.cfg)
        self.register_hook_from_cfg(hooks)
        if is_master():
            self.logger.info(self.get_hook_info())
        self.set_checkpoint_file_to_hook(checkpoint_path, load_all_state,
                                         kwargs.get('strict', False))
        self.model.train()

        self.train_loop(self.train_dataloader)

    def predict(self,
                predict_datasets: Union[Dataset, List[Dataset]],
                saving_fn,
                checkpoint_path=None,
                strict=False):
        """Start prediction.

        Args:
            predict_datasets(Union[Dataset, List[Dataset]]): The datasets used to predict ground truth.

            saving_fn(`Callable`): The callable used to save the prediction values to files. Like:
                >>> class SavingFn:
                >>>     def __init__(self):
                >>>         self.filename = '/tmp/results.txt'
                >>>
                >>>     def __call__(self, inputs, outputs):
                >>>         import numpy as np
                >>>         ids = inputs.ids
                >>>         predictions = np.argmax(outputs['logits'].cpu().numpy(), axis=1)
                >>>         with open(self.filename, 'a') as f:
                >>>             for id, pred in zip(ids, predictions):
                >>>                 f.writelines(f'{id}, {pred}')

                This saving_fn's result will not be collected to one file, Training with multiprocessing please
                consider combining these files manually.

            checkpoint_path(`str`, `optional`): The previous saving checkpoint to read,
                usually it's a `some-file-name.pth` file or a pure PyTorch `some-file.bin` file
                generated by this trainer.

            strict(`boolean`): If strict, any unmatched keys will cause an error.
        """
        if not self._hooks:
            hooks = merge_hooks(self.cfg)
            self.register_hook_from_cfg(hooks)
            if is_master():
                self.logger.info(self.get_hook_info())
        if checkpoint_path is not None:
            from modelscope.trainers.hooks import LoadCheckpointHook
            LoadCheckpointHook.load_checkpoint(
                checkpoint_path, self, strict=strict)
        self.model.eval()
        self._mode = ModeKeys.EVAL
        predict_dataloader = self.get_predict_data_loader(predict_datasets)
        metric_classes = [PredictionSavingWrapper(saving_fn=saving_fn)]

        for m in metric_classes:
            m.trainer = self

        self.evaluation_loop(predict_dataloader, metric_classes)

    def evaluate(self, checkpoint_path=None, saving_fn=None, **kwargs):
        """Start evaluation.

        Args:
            checkpoint_path(`str`, `optional`): The previous saving checkpoint to read,
                usually it's a `some-file-name.pth` file or a pure PyTorch `some-file.bin` file
                generated by this trainer.

            saving_fn(`Callable`): The callable used to save the prediction values to files. Like:
                >>> class SavingFn:
                >>>     def __init__(self):
                >>>         self.filename = '/tmp/results.txt'
                >>>
                >>>     def __call__(self, inputs, outputs):
                >>>         import numpy as np
                >>>         ids = inputs.ids
                >>>         predictions = np.argmax(outputs['logits'].cpu().numpy(), axis=1)
                >>>         with open(self.filename, 'a') as f:
                >>>             for id, pred in zip(ids, predictions):
                >>>                 f.writelines(f'{id}, {pred}')
            kwargs:
                strict(`boolean`): If strict, any unmatched keys will cause an error.
        """
        if not self._hooks:
            hooks = merge_hooks(self.cfg)
            self.register_hook_from_cfg(hooks)
            if is_master():
                self.logger.info(self.get_hook_info())
        if checkpoint_path is not None:
            from modelscope.trainers.hooks import LoadCheckpointHook
            LoadCheckpointHook.load_checkpoint(
                checkpoint_path, self, strict=kwargs.get('strict', False))
        self.model.eval()
        self._mode = ModeKeys.EVAL
        self.eval_dataloader = self.get_eval_data_loader()
        self.data_loader = self.eval_dataloader
        metric_classes = [build_metric(metric) for metric in self.metrics]
        if saving_fn is not None:
            metric_classes.append(PredictionSavingWrapper(saving_fn=saving_fn))
        for m in metric_classes:
            m.trainer = self

        metric_values = self.evaluation_loop(self.eval_dataloader,
                                             metric_classes)

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
        model = Model.from_pretrained(self.model_dir, cfg_dict=self.cfg)
        if not isinstance(model, nn.Module) and hasattr(model, 'model'):
            return model.model
        elif isinstance(model, nn.Module):
            return model

    def to_parallel(self, model) -> Union[nn.Module, TorchModel]:
        # config format to reserve custom ddp
        if self.cfg.get('parallel', None) is not None:
            dp_cfg = deepcopy(self.cfg['parallel'])
            dp_cfg.update(
                dict(module=model, device_ids=[torch.cuda.current_device()]))
            return build_parallel(dp_cfg)

        dp_cfg = dict(
            type='DistributedDataParallel',
            module=model,
            find_unused_parameters=True,
            device_ids=[torch.cuda.current_device()])

        if is_megatron_initialized():
            from megatron_util import mpu
            dp_cfg.update({
                'output_device': torch.cuda.current_device(),
                'process_group': mpu.get_data_parallel_group()
            })

        return build_parallel(dp_cfg)

    def unwrap_module(self, model) -> Union[nn.Module, TorchModel]:
        """Unwrap the model until it's a naked nn.Module.

        Args:
            model: An module.
        """
        if hasattr(model, 'module'):
            return self.unwrap_module(model.module)
        else:
            assert isinstance(model, torch.nn.Module)
            return model

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

        receive_dict_inputs = func_receive_dict_inputs(
            self.unwrap_module(self.model).forward)

        if isinstance(inputs, Mapping) and not receive_dict_inputs:
            train_outputs = model.forward(**inputs)
        else:
            train_outputs = model.forward(inputs)

        if isinstance(train_outputs, ModelOutputBase):
            train_outputs = train_outputs.to_dict()
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
                    if is_dist():
                        value = value.data.clone().to('cuda')
                        dist.all_reduce(value.div_(dist.get_world_size()))
                    log_vars.update({key: value.item()})
            self.log_buffer.update(log_vars)
        else:
            self.log_buffer.update(train_outputs['log_vars'])

        self.train_outputs = train_outputs

    def prediction_step(self, model, inputs):
        """Deprecated method
        """
        self.logger.warn('This prediction_step method is deprecated.')
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
            collate_fn=self.train_data_collator,
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

        default_config = {'shuffle': False}
        default_config.update(self.cfg.evaluation.get('dataloader', {}))
        data_loader = self._build_dataloader_with_dataset(
            self.eval_dataset,
            dist=self._dist,
            seed=self._seed,
            collate_fn=self.eval_data_collator,
            **default_config)
        return data_loader

    def get_predict_data_loader(self, predict_datasets: Union[Dataset,
                                                              List[Dataset]]):
        """ Builder torch dataloader for prediction with the config of evaluation.

        Args:
            predict_datasets(Union[Dataset, List[Dataset]]): The datasets used to predict ground truth.
        """
        dataset = self.to_task_dataset(
            predict_datasets,
            mode=ModeKeys.EVAL,
            preprocessor=self.eval_preprocessor)

        default_config = {'shuffle': False}
        default_config.update(self.cfg.evaluation.get('dataloader', {}))
        data_loader = self._build_dataloader_with_dataset(
            dataset,
            dist=self._dist,
            seed=self._seed,
            collate_fn=self.eval_data_collator,
            **default_config)
        return data_loader

    def build_dataset(self, data_cfg, mode, preprocessor=None):
        """ Build torch dataset object using data config
        """
        # TODO: support MsDataset load for cv
        if hasattr(data_cfg, 'name'):
            dataset_name = data_cfg.pop('name')
            dataset = MsDataset.load(
                dataset_name=dataset_name,
                **data_cfg,
            )
            cfg = ConfigDict(type=self.cfg.model.type, mode=mode)
            torch_dataset = dataset.to_torch_dataset(
                task_data_config=cfg,
                task_name=self.cfg.task,
                preprocessors=preprocessor)
        else:
            torch_dataset = build_task_dataset(data_cfg, self.cfg.task)
        dataset = self.to_task_dataset(torch_dataset, mode)
        return dataset

    def build_optimizer(self, cfg: ConfigDict, default_args: dict = None):
        try:
            return build_optimizer(
                self.unwrap_module(self.model),
                cfg=cfg,
                default_args=default_args)
        except KeyError as e:
            self.logger.error(
                f'Build optimizer error, the optimizer {cfg} is a torch native component, '
                f'please check if your torch with version: {torch.__version__} matches the config.'
            )
            raise e

    def build_lr_scheduler(self, cfg: ConfigDict, default_args: dict = None):
        try:
            return build_lr_scheduler(cfg=cfg, default_args=default_args)
        except KeyError as e:
            self.logger.error(
                f'Build lr_scheduler error, the lr_scheduler {cfg} is a torch native component, '
                f'please check if your torch with version: {torch.__version__} matches the config.'
            )
            raise e

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
            optimizer = self.build_optimizer(cfg=optimizer_cfg)

        if lr_scheduler is None:
            lr_scheduler_cfg = self.cfg.train.get('lr_scheduler', None)
        else:
            lr_scheduler_cfg = None

        lr_options = {}
        if lr_scheduler_cfg is not None:
            assert optimizer is not None
            lr_options = lr_scheduler_cfg.pop('options', {})
            lr_scheduler = self.build_lr_scheduler(
                cfg=lr_scheduler_cfg, default_args={'optimizer': optimizer})

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        return self.optimizer, self.lr_scheduler, optim_options, lr_options

    def register_optimizers_hook(self):
        """ Register optimizer hook and lr scheduler hook.
        """
        _, lr_scheduler, optim_options, lr_options = self.create_optimizer_and_scheduler(
        )

        optim_hook = self.cfg.train.get('optimizer_hook', {})
        lr_hook = self.cfg.train.get('lr_scheduler_hook', {})

        # adapt to `ReduceLROnPlateau`
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        if isinstance(lr_scheduler, ReduceLROnPlateau) and not lr_hook:
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

        def _fit_to_old_keys():
            """This function used to fit `optimizer_hook` key and `lr_scheduler_hook` key for easycv configs.

            The logic is:
                If the optimizer_hook is provided and it's not TorchAMPOptimizerHook or ApexAMPOptimizerHook,
                (which means the hook is a complete one for optimization, which does not need the OptimizerHook),
                The OptimizerHook will not be registered, or else the OptimizerHook will be registered.

                Same logic to the LrSchedulerHook, the only difference is the condition of lr_scheduler_hook is
                PlateauLrSchedulerHook.

                If TorchAMPOptimizerHook or ApexAMPOptimizerHook is provided, self.use_fp16 will be set to False
                in case of the duplication of registration.

            """
            if lr_hook:
                self.register_hook_from_cfg([lr_hook])

            _lr_options = None
            if not lr_hook or lr_hook.get('type') == 'PlateauLrSchedulerHook':
                lr_hook.pop('type', None)
                _lr_options = {**lr_options, **lr_hook}

            if optim_hook:
                self.register_hook_from_cfg([optim_hook])

            _optim_options = None
            if optim_hook.get('type') in ('TorchAMPOptimizerHook',
                                          'ApexAMPOptimizerHook'):
                self.use_fp16 = False
            if not optim_hook or optim_hook.get('type') in (
                    'TorchAMPOptimizerHook', 'ApexAMPOptimizerHook'):
                optim_hook.pop('type', None)
                _optim_options = {**optim_options, **optim_hook}

            return _optim_options, _lr_options

        optim_options, lr_options = _fit_to_old_keys()

        if optim_options is not None:
            self.register_hook_from_cfg(
                [dict(type='OptimizerHook', **optim_options)])
        if lr_options is not None:
            self.register_hook_from_cfg(
                [dict(type='LrSchedulerHook', **lr_options)])
        if self.use_fp16:
            self.register_hook_from_cfg(
                [dict(type='TorchAMPOptimizerHook', **optim_options)])

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

        if dist and not isinstance(dataset, torch.utils.data.IterableDataset):
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
            pin_memory=kwargs.pop('pin_memory', False),
            worker_init_fn=init_fn,
            **kwargs)

        return data_loader

    def train_loop(self, data_loader):
        """ Training loop used by `EpochBasedTrainer.train()`
        """
        self.invoke_hook(TrainerStages.before_run)
        kwargs = {}
        self.model.train()
        for _ in range(self._epoch, self._max_epochs):
            self.invoke_hook(TrainerStages.before_train_epoch)
            for i, data_batch in enumerate(data_loader):
                if i < self.inner_iter:
                    # inner_iter may be read out from the checkpoint file, so skip the trained iters in the epoch.
                    continue
                data_batch = to_device(data_batch, self.device)
                self.data_batch = data_batch
                self._inner_iter = i
                self.invoke_hook(TrainerStages.before_train_iter)
                self.train_step(self.model, data_batch, **kwargs)
                self.invoke_hook(TrainerStages.after_train_iter)
                # Value changed after the hooks are invoked, do not move them above the invoke_hook code.
                del self.data_batch
                self._iter += 1
                self._mode = ModeKeys.TRAIN

                if i + 1 >= self.iters_per_epoch:
                    break

            self.invoke_hook(TrainerStages.after_train_epoch)
            # Value changed after the hooks are invoked, do not move them above the invoke_hook code.
            self._inner_iter = 0
            self._epoch += 1
            if self._stop_training:
                break

        self.invoke_hook(TrainerStages.after_run)

    def evaluation_step(self, data):
        """Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        """
        self.model.eval()

        receive_dict_inputs = func_receive_dict_inputs(
            self.unwrap_module(self.model).forward)

        with torch.no_grad():
            if isinstance(data, Mapping) and not receive_dict_inputs:
                result = self.model.forward(**data)
            else:
                result = self.model.forward(data)
        return result

    def evaluation_loop(self, data_loader, metric_classes):
        """ Evaluation loop used by `EpochBasedTrainer.evaluate()`.

        """
        vis_closure = None
        if hasattr(self.cfg.evaluation, 'visualization'):
            vis_cfg = self.cfg.evaluation.visualization
            vis_closure = partial(
                self.visualization, dataset=self.eval_dataset, **vis_cfg)

        if self._dist:
            from modelscope.trainers.utils.inference import multi_gpu_test
            # list of batched result and data samples
            metric_values = multi_gpu_test(
                self,
                data_loader,
                device=self.device,
                metric_classes=metric_classes,
                vis_closure=vis_closure,
                tmpdir=self.cfg.evaluation.get('cache_dir', None),
                gpu_collect=self.cfg.evaluation.get('gpu_collect', False),
                data_loader_iters_per_gpu=self._eval_iters_per_epoch)
        else:
            from modelscope.trainers.utils.inference import single_gpu_test
            metric_values = single_gpu_test(
                self,
                data_loader,
                device=self.device,
                metric_classes=metric_classes,
                vis_closure=vis_closure,
                data_loader_iters=self._eval_iters_per_epoch)

        return metric_values

    def visualization(self, batch_result, dataset, **kwargs):
        """ visualization function for evaluation results.

        Examples:
            >>> # draw list of images as numpy array
            >>> images = draw_images(num_of_visualization)

            >>> # set displayed name for each image
            >>> filenames = get_image_display_names()
            >>> vis_results = {'images': images, 'filenames' : filenames}

            >>> # visualization results will be displayed in group named eva_vis
            >>> self.visualization_buffer.output['eval_vis'] = vis_results

        Args:
            results (list(dict)):  a list of result dict.
            dataset (Dataset): torch dataset object to access original data.
        """
        # TODO @wenmeng.zwm add visualization support for cv evaluation
        raise NotImplementedError(
            'visualization for evaluation will be supported in the future')

    def register_hook(self, hook: Hook) -> None:
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
        """
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            p = hook.PRIORITY if hasattr(hook, 'PRIORITY') else Priority.NORMAL
            p_i = self._hooks[i].PRIORITY if hasattr(
                self._hooks[i], 'PRIORITY') else Priority.NORMAL

            if get_priority(p) > get_priority(p_i):
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def register_hook_from_cfg(self, hook_cfg: List) -> List:
        """Register a hook from its cfg.

        Args:
            hook_cfg (dict): Hook config. It should have at least keys 'type'
              and 'priority' indicating its type and priority.

        Note:
            The specific hook class to register should not use 'type' and
            'priority' arguments during initialization.

        Returns:
            A list of instances of registered hooks.
        """
        hook_cfg = hook_cfg.copy()
        assert isinstance(hook_cfg, list)
        hooks = []
        for cfg_i in hook_cfg:
            hook = build_from_cfg(cfg_i, HOOKS)
            if hasattr(hook, 'register_strategy'):
                hook.register_strategy()
            self.register_hook(hook)
            hooks.append(hook)
        return hooks

    def get_hook(self, cls):
        return [h for h in self._hooks if h.__class__ == cls]

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
                priority = Priority(hook.PRIORITY).name  # type: ignore
            except Exception:
                priority = Priority.NORMAL  # type: ignore
            classname = hook.__class__.__name__
            hook_info = f'({priority:<12}) {classname:<35}'
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f'Stage: {stage}:\n    '
                info += '\n    '.join(hook_infos)
                info += '\n -------------------- '
                stage_hook_infos.append(info)
        stage_hook_infos = '\n'.join(stage_hook_infos)

        strategy_info = '\n --- Hook strategies info --- \n'
        for consumer, methods in Hook._strategies.items():
            strategy_info += f'Method: {consumer} ' \
                             f'replaced by: ' \
                             f'{[method.__self__.__class__.__name__ + "." + method.__name__ for method in methods]}\n'
        strategy_info += '\n --- Hook strategies info end --- \n'
        return stage_hook_infos + strategy_info


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    set_random_seed(worker_seed)
