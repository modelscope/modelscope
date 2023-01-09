# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from modelscope.metainfo import Trainers
from modelscope.metrics.builder import build_metric
from modelscope.models.base import Model, TorchModel
from modelscope.msdatasets import MsDataset
from modelscope.preprocessors import Preprocessor
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import (DEFAULT_MODEL_REVISION, ModeKeys,
                                       ModelFile)
from modelscope.utils.hub import parse_label_mapping
from .base import TRAINERS
from .trainer import EpochBasedTrainer


@dataclass
class NlpTrainerArguments:
    """The arguments for the nlp trainer.

    All the arguments listed here have None default values, which means follow the default value in the input
    cfg dict.
    """

    work_dir: Optional[str] = field(
        default=None, metadata={'help': 'The work dir(key: train.work_dir)'})

    task: Optional[str] = field(
        default=None, metadata={'help': 'The task type(key: task)'})

    preprocessor_type: Optional[str] = field(
        default=None,
        metadata={'help': 'The preprocessor type(key: preprocessor.type)'})

    train_first_sequence: str = field(
        default=None,
        metadata={
            'help':
            'The key of first sentence for the training dataset(key:preprocessor.train.'
            'first_sequence/dataset.train.first_sequence)'
        })

    train_second_sequence: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'The key of second sentence for the training dataset(key:preprocessor.train.'
            'second_sequence/dataset.train.second_sequence)'
        })

    train_label: str = field(
        default=None,
        metadata={
            'help':
            'The key of label for the training dataset(key:preprocessor.train.'
            'second_sequence/dataset.train.second_sequence)'
        })

    eval_first_sequence: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'The key of first sentence for the eval dataset(key:preprocessor.val.'
            'first_sequence/dataset.val.first_sequence), '
            'if not provided, the trainer will use the train_first_sequence for evaluation'
        })

    eval_second_sequence: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'The key of second sentence for the eval dataset(key:preprocessor.val.'
            'second_sequence/dataset.val.second_sequence),'
            'if not provided, the trainer will use the train_second_sequence for evaluation'
        })

    eval_label: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'The key of label for the eval dataset(key:preprocessor.val.'
            'second_sequence/dataset.val.second_sequence),'
            'if not provided, the trainer will use the train_label for evaluation'
        })

    labels: Optional[List] = field(
        default=None,
        metadata={
            'help':
            'The labels list of the dataset(key:dataset.train.labels),'
            'This parameter has the same effect with "label2id"'
        })

    max_epochs: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'The max_epochs of the training loop(key: train.max_epochs)'
        })

    train_batch_size_per_gpu: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'The train batch size per gpu(key: train.dataloader.batch_size_per_gpu)'
        })

    train_workers_per_gpu: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'The number of workers per gpu(key: train.dataloader.workers_per_gpu)'
        })

    train_shuffle: Optional[bool] = field(
        default=None,
        metadata={
            'help':
            'Shuffle the train dataset or not(key: train.dataloader.shuffle)'
        })

    eval_batch_size_per_gpu: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'The eval batch size per gpu(key: evaluation.dataloader.batch_size_per_gpu)'
        })

    eval_workers_per_gpu: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'The number of workers per gpu(key: evaluation.dataloader.workers_per_gpu)'
        })

    eval_shuffle: Optional[bool] = field(
        default=None,
        metadata={
            'help':
            'Shuffle the eval dataset or not(key: evaluation.dataloader.shuffle)'
        })

    optimizer_args: Optional[Dict] = field(
        default=None,
        metadata={'help': 'The optimizer config dict(key: train.optimizer)'})

    lr_scheduler_args: Optional[Dict] = field(
        default=None,
        metadata={
            'help': 'The lr_scheduler config dict(key: train.lr_scheduler)'
        })

    checkpoint_saving_type: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'The checkpoint saving type(key: The ckpt hook dict in train.hooks), '
            'valid options: "BestCkptSaverHook", "CheckpointHook"'
        })

    checkpoint_by_epoch: Optional[bool] = field(
        default=None,
        metadata={
            'help':
            'Saving checkpoint by epoch or not(key: The by_epoch key in '
            'ckpt hook dict in train.hooks)'
        })

    checkpoint_interval: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'The checkpoint saving interval(key: The interval key in '
            'ckpt hook dict in train.hooks)'
        })

    metric_key: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'The metric key for the BestCkptSaverHook(key: The metric_key key in '
            'ckpt hook dict in train.hooks), if the checkpoint_saving_type is "CheckpointHook" or '
            '"None", the metric_key key has no effects'
        })

    evaluation_type: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'The evaluation type(key: The evaluation hook dict in train.hooks), '
            'valid options: "EvaluationHook", "None"'
        })

    evaluation_by_epoch: Optional[bool] = field(
        default=None,
        metadata={
            'help':
            'Evaluating by epoch or not(key: The by_epoch key in '
            'evaluation hook dict in train.hooks)'
        })

    evaluation_interval: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'The evaluating interval(key: The interval key in '
            'evaluation hook dict in train.hooks)'
        })

    metrics: Optional[List[str]] = field(
        default=None,
        metadata={'help': 'The metrics class keys(key: evaluation.metrics)'})

    default_train_config = ConfigDict({
        'work_dir':
        '/tmp',
        'max_epochs':
        5,
        'dataloader': {
            'batch_size_per_gpu': 32,
            'workers_per_gpu': 0
        },
        'optimizer': {
            'type': 'AdamW',
            'lr': 2e-5,
            'options': {}
        },
        'lr_scheduler': {
            'type': 'LinearLR',
            'start_factor': 1.0,
            'end_factor': 0.0,
            'total_iters': 10000,
            'options': {
                'by_epoch': False
            }
        },
        'hooks': [{
            'type': 'CheckpointHook',
            'by_epoch': False,
            'interval': 100
        }, {
            'type': 'TextLoggerHook',
            'interval': 1
        }, {
            'type': 'IterTimerHook'
        }, {
            'type': 'EvaluationHook',
            'by_epoch': False,
            'interval': 100
        }]
    })

    def __call__(self, cfg):
        """

        Args:
            cfg(`Config`): The cfg to be modified.

        Returns:
            The cfg after modification.
        """

        if self.task is not None:
            cfg.task = self.task

        if self.preprocessor_type is not None:
            if not hasattr(cfg, 'preprocessor'):
                cfg.preprocessor = ConfigDict()
            cfg.preprocessor.type = self.preprocessor_type

        if self.train_first_sequence is not None or self.train_second_sequence \
                is not None or self.train_label is not None or self.labels is not None:
            if not hasattr(cfg, 'dataset'):
                cfg.dataset = ConfigDict()
            if not hasattr(cfg.dataset, 'train'):
                cfg.dataset.train = ConfigDict()
            if self.train_first_sequence is not None:
                cfg.dataset.train.first_sequence = self.train_first_sequence
            if self.train_second_sequence is not None:
                cfg.dataset.train.second_sequence = self.train_second_sequence
            if self.train_label is not None:
                cfg.dataset.train.label = self.train_label
            if self.labels is not None:
                cfg.dataset.train.labels = self.labels

        if self.eval_first_sequence is not None or self.eval_second_sequence \
                is not None or self.eval_label is not None:
            if not hasattr(cfg, 'dataset'):
                cfg.dataset = ConfigDict()
            if not hasattr(cfg.dataset, 'val'):
                cfg.dataset.val = ConfigDict()
            if self.eval_first_sequence is not None:
                cfg.dataset.val.first_sequence = self.eval_first_sequence
            if self.eval_second_sequence is not None:
                cfg.dataset.val.second_sequence = self.eval_second_sequence
            if self.eval_label is not None:
                cfg.dataset.val.label = self.eval_label

        if self.max_epochs is not None or self.train_batch_size_per_gpu is not None \
                or self.train_shuffle is not None or self.optimizer_args is not None \
                or self.work_dir is not None or self.lr_scheduler_args is not None\
                or self.train_workers_per_gpu is not None:
            if not hasattr(cfg, 'train'):
                cfg.train = deepcopy(self.default_train_config)
            if not hasattr(cfg.train, 'dataloader'):
                cfg.train.dataloader = deepcopy(
                    self.default_train_config.dataloader)
            if not hasattr(cfg.train, 'optimizer'):
                cfg.train.optimizer = deepcopy(
                    self.default_train_config.optimizer)
            if not hasattr(cfg.train, 'lr_scheduler'):
                cfg.train.lr_scheduler = deepcopy(
                    self.default_train_config.lr_scheduler)
            if self.work_dir is not None:
                cfg.train.work_dir = self.work_dir
            if self.max_epochs is not None:
                cfg.train.max_epochs = self.max_epochs
            if self.train_batch_size_per_gpu is not None:
                cfg.train.dataloader.batch_size_per_gpu = self.train_batch_size_per_gpu
            if self.train_workers_per_gpu is not None:
                cfg.train.dataloader.workers_per_gpu = self.train_workers_per_gpu
            if self.train_shuffle is not None:
                cfg.train.dataloader.shuffle = self.train_shuffle
            if self.optimizer_args is not None:
                if cfg.train.optimizer.type != self.optimizer_args.get(
                        'type', cfg.train.optimizer.type):
                    cfg.train.optimizer = ConfigDict(
                        deepcopy(self.optimizer_args))
                else:
                    cfg.train.optimizer = Config._merge_a_into_b(
                        self.optimizer_args, cfg.train.optimizer, force=True)
            if self.lr_scheduler_args is not None:
                if cfg.train.lr_scheduler.type != self.lr_scheduler_args.get(
                        'type', cfg.train.lr_scheduler.type):
                    cfg.train.lr_scheduler = ConfigDict(
                        deepcopy(self.lr_scheduler_args))
                else:
                    cfg.train.lr_scheduler = Config._merge_a_into_b(
                        self.lr_scheduler_args,
                        cfg.train.lr_scheduler,
                        force=True)

        if self.checkpoint_saving_type is not None or self.checkpoint_by_epoch is not None \
                or self.checkpoint_interval is not None or self.metric_key is not None:
            if not any([
                    self.checkpoint_saving_type == hook['type']
                    for hook in cfg.train.hooks
            ]):
                cfg.train.hooks = list(
                    filter(
                        lambda hook: hook['type'] not in
                        ['CheckpointHook', 'BestCkptSaverHook'],
                        cfg.train.hooks))
                cfg.train.hooks.append(
                    deepcopy(self.default_train_config.hooks[0]))
                cfg.train.hooks[-1].type = self.checkpoint_saving_type
            checkpoint_hook = list(
                filter(
                    lambda hook: hook[
                        'type'] in ['CheckpointHook', 'BestCkptSaverHook'],
                    cfg.train.hooks))[0]
            if self.checkpoint_by_epoch is not None:
                checkpoint_hook['by_epoch'] = self.checkpoint_by_epoch
            if self.checkpoint_interval is not None:
                checkpoint_hook['interval'] = self.checkpoint_interval
            if checkpoint_hook['type'] == 'BestCkptSaverHook':
                assert self.metric_key is not None, 'The metric_key must be provided ' \
                                                    'if the ckpt saving hook is "BestCkptSaverHook"'
                checkpoint_hook['metric_key'] = self.metric_key

        if self.evaluation_type is not None or self.evaluation_by_epoch is not None \
                or self.evaluation_interval is not None or self.eval_batch_size_per_gpu is not None or \
                self.eval_shuffle is not None or self.metrics is not None:
            if self.evaluation_type is not None and not any([
                    self.evaluation_type == hook['type']
                    for hook in cfg.train.hooks
            ]):
                cfg.train.hooks = list(
                    filter(lambda hook: hook['type'] not in ['EvaluationHook'],
                           cfg.train.hooks))
                if self.evaluation_type != 'None':
                    cfg.train.hooks.append(
                        deepcopy(self.default_train_config.hooks[3]))
                    cfg.train.hooks[-1].type = self.evaluation_type

            evaluation_hook = list(
                filter(lambda hook: hook['type'] in ['EvaluationHook'],
                       cfg.train.hooks))
            evaluation_hook = evaluation_hook[0] if len(
                evaluation_hook) > 0 else None

            if evaluation_hook is not None and self.evaluation_by_epoch is not None:
                evaluation_hook['by_epoch'] = self.evaluation_by_epoch
            if evaluation_hook is not None and self.evaluation_interval is not None:
                evaluation_hook['interval'] = self.evaluation_interval

            if not hasattr(cfg, 'evaluation'):
                cfg.evaluation = ConfigDict({
                    'dataloader': {
                        'batch_size_per_gpu': 32,
                        'workers_per_gpu': 0,
                        'shuffle': False
                    }
                })

            if self.metrics is not None:
                cfg.evaluation.metrics = self.metrics
            if self.eval_batch_size_per_gpu is not None:
                cfg.evaluation.dataloader.batch_size_per_gpu = self.eval_batch_size_per_gpu
            if self.eval_workers_per_gpu is not None:
                cfg.evaluation.dataloader.workers_per_gpu = self.eval_workers_per_gpu
            if self.eval_shuffle is not None:
                cfg.evaluation.dataloader.shuffle = self.eval_shuffle

        return cfg


@TRAINERS.register_module(module_name=Trainers.nlp_base_trainer)
class NlpEpochBasedTrainer(EpochBasedTrainer):
    """Add code to adapt with nlp models.

    This trainer will accept the information of labels&text keys in the cfg, and then initialize
    the nlp models/preprocessors with this information.

    Labels&text key information may be carried in the cfg like this:

    >>> cfg = {
    >>>     ...
    >>>     "dataset": {
    >>>         "train": {
    >>>             "first_sequence": "text1",
    >>>             "second_sequence": "text2",
    >>>             "label": "label",
    >>>             "labels": [1, 2, 3, 4],
    >>>         },
    >>>         "val": {
    >>>             "first_sequence": "text3",
    >>>             "second_sequence": "text4",
    >>>             "label": "label2",
    >>>         },
    >>>     }
    >>> }

    To view some actual finetune examples, please check the test files listed below:
    tests/trainers/test_finetune_sequence_classification.py
    tests/trainers/test_finetune_token_classification.py
    """

    def __init__(self, *args, **kwargs):
        self.label2id = None
        self.id2label = None
        self.num_labels = None
        self.train_keys = None
        self.eval_keys = None
        super().__init__(*args, **kwargs)

    def prepare_labels(self, cfg):
        try:
            labels = cfg.dataset.train.labels
            self.label2id = {label: idx for idx, label in enumerate(labels)}
            self.id2label = {idx: label for idx, label in enumerate(labels)}
            self.num_labels = len(labels)
        except AttributeError:
            pass

        def build_dataset_keys(cfg):
            if cfg is not None:
                input_keys = {
                    'first_sequence': getattr(cfg, 'first_sequence', None),
                    'second_sequence': getattr(cfg, 'second_sequence', None),
                    'label': getattr(cfg, 'label', None),
                }
            else:
                input_keys = {}

            return {k: v for k, v in input_keys.items() if v is not None}

        self.train_keys = build_dataset_keys(cfg.safe_get('dataset.train'))
        self.eval_keys = build_dataset_keys(cfg.safe_get('dataset.val'))
        if len(self.eval_keys) == 0:
            self.eval_keys = self.train_keys

    def rebuild_config(self, cfg: Config):
        if self.cfg_modify_fn is not None:
            cfg = self.cfg_modify_fn(cfg)
        self.prepare_labels(cfg)
        if not hasattr(cfg.model, 'label2id') and not hasattr(
                cfg.model, 'id2label'):
            if self.id2label is not None:
                cfg.model['id2label'] = self.id2label
            if self.label2id is not None:
                cfg.model['label2id'] = self.label2id
        return cfg

    def build_model(self) -> Union[nn.Module, TorchModel]:
        """ Instantiate a pytorch model and return.

        By default, we will create a model using config from configuration file. You can
        override this method in a subclass.

        """
        model_args = {} if self.num_labels is None else {
            'num_labels': self.num_labels
        }
        model = Model.from_pretrained(
            self.model_dir, cfg_dict=self.cfg, **model_args)
        if not isinstance(model, nn.Module) and hasattr(model, 'model'):
            return model.model
        elif isinstance(model, nn.Module):
            return model

    def build_preprocessor(self) -> Tuple[Preprocessor, Preprocessor]:
        """Build the preprocessor.

        User can override this method to implement custom logits.

        Returns: The preprocessor instance.

        """

        # Compatible with old logic
        extra_args = {} if self.label2id is None else {
            'label2id': self.label2id
        }

        train_preprocessor = Preprocessor.from_pretrained(
            self.model_dir,
            cfg_dict=self.cfg,
            preprocessor_mode=ModeKeys.TRAIN,
            **extra_args,
            **self.train_keys,
            mode=ModeKeys.TRAIN,
            use_fast=True)
        eval_preprocessor = Preprocessor.from_pretrained(
            self.model_dir,
            cfg_dict=self.cfg,
            preprocessor_mode=ModeKeys.EVAL,
            **extra_args,
            **self.eval_keys,
            mode=ModeKeys.EVAL,
            use_fast=True)
        return train_preprocessor, eval_preprocessor


@TRAINERS.register_module(module_name=Trainers.nlp_veco_trainer)
class VecoTrainer(NlpEpochBasedTrainer):

    def evaluate(self, checkpoint_path=None):
        """Veco evaluates the datasets one by one.

        """
        from modelscope.msdatasets.task_datasets import VecoDataset
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            from modelscope.trainers.hooks import CheckpointHook
            CheckpointHook.load_checkpoint(checkpoint_path, self)
        self.model.eval()
        self._mode = ModeKeys.EVAL
        metric_values = {}

        if self.eval_dataset is None:
            val_data = self.cfg.dataset.val
            self.eval_dataset = self.build_dataset(
                val_data, mode=ModeKeys.EVAL)

        idx = 0
        dataset_cnt = 1
        if isinstance(self.eval_dataset, VecoDataset):
            self.eval_dataset.switch_dataset(idx)
            dataset_cnt = len(self.eval_dataset.datasets)

        while True:
            self.eval_dataloader = self._build_dataloader_with_dataset(
                self.eval_dataset, **self.cfg.evaluation.get('dataloader', {}))
            self.data_loader = self.eval_dataloader

            metric_classes = [build_metric(metric) for metric in self.metrics]
            for m in metric_classes:
                m.trainer = self
            self.evaluation_loop(self.eval_dataloader, metric_classes)

            for m_idx, metric_cls in enumerate(metric_classes):
                if f'eval_dataset[{idx}]' not in metric_values:
                    metric_values[f'eval_dataset[{idx}]'] = {}
                metric_values[f'eval_dataset[{idx}]'][
                    self.metrics[m_idx]] = metric_cls.evaluate()

            idx += 1
            if idx < dataset_cnt:
                self.eval_dataset.switch_dataset(idx)
            else:
                break

        for metric_name in self.metrics:
            all_metrics = [m[metric_name] for m in metric_values.values()]
            for key in all_metrics[0].keys():
                metric_values[key] = np.average(
                    [metric[key] for metric in all_metrics])

        return metric_values
