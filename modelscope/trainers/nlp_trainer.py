# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Tuple, Union

import numpy as np
from torch import nn

from modelscope.metainfo import Trainers
from modelscope.metrics.builder import build_metric
from modelscope.models.base import Model, TorchModel
from modelscope.preprocessors import Preprocessor
from modelscope.utils.config import Config
from modelscope.utils.constant import ModeKeys
from .base import TRAINERS
from .trainer import EpochBasedTrainer


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
        from modelscope.msdatasets.dataset_cls.custom_datasets import VecoDataset
        if checkpoint_path is not None:
            from modelscope.trainers.hooks import LoadCheckpointHook
            LoadCheckpointHook.load_checkpoint(checkpoint_path, self)
        self.model.eval()
        self._mode = ModeKeys.EVAL
        metric_values = {}

        if self.eval_dataset is None:
            self.eval_dataset = self.build_dataset_from_cfg(
                model_cfg=self.cfg,
                mode=self._mode,
                preprocessor=self.eval_preprocessor)

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
