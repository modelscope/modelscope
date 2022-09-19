import os
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.metrics.builder import build_metric
from modelscope.models.base import Model, TorchModel
from modelscope.msdatasets import MsDataset
from modelscope.preprocessors import Preprocessor, build_preprocessor
from modelscope.utils.config import Config
from modelscope.utils.constant import (DEFAULT_MODEL_REVISION, ModeKeys,
                                       ModelFile, Tasks)
from modelscope.utils.hub import parse_label_mapping
from .base import TRAINERS
from .trainer import EpochBasedTrainer


@TRAINERS.register_module(module_name=Trainers.nlp_base_trainer)
class NlpEpochBasedTrainer(EpochBasedTrainer):

    def __init__(
            self,
            model: Optional[Union[TorchModel, nn.Module, str]] = None,
            cfg_file: Optional[str] = None,
            cfg_modify_fn: Optional[Callable] = None,
            arg_parse_fn: Optional[Callable] = None,
            data_collator: Optional[Callable] = None,
            train_dataset: Optional[Union[MsDataset, Dataset]] = None,
            eval_dataset: Optional[Union[MsDataset, Dataset]] = None,
            preprocessor: Optional[Preprocessor] = None,
            optimizers: Tuple[torch.optim.Optimizer,
                              torch.optim.lr_scheduler._LRScheduler] = (None,
                                                                        None),
            model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
            **kwargs):
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
        >>>             "labels": [1, 2, 3, 4]
        >>>         }
        >>>     }
        >>> }


        Args:
            cfg_modify_fn: An input fn which is used to modify the cfg read out of the file.

            Example:
            >>> def cfg_modify_fn(cfg):
            >>>     cfg.preprocessor.first_sequence= 'text1'
            >>>     cfg.preprocessor.second_sequence='text2'
            >>>     return cfg

            To view some actual finetune examples, please check the test files listed below:
            tests/trainers/test_finetune_sequence_classification.py
            tests/trainers/test_finetune_token_classification.py
        """

        if isinstance(model, str):
            if os.path.exists(model):
                model_dir = model if os.path.isdir(model) else os.path.dirname(
                    model)
            else:
                model_dir = snapshot_download(model, revision=model_revision)
            cfg_file = os.path.join(model_dir, ModelFile.CONFIGURATION)
        else:
            assert cfg_file is not None, 'Config file should not be None if model is an nn.Module class'
            model_dir = os.path.dirname(cfg_file)

        self.label2id = None
        self.id2label = None
        self.num_labels = None
        self.cfg_modify_fn = cfg_modify_fn
        self.cfg = self.rebuild_config(Config.from_file(cfg_file))

        label2id = parse_label_mapping(model_dir)
        if label2id is not None:
            self.label2id = label2id
            self.id2label = {id: label for label, id in label2id.items()}
            self.num_labels = len(label2id)
        else:
            try:
                labels = self.cfg.dataset.train.labels
                if labels is not None and len(labels) > 0:
                    self.label2id = {
                        label: idx
                        for idx, label in enumerate(labels)
                    }
                    self.id2label = {
                        idx: label
                        for idx, label in enumerate(labels)
                    }
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

        self.train_keys = build_dataset_keys(
            self.cfg.dataset.train if hasattr(self.cfg, 'dataset')
            and hasattr(self.cfg.dataset, 'train') else None)
        self.eval_keys = build_dataset_keys(
            self.cfg.dataset.val if hasattr(self.cfg, 'dataset')
            and hasattr(self.cfg.dataset, 'val') else None)
        if len(self.eval_keys) == 0:
            self.eval_keys = self.train_keys

        super().__init__(
            model=model_dir,
            cfg_file=cfg_file,
            arg_parse_fn=arg_parse_fn,
            data_collator=data_collator,
            preprocessor=preprocessor,
            optimizers=optimizers,
            model_revision=model_revision,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs)

    def rebuild_config(self, cfg: Config):
        if self.cfg_modify_fn is not None:
            cfg = self.cfg_modify_fn(cfg)
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
        model_args = {} if self.label2id is None else {
            'label2id': self.label2id
        }

        field_name = Tasks.find_field_by_task(self.cfg.task)
        train_preprocessor, eval_preprocessor = None, None
        _train_cfg, _eval_cfg = {}, {}

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
            _train_cfg.update({
                'model_dir': self.model_dir,
                **model_args,
                **self.train_keys, 'mode': ModeKeys.TRAIN
            })
            train_preprocessor = build_preprocessor(_train_cfg, field_name)
        if len(_eval_cfg):
            _eval_cfg.update({
                'model_dir': self.model_dir,
                **model_args,
                **self.eval_keys, 'mode': ModeKeys.EVAL
            })
            eval_preprocessor = build_preprocessor(_eval_cfg, field_name)

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
            metric_values[metric_name] = np.average(
                [m[metric_name] for m in metric_values.values()])

        return metric_values
