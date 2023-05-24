# Copyright (c) Alibaba, Inc. and its affiliates.
"""PyTorch trainer for UniTE model."""

import os.path as osp
import random
from math import ceil
from os import mkdir
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from pandas import DataFrame
from torch.nn.functional import pad
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.utils.data import (BatchSampler, DataLoader, Dataset, Sampler,
                              SequentialSampler, SubsetRandomSampler)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from modelscope.metainfo import Metrics, Trainers
from modelscope.metrics import Metric
from modelscope.metrics.builder import MetricKeys, build_metric
from modelscope.models.base import TorchModel
from modelscope.models.nlp.unite.configuration import InputFormat
from modelscope.models.nlp.unite.translation_evaluation import (
    UniTEForTranslationEvaluation, combine_input_sentences)
from modelscope.msdatasets import MsDataset
from modelscope.preprocessors import Preprocessor
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.hooks import Hook
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.config import ConfigDict
from modelscope.utils.constant import (ConfigKeys, Fields, ModeKeys, ModelFile,
                                       TrainerStages)
from modelscope.utils.device import create_device
from modelscope.utils.logger import get_logger

logger = get_logger()


class TranslationEvaluationTrainingSampler(Sampler):

    def __init__(self, num_of_samples: int,
                 batch_size_for_each_input_format: int):
        r"""Build a sampler for model training with translation evaluation trainer.
        The trainer should derive samples for each subset of the entire dataset.

        Args:
            num_of_samples: The number of samples in total.
            batch_size_for_each_input_format: During training, the batch size for each input format

        Returns:
            A data sampler for translation evaluation model training.

        """

        self.num_of_samples = num_of_samples
        self.batch_size_for_each_input_format = batch_size_for_each_input_format

        self.num_of_samples_for_each_input_format = self.num_of_samples // 3
        num_of_samples_to_use = self.num_of_samples_for_each_input_format * 3

        logger.info(
            '%d samples are given for training. '
            'Using %d samples for each input format. '
            'Leaving the last %d samples unused.' %
            (self.num_of_samples, self.num_of_samples_for_each_input_format,
             self.num_of_samples - num_of_samples_to_use))
        self.num_of_samples = num_of_samples_to_use

        random_permutations = torch.randperm(
            self.num_of_samples).cpu().tolist()

        self.subset_iterators = dict()
        self.subset_samplers = dict()
        self.indices_for_each_input_format = dict()
        for input_format_index, input_format in \
                enumerate((InputFormat.SRC_REF, InputFormat.SRC, InputFormat.REF)):
            start_idx = input_format_index * self.num_of_samples_for_each_input_format
            end_idx = start_idx + self.num_of_samples_for_each_input_format
            self.indices_for_each_input_format[
                input_format] = random_permutations[start_idx:end_idx]
            self.subset_samplers[input_format] = \
                BatchSampler(SubsetRandomSampler(self.indices_for_each_input_format[input_format]),
                             batch_size=self.batch_size_for_each_input_format,
                             drop_last=True)
            self.subset_iterators[input_format] = iter(
                self.subset_samplers[input_format])

        self.num_of_sampled_batches = 0

        if self.__len__() == 0:
            raise ValueError(
                'The dataset doesn\'t contain enough examples to form a single batch.',
                'Please reduce the batch_size or use more examples for training.'
            )

        return

    def __iter__(self):
        while True:
            try:
                if self.num_of_sampled_batches == self.__len__():
                    for input_format in (InputFormat.SRC_REF, InputFormat.SRC,
                                         InputFormat.REF):
                        while True:
                            try:
                                next(self.subset_iterators[input_format])
                            except StopIteration:
                                self.subset_iterators[input_format] = \
                                    iter(self.subset_samplers[input_format])
                                break

                    self.num_of_sampled_batches = 0

                output = list()
                for input_format_idx, input_format in \
                        enumerate((InputFormat.SRC_REF, InputFormat.SRC, InputFormat.REF)):
                    output += next(self.subset_iterators[input_format])

                self.num_of_sampled_batches += 1

                yield output
            except StopIteration:
                break

    def __len__(self) -> int:
        return self.num_of_samples_for_each_input_format // self.batch_size_for_each_input_format


def convert_csv_dict_to_input(
        batch: List[Dict[str, Any]],
        preprocessor: Preprocessor) -> Tuple[List[torch.Tensor]]:

    input_dict = dict()

    for key in batch[0].keys():
        input_dict[key] = list(x[key] for x in batch)

    input_dict = preprocessor(input_dict)

    return input_dict


def data_collate_fn(batch: List[Dict[str, Any]], batch_size: int,
                    preprocessor: Preprocessor) -> List[Dict[str, Any]]:

    output_dict = dict()
    output_dict['input_format'] = list()

    if preprocessor.mode == ModeKeys.TRAIN:
        for input_format_index, input_format in \
                enumerate((InputFormat.SRC_REF, InputFormat.SRC, InputFormat.REF)):
            start_idx = input_format_index * batch_size
            end_idx = start_idx + batch_size
            batch_to_process = batch[start_idx:end_idx]
            output_dict['input_format'] += [input_format] * batch_size
            preprocessor.change_input_format(input_format)
            batch_to_process = convert_csv_dict_to_input(
                batch_to_process, preprocessor)

            for key, value in batch_to_process.items():
                if key not in output_dict.keys():
                    output_dict[key] = list()
                output_dict[key].append(value)
    elif preprocessor.mode == ModeKeys.EVAL:
        output_dict['input_format'] += [preprocessor.input_format] * len(batch)
        batch = convert_csv_dict_to_input(batch, preprocessor)

        for key, value in batch.items():
            if key not in output_dict.keys():
                output_dict[key] = list()
            output_dict[key].append(value)
    else:
        raise ValueError(
            'During training, %s mode is not allowed for preprocessor.'
            % preprocessor.mode)

    input_max_lengths = max(x.size(-1) for x in output_dict['input_ids'])
    output_dict['input_ids'] = list(
        pad(x,
            pad=(0, input_max_lengths - x.size(-1)),
            value=preprocessor.pad_token_id) for x in output_dict['input_ids'])

    output_dict['input_ids'] = torch.cat(output_dict['input_ids'], dim=0)
    output_dict['score'] = torch.Tensor(output_dict['score']).view(-1)

    if preprocessor.mode == ModeKeys.EVAL:
        output_dict['lp'] = sum(output_dict['lp'], list())
        output_dict['raw_score'] = sum(output_dict['raw_score'], list())
        output_dict['segment_id'] = sum(output_dict['segment_id'], list())

    return output_dict


@TRAINERS.register_module(module_name=Trainers.translation_evaluation_trainer)
class TranslationEvaluationTrainer(EpochBasedTrainer):

    def __init__(self,
                 model: Optional[Union[TorchModel, torch.nn.Module,
                                       str]] = None,
                 cfg_file: Optional[str] = None,
                 device: str = 'gpu',
                 *args,
                 **kwargs):
        r"""Build a translation evaluation trainer with a model dir or a model id in the model hub.

        Args:
            model: A Model instance.
            cfg_file: The path for the configuration file (configuration.json).
            device: Used device for this trainer.

        """

        def data_collator_for_train(x):
            return data_collate_fn(
                x,
                batch_size=self.cfg.train.batch_size,
                preprocessor=self.train_preprocessor)

        def data_collator_for_eval(x):
            return data_collate_fn(
                x,
                batch_size=self.cfg.evaluation.batch_size,
                preprocessor=self.eval_preprocessor)

        data_collator = {
            ConfigKeys.train: data_collator_for_train,
            ConfigKeys.val: data_collator_for_eval
        }

        super().__init__(
            model,
            cfg_file=cfg_file,
            data_collator=data_collator,
            *args,
            **kwargs)

        self.train_dataloader = None
        self.eval_dataloader = None

        return

    def build_optimizer(self, cfg: ConfigDict) -> Optimizer:
        r"""Sets the optimizers to be used during training."""
        if self.cfg.train.optimizer.type != 'AdamW':
            return super().build_optimizer(cfg)

        # Freezing embedding layers for more efficient training.
        for param in self.model.encoder.embeddings.parameters():
            param.requires_grad = False

        logger.info('Building AdamW optimizer ...')
        learning_rates_and_parameters = list({
            'params':
            self.model.encoder.encoder.layer[i].parameters(),
            'lr':
            self.cfg.train.optimizer.plm_lr
            * self.cfg.train.optimizer.plm_lr_layerwise_decay**i,
        } for i in range(0, self.cfg.model.num_hidden_layers))

        learning_rates_and_parameters.append({
            'params':
            self.model.encoder.embeddings.parameters(),
            'lr':
            self.cfg.train.optimizer.plm_lr,
        })

        learning_rates_and_parameters.append({
            'params':
            self.model.estimator.parameters(),
            'lr':
            self.cfg.train.optimizer.mlp_lr
        })

        learning_rates_and_parameters.append({
            'params':
            self.model.layerwise_attention.parameters(),
            'lr':
            self.cfg.train.optimizer.mlp_lr,
        })

        optimizer = AdamW(
            learning_rates_and_parameters,
            lr=self.cfg.train.optimizer.plm_lr,
            betas=self.cfg.train.optimizer.betas,
            eps=self.cfg.train.optimizer.eps,
            weight_decay=self.cfg.train.optimizer.weight_decay,
        )

        return optimizer

    def get_train_dataloader(self) -> DataLoader:
        logger.info('Building dataloader for training ...')

        if self.train_dataset is None:
            logger.info('Reading train csv file from %s ...'
                        % self.cfg.dataset.train.name)
            self.train_dataset = MsDataset.load(
                osp.join(self.model_dir, self.cfg.dataset.train.name),
                split=self.cfg.dataset.train.split)

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=TranslationEvaluationTrainingSampler(
                len(self.train_dataset),
                batch_size_for_each_input_format=self.cfg.train.batch_size),
            num_workers=4,
            collate_fn=self.train_data_collator,
            generator=None)

        logger.info('Reading done, %d items in total'
                    % len(self.train_dataset))

        return train_dataloader

    def get_eval_data_loader(self) -> DataLoader:
        logger.info('Building dataloader for evaluating ...')

        if self.eval_dataset is None:
            logger.info('Reading eval csv file from %s ...'
                        % self.cfg.dataset.valid.name)

            self.eval_dataset = MsDataset.load(
                osp.join(self.model_dir, self.cfg.dataset.valid.name),
                split=self.cfg.dataset.valid.split)

        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_sampler=BatchSampler(
                SequentialSampler(range(0, len(self.eval_dataset))),
                batch_size=self.cfg.evaluation.batch_size,
                drop_last=False),
            num_workers=4,
            collate_fn=self.eval_data_collator,
            generator=None)

        logger.info('Reading done, %d items in total' % len(self.eval_dataset))

        return eval_dataloader

    def evaluation_loop(self, data_loader, metric_classes):
        """ Evaluation loop used by `TranslationEvaluationTrainer.evaluate()`.

        The evaluation process of UniTE model should be arranged with three loops,
        corresponding to the input formats of `InputFormat.SRC_REF`, `InputFormat.REF`,
        and `InputFormat.SRC`.

        Here we directly copy the codes of `EpochBasedTrainer.evaluation_loop`, and change
        the input format during each evaluation subloop.
        """
        vis_closure = None
        if hasattr(self.cfg.evaluation, 'visualization'):
            vis_cfg = self.cfg.evaluation.visualization
            vis_closure = partial(
                self.visualization, dataset=self.eval_dataset, **vis_cfg)

        self.invoke_hook(TrainerStages.before_val)
        metric_values = dict()

        for input_format in (InputFormat.SRC_REF, InputFormat.SRC,
                             InputFormat.REF):
            self.eval_preprocessor.change_input_format(input_format)

            if self._dist:
                from modelscope.trainers.utils.inference import multi_gpu_test
                # list of batched result and data samples
                metric_values.update(
                    multi_gpu_test(
                        self,
                        data_loader,
                        device=self.device,
                        metric_classes=metric_classes,
                        vis_closure=vis_closure,
                        tmpdir=self.cfg.evaluation.get('cache_dir', None),
                        gpu_collect=self.cfg.evaluation.get(
                            'gpu_collect', False),
                        data_loader_iters_per_gpu=self._eval_iters_per_epoch))
            else:
                from modelscope.trainers.utils.inference import single_gpu_test
                metric_values.update(
                    single_gpu_test(
                        self,
                        data_loader,
                        device=self.device,
                        metric_classes=metric_classes,
                        vis_closure=vis_closure,
                        data_loader_iters=self._eval_iters_per_epoch))

            for m in metric_classes:
                if hasattr(m, 'clear') and callable(m.clear):
                    m.clear()

        self.invoke_hook(TrainerStages.after_val)
        return metric_values
