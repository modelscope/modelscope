# Copyright (c) Alibaba, Inc. and its affiliates.
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import torch
from datasets import Dataset, IterableDataset, concatenate_datasets
from torch.utils.data import ConcatDataset
from transformers import DataCollatorWithPadding

from modelscope.metainfo import Models
from modelscope.utils.constant import ModeKeys, Tasks
from .base import TaskDataset
from .builder import TASK_DATASETS
from .torch_base_dataset import TorchTaskDataset


@TASK_DATASETS.register_module(
    group_key=Tasks.text_ranking, module_name=Models.bert)
@TASK_DATASETS.register_module(
    group_key=Tasks.sentence_embedding, module_name=Models.bert)
class TextRankingDataset(TorchTaskDataset):

    def __init__(self,
                 datasets: Union[Any, List[Any]],
                 mode,
                 preprocessor=None,
                 *args,
                 **kwargs):
        self.seed = kwargs.get('seed', 42)
        self.permutation = None
        self.datasets = None
        self.dataset_config = kwargs
        self.query_sequence = self.dataset_config.get('query_sequence',
                                                      'query')
        self.pos_sequence = self.dataset_config.get('pos_sequence',
                                                    'positive_passages')
        self.neg_sequence = self.dataset_config.get('neg_sequence',
                                                    'negative_passages')
        self.text_fileds = self.dataset_config.get('text_fileds',
                                                   ['title', 'text'])
        self.qid_field = self.dataset_config.get('qid_field', 'query_id')
        if mode == ModeKeys.TRAIN:
            self.neg_samples = self.dataset_config.get('neg_sample', 4)

        super().__init__(datasets, mode, preprocessor, **kwargs)

    def __getitem__(self, index) -> Any:
        if self.mode == ModeKeys.TRAIN:
            return self.__get_train_item__(index)
        else:
            return self.__get_test_item__(index)

    def __get_test_item__(self, index):
        group = self._inner_dataset[index]
        labels = []

        qry = group[self.query_sequence]

        pos_sequences = group[self.pos_sequence]
        pos_sequences = [
            ' '.join([ele[key] for key in self.text_fileds])
            for ele in pos_sequences
        ]
        labels.extend([1] * len(pos_sequences))

        neg_sequences = group[self.neg_sequence]
        neg_sequences = [
            ' '.join([ele[key] for key in self.text_fileds])
            for ele in neg_sequences
        ]

        labels.extend([0] * len(neg_sequences))
        qid = group[self.qid_field]

        examples = pos_sequences + neg_sequences
        sample = {
            'qid': torch.LongTensor([int(qid)] * len(labels)),
            self.preprocessor.first_sequence: qry,
            self.preprocessor.second_sequence: examples,
            'labels': torch.LongTensor(labels)
        }
        return self.prepare_sample(sample)

    def __get_train_item__(self, index):
        group = self._inner_dataset[index]

        qry = group[self.query_sequence]

        pos_sequences = group[self.pos_sequence]
        pos_sequences = [
            ' '.join([ele[key] for key in self.text_fileds])
            for ele in pos_sequences
        ]

        neg_sequences = group[self.neg_sequence]
        neg_sequences = [
            ' '.join([ele[key] for key in self.text_fileds])
            for ele in neg_sequences
        ]

        pos_psg = random.choice(pos_sequences)

        if len(neg_sequences) < self.neg_samples:
            negs = random.choices(neg_sequences, k=self.neg_samples)
        else:
            negs = random.sample(neg_sequences, k=self.neg_samples)
        examples = [pos_psg] + negs
        sample = {
            self.preprocessor.first_sequence: qry,
            self.preprocessor.second_sequence: examples,
        }
        return self.prepare_sample(sample)

    def __len__(self):
        return len(self._inner_dataset)

    def prepare_dataset(self, datasets: Union[Any, List[Any]]) -> Any:
        """Prepare a dataset.

        User can process the input datasets in a whole dataset perspective.
        This method gives a default implementation of datasets merging, user can override this
        method to write custom logics.

        Args:
            datasets: The original dataset(s)

        Returns: A single dataset, which may be created after merging.

        """
        if isinstance(datasets, List):
            if len(datasets) == 1:
                return datasets[0]
            elif len(datasets) > 1:
                return ConcatDataset(datasets)
        else:
            return datasets

    def prepare_sample(self, data):
        """Preprocess the data fetched from the inner_dataset.

        If the preprocessor is None, the original data will be returned, else the preprocessor will be called.
        User can override this method to implement custom logics.

        Args:
            data: The data fetched from the dataset.

        Returns: The processed data.

        """
        return self.preprocessor(
            data) if self.preprocessor is not None else data
