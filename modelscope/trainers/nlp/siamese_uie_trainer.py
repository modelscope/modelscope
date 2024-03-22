# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import random
import time
from collections import defaultdict
from math import ceil
from typing import Callable, Dict, List, Optional, Tuple, Union

import json
import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.utils.data import Dataset

from modelscope.metainfo import Trainers
from modelscope.models.base import TorchModel
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.preprocessors.base import Preprocessor
from modelscope.trainers import EpochBasedTrainer, NlpEpochBasedTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.utils.config import Config
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, ModeKeys, Tasks
from modelscope.utils.file_utils import func_receive_dict_inputs
from modelscope.utils.logger import get_logger
from ..parallel.utils import is_parallel

PATH = None
logger = get_logger(PATH)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@TRAINERS.register_module(module_name=Trainers.siamese_uie_trainer)
class SiameseUIETrainer(EpochBasedTrainer):

    def __init__(
            self,
            model: Optional[Union[TorchModel, nn.Module, str]] = None,
            cfg_file: Optional[str] = None,
            cfg_modify_fn: Optional[Callable] = None,
            train_dataset: Optional[Union[MsDataset, Dataset]] = None,
            eval_dataset: Optional[Union[MsDataset, Dataset]] = None,
            preprocessor: Optional[Union[Preprocessor,
                                         Dict[str, Preprocessor]]] = None,
            optimizers: Tuple[torch.optim.Optimizer,
                              torch.optim.lr_scheduler._LRScheduler] = (None,
                                                                        None),
            model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
            seed: int = 42,
            negative_sampling_rate=1,
            slide_len=352,
            max_len=384,
            hint_max_len=128,
            **kwargs):
        """Epoch based Trainer, a training helper for PyTorch.

        Args:
            model (:obj:`torch.nn.Module` or :obj:`TorchModel` or `str`): The model to be run, or a valid model dir
                or a model id. If model is None, build_model method will be called.
            cfg_file(str): The local config file.
            cfg_modify_fn (function): Optional[Callable] = None, config function
            train_dataset (`MsDataset` or `torch.utils.data.Dataset`, *optional*):
                The dataset to use for training.

                Note that if it's a `torch.utils.data.IterableDataset` with some randomization and you are training in a
                distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
                `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will
                manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
                sets the seed of the RNGs used.
            eval_dataset (`MsDataset` or `torch.utils.data.Dataset`, *optional*): The dataset to use for evaluation.
            preprocessor (:obj:`Preprocessor`, *optional*): The optional preprocessor.
                NOTE: If the preprocessor has been called before the dataset fed into this
                trainer by user's custom code,
                this parameter should be None, meanwhile remove the 'preprocessor' key from the cfg_file.
                Else the preprocessor will be instantiated from the cfg_file or assigned from this parameter and
                this preprocessing action will be executed every time the dataset's __getitem__ is called.
            optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]`, *optional*): A tuple
                containing the optimizer and the scheduler to use.
            model_revision (str): The model version to use in modelhub.
            negative_sampling_rate (float): The rate to do negative sampling.
            slide_len (int): The length to slide.
            max_len (int): The max length of prompt + text.
            hint_max_len (int): The max length of prompt.
            seed (int): The optional random seed for torch, cuda, numpy and random.
        """
        print('*******************')
        self.slide_len = slide_len
        self.max_len = max_len
        self.hint_max_len = hint_max_len
        self.negative_sampling_rate = negative_sampling_rate

        super().__init__(
            model=model,
            cfg_file=cfg_file,
            cfg_modify_fn=cfg_modify_fn,
            data_collator=self._nn_collate_fn,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            preprocessor=preprocessor,
            optimizers=optimizers,
            model_revision=model_revision,
            seed=seed,
            **kwargs)

    def build_dataset(self,
                      datasets: Union[torch.utils.data.Dataset, MsDataset,
                                      List[torch.utils.data.Dataset]],
                      model_cfg: Config,
                      mode: str,
                      preprocessor: Optional[Preprocessor] = None,
                      **kwargs):
        if mode == ModeKeys.TRAIN:
            datasets = self.load_dataset(datasets)
        return super(SiameseUIETrainer, self).build_dataset(
            datasets=datasets,
            model_cfg=self.cfg,
            mode=mode,
            preprocessor=preprocessor,
            **kwargs)

    def get_train_dataloader(self):
        """ Builder torch dataloader for training.

        We provide a reasonable default that works well. If you want to use something else, you can change
        the config for data.train in configuration file, or subclass and override this method
        (or `get_train_dataloader` in a subclass.
        """
        self.train_dataset.preprocessor = None
        data_loader = self._build_dataloader_with_dataset(
            self.train_dataset,
            dist=self._dist,
            seed=self._seed,
            collate_fn=self.train_data_collator,
            **self.cfg.train.get('dataloader', {}))
        return data_loader

    def get_brother_type_map(self, schema, brother_type_map, prefix_types):
        if not schema:
            return
        for k in schema:
            brother_type_map[tuple(prefix_types
                                   + [k])] += [v for v in schema if v != k]
            self.get_brother_type_map(schema[k], brother_type_map,
                                      prefix_types + [k])

    def load_dataset(self, raw_dataset):
        data = []
        for num_line, raw_sample in enumerate(raw_dataset):
            raw_sample['info_list'] = json.loads(raw_sample['info_list'])
            raw_sample['schema'] = json.loads(raw_sample['schema'])
            hint_spans_map = defaultdict(list)
            # positive sampling
            for info in raw_sample['info_list']:
                hint = ''
                for item in info:
                    hint += f'{item["type"]}: '
                    span = {'span': item['span'], 'offset': item['offset']}
                    if span not in hint_spans_map[hint]:
                        hint_spans_map[hint].append(span)
                    hint += f'{item["span"]}, '
            # negative sampling
            brother_type_map = defaultdict(list)
            self.get_brother_type_map(raw_sample['schema'], brother_type_map,
                                      [])

            for info in raw_sample['info_list']:
                hint = ''
                for i, item in enumerate(info):
                    key = tuple([info[j]['type'] for j in range(i + 1)])
                    for st in brother_type_map.get(key, []):
                        neg_hint = hint + f'{st}: '
                        if neg_hint not in hint_spans_map and random.random(
                        ) < self.negative_sampling_rate:
                            hint_spans_map[neg_hint] = []
                    hint += f'{item["type"]}: '
                    hint += f'{item["span"]}, '
            # info list为空
            for k in raw_sample['schema']:
                neg_hint = f'{k}: '
                if neg_hint not in hint_spans_map and random.random(
                ) < self.negative_sampling_rate:
                    hint_spans_map[neg_hint] = []

            for i, hint in enumerate(hint_spans_map):
                sample = {
                    'id': f'{raw_sample["id"]}-{i}',
                    'hint': hint,
                    'text': raw_sample['text'],
                    'spans': hint_spans_map[hint]
                }
                uuid = sample['id']
                text = sample['text']
                tokenized_input = self.train_preprocessor([text])[0]
                tokenized_hint = self.train_preprocessor(
                    [hint], max_length=self.hint_max_len, truncation=True)[0]
                sample['offsets'] = tokenized_input.offsets
                entities = sample.get('spans', [])
                head_labels, tail_labels = self._get_labels(
                    text, tokenized_input, sample['offsets'], entities)

                split_num = ceil(
                    (len(tokenized_input) - self.max_len) / self.slide_len
                ) + 1 if len(tokenized_input) > self.max_len else 1
                for j in range(split_num):
                    a, b = j * self.slide_len, j * self.slide_len + self.max_len
                    item = {
                        'id': uuid,
                        'shift': a,
                        'tokens': tokenized_input.tokens[a:b],
                        'token_ids': tokenized_input.ids[a:b],
                        'hint_tokens': tokenized_hint.tokens,
                        'hint_token_ids': tokenized_hint.ids,
                        'attention_masks': tokenized_input.attention_mask[a:b],
                        'cross_attention_masks': tokenized_hint.attention_mask,
                        'head_labels': head_labels[a:b],
                        'tail_labels': tail_labels[a:b]
                    }
                    data.append(item)

        from datasets import Dataset
        train_dataset = Dataset.from_list(data)
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(
                f'Sample {index} of the training set: {train_dataset[index]}.')
        return train_dataset

    def _get_labels(self, text, tokenized_input, offsets, entities):
        num_tokens = len(tokenized_input)
        head_labels = [0] * num_tokens
        tail_labels = [0] * num_tokens
        char_index_to_token_index_map = {}
        for i in range(len(offsets)):
            offset = offsets[i]
            for j in range(offset[0], offset[1]):
                char_index_to_token_index_map[j] = i
        for e in entities:
            h, t = e['offset']
            t -= 1
            while h not in char_index_to_token_index_map:
                h += 1
                if h > len(text):
                    print('h', e['offset'], e['span'],
                          text[e['offset'][0]:e['offset'][1]])
                    break
            while t not in char_index_to_token_index_map:
                t -= 1
                if t < 0:
                    print('t', e['offset'], e['span'],
                          text[e['offset'][0]:e['offset'][1]])
                    break
            if h > len(text) or t < 0:
                continue
            token_head = char_index_to_token_index_map[h]
            token_tail = char_index_to_token_index_map[t]
            head_labels[token_head] = 1
            tail_labels[token_tail] = 1
        return head_labels, tail_labels

    def _padding(self, data, val=0):
        res = []
        for seq in data:
            res.append(seq + [val] * (self.max_len - len(seq)))
        return res

    def _nn_collate_fn(self, batch):
        token_ids = torch.tensor(
            self._padding([item['token_ids'] for item in batch]),
            dtype=torch.long)
        hint_token_ids = torch.tensor(
            self._padding([item['hint_token_ids'] for item in batch]),
            dtype=torch.long)
        attention_masks = torch.tensor(
            self._padding([item['attention_masks'] for item in batch]),
            dtype=torch.long)
        cross_attention_masks = torch.tensor(
            self._padding([item['cross_attention_masks'] for item in batch]),
            dtype=torch.long)
        head_labels = torch.tensor(
            self._padding([item['head_labels'] for item in batch]),
            dtype=torch.float)
        tail_labels = torch.tensor(
            self._padding([item['tail_labels'] for item in batch]),
            dtype=torch.float)
        # the content of `batch` is like batch_size * [token_ids, head_labels, tail_labels]
        # for fp16 acceleration, truncate seq_len to multiples of 8
        batch_max_len = token_ids.gt(0).sum(dim=-1).max().item()
        batch_max_len += (8 - batch_max_len % 8) % 8
        truncate_len = min(self.max_len, batch_max_len)
        token_ids = token_ids[:, :truncate_len]
        attention_masks = attention_masks[:, :truncate_len]
        head_labels = head_labels[:, :truncate_len]
        tail_labels = tail_labels[:, :truncate_len]

        # for fp16 acceleration, truncate seq_len to multiples of 8
        batch_max_len = hint_token_ids.gt(0).sum(dim=-1).max().item()
        batch_max_len += (8 - batch_max_len % 8) % 8
        hint_truncate_len = min(self.hint_max_len, batch_max_len)
        hint_token_ids = hint_token_ids[:, :hint_truncate_len]
        cross_attention_masks = cross_attention_masks[:, :hint_truncate_len]

        return {
            'input_ids': token_ids,
            'attention_masks': attention_masks,
            'hint_ids': hint_token_ids,
            'cross_attention_masks': cross_attention_masks,
            'head_labels': head_labels,
            'tail_labels': tail_labels
        }

    def evaluate(self,
                 checkpoint_path: Optional[str] = None,
                 *args,
                 **kwargs) -> Dict[str, float]:
        """evaluate a dataset

        evaluate a dataset via a specific model from the `checkpoint_path` path, if the `checkpoint_path`
        does not exist, read from the config file.

        Args:
            checkpoint_path (Optional[str], optional): the model path. Defaults to None.

        Returns:
            Dict[str, float]: the results about the evaluation
            Example:
            {"accuracy": 0.5091743119266054, "f1": 0.673780487804878}
        """
        pipeline_uie = pipeline(
            Tasks.siamese_uie, self.model, device=str(self.device))
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            from modelscope.trainers.hooks import LoadCheckpointHook
            LoadCheckpointHook.load_checkpoint(checkpoint_path, self)
        self.model.eval()
        self._mode = ModeKeys.EVAL
        self.eval_dataloader = self.train_dataloader
        num_pred = num_recall = num_correct = 1e-10
        self.eval_dataset.preprocessor = None
        for sample in self.eval_dataset:
            text = sample['text']
            schema = json.loads(sample['schema'])
            gold_info_list = json.loads(sample['info_list'])
            pred_info_list = pipeline_uie(input=text, schema=schema)['output']
            pred_info_list_set = set([str(item) for item in pred_info_list])
            gold_info_list_set = set([str(item) for item in gold_info_list])
            a, b, c = len(pred_info_list_set), len(gold_info_list_set), len(
                pred_info_list_set.intersection(gold_info_list_set))
            num_pred += a
            num_recall += b
            num_correct += c
        precision, recall, f1 = self.compute_metrics(num_pred, num_recall,
                                                     num_correct)
        return {'precision': precision, 'recall': recall, 'f1': f1}

    def get_metrics(self) -> List[Union[str, Dict]]:
        """Get the metric class types.

        The first choice will be the metrics configured in the config file, if not found, the default metrics will be
        used.
        If no metrics is found and the eval dataset exists, the method will raise an error.

        Returns: The metric types.

        """
        return self.compute_metrics

    def compute_metrics(self, num_pred, num_recall, num_correct):
        if num_pred == num_recall == 1e-10:
            return 1, 1, 1
        precision = num_correct / float(num_pred)
        recall = num_correct / float(num_recall)
        f1 = 2 * precision * recall / (precision + recall)
        # print(num_pred, num_recall, num_correct)
        if num_correct == 1e-10:
            return 0, 0, 0
        return precision, recall, f1
