# Copyright (c) Alibaba, Inc. and its affiliates.

import contextlib
from collections import defaultdict
from dataclasses import dataclass
from distutils.version import LooseVersion
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from modelscope.metainfo import Trainers
from modelscope.models.base import TorchModel
from modelscope.msdatasets import MsDataset
from modelscope.preprocessors import Preprocessor
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.nlp_trainer import EpochBasedTrainer
from modelscope.trainers.trainer import worker_init_fn
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, ModeKeys
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import get_dist_info

logger = get_logger()


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class EpisodeSampler(torch.utils.data.BatchSampler):

    def __init__(self, dataset, k_shot, n_way, r_query, min_labels, seed,
                 n_iter, rank, world_size):
        self.dataset = dataset
        self.k_shot = k_shot
        self.n_way = n_way
        self.r_query = r_query
        self.min_labels = min_labels
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.step = 0
        self.label_field = 'label'
        self.text_field = 'text'
        self.domain_field = 'domain'
        self.default_domain = 'default_domain'
        self.episode = n_iter
        domain_label_sampleid = {}
        bad_sample_ids = self.get_bad_sampleids(dataset)
        if dataset.mode == 'train':
            logger.info(
                f'num. of bad sample ids:{len(bad_sample_ids)}/{len(dataset)}')
        for sample_index, sample in enumerate(dataset):
            if sample_index in bad_sample_ids:
                continue
            label = self._get_field(sample, self.label_field)
            text = self._get_field(sample, self.text_field)
            if label is None or text is None:
                continue
            domain = self._get_field(sample, self.domain_field,
                                     self.default_domain)
            label_tokens = domain_label_sampleid.get(domain, {})
            domain_label_sampleid[domain] = label_tokens
            sample_list = label_tokens.get(label, [])
            label_tokens[label] = sample_list
            sample_list.append(sample_index)
        self.domain_label_tokens = self.remove_invalid_labels(
            domain_label_sampleid)
        self.domains = sorted(list(self.domain_label_tokens.keys()))
        domain_label_cnt = [
            len(self.domain_label_tokens[domain]) for domain in self.domains
        ]
        total = float(sum(domain_label_cnt))
        self.domain_to_prob = [
            domain_label_cnt[i] / total
            for i, domain in enumerate(self.domains)
        ]
        data_size = 0
        for domain, label_tokens in self.domain_label_tokens.items():
            for label, tokens in label_tokens.items():
                data_size += len(tokens)
        if dataset.mode == 'train':
            logger.info(
                f'{dataset.mode}: label size:{total}, data size:{data_size}, \
                domain_size:{len(self.domain_label_tokens)}')
        self.mode = dataset.mode

    def __iter__(self):
        for i in range(self.episode):
            seed = self.step * self.world_size + self.rank
            with numpy_seed(*(seed, self.seed)):
                self.step += 1
                domain = np.random.choice(
                    self.domains, p=self.domain_to_prob, size=1,
                    replace=False)[0]
                all_labels = sorted(
                    list(self.domain_label_tokens[domain].keys()))
                N = min(self.n_way, len(all_labels))
                labels = np.random.choice(
                    all_labels, size=min(N, len(all_labels)),
                    replace=False).tolist()
                batch = []
                for label in labels[:N]:
                    candidates = self.domain_label_tokens[domain][label]
                    num_samples = self.k_shot + self.r_query
                    K = min(len(candidates), int(num_samples))
                    tmp = np.random.choice(
                        candidates, size=K, replace=False).tolist()
                    batch.extend(tmp)
                batch = [int(n) for n in batch]
                yield batch

    def _get_field(self, obj, key, default=None):
        value = obj.get(key, default)
        if value is not None:
            return str(value)
        return None

    def remove_invalid_labels(self, domain_label_sampleid):
        removed_labels = set()
        removed_domains = set()
        result = {}
        for domain, label_to_samples in domain_label_sampleid.items():
            result[domain] = {}
            for label, samples in label_to_samples.items():
                if len(samples) < self.k_shot:
                    removed_labels.add(label)
                else:
                    result[domain][label] = samples
            if len(result[domain]) < self.min_labels:
                del result[domain]
                removed_domains.add(domain)
        return result

    def get_bad_sampleids(self, dataset):
        domain_text_to_samples = defaultdict(lambda: defaultdict(list))
        for local_index, sample in enumerate(dataset):
            domain = self._get_field(
                sample, self.domain_field, default=self.default_domain)
            idx = self._get_field(sample, self.text_field, default='')
            domain_text_to_samples[domain][idx].append(
                (local_index, self._get_field(sample, self.label_field)))

        overall_conflict_result = []
        overall_duplicate_result = []
        for domain, text_to_samples in domain_text_to_samples.items():
            conflict_result = []
            duplicate_result = []
            for text, samples in text_to_samples.items():
                label_cnt = set([item[1] for item in samples])
                if len(label_cnt) >= 2:
                    conflict_result.extend([item[0] for item in samples])
                else:
                    duplicate_result.extend([item[0] for item in samples[1:]])
            overall_conflict_result.extend(conflict_result)
            overall_duplicate_result.extend(duplicate_result)

        result = set(list(overall_duplicate_result))
        # remove conflict data which the same query has different label
        result.update(set(list(overall_conflict_result)))
        return result

    def __len__(self):
        return self.episode


@dataclass
class FewShotCollator():

    def __init__(self, preprocessor: Preprocessor, k_shot):
        self.preprocessor = preprocessor
        self.k_shot = k_shot
        self.label_field = 'label'
        self.text_field = 'text'
        self.domain_field = 'domain'

    def _get_field(self, obj, key, default=None):
        return getattr(obj, key, default) or obj.get(key, default)

    def __call__(self, samples):
        label_to_texts = defaultdict(list)
        for sample in samples:
            text = self._get_field(sample, self.text_field)
            label = self._get_field(sample, self.label_field)
            label_to_texts[label].append(text)
        query_set = []
        query_labels = []
        support_set = []
        for label, texts in label_to_texts.items():
            s = texts[:self.k_shot]
            q = texts[self.k_shot:]
            query_set.extend(q)
            support_set.extend([{
                self.text_field: t,
                self.label_field: label
            } for t in s])
            query_labels.extend([label] * len(q))
        sample = {
            'query_set': query_set,
            'support_set': support_set,
            'query_label': query_labels
        }
        result = self.preprocessor(sample, mode=ModeKeys.INFERENCE)
        return result


class FaqDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)


@TRAINERS.register_module(module_name=Trainers.faq_question_answering_trainer)
class FaqQuestionAnsweringTrainer(EpochBasedTrainer):

    def __init__(
            self,
            model: Optional[Union[TorchModel, nn.Module, str]] = None,
            cfg_file: Optional[str] = None,
            cfg_modify_fn: Optional[Callable] = None,
            arg_parse_fn: Optional[Callable] = None,
            data_collator: Optional[Union[Callable, Dict[str,
                                                         Callable]]] = None,
            train_dataset: Optional[Union[MsDataset, Dataset, List]] = None,
            eval_dataset: Optional[Union[MsDataset, Dataset, List]] = None,
            preprocessor: Optional[Union[Preprocessor,
                                         Dict[str, Preprocessor]]] = None,
            optimizers: Tuple[torch.optim.Optimizer,
                              torch.optim.lr_scheduler._LRScheduler] = (None,
                                                                        None),
            model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
            seed: int = 42,
            **kwargs):
        if isinstance(train_dataset, list):
            train_dataset = FaqDataset(train_dataset)
        if isinstance(eval_dataset, list):
            eval_dataset = FaqDataset(eval_dataset)
        super(FaqQuestionAnsweringTrainer,
              self).__init__(model, cfg_file, cfg_modify_fn, arg_parse_fn,
                             data_collator, train_dataset, eval_dataset,
                             preprocessor, optimizers, model_revision, seed,
                             **kwargs)
        k_shot = self.cfg.safe_get('train.sampler.k_shot')
        self.train_data_collator = FewShotCollator(self.train_preprocessor,
                                                   k_shot)
        self.eval_data_collator = FewShotCollator(self.eval_preprocessor,
                                                  k_shot)

    @property
    def max_iters(self):
        return self._train_iters_per_epoch * self.max_epochs

    @property
    def inner_iter(self) -> int:
        return 0

    def _build_dataloader_with_dataset(self,
                                       dataset: Dataset,
                                       workers_per_gpu: int,
                                       dist: bool = False,
                                       shuffle: bool = True,
                                       seed: int = 0,
                                       persistent_workers=False,
                                       **kwargs) -> DataLoader:
        rank, world_size = get_dist_info()
        sampler = None
        sampler_cfg = self.cfg.safe_get('train.sampler', {})
        sampler_cfg['seed'] = seed
        if dataset.mode == ModeKeys.TRAIN:
            sampler_cfg['n_iter'] = self.cfg.safe_get(
                'train.train_iters_per_epoch')
        else:
            sampler_cfg['n_iter'] = self.cfg.safe_get(
                'evaluation.val_iters_per_epoch')
        sampler_cfg['rank'] = rank
        sampler_cfg['world_size'] = world_size
        batch_sampler = EpisodeSampler(dataset, **sampler_cfg)

        init_fn = partial(
            worker_init_fn, num_workers=workers_per_gpu, rank=rank,
            seed=seed) if seed is not None else None

        if LooseVersion(torch.__version__) >= LooseVersion('1.7.0'):
            kwargs['persistent_workers'] = persistent_workers
        elif persistent_workers is True:
            self.logger.warning(
                'persistent_workers is invalid because your pytorch '
                'version is lower than 1.7.0')
        data_loader = DataLoader(
            dataset,
            sampler=sampler,
            num_workers=workers_per_gpu,
            batch_sampler=batch_sampler,
            pin_memory=kwargs.pop('pin_memory', False),
            worker_init_fn=init_fn,
            **kwargs)

        return data_loader
