# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""parses arguments and preps data loader"""

import copy
import os
import random
from bisect import bisect_right
from itertools import accumulate

import numpy as np
import torch
import torch.utils.data
from megatron_util import mpu, print_rank_0

from . import data_utils
from .blocklm_utils import ConstructBlockStrategy
from .data_utils.tokenization import make_tokenizer


class MultiTaskDataset(torch.utils.data.Dataset):

    def __init__(self,
                 tasks,
                 datasets,
                 reweight=True,
                 temperature=0.8,
                 max_limit=200000):
        super(MultiTaskDataset, self).__init__()
        self.tasks = tasks
        self.datasets = datasets
        self.reweight = reweight
        self.temperature = temperature
        self.lens = [len(dataset) for dataset in datasets]
        self.weights = np.array(
            [min(length, max_limit)**temperature for length in self.lens])
        self.total_len = sum(self.lens)
        self.cumulative_lens = list(accumulate(self.lens))
        if self.reweight:
            print_rank_0(list(zip(self.tasks, self.lens, self.weights)))
        else:
            print_rank_0(list(zip(self.tasks, self.lens)))
        self.weights /= self.weights.sum()

    def __len__(self):
        return self.total_len * 1000

    @staticmethod
    def pet_wrapper(data):
        text = data['text']
        loss_mask = data['logit_mask']
        target = data['target']
        attention_mask = data['mask']
        position_id = data['position']
        label = data['label']
        if len(text.shape) == 2:
            text = text[label]
            loss_mask = loss_mask[label]
            target = target[label]
            attention_mask = attention_mask[label]
            position_id = position_id[label]
        else:
            target = target[label]
        if not target.shape:
            target = target.repeat(len(text))
        return {
            'text': text,
            'target': target,
            'loss_mask': loss_mask,
            'position_id': position_id,
            'attention_mask': attention_mask
        }

    def __getitem__(self, idx):
        if self.reweight:
            rng = random.Random(idx)
            rng = np.random.RandomState(
                seed=[rng.randint(0, 2**32 - 1) for _ in range(16)])
            dataset_idx = rng.choice(
                np.arange(len(self.datasets)), p=self.weights)
            dataset = self.datasets[dataset_idx]
            sample_idx = rng.choice(np.arange(len(dataset)))
            item = self.datasets[dataset_idx][sample_idx]
        else:
            dataset_idx = bisect_right(self.cumulative_lens, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_lens[dataset_idx - 1]
            item = self.datasets[dataset_idx][sample_idx]
        item = self.pet_wrapper(item)
        return item


class DataConfig:

    def __init__(self, defaults=None):
        super(DataConfig, self).__init__()
        if defaults is None:
            defaults = {}
        self.defaults = defaults

    def apply(self, args, tokenizer):
        if torch.distributed.get_rank() == 0:
            print('configuring data')
        self.apply_defaults(args)
        return make_loaders(args, tokenizer)

    def set_defaults(self, **kwargs):
        for k, v in kwargs.items():
            self.defaults[k] = v

    def apply_defaults(self, args):
        for k, v in self.defaults.items():
            k = k.replace('-', '_')
            if not hasattr(args, k):
                setattr(args, k, v)


def prepare_tokenizer(args):
    add_sentinel_token = 0
    if args.sentinel_token:
        add_sentinel_token = args.max_position_embeddings
    tokenizer = make_tokenizer(
        args.tokenizer_type,
        None,
        args.tokenizer_path,
        args.vocab_size,
        args.tokenizer_model_type,
        add_block_symbols=args.block_lm,
        cache_dir=args.cache_dir,
        add_sentinel_token=add_sentinel_token,
        add_task_mask=args.task_mask,
        add_decoder_mask=args.block_mask_prob > 0.0
        or args.context_mask_ratio > 0.0)
    if mpu.get_model_parallel_rank() == 0:
        num_tokens = tokenizer.num_tokens
        eod_token = tokenizer.get_command('eos').Id
        assert eod_token == tokenizer.get_command('pad').Id
        before = num_tokens
        after = before
        multiple = args.make_vocab_size_divisible_by
        while (after % multiple) != 0:
            after += 1
        print_rank_0('> padded vocab (size: {}) with {} dummy '
                     'tokens (new size: {})'.format(before, after - before,
                                                    after))
        print_rank_0('> found end-of-document token: {}'.format(eod_token))
        token_counts = torch.cuda.LongTensor([after, eod_token])
    else:
        token_counts = torch.cuda.LongTensor([0, 0])
    # Broadcast num tokens.
    torch.distributed.broadcast(
        token_counts,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group())
    num_tokens = token_counts[0].item()
    eod_token = token_counts[1].item()
    args.vocab_size, args.eod_token = num_tokens, eod_token
    return tokenizer


def make_data_loader(dataset,
                     tokenizer,
                     batch_size,
                     num_iters,
                     args,
                     shuffle=False,
                     block_collate=False):
    world_size = torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group())
    rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    if args.loader_scatter is not None:
        rank = rank // args.loader_scatter
        world_size = world_size // args.loader_scatter
        batch_size = batch_size // args.loader_scatter
    distributed = world_size > 1
    if args.transformer_xl:
        batch_sampler = data_utils.samplers.DistributedSequentialSampler(
            len(dataset), num_iters, batch_size, rank, world_size)
    else:
        if shuffle:
            sampler = data_utils.samplers.RandomSampler(
                dataset,
                replacement=True,
                num_samples=batch_size * args.train_iters
                * args.gradient_accumulation_steps)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        drop_last = distributed
        # the GPUs in the same model parallel group receive the same data
        if distributed:
            batch_sampler = data_utils.samplers.DistributedBatchSampler(
                sampler,
                batch_size,
                drop_last,
                rank,
                world_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps)
        else:
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, batch_size, drop_last)
    collate_fn = None
    if block_collate:
        collate_fn = ConstructBlockStrategy(
            args,
            tokenizer,
            args.seq_length,
            bert_prob=args.bert_prob,
            gap_sentence_prob=args.gap_sentence_prob,
            gap_sentence_ratio=args.gap_sentence_ratio,
            gpt_infill_prob=args.gpt_infill_prob,
            average_block_length=args.avg_block_length,
            gpt_min_ratio=args.gpt_min_ratio,
            block_mask_prob=args.block_mask_prob,
            context_mask_ratio=args.context_mask_ratio,
            short_seq_prob=args.short_seq_prob,
            single_span_prob=args.single_span_prob,
            shuffle_blocks=not args.no_shuffle_block,
            block_position_encoding=not args.no_block_position,
            sentinel_token=args.sentinel_token,
            encoder_decoder=args.encoder_decoder,
            task_mask=args.task_mask,
            random_position=args.random_position,
            masked_lm=args.masked_lm).construct_blocks
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn)

    return data_loader


def make_tfrecord_loaders(args):
    """Load train/val/test dataset from shuffled TFRecords"""

    import data_utils.tf_dl
    data_set_args = {
        'batch_size': args.batch_size,
        'max_seq_len': args.seq_length,
        'max_preds_per_seq': args.max_preds_per_seq,
        'train': True,
        'num_workers': max(args.num_workers, 1),
        'seed': args.seed + args.rank + 1,
        'threaded_dl': args.num_workers > 0
    }
    train = data_utils.tf_dl.TFRecordDataLoader(args.train_data,
                                                **data_set_args)
    data_set_args['train'] = False
    if args.eval_seq_length is not None:
        data_set_args['max_seq_len'] = args.eval_seq_length
    if args.eval_max_preds_per_seq is not None:
        data_set_args['max_preds_per_seq'] = args.eval_max_preds_per_seq
    valid = None
    if args.valid_data is not None:
        valid = data_utils.tf_dl.TFRecordDataLoader(args.valid_data,
                                                    **data_set_args)
    test = None
    if args.test_data is not None:
        test = data_utils.tf_dl.TFRecordDataLoader(args.test_data,
                                                   **data_set_args)
    tokenizer = data_utils.make_tokenizer(
        args.tokenizer_type,
        train,
        args.tokenizer_path,
        args.vocab_size,
        args.tokenizer_model_type,
        cache_dir=args.cache_dir)

    return (train, valid, test), tokenizer


def make_loaders(args, tokenizer):
    """makes training/val/test"""

    if args.use_tfrecords:
        return make_tfrecord_loaders(args)
    world_size = torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group())
    if args.loader_scatter is not None:
        assert world_size % args.loader_scatter == 0
    batch_size = args.batch_size * world_size
    eval_batch_size = batch_size
    if args.eval_batch_size is not None:
        eval_batch_size = args.eval_batch_size * world_size
    seq_length = args.seq_length
    if seq_length < 0:
        seq_length = seq_length * world_size
    eval_seq_length = args.eval_seq_length
    if eval_seq_length is not None and eval_seq_length < 0:
        eval_seq_length = eval_seq_length * world_size
    split = get_split(args)
    data_set_args = {
        'path': args.train_data,
        'seq_length': seq_length,
        'mem_length': args.mem_length,
        'delim': args.delim,
        'text_key': args.text_key,
        'label_key': 'label',
        'ds_type': args.data_set_type,
        'split': split,
        'loose': args.loose_json,
        'max_preds_per_seq': args.max_preds_per_seq,
        'presplit_sentences': args.presplit_sentences,
        'sample_one_document': args.sample_one_document,
        'filter_english': args.filter_english,
        'pre_tokenize': not args.no_pre_tokenize,
        'tokenizer': tokenizer,
        'save_splits': args.save_splits,
        'load_splits': args.load_splits,
        'save_test_data': args.save_test_data,
        'no_lazy_loader': args.no_lazy_loader,
        'loader_scatter': args.loader_scatter,
        'data_parallel_rank': mpu.get_data_parallel_rank(),
        'non_sentence_start': args.non_sentence_start,
        'half_lazy_loader': args.half_lazy_loader
    }

    eval_set_args = copy.copy(data_set_args)
    eval_set_args['split'] = [1.]
    # if optional eval args were set then replace their
    # equivalent values in the arg dict
    if eval_seq_length:
        eval_set_args['seq_length'] = eval_seq_length
    if args.eval_max_preds_per_seq:
        eval_set_args['max_preds_per_seq'] = args.eval_max_preds_per_seq
    if args.eval_text_key is not None:
        eval_set_args['text_key'] = args.eval_text_key

    # make datasets splits and tokenizer
    train, valid, test = None, None, None

    if args.train_data is not None:
        train = data_utils.make_dataset(**data_set_args)
        if data_utils.should_split(split):
            train, valid, test = train
        eval_set_args['tokenizer'] = tokenizer

    # make training and val dataset if necessary
    if valid is None and args.valid_data is not None:
        eval_set_args['path'] = args.valid_data
        valid = data_utils.make_dataset(**eval_set_args)
        eval_set_args['tokenizer'] = tokenizer
    if test is None and args.test_data is not None:
        eval_set_args['path'] = args.test_data
        test = data_utils.make_dataset(**eval_set_args)

    # wrap datasets with data loader
    use_block = args.block_lm or args.encoder_decoder

    if train is not None and args.batch_size > 0:
        train = make_data_loader(
            train,
            tokenizer,
            batch_size,
            args.train_iters,
            args,
            shuffle=args.shuffle,
            block_collate=use_block)
        args.do_train = True
    else:
        args.do_train = False
    eval_batch_size = eval_batch_size if eval_batch_size != 0 else batch_size
    if valid is not None:
        valid = make_data_loader(
            valid,
            tokenizer,
            eval_batch_size,
            args.train_iters,
            args,
            shuffle=args.shuffle,
            block_collate=use_block)
        args.do_valid = True
    else:
        args.do_valid = False
    if test is not None:
        test = make_data_loader(
            test,
            tokenizer,
            eval_batch_size,
            len(test) // eval_batch_size + 1,
            args,
            shuffle=args.shuffle,
            block_collate=use_block)
        args.do_test = True
    else:
        args.do_test = False

    return train, valid, test


def build_multi_task_dataset(args, tokenizer):
    task_dirs = {
        'mnli': 'MNLI',
        'cola': 'CoLA',
        'mrpc': 'MRPC',
        'qnli': 'QNLI',
        'qqp': 'QQP',
        'sst2': 'SST-2',
        'agnews': 'Agnews',
        'yelp-polarity': 'yelp_review_polarity_csv',
        'yelp-full': 'yelp_review_full_csv',
        'yahoo': 'Yahoo',
        'squad': 'SQuAD',
        'race': 'RACE'
    }
    train, valid = None, None
    if mpu.get_model_parallel_rank() == 0:
        multi_seq_length = args.seq_length
        if args.multi_seq_length is not None:
            multi_seq_length = args.multi_seq_length
        train_datasets, valid_datasets = [], []
        for task in args.multi_task_data:
            task = task.lower()
            data_dir = os.path.join(args.data_dir, task_dirs[task])
            train_datasets.append(
                SuperGlueDataset(
                    args,
                    task,
                    data_dir,
                    multi_seq_length,
                    'train',
                    tokenizer,
                    pattern_ensemble=True))
            valid_datasets.append(
                SuperGlueDataset(
                    args,
                    task,
                    data_dir,
                    multi_seq_length,
                    'dev',
                    tokenizer,
                    pattern_ensemble=True))
        train = MultiTaskDataset(args.multi_task_data, train_datasets)
        valid = MultiTaskDataset(args.multi_task_data, valid_datasets)
        world_size = torch.distributed.get_world_size(
            group=mpu.get_data_parallel_group())
        multi_batch_size = args.batch_size * world_size
        if args.multi_batch_size is not None:
            multi_batch_size = args.multi_batch_size * world_size
        train = make_data_loader(
            train,
            tokenizer,
            multi_batch_size,
            args.train_iters,
            args,
            shuffle=True)
        valid = make_data_loader(
            valid,
            tokenizer,
            multi_batch_size,
            args.train_iters,
            args,
            shuffle=True)
    return train, valid


def get_split(args):
    """
    Get dataset splits from comma separated string list
    """
    splits = []
    if args.split.find(',') != -1:
        splits = [float(s) for s in args.split.split(',')]
    elif args.split.find('/') != -1:
        splits = [float(s) for s in args.split.split('/')]
    else:
        splits = [float(args.split)]
    split_total = sum(splits)
    if split_total < 1.:
        splits.append(1 - split_total)
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    if args.valid_data is not None:
        splits[1] = 0.
    if args.test_data is not None:
        splits[2] = 0.
    final_sum = sum(splits)
    return [s / final_sum for s in splits]


def configure_data():
    """add cmdline flags for configuring datasets"""
    # These are options that are used by data_utils, but are either
    # deprecated or not meant to be exposed to the command line user.
    # These options are intneded to be set in code by specific scripts.
    defaults = {
        'world_size': 1,
        'rank': -1,
        'persist_state': 0,
        'lazy': False,
        'transpose': False,
        'data_set_type': 'supervised',
        'seq_length': 256,
        'eval_seq_length': 256,
        'samples_per_shard': 100
    }

    return DataConfig(defaults=defaults)
