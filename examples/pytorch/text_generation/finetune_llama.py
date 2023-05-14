#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import utils
import copy
import logging
import shutil
import torch
import tempfile
import unittest
from dataclasses import dataclass

from modelscope.metainfo import Trainers
from modelscope.models.nlp.llama import (LlamaForTextGeneration,
                                         LlamaTokenizerFast)
from modelscope.msdatasets.dataset_cls.custom_datasets.torch_custom_dataset import \
    TorchCustomDataset
from modelscope.trainers import build_trainer
from modelscope.utils.test_utils import DistributedTestCase, test_level

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(sources, targets, tokenizer):
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)



def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



class SupervisedDataset(TorchCustomDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer):
        logging.warning('Loading data...')
        list_data_dict = utils.jload(data_path)

        logging.warning('Formatting inputs...')
        prompt_input, prompt_no_input = PROMPT_DICT[
            'prompt_input'], PROMPT_DICT['prompt_no_input']
        sources = [
            prompt_input.format_map(example) if example.get('input', '') != ''
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}"
            for example in list_data_dict
        ]

        logging.warning('Tokenizing inputs... This may take some time...')
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict['input_ids']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: LlamaTokenizerFast 

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )




if __name__ == '__main__':

    def cfg_modify_fn(cfg):
        cfg.train.lr_scheduler = {
            'type': 'CosineAnnealingLR',
            'T_max': 1,
            'options': {
                'by_epoch': False
            }
        }
        cfg.train.optimizer = {
            'type': 'AdamW',
            'lr': 2e-5,
            'weight_decay': 0.0,
            'options': {
                'warmup': {
                    'type': 'LinearWarmup',
                    'warmup_ratio': 0.03
                }
            }
        }
        cfg.train['bf16'] = True
        cfg.train['gradient_accumulation_steps'] = 8
        cfg.train['checkpoint']: {'interval': 1, 'by_epoch': False}
        cfg.train.dataloader = {'batch_size_per_gpu': 4, 'workers_per_gpu': 2}
        cfg.train.hooks.append({
            'type': 'DeepspeedHook',
            'config':
            '/root/work/stanford_alpaca/configs/default_offload_opt_param.json',
            'save_zero_checkpoint': True,
            'with_mpu': False,
        })

        cfg.preprocessor.sequence_length = 512
        return cfg

    model_name_or_path = '/run/model/llama-7b'
    model = LlamaForTextGeneration.from_pretrained(
        model_name_or_path,
        cache_dir='/run/model/ms_out',
    )

    tokenizer = LlamaTokenizerFast.from_pretrained(
        model_name_or_path,
        cache_dir='/run/model/ms_out',
        model_max_length=512,
        padding_side='right',
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path='/root/work/stanford_alpaca/alpaca_data.json')
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    kwargs = dict(
        model=model,
        cfg_file=os.path.join(model_name_or_path, 'configuration.json'),
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        max_epochs=1,
        launcher='pytorch',
        work_dir='/run/model/ms_out',
        cfg_modify_fn=cfg_modify_fn)

    # Construct trainer and train
    trainer = build_trainer(
        name=Trainers.text_generation_trainer, default_args=kwargs)
    trainer.train()
