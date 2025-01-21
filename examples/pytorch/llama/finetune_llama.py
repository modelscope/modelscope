#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import logging
import os
from dataclasses import dataclass, field

import json
import torch
from swift import LoRAConfig, Swift

from modelscope import TrainingArgs, build_dataset_from_file
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.models.nlp.llama import LlamaForTextGeneration, LlamaTokenizer
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.dataset_cls.custom_datasets.torch_custom_dataset import \
    TorchCustomDataset
from modelscope.trainers import build_trainer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_UNK_TOKEN = '<unk>'
PROMPT_DICT = {
    'prompt_input':
    ('Below is an instruction that describes a task, paired with an input that provides further context. '
     'Write a response that appropriately completes the request.\n\n'
     '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
     ),
    'prompt_no_input':
    ('Below is an instruction that describes a task. '
     'Write a response that appropriately completes the request.\n\n'
     '### Instruction:\n{instruction}\n\n### Response:'),
}


@dataclass(init=False)
class TextGenerationArguments(TrainingArgs):
    instruction: str = field(
        default='instruction',
        metadata={
            'help': 'The instruction text key of dataset',
        })

    input: str = field(
        default='input', metadata={
            'help': 'The input text key of dataset',
        })

    output: str = field(
        default='output',
        metadata={
            'help': 'The output text key of dataset',
        })

    src_txt: str = field(
        default=None,
        metadata={
            'help': 'The source text key of preprocessor',
            'cfg_node': 'preprocessor.src_txt'
        })

    deepspeed: str = field(
        default=None,
        metadata={
            'help': 'The location of DeepSpeed json config file.',
        })

    use_lora: int = field(
        default=0,
        metadata={'help': 'Whether to use lora to train the model.'},
    )

    lora_rank: int = field(
        default=32,
        metadata={'help': 'The lora rank'},
    )

    lora_alpha: int = field(
        default=32,
        metadata={'help': 'The lora alpha'},
    )

    lora_dropout: float = field(
        default=0.05,
        metadata={'help': 'The lora dropout'},
    )

    device_map: str = field(
        default=None,
        metadata={
            'help': 'A map that specifies where each submodule should go.'
        })

    zero_stage: int = field(
        default=None, metadata={'help': 'The stage of zero_optimization'})


def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors='pt',
            padding='longest',
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
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
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized['input_ids']
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized['input_ids_lens']):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer,
                                         model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class SupervisedDataset(TorchCustomDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, list_data_dict, tokenizer):
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
        if isinstance(i, int):
            return dict(input_ids=self.input_ids[i], labels=self.labels[i])
        elif isinstance(i, slice):
            return SliceSupervisedDataset(self.input_ids, self.labels, i)
        else:
            raise TypeError(f'Unsupported input type: {type(i)}')


class SliceSupervisedDataset(TorchCustomDataset):

    def __init__(self, input_ids, labels, slice_):
        self.input_ids = input_ids[slice_]
        self.labels = labels[slice_]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: LlamaTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ('input_ids', 'labels'))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


training_args = TextGenerationArguments().parse_cli()
config, args = training_args.to_config()
print(args)

if __name__ == '__main__':

    def cfg_modify_fn(cfg):
        if args.use_model_config:
            cfg.merge_from_dict(config)
        else:
            cfg = config
        cfg.train.lr_scheduler = {
            'type': 'CosineAnnealingLR',
            'T_max': 1,
            'options': {
                'by_epoch': False
            }
        }
        cfg.train.optimizer = {
            'type': 'AdamW',
            'lr': training_args.lr,
            'weight_decay': 0.0,
            'options': {
                'cumulative_iters': 8,
                'warmup': {
                    'type': 'LinearWarmup',
                    'warmup_ratio': 0.03
                }
            }
        }
        cfg.train.logging = {
            'interval': training_args.logging_interval,
            'by_epoch': False
        }
        cfg.train['bf16'] = True
        cfg.train.dataloader = {
            'batch_size_per_gpu': training_args.per_device_train_batch_size,
            'workers_per_gpu': 1
        }
        if 'hooks' not in cfg.train:
            cfg.train['hooks'] = []
        if args.deepspeed is not None:
            cfg.train.hooks.append({
                'type': 'DeepspeedHook',
                'config': args.deepspeed,
                'save_zero_checkpoint': True,
                'with_mpu': False,
            })
        if args.zero_stage is not None:
            cfg.train.hooks[-1]['zero_stage'] = args.zero_stage

        cfg.preprocessor.sequence_length = 512
        return cfg

    model_path = args.model if os.path.exists(
        args.model) else snapshot_download(args.model)

    dataset_mapping_dict = {
        args.instruction: 'instruction',
        args.input: 'input',
        args.output: 'output'
    }
    if args.dataset_json_file is None:
        if args.train_dataset_name is not None and args.val_dataset_name is not None:
            train_dataset = MsDataset.load(
                args.train_dataset_name,
                subset_name=args.train_subset_name,
                split=args.train_split,
                namespace=args.train_dataset_namespace).remap_columns(
                    dataset_mapping_dict)
            validation_dataset = MsDataset.load(
                args.val_dataset_name,
                subset_name=args.val_subset_name,
                split=args.val_split,
                namespace=args.val_dataset_namespace).remap_columns(
                    dataset_mapping_dict)
        elif args.train_dataset_name is not None and args.val_dataset_name is None:
            ms_dataset = MsDataset.load(
                args.train_dataset_name,
                subset_name=args.train_subset_name,
                split=args.train_split,
                namespace=args.train_dataset_namespace).remap_columns(
                    dataset_mapping_dict).train_test_split(
                        test_size=0.02, seed=args.seed)
            train_dataset = ms_dataset['train']
            validation_dataset = ms_dataset['test']
        else:
            data_path = training_args.src_txt if training_args.src_txt else os.path.join(
                model_path, 'alpaca_data.json')
            ms_dataset = MsDataset.load(
                'json', data_files=data_path).remap_columns(
                    dataset_mapping_dict).train_test_split(
                        test_size=0.02, seed=args.seed)
            train_dataset = ms_dataset['train']
            validation_dataset = ms_dataset['test']
    else:
        train_dataset, validation_dataset = build_dataset_from_file(
            args.dataset_json_file)

    model = LlamaForTextGeneration.from_pretrained(
        model_path, device_map=args.device_map)

    if args.use_lora != 0:
        lora_config = LoRAConfig(
            target_modules=['q_proj', 'k_proj', 'v_proj'],
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout)
        model = model.bfloat16()
        model = Swift.prepare_model(model, lora_config)

    tokenizer = LlamaTokenizer.from_pretrained(
        model_path,
        model_max_length=512,
        padding_side='right',
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None or tokenizer.pad_token == '':
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None or tokenizer.eos_token == '':
        special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None or tokenizer.bos_token == '':
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None or tokenizer.unk_token == '':
        special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, list_data_dict=train_dataset)
    validation_dataset = SupervisedDataset(
        tokenizer=tokenizer, list_data_dict=validation_dataset)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    kwargs = dict(
        model=model,
        cfg_file=os.path.join(model_path, 'configuration.json'),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        cfg_modify_fn=cfg_modify_fn)

    # Construct trainer and train
    trainer = build_trainer(
        name=Trainers.text_generation_trainer, default_args=kwargs)
    trainer.train()

    # prepare for inference
    if args.deepspeed and args.zero_stage is None and int(
            os.environ.get('LOCAL_RANK', 0)) == 0:
        work_dir = config.train.work_dir
        tokenizer.save_pretrained(os.path.join(work_dir, 'output'))
        os.system(f'rm {work_dir}/output/pytorch_model*')
        os.system(
            f'python3 {work_dir}/zero_to_fp32.py {work_dir} {work_dir}/output/pytorch_model.bin'
        )
        os.system(
            f'cp {model_path}/configuration.json {work_dir}/output/configuration.json'
        )
        with open(f'{model_path}/config.json', 'r') as f:
            config = json.load(f)
            config['vocab_size'] = len(tokenizer)
        with open(f'{work_dir}/output/config.json', 'w') as f:
            json.dump(config, f)
