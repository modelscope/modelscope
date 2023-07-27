import os
from dataclasses import dataclass, field

import numpy as np
import torch
from chatglm_trainer import Seq2SeqTrainer
from text_generation_metric import TextGenerationMetric
from transformers import DataCollatorForSeq2Seq

from modelscope import build_dataset_from_file, snapshot_download
from modelscope.metainfo import Models
from modelscope.models import Model
from modelscope.msdatasets import MsDataset
from modelscope.swift import Swift
from modelscope.swift.lora import LoRAConfig
from modelscope.trainers.training_args import TrainingArgs
from modelscope.utils.config import ConfigDict
from modelscope.utils.hub import read_config


@dataclass(init=False)
class Chatglm6bArguments(TrainingArgs):
    ptuning_checkpoint: str = field(
        default=None,
        metadata={
            'help': 'The p-tuning checkpoint previously trained.',
        })

    pre_seq_len: int = field(
        default=None, metadata={
            'help': 'The p-tuning sequence length',
        })

    prefix_projection: bool = field(
        default=False, metadata={
            'help': '',
        })

    quantization_bit: int = field(
        default=None, metadata={
            'help': 'Quantized bit',
        })

    prompt_column: str = field(
        default=None,
        metadata={
            'help':
            'The name of the column in the datasets containing the full texts (for summarization).'
        },
    )

    response_column: str = field(
        default=None,
        metadata={
            'help':
            'The name of the column in the datasets containing the summaries (for summarization).'
        },
    )

    history_column: str = field(
        default=None,
        metadata={
            'help':
            'The name of the column in the datasets containing the history of chat.'
        },
    )

    source_prefix: str = field(
        default='',
        metadata={
            'help':
            'A prefix to add before every source text (useful for T5 models).'
        })

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            'help':
            'Whether to ignore the tokens corresponding to padded labels in the loss computation or not.'
        },
    )

    max_source_length: int = field(
        default=1024,
        metadata={
            'help':
            ('The maximum total input sequence length after tokenization. Sequences longer '
             'than this will be truncated, sequences shorter will be padded.')
        },
    )

    max_target_length: int = field(
        default=128,
        metadata={
            'help':
            ('The maximum total sequence length for target text after tokenization. Sequences longer '
             'than this will be truncated, sequences shorter will be padded.')
        },
    )

    max_train_samples: int = field(
        default=None,
        metadata={
            'help':
            ('For debugging purposes or quicker training, truncate the number of training examples to this '
             'value if set.')
        },
    )

    max_eval_samples: int = field(
        default=None,
        metadata={
            'help':
            ('For debugging purposes or quicker training, truncate the number of evaluation examples to this '
             'value if set.')
        },
    )

    preprocessing_num_workers: int = field(
        default=None,
        metadata={
            'help': 'The number of processes to use for the preprocessing.'
        },
    )

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
        metadata={'help': 'The lora alpha'},
    )

    use_amp: int = field(
        default=0,
        metadata={
            'help':
            'Whether to use amp(automatic mixed precision) to train the model.'
        },
    )


args = Chatglm6bArguments(eval_metrics='chatglm').parse_cli()
print(args)
config, _ = args.to_config(ignore_default_config=args.use_model_config)
config.dump('./configuration.json')

if config['model']['type'] == 'chatglm6b':
    from modelscope.models.nlp import ChatGLMTokenizer
else:
    from modelscope.models.nlp import ChatGLM2Tokenizer as ChatGLMTokenizer


def cfg_modify_fn(cfg):
    if args.use_model_config:
        cfg.merge_from_dict(config)
    else:
        cfg = config
    if args.use_amp:
        if not getattr(cfg.train, 'hooks', None):
            cfg.train.hooks = []
        cfg.train.hooks.append({
            'type': 'TorchAMPOptimizerHook',
            # Optional loss_scale parameter here.
        })
    if cfg.train.lr_scheduler.type == 'LinearLR':
        cfg.train.lr_scheduler['total_iters'] = \
            int(len(train_dataset) / cfg.train.dataloader.batch_size_per_gpu) * cfg.train.max_epochs
    cfg['gen_kwargs'] = {
        'do_sample': True,
        'top_p': 0.7,
        'max_length': 512,
        'temperature': 0.95
    }
    return cfg


if args.dataset_json_file is None:
    train_dataset = MsDataset.load(
        args.train_dataset_name,
        subset_name=args.train_subset_name,
        split=args.train_split,
        namespace=args.train_dataset_namespace)
    validation_dataset = MsDataset.load(
        args.val_dataset_name,
        subset_name=args.val_subset_name,
        split=args.val_split,
        namespace=args.val_dataset_namespace)
else:
    train_dataset, validation_dataset = build_dataset_from_file(
        args.dataset_json_file)

model_dir = snapshot_download(args.model)
model_config = read_config(model_dir)
model_config['model'] = ConfigDict({
    'type': config['model']['type'],
})

model_config['model']['pre_seq_len'] = args.pre_seq_len
model_config['model']['prefix_projection'] = args.prefix_projection
tokenizer = ChatGLMTokenizer.from_pretrained(model_dir, trust_remote_code=True)

device_map_kwargs = {}
if args.use_lora != 0 and torch.cuda.device_count() > 1:
    device_map_kwargs['device_map'] = 'auto'
model = Model.from_pretrained(
    model_dir, cfg_dict=model_config, **device_map_kwargs)

if args.ptuning_checkpoint is not None:
    # Evaluation
    # Loading extra state dict of prefix encoder

    prefix_state_dict = torch.load(
        os.path.join(args.ptuning_checkpoint, 'pytorch_model.bin'))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith('transformer.prefix_encoder.'):
            new_prefix_state_dict[k[len('transformer.prefix_encoder.'):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

if args.quantization_bit is not None:
    print(f'Quantized to {args.quantization_bit} bit')
    model = model.quantize(args.quantization_bit)
if args.pre_seq_len is not None:
    # P-tuning v2
    model = model.half()
    model.transformer.prefix_encoder.float()
elif not args.use_lora:
    # Finetune
    model = model.float()

if args.use_lora != 0:
    lora_config = LoRAConfig(
        replace_modules=['attention.query_key_value'],
        rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout)
    if args.use_amp:
        model = model.float()
    else:
        model = model.bfloat16()
    Swift.prepare_model(model, lora_config)

prefix = args.source_prefix if args.source_prefix is not None else ''

# Get the column names for input/target.
prompt_column = args.prompt_column
response_column = args.response_column
history_column = args.history_column

# Temporarily set max_target_length for training.
max_target_length = args.max_target_length

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
trainable_params = sum([np.prod(p.size()) for p in model_parameters])

model_parameters = filter(lambda p: not p.requires_grad, model.parameters())
non_trainable_params = sum([np.prod(p.size()) for p in model_parameters])

print('trainable_params:{} ({:.2f}%), non_trainable_params:{}'.format(
    trainable_params, trainable_params / non_trainable_params * 100,
    non_trainable_params))


def preprocess_function_eval(examples):
    inputs, targets = [], []
    for i in range(len(examples[prompt_column])):
        if examples[prompt_column][i] and examples[response_column][i]:
            query = examples[prompt_column][i]
            if history_column is None or len(examples[history_column][i]) == 0:
                prompt = query
            else:
                prompt = ''
                history = examples[history_column][i]
                for turn_idx, (old_query, response) in enumerate(history):
                    prompt += '[Round {}]\n问：{}\n答：{}\n'.format(
                        turn_idx, old_query, response)
                prompt += '[Round {}]\n问：{}\n答：'.format(len(history), query)
            inputs.append(prompt)
            targets.append(examples[response_column][i])

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(
        inputs,
        max_length=args.max_source_length,
        truncation=True,
        padding=True)
    labels = tokenizer(
        text_target=targets, max_length=max_target_length, truncation=True)

    if args.ignore_pad_token_for_loss:
        labels['input_ids'] = [[(lb if lb != tokenizer.pad_token_id else -100)
                                for lb in label]
                               for label in labels['input_ids']]
    model_inputs['labels'] = labels['input_ids']

    return model_inputs


def preprocess_function_train(examples):
    max_seq_length = args.max_source_length + args.max_target_length

    model_inputs = {
        'input_ids': [],
        'labels': [],
    }
    for i in range(len(examples[prompt_column])):
        if examples[prompt_column][i] and examples[response_column][i]:
            query, answer = examples[prompt_column][i], examples[
                response_column][i]

            if history_column is None:
                prompt = query
            else:
                prompt = ''
                history = examples[history_column][i]
                for turn_idx, (old_query, response) in enumerate(history):
                    prompt += '[Round {}]\n问：{}\n答：{}\n'.format(
                        turn_idx, old_query, response)
                prompt += '[Round {}]\n问：{}\n答：'.format(len(history), query)

            prompt = prefix + prompt
            a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            if len(a_ids) > args.max_source_length - 1:
                a_ids = a_ids[:args.max_source_length - 1]

            if len(b_ids) > args.max_target_length - 2:
                b_ids = b_ids[:args.max_target_length - 2]

            input_ids = tokenizer.build_inputs_with_special_tokens(
                a_ids, b_ids)

            if config['model']['type'] == 'chatglm6b':
                context_length = input_ids.index(tokenizer.bos_token_id)
            else:
                context_length = len(a_ids) + 2
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position + 1:]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            if args.ignore_pad_token_for_loss:
                labels = [(lb if lb != tokenizer.pad_token_id else -100)
                          for lb in labels]

            model_inputs['input_ids'].append(input_ids)
            model_inputs['labels'].append(labels)

    return model_inputs


train_dataset = train_dataset.to_hf_dataset().map(
    preprocess_function_train,
    batched=True,
    num_proc=args.preprocessing_num_workers,
    desc='Running tokenizer on train dataset',
)

validation_dataset = validation_dataset.to_hf_dataset().map(
    preprocess_function_eval,
    batched=True,
    num_proc=args.preprocessing_num_workers,
    desc='Running tokenizer on eval dataset',
)

# Data collator
label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=None,
    padding=False)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# import torch
# model = torch.nn.DataParallel(model).cuda()
trainer = Seq2SeqTrainer(
    model=model,
    cfg_file='./configuration.json',
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    seed=args.seed,
    data_collator=data_collator,
    remove_unused_data=True,
    cfg_modify_fn=cfg_modify_fn)
trainer.tokenizer = tokenizer
trainer.train()
