import datetime as dt
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from functools import partial
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from _parser import *
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from numpy.random import RandomState
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.optim import lr_scheduler as lrs
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import Dataset
from torchmetrics import Accuracy, MeanMetric
from tqdm import tqdm
from transformers import GenerationConfig, HfArgumentParser, TextStreamer

from modelscope import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, Model,
                        MsDataset, get_logger, read_config, snapshot_download)
from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS
from modelscope.models.nlp.chatglm2 import ChatGLM2Config, ChatGLM2Tokenizer
from modelscope.models.nlp.llama2 import Llama2Config, Llama2Tokenizer
from modelscope.swift import LoRAConfig, Swift
from modelscope.trainers import EpochBasedTrainer
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.registry import default_group

_COLOR, _COLOR_S = '#FFE2D9', '#FF7043'

DEFAULT_PROMPT = """Here's a conversation between a human and an AI assistant. \
The AI assistant provides detailed, friendly answers for the human.

### Human:
{instruction}

### AI:
"""

logger = get_logger()
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def _get_version(work_dir: str) -> int:
    if os.path.isdir(work_dir):
        fnames = os.listdir(work_dir)
    else:
        fnames = []
    v_list = [-1]
    for fname in fnames:
        m = re.match(r'v(\d+)', fname)
        if m is None:
            continue
        v = m.group(1)
        v_list.append(int(v))
    return max(v_list) + 1


def get_work_dir(work_dir: str) -> str:
    """add version"""
    work_dir = os.path.abspath(work_dir)
    version = _get_version(work_dir)
    time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')

    work_dir = os.path.join(work_dir, f'v{version}-{time}')
    logger.info(f'work_dir: {work_dir}')
    return work_dir


def seed_everything(seed: Optional[int] = None, gpu_dtm: bool = False) -> int:
    if seed is None:
        seed_max = np.iinfo(np.int32).max
        seed = random.randint(0, seed_max)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f'Global seed set to {seed}')
    if gpu_dtm:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f'Setting deterministic: {True}, benchmark: {False}')
    return seed


def get_T_max(dataset_len: int, batch_size: int, max_epochs: int,
              drop_last: bool) -> int:
    """Calculate T_max in CosineAnnealingLR"""
    if drop_last:
        T_max = dataset_len // batch_size
    else:
        T_max = math.ceil(dataset_len / batch_size)
    T_max *= max_epochs
    return T_max


def tokenize_function(example: Dict[str, Optional[str]],
                      tokenizer,
                      prompt: str = DEFAULT_PROMPT,
                      max_length: Optional[int] = 2048) -> Dict[str, Any]:
    instruction: str = example['instruction']
    output = example['output']
    src_text = prompt.format(instruction=instruction)
    src_input_ids: List[int] = tokenizer(
        src_text, return_attention_mask=False,
        add_special_tokens=True)['input_ids']

    tgt_input_ids = []
    if output is not None:
        tgt_input_ids += tokenizer(
            output, return_attention_mask=False,
            add_special_tokens=False)['input_ids']
        tgt_input_ids += [tokenizer.eos_token_id]
        labels = [-100] * len(src_input_ids) + tgt_input_ids
    else:
        labels = None
    input_ids = src_input_ids + tgt_input_ids

    if max_length is not None:
        input_ids = input_ids[-max_length:]
        if labels is not None:
            labels = labels[-max_length:]

    return {'input_ids': input_ids, 'labels': labels}


def stat_dataset(dataset: HfDataset) -> None:
    """Statistical analysis was performed on the dataset"""
    _token_len = []
    for d in dataset:
        _token_len.append(len(d['input_ids']))
    _token_len = np.array(_token_len)
    mean = _token_len.mean().item()
    std = _token_len.std().item()
    min_ = _token_len.min().item()
    max_ = _token_len.max().item()
    logger.info(
        f'Dataset Token Length: {mean:.6f}±{std:.6f}, min={min_:.6f}, max={max_:.6f}, size={_token_len.shape[0]}'
    )


def print_example(example: Dict[str, Any], tokenizer) -> None:
    input_ids, labels = example['input_ids'], example['labels']
    print(f'[INPUT_IDS] {input_ids}')
    print(f'[INPUT] {tokenizer.decode(input_ids)}')
    print()
    print(f'[LABLES_IDS] {labels}')
    print(
        f'[LABLES] {tokenizer.decode([lb if lb != -100 else 0 for lb in labels])}'
    )


def data_collate_fn(batch: List[Dict[str, Any]], tokenizer) -> Dict[str, Any]:
    input_ids = [torch.tensor(b['input_ids']) for b in batch]
    labels = [torch.tensor(b['labels']) for b in batch]
    attention_mask = [
        torch.ones(len(input_ids[i]), dtype=torch.int64)
        for i in range(len(input_ids))
    ]

    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(
        attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def print_model_info(model: Module, name: Optional[str] = None) -> None:
    if name is None:
        name = model.__class__.__name__

    n_params = sum(p.numel() for p in model.parameters())
    n_grads = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buffers = sum(p.numel() for p in model.buffers())

    n_params /= 1e6
    n_grads /= 1e6
    n_buffers /= 1e6
    s = [
        f'{name}: ',
        f'{n_params:.4f}M Params ({n_grads:.4f}M Trainable), ',
        f'{n_buffers:.4f}M Buffers',
    ]
    s += '.'
    logger.info(''.join(s))


def show_freeze_layers(model: Module, max_lines: int = 20) -> None:
    named_p = list(model.named_parameters())
    for i, (n, p) in enumerate(named_p):
        if i >= max_lines:
            logger.info('...')
            break
        logger.info(f'{n}: requires_grad={p.requires_grad}')


@METRICS.register_module(group_key=default_group, module_name='my_metric')
class MyMetric(Metric):

    def __init__(self, vocab_size: int):
        self.acc = Accuracy('multiclass', num_classes=vocab_size)
        self.loss = MeanMetric()

    def add(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        loss: Tensor = outputs.loss
        self.loss.update(loss.cpu())

        labels: Tensor = inputs['labels']
        labels = labels[:, 1:]
        labels_mask = labels != -100
        logits: Tensor = outputs.logits[:, :-1]
        logits = logits[labels_mask].contiguous().view(-1, logits.shape[-1])
        pred = logits.argmax(dim=-1)
        labels = labels[labels_mask].to(logits.device)
        self.acc.update(pred.cpu(), labels.cpu())

    def evaluate(self):
        return {
            'acc': self.acc.compute().item(),
            'loss': self.loss.compute().item()
        }

    def merge(self, other: 'MyMetric') -> None:
        """This script does not support ddp. TODO"""
        raise NotImplementedError


def _add_special_token(tokenizer):
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = 2
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = 1
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    logger.info(f'bos_token_id: {tokenizer.bos_token_id}, '
                f'eos_token_id: {tokenizer.eos_token_id}, '
                f'pad_token_id: {tokenizer.pad_token_id}')


def get_baichuan_model_tokenizer(model_dir: str,
                                 load_model: bool = True,
                                 add_special_token: bool = True,
                                 torch_dtype: Dtype = torch.float16):
    """load from an independent repository"""
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    model_config.torch_dtype = torch_dtype
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            device_map='auto',
            torch_dtype=torch_dtype,
            trust_remote_code=True)

    if add_special_token:
        _add_special_token(tokenizer)
    return model, tokenizer


def get_chatglm2_model_tokenizer(model_dir: str,
                                 load_model: bool = True,
                                 add_special_token: bool = True,
                                 torch_dtype: Dtype = torch.float16):
    """load from ms library"""
    config = read_config(model_dir)
    logger.info(config)
    model_config = ChatGLM2Config.from_pretrained(model_dir)
    model_config.torch_dtype = torch_dtype
    logger.info(model_config)
    tokenizer = ChatGLM2Tokenizer.from_pretrained(model_dir)
    model = None
    if load_model:
        model = Model.from_pretrained(
            model_dir,
            cfg_dict=config,
            config=model_config,
            device_map='auto',
            torch_dtype=torch_dtype)
    if add_special_token:
        _add_special_token(tokenizer)
    return model, tokenizer


def get_llama2_model_tokenizer(model_dir: str,
                               load_model: bool = True,
                               add_special_token: bool = True,
                               torch_dtype: Dtype = torch.float16):
    config = read_config(model_dir)
    logger.info(config)
    model_config = Llama2Config.from_pretrained(model_dir)
    model_config.torch_dtype = torch_dtype
    logger.info(model_config)
    tokenizer = Llama2Tokenizer.from_pretrained(model_dir)
    model = None
    if load_model:
        model = Model.from_pretrained(
            model_dir,
            cfg_dict=config,
            config=model_config,
            device_map='auto',
            torch_dtype=torch_dtype)
    if add_special_token:
        _add_special_token(tokenizer)
    return model, tokenizer


MODEL_MAPPER = {
    'baichuan-7b': {
        'model_id': 'baichuan-inc/baichuan-7B',
        'revision': 'v1.0.7',
        'get_function': get_baichuan_model_tokenizer,
    },
    'baichuan-13b': {
        'model_id': 'baichuan-inc/Baichuan-13B-Base',
        'revision': 'v1.0.3',
        'get_function': get_baichuan_model_tokenizer,
    },
    'chatglm2': {
        'model_id': 'ZhipuAI/chatglm2-6b',
        'revision': 'v1.0.6',
        'get_function': get_chatglm2_model_tokenizer,
    },
    'llama2-7b': {
        'model_id': 'modelscope/Llama-2-7b-ms',
        'revision': 'v1.0.2',
        'ignore_file_pattern': [r'.+\.bin$'],
        'get_function': get_llama2_model_tokenizer,
    },
}


def get_model_tokenizer(model_type: str,
                        load_model: bool = True,
                        add_special_token: bool = True,
                        torch_dtype: Dtype = torch.float16):
    data = MODEL_MAPPER.get(model_type)
    if data is None:
        raise ValueError(f'model_type: {model_type}')
    model_id = data['model_id']
    revision = data['revision']
    get_function = data['get_function']
    ignore_file_pattern = data.get('ignore_file_pattern', [])
    model_dir = snapshot_download(
        model_id, revision, ignore_file_pattern=ignore_file_pattern)
    model, tokenizer = get_function(model_dir, load_model, add_special_token,
                                    torch_dtype)
    return model, tokenizer, model_dir


def _processing_alpaca(dataset: HfDataset) -> HfDataset:
    instruction = dataset['instruction']
    input_ = dataset['input']
    res = []
    for inst, inp in zip(instruction, input_):
        if inp is not None and inp != '':
            if inp.startswith('输入：'):
                inp = inp[3:]
            inst = f'{inst}\n{inp}'
        res.append(inst)
    dataset = HfDataset.from_dict({
        'instruction': res,
        'output': dataset['output']
    })
    return dataset


def get_alpaca_en_dataset() -> HfDataset:
    dataset_en: HfDataset = MsDataset.load(
        'AI-ModelScope/alpaca-gpt4-data-en', split='train').to_hf_dataset()
    dataset_en = dataset_en.remove_columns(['text'])
    return _processing_alpaca(dataset_en)


def get_alpaca_zh_dataset() -> HfDataset:
    dataset_zh: HfDataset = MsDataset.load(
        'AI-ModelScope/alpaca-gpt4-data-zh', split='train').to_hf_dataset()
    return _processing_alpaca(dataset_zh)


def process_dataset(dataset: HfDataset, dataset_test_size: float,
                    dataset_sample: Optional[int],
                    dataset_seed: int) -> Tuple[HfDataset, HfDataset]:
    random_state = np.random.RandomState(dataset_seed)
    if dataset_sample is not None:
        index = random_state.permutation(len(dataset))[:dataset_sample]
        dataset = dataset.select(index)
    dataset = dataset.train_test_split(
        dataset_test_size, seed=get_seed(random_state))
    return dataset['train'], dataset['test']


DATASET_MAPPER = {
    'alpaca-en': get_alpaca_en_dataset,
    'alpaca-zh': get_alpaca_zh_dataset,
}


def get_dataset(dataset_names: str) -> HfDataset:
    dataset_name_list = dataset_names.split(',')
    dataset_list = []
    for dataset_name in dataset_name_list:
        get_function = DATASET_MAPPER[dataset_name]
        dataset_list.append(get_function())
    dataset = concatenate_datasets(dataset_list)
    return dataset


def get_seed(random_state: RandomState) -> int:
    seed_max = np.iinfo(np.int32).max
    seed = random_state.randint(0, seed_max)
    return seed


Item = Dict[str, float]


def read_tensorboard_file(fpath: str) -> Dict[str, List[Item]]:
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f'fpath: {fpath}')
    ea = EventAccumulator(fpath)
    ea.Reload()
    res = {}
    tags = ea.Tags()['scalars']
    for tag in tags:
        values = ea.Scalars(tag)
        r = []
        for v in values:
            r.append({'step': v.step, 'value': v.value})
        res[tag] = r
    return res


def tensorboard_smoothing(values: List[float],
                          smooth: float = 0.9) -> List[float]:
    norm_factor = 1
    x = 0
    res = []
    for i in range(len(values)):
        x = x * smooth + values[i]  # Exponential decay
        res.append(x / norm_factor)

        norm_factor *= smooth
        norm_factor += 1
    return res


def plot_images(tb_dir: str,
                smooth_key: List[str],
                smooth_val: float = 0.9,
                figsize: Tuple[int, int] = (8, 5),
                dpi: int = 100) -> None:
    images_dir = os.path.join(os.path.dirname(tb_dir), 'images')
    os.makedirs(images_dir, exist_ok=True)

    fname = os.listdir(tb_dir)[0]
    tb_path = os.path.join(tb_dir, fname)
    data = read_tensorboard_file(tb_path)

    for k in data.keys():
        _data = data[k]
        steps = [d['step'] for d in _data]
        values = [d['value'] for d in _data]
        if len(values) == 0:
            continue
        _, ax = plt.subplots(1, 1, squeeze=True, figsize=figsize, dpi=dpi)
        ax.set_title(k)
        if len(values) == 1:
            ax.scatter(steps, values, color=_COLOR_S)
        elif k in smooth_key:
            ax.plot(steps, values, color=_COLOR)
            values_s = tensorboard_smoothing(values, smooth_val)
            ax.plot(steps, values_s, color=_COLOR_S)
        else:
            ax.plot(steps, values, color=_COLOR_S)
        fpath = os.path.join(images_dir, k.replace('/', '_'))
        plt.savefig(fpath, dpi=dpi, bbox_inches='tight')


def inference(input_ids: List[int],
              model,
              tokenizer,
              streamer: Optional[TextStreamer] = None,
              generation_config: Optional[GenerationConfig] = None,
              tag: str = '[INFERENCE]') -> str:
    print(f'{tag}{tokenizer.decode(input_ids)}', end='')
    input_ids = torch.tensor(input_ids)[None].cuda()
    attention_mask = torch.ones_like(input_ids)
    generate_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        generation_config=generation_config)
    output_text = tokenizer.decode(generate_ids[0])
    return output_text
