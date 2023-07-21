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
#
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
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
#
from torchmetrics import Accuracy, MeanMetric
#
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, HfArgumentParser, TextStreamer)

#
from modelscope import (Model, MsDataset, get_logger, read_config,
                        snapshot_download)
from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS
from modelscope.models.nlp.chatglm2 import ChatGLM2Tokenizer
from modelscope.models.nlp.llama2 import Llama2Tokenizer
from modelscope.swift import LoRAConfig, Swift
from modelscope.trainers import EpochBasedTrainer
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.registry import default_group

#
COLOR, COLOR_S = '#FFE2D9', '#FF7043'

PROMPT = """Here's a conversation between a human and an AI assistant. \
The AI assistant provides detailed, friendly answers for the human.

### Human:
{instruction}

### AI:
"""

logger = get_logger()
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
#


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
    #
    work_dir = os.path.join(work_dir, f'v{version}-{time}')
    logger.info(f'work_dir: {work_dir}')
    return work_dir


def _format_device(device: Union[List[int], str]) -> Tuple[List[int], str]:
    if isinstance(device, list):
        device_ids = device
        device_str = ','.join([str(d) for d in device])
    else:
        device_ids = [int(d) for d in device.split(',') if d != '-1']
        device_str = device
    device_str = device_str.replace(' ', '')
    return device_ids, device_str


def select_device(device: Union[List[int], str]) -> Device:
    """Call this function before cuda is initialized.
    device: e.g. []: 'cpu', [0], [0, 1, 2]
        e.g. '-1': 'cpu', '0', '0,1,2'
    """
    if torch.cuda.is_initialized():
        logger.warning('CUDA has been initialized! Device selection fails!')
        return torch.device('cuda:0')
    #
    device_ids, device_str = _format_device(device)
    #
    os.environ['CUDA_VISIBLE_DEVICES'] = device_str
    log_s = 'Using device: '
    if len(device_ids) == 0:
        master_device: str = 'cpu'
        log_s += 'cpu'
    else:
        assert torch.cuda.is_available(
        ) and torch.cuda.device_count() >= len(device_ids)
        master_device = 'cuda:0'
        log_s += f'cuda:{device_str}'
    logger.info(log_s)
    return torch.device(master_device)


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
                      max_length: Optional[int] = 2048) -> Dict[str, Any]:
    instruction: str = example['instruction']
    input_ = example['input']
    if input_ is not None and input_ != '':
        # instruction = instruction + '\n'
        if input_.startswith('输入：'):
            instruction = instruction + input_[3:]
        else:
            instruction = instruction + input_
    output = example['output']
    src_text = PROMPT.format(instruction=instruction)
    src_input_ids: List[int] = tokenizer(
        src_text, return_attention_mask=False,
        add_special_tokens=True)['input_ids']
    #
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
    #
    if max_length is not None:
        input_ids = input_ids[-max_length:]
        if labels is not None:
            labels = labels[-max_length:]
    #
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
    #
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
    #
    n_params = sum(p.numel() for p in model.parameters())
    n_grads = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buffers = sum(p.numel() for p in model.buffers())
    #
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
        self.loss.update(loss)
        #
        labels: Tensor = inputs['labels']
        labels = labels[:, 1:]
        labels_mask = labels != -100
        logits: Tensor = outputs.logits[:, :-1]
        logits = logits[labels_mask].contiguous().view(-1, logits.shape[-1])
        pred = logits.argmax(dim=-1)
        labels = labels[labels_mask].to(logits.device)
        self.acc.update(pred, labels)

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
                                 add_special_token: bool = True):
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    model_config.torch_dtype = torch.float16
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            device_map='auto',
            torch_dtype=torch.float16,
            trust_remote_code=True)
    #
    if add_special_token:
        _add_special_token(tokenizer)
    return model, tokenizer


def get_chatglm2_model_tokenizer(model_dir: str,
                                 load_model: bool = True,
                                 add_special_token: bool = True):
    config = read_config(model_dir)
    tokenizer = ChatGLM2Tokenizer.from_pretrained(model_dir)
    model = None
    if load_model:
        model = Model.from_pretrained(
            model_dir,
            cfg_dict=config,
            device_map='auto',
            torch_dtype=torch.float16)
    if add_special_token:
        _add_special_token(tokenizer)
    return model, tokenizer


def get_llama2_model_tokenizer(model_dir: str,
                               load_model: bool = True,
                               add_special_token: bool = True):
    config = read_config(model_dir)
    tokenizer = Llama2Tokenizer.from_pretrained(model_dir)
    model = None
    if load_model:
        model = Model.from_pretrained(
            model_dir,
            cfg_dict=config,
            device_map='auto',
            torch_dtype=torch.float16)
    if add_special_token:
        _add_special_token(tokenizer)
    return model, tokenizer


def get_model_tokenizer(model_type: str):
    # ### Loading Model and Tokenizer
    if model_type == 'baichuan-7b':
        model_dir = snapshot_download('baichuan-inc/baichuan-7B', 'v1.0.7')
        model, tokenizer = get_baichuan_model_tokenizer(model_dir)
    elif model_type == 'baichuan-13b':
        model_dir = snapshot_download('baichuan-inc/Baichuan-13B-Base',
                                      'v1.0.3')
        model, tokenizer = get_baichuan_model_tokenizer(model_dir)
    elif model_type == 'chatglm2':
        model_dir = snapshot_download('ZhipuAI/chatglm2-6b', 'v1.0.6')
        model, tokenizer = get_chatglm2_model_tokenizer(model_dir)
    elif model_type == 'llama2-7b':
        model_dir = snapshot_download('modelscope/Llama-2-7b-ms', 'v1.0.2')
        model, tokenizer = get_llama2_model_tokenizer(model_dir)
    else:
        raise ValueError(f'model_type: {model_type}')
    return model, tokenizer


def get_alpaca_en_zh_dataset(
        tokenize_function,
        only_val: bool = False,
        test_split_p: float = 0.01,
        split_seed: int = 42,
        data_sample: Optional[int] = None) -> Tuple[HfDataset, HfDataset]:
    dataset_en: HfDataset = MsDataset.load(
        'AI-ModelScope/alpaca-gpt4-data-en', split='train').to_hf_dataset()
    dataset_zh: HfDataset = MsDataset.load(
        'AI-ModelScope/alpaca-gpt4-data-zh', split='train').to_hf_dataset()
    dataset_en = dataset_en.remove_columns(['text'])
    dataset: HfDataset = concatenate_datasets([dataset_zh, dataset_en])
    #
    if data_sample is not None:
        dataset = dataset.select(range(data_sample))
    dataset = dataset.train_test_split(test_split_p, seed=split_seed)
    if only_val:
        dataset = dataset['test']
    if tokenize_function is not None:
        dataset = dataset.map(tokenize_function)
        dataset = dataset.remove_columns(['instruction', 'input', 'output'])
    #
    if only_val:
        return None, dataset
    else:
        return dataset['train'], dataset['test']


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
        #
        norm_factor *= smooth
        norm_factor += 1
    return res


def plot_image(tb_dir: str,
               smooth_key: List[str],
               smooth_val: float = 0.9,
               figsize: Tuple[int, int] = (8, 5),
               dpi: int = 100) -> None:
    image_dir = os.path.join(os.path.dirname(tb_dir), 'images')
    os.makedirs(image_dir, exist_ok=True)
    #
    fname = os.listdir(tb_dir)[0]
    tb_path = os.path.join(tb_dir, fname)
    data = read_tensorboard_file(tb_path)
    #
    for k in data.keys():
        _data = data[k]
        steps = [d['step'] for d in _data]
        values = [d['value'] for d in _data]
        if len(values) == 0:
            continue
        _, ax = plt.subplots(1, 1, squeeze=True, figsize=figsize, dpi=dpi)
        ax.set_title(k)
        if len(values) == 1:
            ax.scatter(steps, values, color=COLOR_S)
        elif k in smooth_key:
            ax.plot(steps, values, color=COLOR)
            values_s = tensorboard_smoothing(values, smooth_val)
            ax.plot(steps, values_s, color=COLOR_S)
        else:
            ax.plot(steps, values, color=COLOR_S)
        fpath = os.path.join(image_dir, k.replace('/', '_'))
        plt.savefig(fpath, dpi=dpi, bbox_inches='tight')
