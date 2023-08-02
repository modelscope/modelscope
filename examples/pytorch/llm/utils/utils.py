import datetime as dt
import math
import os
import random
import re
from typing import Any, Counter, Dict, List, Optional, Tuple, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset as HfDataset
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import Accuracy, MeanMetric
from tqdm import tqdm
from transformers import GenerationConfig, HfArgumentParser, TextStreamer

from modelscope import get_logger
from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS
from modelscope.utils.registry import default_group

COLOR, COLOR_S = '#FFE2D9', '#FF7043'

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
    output = example.get('output')
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
    n_mask = Counter(labels)[-100]
    print(f'[LABLES_IDS] {labels}')
    print(f'[LABLES] <-100 * {n_mask}>{tokenizer.decode(labels[n_mask:])}')


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


def show_freeze_layers(model: Module, max_lines: Optional[int] = 20) -> None:
    named_p = list(model.named_parameters())
    for i, (n, p) in enumerate(named_p):
        if max_lines is not None and i >= max_lines:
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
            ax.scatter(steps, values, color=COLOR_S)
        elif k in smooth_key:
            ax.plot(steps, values, color=COLOR)
            values_s = tensorboard_smoothing(values, smooth_val)
            ax.plot(steps, values_s, color=COLOR_S)
        else:
            ax.plot(steps, values, color=COLOR_S)
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


_T = TypeVar('_T')


def parse_args(class_type: Type[_T],
               argv: Optional[List[str]] = None) -> Tuple[_T, List[str]]:
    parser = HfArgumentParser([class_type])
    args, remaining_args = parser.parse_args_into_dataclasses(
        argv, return_remaining_strings=True)
    logger.info(f'args: {args}')
    return args, remaining_args
