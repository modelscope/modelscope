import ast
import datetime as dt
import math
import os
import random
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
#
import torch
from matplotlib.figure import Figure
from swift import LoRAConfig, Swift
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from torch import Tensor
from torch import device as Device
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
#
from torchmetrics import Accuracy, MeanMetric
#
from tqdm import tqdm

#
from modelscope import Model, MsDataset, get_logger, read_config
from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS
from modelscope.models.nlp.chatglm2 import ChatGLM2Tokenizer
from modelscope.msdatasets.dataset_cls.custom_datasets import \
    TorchCustomDataset
from modelscope.utils.config import ConfigDict
from modelscope.utils.registry import default_group

#
PROMPT = """System: {system}
Human: {user}
AI: """
MAX_LENGTH = 2048
TEST_MAX_LENGTH = MAX_LENGTH

COLOR, COLOR_S = '#FFE2D9', '#FF7043'
logger = get_logger()
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


def tokenize_function(system: str, user: str, assistant: Optional[str],
                      tokenizer) -> Dict[str, Any]:
    """Only applicable to baichuan and chatglm2. Other models need to be tested"""
    src_text = PROMPT.format(system=system, user=user)
    src_input_ids: List[int] = tokenizer(
        src_text, return_attention_mask=False,
        add_special_tokens=True)['input_ids']
    #
    tgt_input_ids: List[int] = []
    if assistant is not None:
        tgt_input_ids += tokenizer(
            assistant, return_attention_mask=False,
            add_special_tokens=False)['input_ids']
        tgt_input_ids += [tokenizer.eos_token_id]
        labels = [-100] * len(src_input_ids) + tgt_input_ids
    else:
        labels = None
    input_ids = src_input_ids + tgt_input_ids
    #
    if assistant is not None:
        if len(input_ids) > MAX_LENGTH:
            return {}
    else:
        input_ids = input_ids[-TEST_MAX_LENGTH:]
    #
    return {'input_ids': input_ids, 'labels': labels}


class MyDataset(TorchCustomDataset):

    def __init__(self, system: List[str], user: List[str],
                 assistant: List[str], tokenize_function) -> None:
        self._data = []
        for i in tqdm(range(len(system))):
            _d = tokenize_function(system[i], user[i], assistant[i])
            if len(_d) == 0:
                continue
            self._data.append(_d)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)


def stat_dataset(dataset: 'MyDataset') -> None:
    """Statistical analysis was performed on the data set"""
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


def print_examples(examples: Dict[str, Any], tokenizer) -> None:
    input_ids, labels = examples['input_ids'], examples['labels']
    print(f'[INPUT_IDS] {tokenizer.decode(input_ids)}')
    print()
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
        """This script does not support ddp"""
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


def get_baichuan7B_model_tokenizer(model_dir: str,
                                   load_model: bool = True,
                                   add_special_token: bool = True):
    sys.path.insert(0, model_dir)
    from configuration_baichuan import BaiChuanConfig
    from tokenization_baichuan import BaiChuanTokenizer
    from modeling_baichuan import BaiChuanForCausalLM
    model_config = BaiChuanConfig.from_pretrained(model_dir)
    model_config.torch_dtype = torch.float16
    logger.info(f'model_config: {model_config}')
    tokenizer = BaiChuanTokenizer.from_pretrained(model_dir)
    model = None
    if load_model:
        model = BaiChuanForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            device_map='auto',
            torch_dtype=torch.float16)
    #
    if add_special_token:
        _add_special_token(tokenizer)
    return model, tokenizer


def get_chatglm2_model_tokenizer(model_dir: str,
                                 load_model: bool = True,
                                 add_special_token: bool = True):
    config = read_config(model_dir)
    config['model'] = ConfigDict({'type': 'chatglm2-6b'})
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


def make_dataset(
    split: str, tokenize_function: Callable[[str, str, Optional[str]],
                                            Dict[str, Any]]
) -> MyDataset:
    """
    split: Literal['train', 'validation']
    """
    dataset = MsDataset.load(
        'modelscope/ms_hackathon_23_agent_train_dev', split=split)
    system = []
    user = []
    assistant = []
    for d in dataset:
        content = ast.literal_eval(d['conversations'])
        s = content[0]['value']
        assert len(content) % 2 == 1
        for i in range(len(content) // 2):
            system.append(s)
            user.append(content[2 * i + 1]['value'])
            assistant.append(content[2 * i + 2]['value'])
    return MyDataset(system, user, assistant, tokenize_function)


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


def plot_image(data: Dict[str, List[Item]], key_name: str,
               smooth: float) -> Figure:
    _data = data[key_name]
    steps = [d['step'] for d in _data]
    values = [d['value'] for d in _data]
    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(8, 5), dpi=100)
    ax.set_title(key_name)
    if smooth != 0:
        ax.plot(steps, values, color=COLOR)
        values_s = tensorboard_smoothing(values, smooth)
        ax.plot(steps, values_s, color=COLOR_S)
    else:
        ax.plot(steps, values, color=COLOR_S)
    return fig
