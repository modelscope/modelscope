import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type, TypeVar, Union

import torch
from torch import device as Device
from transformers import HfArgumentParser

from modelscope import get_logger

logger = get_logger()


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

    device_ids, device_str = _format_device(device)
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


_T = TypeVar('_T')


def parse_args(class_type: Type[_T],
               argv: Optional[List[str]] = None) -> Tuple[_T, List[str]]:
    parser = HfArgumentParser([class_type])
    args, remaining_args = parser.parse_args_into_dataclasses(
        argv, return_remaining_strings=True)
    logger.info(f'args: {args}')
    return args, remaining_args


@dataclass
class DeviceArguments:
    device: str = '0'  # e.g. '-1'; '0'; '0,1'


def parse_device(argv: Optional[List[str]] = None) -> List[str]:
    args, remaining_args = parse_args(DeviceArguments, argv)
    select_device(args.device)
    return remaining_args
