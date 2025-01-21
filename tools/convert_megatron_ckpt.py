# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import os

from modelscope.models import Model
from modelscope.utils.megatron_utils import convert_megatron_checkpoint


def unwrap_model(model):
    for name in ('model', 'module', 'dist_model'):
        while hasattr(model, name):
            model = getattr(model, name)
    return model


parser = argparse.ArgumentParser(
    description='Split or merge your megatron_based checkpoint.')
parser.add_argument(
    '--model_dir', type=str, required=True, help='Checkpoint to be converted.')
parser.add_argument(
    '--target_dir', type=str, required=True, help='Target save path.')
args = parser.parse_args()

model = Model.from_pretrained(
    args.model_dir,
    rank=int(os.getenv('RANK')),
    megatron_cfg={'tensor_model_parallel_size': int(os.getenv('WORLD_SIZE'))})
unwrapped_model = unwrap_model(model)

convert_megatron_checkpoint(unwrapped_model, model.model_dir, args.target_dir)
