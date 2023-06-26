# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
# Copyright (c) Alibaba, Inc. and its affiliates.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse
from typing import Dict, Optional

import torch
import tqdm
import transformers

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.utils.checkpoint import save_pretrained
from modelscope.utils.logger import get_logger

logger = get_logger()


def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer,
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


def make_same_shape(model_raw: Model, model_convert: Model, tokenizer_raw,
                    tokenizer_convert):
    if model_raw.__class__ != model_convert.__class__:
        logger.error(
            f'weight diff: These two models should be of the same class. model_raw:'
            f'{model_raw.__class__} vs model_convert: {model_convert.__class__}.'
        )

    special_tokens = {}
    for k, v in tokenizer_convert.special_tokens_map_extended.items():
        if k not in tokenizer_raw.special_tokens_map_extended:
            special_tokens[k] = v

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens,
        model=model_raw,
        tokenizer=tokenizer_raw,
    )

    state_dict_tuned = model_convert.state_dict()
    state_dict_raw = model_raw.state_dict()
    for key in tqdm.tqdm(state_dict_tuned):
        if state_dict_tuned[key].shape != state_dict_raw[key].shape:
            logger.error(
                f'weight diff: shape mismatch. {key}, model_raw shape: {state_dict_raw[key].shape}'
                f' vs model_convert shape: {state_dict_tuned[key].shape}.')


def _weight_diff(model_raw,
                 model_convert,
                 tokenizer_raw,
                 tokenizer_convert,
                 path_to_save=None,
                 make_diff_or_recover='diff'):
    make_same_shape(model_raw, model_convert, tokenizer_raw, tokenizer_convert)

    state_dict_raw = model_raw.state_dict()
    state_dict_convert = model_convert.state_dict()
    if make_diff_or_recover == 'diff':
        for key in tqdm.tqdm(state_dict_convert):
            state_dict_convert[key].add_(-state_dict_raw[key])
    elif make_diff_or_recover == 'recover':
        for key in tqdm.tqdm(state_dict_convert):
            state_dict_convert[key].add_(state_dict_raw[key])

    if path_to_save:
        model_convert.save_pretrained(path_to_save, 'pytorch_model.bin')
        tokenizer_convert.save_pretrained(path_to_save)

    return model_convert, tokenizer_convert


@torch.inference_mode()
def weight_diff(path_raw: str,
                path_convert: str,
                path_to_save: str,
                make_diff_or_recover,
                device='cpu'):
    """Make the weight diff.

    This function is given to present full transparency of how the weight diff was created.
    """
    model_raw = Model.from_pretrained(path_raw, device=device)
    model_convert = Model.from_pretrained(path_convert, device=device)

    tokenizer_raw: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_raw)
    tokenizer_convert: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_convert)

    return _weight_diff(
        model_raw,
        model_convert,
        tokenizer_raw,
        tokenizer_convert,
        path_to_save=path_to_save,
        make_diff_or_recover=make_diff_or_recover)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'Make the weight diff between the raw model and tuned model, or recover tuned weights from the '
        'released weight diff.')

    parser.add_argument(
        'make_diff_or_recover',
        choices=['diff', 'recover'],
        help=
        'model selection, make weight diff or recover weights from the weight diff.'
    )
    parser.add_argument(
        'path_raw', type=str, help='path to the raw pretrained model.')
    parser.add_argument(
        'path_convert',
        type=str,
        help=
        'path to the tuned model in mode `diff`, or path to the diff model in mode `recover`.'
    )
    parser.add_argument(
        'path_to_save',
        type=str,
        help='path to save the diff or recover output files.')
    args = parser.parse_args()

    weight_diff(args.path_raw, args.path_convert, args.path_to_save,
                args.make_diff_or_recover)
