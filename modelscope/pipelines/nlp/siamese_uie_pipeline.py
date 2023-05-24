# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os
import pathlib
from copy import deepcopy
from math import ceil
from time import time
from typing import Any, Dict, Generator, List, Mapping, Optional, Union

import json
import torch
from scipy.special import softmax
from torch.cuda.amp import autocast
from tqdm import tqdm

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor, SiameseUiePreprocessor
from modelscope.utils.constant import ModelFile, Tasks

Input = Union[str, tuple, MsDataset, 'Image.Image', 'numpy.ndarray']

logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

__all__ = ['SiameseUiePipeline']


@PIPELINES.register_module(
    Tasks.siamese_uie, module_name=Pipelines.siamese_uie)
class SiameseUiePipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 config_file: str = None,
                 device: str = 'cpu',
                 auto_collate=True,
                 **kwargs):
        """Use `model` and `preprocessor` to create a generation pipeline for prediction.

        Args:
            model (str or Model): Supply either a local model dir which supported the text generation task,
            or a model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.

        Examples:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline(Tasks.siamese_uie,
            >>>    model='damo/nlp_structbert_siamese-uie_chinese-base')
            >>> sentence = '1944年毕业于北大的名古屋铁道会长谷口清太郎等人在日本积极筹资，共筹款2.7亿日元，参加捐款的日本企业有69家。'
            >>> print(pipeline_ins(sentence, schema={'人物': None, '地理位置': None, '组织机构': None}))

            To view other examples plese check tests/pipelines/test_siamese_uie.py.
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate,
            compile=kwargs.pop('compile', False),
            compile_options=kwargs.pop('compile_options', {}))

        assert isinstance(self.model, Model), \
            f'please check whether model config exists in {ModelFile.CONFIGURATION}'

        if self.preprocessor is None:
            self.preprocessor = Preprocessor.from_pretrained(
                self.model.model_dir, **kwargs)
        self.model.eval()
        self.slide_len = 352
        self.max_len = 384
        self.hint_max_len = 128
        self.inference_batch_size = 8
        self.threshold = 0.5

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def __call__(self, input: Union[Input, List[Input]], *args,
                 **kwargs) -> Union[Dict[str, Any], Generator]:
        """
        Args:
            input(str): sentence to extract
            schema: (dict or str) schema of uie task
        Default Returns:
            List[List]:  predicted info list i.e.
            [[{'type': '人物', 'span': '谷口清太郎', 'offset': [18, 23]}],
            [{'type': '地理位置', 'span': '日本', 'offset': [26, 28]}],
            [{'type': '地理位置', 'span': '日本', 'offset': [48, 50]}],
            [{'type': '组织机构', 'span': '北大', 'offset': [8, 10]}],
            [{'type': '组织机构', 'span': '名古屋铁道', 'offset': [11, 16]}]]
        """
        # override __call__ because siamese-uie needs to recursively tokenize prompt
        if 'batch_size' in kwargs:
            batch_size = kwargs.pop('batch_size')
            if batch_size and batch_size > 1:
                raise Exception('This pipeline do not support batch inference')
        # place model to cpu or gpu
        if self.model:
            if not self._model_prepare:
                self.prepare_model()

        # sanitize the parameters
        text = input
        schema = kwargs.pop('schema')
        if type(schema) == str:
            schema = json.loads(schema)
        output_all_prefix = kwargs.pop('output_all_prefix', False)
        tokenized_text = self.preprocessor([text])[0]
        pred_info_list = []
        prefix_info = []
        self.forward(text, tokenized_text, prefix_info, schema, pred_info_list,
                     output_all_prefix)
        return {'output': pred_info_list}

    def _pad(self, input_ids, pad_token_id):
        input_ids[-1] += [pad_token_id] * (self.max_len - len(input_ids[-1]))
        return input_ids

    def tokenize_sample(self, text, tokenized_text, hints):
        tokenized_hints = self.preprocessor(
            hints, padding=True, truncation=True, max_length=self.hint_max_len)
        tokenized_data = []
        split_num = ceil(
            (len(tokenized_text) - self.max_len)
            / self.slide_len) + 1 if len(tokenized_text) > self.max_len else 1
        token_ids = [
            tokenized_text.ids[j * self.slide_len:j * self.slide_len
                               + self.max_len] for j in range(split_num)
        ]
        attention_masks = [
            tokenized_text.attention_mask[j * self.slide_len:j * self.slide_len
                                          + self.max_len]
            for j in range(split_num)
        ]
        if split_num > 1:
            token_ids = self._pad(token_ids, 0)
            attention_masks = self._pad(attention_masks, 0)
        token_ids = torch.tensor(
            token_ids, dtype=torch.long, device=self.device)
        attention_masks = torch.tensor(
            attention_masks, dtype=torch.long, device=self.device)
        batch_num = token_ids.size(0) // self.inference_batch_size + 1
        all_token_ids = torch.tensor_split(token_ids, batch_num)
        all_attention_masks = torch.tensor_split(attention_masks, batch_num)
        all_sequence_output = []
        with torch.no_grad():
            with autocast():
                for token_ids, attention_masks in zip(all_token_ids,
                                                      all_attention_masks):
                    sequence_output = self.model.get_plm_sequence_output(
                        token_ids, attention_masks)
                    all_sequence_output.append(sequence_output)
        all_sequence_output = torch.cat(all_sequence_output, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)
        for i in range(len(hints)):
            hint = hints[i]
            tokenized_hint = tokenized_hints[i]
            for j in range(split_num):
                a = j * self.slide_len
                item = {
                    'id': hint + '--' + text,
                    'hint': hint,
                    'text': text,
                    'shift': a,
                    'sequence_output': all_sequence_output[j],
                    'hint_token_ids': tokenized_hint.ids,
                    'attention_masks': all_attention_masks[j],
                    'cross_attention_masks': tokenized_hint.attention_mask
                }
                tokenized_data.append(item)
        # size #hint * #sub_text
        return tokenized_data

    def get_tokenized_data_and_data_loader(self, text, tokenized_text, hints):
        tokenized_data = self.tokenize_sample(text, tokenized_text, hints)
        sequence_output = torch.stack(
            [item['sequence_output'] for item in tokenized_data])
        attention_masks = torch.stack(
            [item['attention_masks'] for item in tokenized_data])
        hint_token_ids = torch.tensor(
            [item['hint_token_ids'] for item in tokenized_data],
            dtype=torch.long,
            device=self.device)
        cross_attention_masks = torch.tensor(
            [item['cross_attention_masks'] for item in tokenized_data],
            dtype=torch.long,
            device=self.device)
        # split to batchs
        batch_num = sequence_output.size(0) // self.inference_batch_size + 1
        sequence_output = torch.tensor_split(sequence_output, batch_num)
        attention_masks = torch.tensor_split(attention_masks, batch_num)
        hint_token_ids = torch.tensor_split(hint_token_ids, batch_num)
        cross_attention_masks = torch.tensor_split(cross_attention_masks,
                                                   batch_num)
        return tokenized_data, (sequence_output, attention_masks,
                                hint_token_ids, cross_attention_masks)

    def get_entities(self, text, offsets, head_probs, tail_probs):
        sample_entities = []
        potential_heads = [
            j for j in range(len(head_probs)) if head_probs[j] > self.threshold
        ]
        for ph in potential_heads:
            for pt in range(ph, len(tail_probs)):
                if tail_probs[pt] > self.threshold:
                    char_head = offsets[ph][0]
                    char_tail = offsets[pt][1]
                    e = {
                        'offset': [char_head, char_tail],
                        'span': text[char_head:char_tail]
                    }
                    sample_entities.append(e)
                    break
        sample_entities = sorted(
            sample_entities, key=lambda x: tuple(x['offset']))
        return sample_entities

    def get_prefix_infos(self, text, tokenized_text, prefix_info,
                         schema_types):
        hints = []
        for st in schema_types:
            hint = ''
            for item in prefix_info:
                hint += f'{item["type"]}: {item["span"]}, '
            hint += f'{st}: '
            # print('hint: ', hint)
            hints.append(hint)
        all_valid_tokenized_data, all_tensor_data = self.get_tokenized_data_and_data_loader(
            text, tokenized_text, hints)
        probs = []
        last_uuid = None
        all_pred_entities = []
        all_head_probs = []
        all_tail_probs = []
        with torch.no_grad():
            with autocast():
                for batch_data in zip(*all_tensor_data):
                    batch_head_probs, batch_tail_probs = self.model.fast_inference(
                        *batch_data)
                    batch_head_probs, batch_tail_probs = batch_head_probs.tolist(
                    ), batch_tail_probs.tolist()  # (b, n, l)
                    all_head_probs += batch_head_probs
                    all_tail_probs += batch_tail_probs
        all_valid_tokenized_data.append({'id': 'WhatADifferentUUiD'
                                         })  # 需要加一个冗余数据，不然最后一个样本无法触发条件
        all_head_probs.append(None)
        all_tail_probs.append(None)
        for tokenized_sample, head_probs, tail_probs in zip(
                all_valid_tokenized_data, all_head_probs, all_tail_probs):
            uuid = tokenized_sample['id']
            prob = {
                'shift': tokenized_sample.get('shift', 0),
                'head': head_probs,  # (n, l)
                'tail': tail_probs
            }
            if last_uuid is not None and uuid != last_uuid:
                len_tokens = len(tokenized_text.offsets)
                head_probs = [-1] * len_tokens  # (n, l)
                tail_probs = [-1] * len_tokens
                for prob_tmp in probs:
                    shift = prob_tmp['shift']
                    head = prob_tmp['head']
                    tail = prob_tmp['tail']
                    len_sub = len(head)
                    for j in range(len_sub):
                        if j + shift < len_tokens:
                            head_probs[j + shift] = head[j] if head_probs[
                                j + shift] == -1 else (head_probs[j + shift]
                                                       + head[j]) / 2
                            tail_probs[j + shift] = tail[j] if tail_probs[
                                j + shift] == -1 else (tail_probs[j + shift]
                                                       + tail[j]) / 2
                # print('head_probs', head_probs)
                # print('head_probs', tail_probs)
                offsets = tokenized_text.offsets
                pred_entities = self.get_entities(text, offsets, head_probs,
                                                  tail_probs)
                all_pred_entities.append(pred_entities)
                probs = []
            probs.append(prob)
            last_uuid = uuid
        next_prefix_infos = []
        for st, pred_entities in zip(schema_types, all_pred_entities):
            for e in pred_entities:
                pi = deepcopy(prefix_info)
                item = {'type': st, 'span': e['span'], 'offset': e['offset']}
                pi.append(item)
                next_prefix_infos.append(pi)
        return next_prefix_infos

    def forward(self, text, tokenized_text, prefix_info, curr_schema_dict,
                pred_info_list, output_all_prefix):
        # print('prefix_info', prefix_info)
        next_prefix_infos = self.get_prefix_infos(text, tokenized_text,
                                                  prefix_info,
                                                  curr_schema_dict)
        # print('next_prefix_infos', next_prefix_infos)
        for prefix_info in next_prefix_infos:
            next_schema_dict = curr_schema_dict[prefix_info[-1]['type']]
            if next_schema_dict is None:
                pred_info_list.append(prefix_info)
            else:
                if output_all_prefix:
                    pred_info_list.append(prefix_info)
                self.forward(text, tokenized_text, prefix_info,
                             next_schema_dict, pred_info_list,
                             output_all_prefix)
