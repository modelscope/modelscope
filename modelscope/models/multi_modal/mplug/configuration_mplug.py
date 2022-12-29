# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" MPLUG model configuration """
import os
from typing import Any, Dict, Union

import yaml
from transformers import PretrainedConfig
from transformers.utils import logging

from modelscope.utils.constant import Tasks

logger = logging.get_logger()


class MPlugConfig(PretrainedConfig):

    model_type = 'mplug'

    def __init__(
            self,
            task=Tasks.visual_question_answering,
            bert_config='config_bert.json',
            image_res=504,
            batch_size_train=128,
            vision_width=1024,
            distill=True,
            clip_name='ViT-L-14',  # ViT-B-16 | ViT-L-14
            batch_size_test=64,
            k_test=128,
            alpha=0.4,
            warm_up=True,
            eos='[SEP]',
            optimizer=None,
            schedular=None,
            min_length=1,
            max_length=10,
            beam_size=5,
            add_ocr=False,
            add_object=False,
            text_encoder='bert-base-uncased',
            text_decoder='bert-base-uncased',
            # clip
            clip_embed_dim=768,
            clip_image_resolution=224,
            clip_vision_layers=24,
            clip_vision_width=1024,
            clip_vision_patch_size=14,
            clip_context_length=77,
            clip_vocab_size=49408,
            clip_transformer_width=768,
            clip_transformer_heads=12,
            clip_transformer_layers=12,
            # retrieval
            queue_size=65536,
            embed_dim=256,
            temp=0.07,
            **kwargs):

        super().__init__(**kwargs)
        self.task = task
        self.bert_config = bert_config
        self.image_res = image_res
        self.batch_size_train = batch_size_train
        self.vision_width = vision_width
        self.distill = distill
        self.clip_name = clip_name
        self.batch_size_test = batch_size_test
        self.k_test = k_test
        self.alpha = alpha
        self.warm_up = warm_up
        self.eos = eos
        self.optimizer = optimizer
        self.schedular = schedular
        self.min_length = min_length
        self.max_length = max_length
        self.beam_size = beam_size
        self.add_ocr = add_ocr
        self.add_object = add_object
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        # clip
        self.clip_embed_dim = clip_embed_dim
        self.clip_image_resolution = clip_image_resolution
        self.clip_vision_layers = clip_vision_layers
        self.clip_vision_width = clip_vision_width
        self.clip_vision_patch_size = clip_vision_patch_size
        self.clip_context_length = clip_context_length
        self.clip_vocab_size = clip_vocab_size
        self.clip_transformer_width = clip_transformer_width
        self.clip_transformer_heads = clip_transformer_heads
        self.clip_transformer_layers = clip_transformer_layers
        # retrieval
        self.queue_size = queue_size
        self.embed_dim = embed_dim
        self.temp = temp

    @classmethod
    def from_yaml_file(cls, yaml_file: Union[str,
                                             os.PathLike]) -> Dict[str, Any]:
        with open(yaml_file, 'r', encoding='utf-8') as reader:
            config_dict = yaml.load(reader, Loader=yaml.Loader)
        return cls(**config_dict)


class HiTeAConfig(PretrainedConfig):

    model_type = 'hitea'

    def __init__(
            self,
            task=Tasks.video_question_answering,
            bert_config='config_bert.json',
            image_res=224,
            num_frames=16,
            batch_size_train=32,
            vision_width=768,
            distill=True,
            batch_size_test=64,
            k_test=128,
            alpha=0.4,
            warm_up=True,
            eos='[SEP]',
            optimizer=None,
            schedular=None,
            min_length=1,
            max_length=10,
            beam_size=5,
            text_encoder='bert-base-uncased',
            text_decoder='bert-base-uncased',
            # retrieval
            queue_size=65536,
            embed_dim=256,
            temp=0.07,
            **kwargs):

        super().__init__(**kwargs)
        self.task = task
        self.bert_config = bert_config
        self.image_res = image_res
        self.num_frames = num_frames
        self.batch_size_train = batch_size_train
        self.vision_width = vision_width
        self.distill = distill
        self.batch_size_test = batch_size_test
        self.k_test = k_test
        self.alpha = alpha
        self.warm_up = warm_up
        self.eos = eos
        self.optimizer = optimizer
        self.schedular = schedular
        self.min_length = min_length
        self.max_length = max_length
        self.beam_size = beam_size
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        # retrieval
        self.queue_size = queue_size
        self.embed_dim = embed_dim
        self.temp = temp

    @classmethod
    def from_yaml_file(cls, yaml_file: Union[str,
                                             os.PathLike]) -> Dict[str, Any]:
        with open(yaml_file, 'r', encoding='utf-8') as reader:
            config_dict = yaml.load(reader, Loader=yaml.Loader)
        return cls(**config_dict)
