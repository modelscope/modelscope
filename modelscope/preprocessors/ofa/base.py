# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from os import path as osp

import json
import numpy as np
import torch

from modelscope.models.multi_modal.ofa import OFATokenizer, OFATokenizerZH
from modelscope.utils.trie import Trie
from .utils.random_help import set_torch_seed


class OfaBasePreprocessor:

    def __init__(self, cfg, model_dir):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            cfg(modelscope.utils.config.ConfigDict) : model config
            model_dir (str): model path
        """
        self.cfg = cfg
        self.language = self.cfg.model.get('language', 'en')
        if self.language == 'en':
            tokenizer = OFATokenizer.from_pretrained(model_dir)
        elif self.language in ['zh', 'cn']:
            tokenizer = OFATokenizerZH.from_pretrained(model_dir)
        else:
            raise NotImplementedError
        # there is some diff between here and our ofa code,
        # there will be no need to use param: use_bpe
        tokenizer.add_tokens(['<code_{}>'.format(i) for i in range(8192)])
        tokenizer.add_tokens(['<bin_{}>'.format(i) for i in range(1000)])
        self.tokenizer = tokenizer
        self.bos_item = torch.LongTensor([tokenizer.bos_token_id])
        self.pad_item = torch.LongTensor([tokenizer.pad_token_id])
        self.eos_item = torch.LongTensor([tokenizer.eos_token_id])
        self.tgt_dict = self.src_dict = {
            value: key
            for key, value in tokenizer.get_vocab().items()
        }
        self.max_src_length = cfg.model.get('max_src_length', 256)
        self.max_image_size = cfg.model.get('max_image_size', 512)
        self.language = self.cfg.model.get('language', 'en')
        self.prompt_type = self.cfg.model.get('prompt_type', 'none')
        seed = self.cfg.model.get('seed', 7)
        np.random.seed(seed)
        set_torch_seed(seed)
        imagenet_default_mean_and_std = self.cfg.model.get(
            'imagenet_default_mean_and_std', False)
        if imagenet_default_mean_and_std:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        self.patch_image_size = self.cfg.model.get('patch_image_size', 480)
        self.constraint_trie = None
        self.index2ans = {}
        if self.cfg.model.get('answer2label', False):
            ans2label_file = osp.join(model_dir, self.cfg.model.answer2label)
            ans2label_dict = json.load(open(ans2label_file, 'r'))
            self.constraint_trie = Trie(tokenizer.eos_token_id)
            for i, answer in enumerate(ans2label_dict.keys()):
                answer_item = tokenizer(
                    ' ' + answer,
                    return_tensors='pt',
                    add_special_tokens=False).input_ids.squeeze(0)
                self.constraint_trie.insert([tokenizer.bos_token_id]
                                            + answer_item.tolist()
                                            + [tokenizer.eos_token_id])

    def get_inputs(self, text, add_bos=True, add_eos=True):
        inputs = self.tokenizer(
            text,
            max_length=self.max_src_length,
            add_special_tokens=False,
            return_tensors='pt')['input_ids'].squeeze(0)
        if add_bos:
            inputs = torch.cat([self.bos_item, inputs])
        if add_eos:
            inputs = torch.cat([inputs, self.eos_item])
        return inputs

    @staticmethod
    def pre_caption(caption, max_words=None):
        caption = caption.lower().lstrip(',.!?*#:;~').replace('-', ' ')\
            .replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r'\s{2,}',
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if max_words is not None and len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption

    @staticmethod
    def pre_question(question, max_ques_words):
        question = question.lower().lstrip(',.!?*#:;~').replace('-',
                                                                ' ').replace(
                                                                    '/', ' ')

        question = re.sub(
            r'\s{2,}',
            ' ',
            question,
        )
        question = question.rstrip('\n')
        question = question.strip(' ')

        # truncate question
        question_words = question.split(' ')
        if len(question_words) > max_ques_words:
            question = ' '.join(question_words[:max_ques_words])

        return question
