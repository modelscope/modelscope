# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np
import torch

from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor


class OfaSudokuPreprocessor(OfaBasePreprocessor):
    r"""
    OFA preprocessor for sudoku tasks
    """

    def __init__(self,
                 cfg,
                 model_dir,
                 mode=ModeKeys.INFERENCE,
                 *args,
                 **kwargs):
        """preprocess the data

        Args:
            cfg(modelscope.utils.config.ConfigDict) : model config
            model_dir (str): model path,
            mode: preprocessor mode (model mode)
        """
        super(OfaSudokuPreprocessor, self).__init__(cfg, model_dir, mode,
                                                    *args, **kwargs)

        self.instruction_text = self.cfg.model.get('prompt',
                                                   ' solve the sudoku .')
        self.seg_embedding = self.cfg.get('seg_embedding', False)
        self.max_struct_length = self.cfg.get('max_struct_length', 256)
        if self.seg_embedding:
            self.input_puzzle_row = []
            self.input_puzzle_col = []
            for idx in range(9):
                for jdx in range(9):
                    self.input_puzzle_row.append(jdx + 1)
                    self.input_puzzle_col.append(idx + 1)
                    if not (idx == 8 and jdx == 8):
                        self.input_puzzle_row.append(0)
                        self.input_puzzle_col.append(0)
            self.input_puzzle_col = torch.tensor(self.input_puzzle_col)
            self.input_puzzle_row = torch.tensor(self.input_puzzle_row)

            instruct_seg = torch.zeros_like(
                self.tokenize_text(self.instruction_text))
            input_puzzle_col = torch.cat([self.input_puzzle_col, instruct_seg])
            input_puzzle_row = torch.cat([self.input_puzzle_row, instruct_seg])
            self.input_puzzle_col = torch.cat(
                [self.bos_item, input_puzzle_col, self.eos_item])
            self.input_puzzle_row = torch.cat(
                [self.bos_item, input_puzzle_row, self.eos_item])

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == ModeKeys.TRAIN:
            return self._build_train_sample(data)
        else:
            return self._build_infer_sample(data)

    def _build_train_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        build sample for training tasks.

        step 1. execute the `_build_infer_sample` function to get a batch sample
            for inference.
        step 2. process the label data for training.
        """
        sample = self._build_infer_sample(data)
        target = sample['label']
        target_token_list = target.lower().strip().split()
        target = ' '.join(target_token_list[:self.max_tgt_length])
        sample['target'] = self.tokenize_text(target, add_bos=False)
        sample['prev_output_tokens'] = torch.cat(
            [self.bos_item, sample['target'][:-1]])
        return sample

    def _build_infer_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        build sample for inference tasks.

        step 1. Get the input random masked sudoku text input, which shold be
            generated like below pseudo code.
            >>> sudo = np.random.randint(1, 9, size=(9, 9)) # a pseudo sudoku
            >>> sudo_text = " | ".join(" : ".join(str(c) for c in row) \
            >>>             for row in sudo)
        step 2. Limit the length, tokenize the input text and add the bos token
            to the front of the input as source input.
        step 3. Add a pseodo ids for every input.
        """
        assert 'text' in self.column_map and 'text' in data, \
            'there must be `text` column in task key map and source data'
        text = data[self.column_map['text']]  # equal data['text']
        text = ' '.join(text.lower().strip().split()[:self.max_struct_length])
        src_item = self.tokenize_text(text + self.instruction_text)
        src_item = src_item[:(self.max_src_length + self.max_struct_length)]

        sample = {'id': 0.0, 'source': src_item}

        if self.seg_embedding:
            sample['seg_row_tokens'] = self.input_puzzle_row
            sample['seg_col_tokens'] = self.input_puzzle_col

        if 'solution' in self.column_map and self.column_map[
                'solution'] in data:
            sample['label'] = ' {}'.format(data[self.column_map['solution']])
        return sample
