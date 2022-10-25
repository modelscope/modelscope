# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from typing import Any, Dict

import sentencepiece as spm
import torch

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sentence_piece)
class SentencePiecePreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        import os

        super().__init__(*args, **kwargs)
        self.tokenizer = None
        for file_name in os.listdir(model_dir):
            if file_name.endswith('.model'):
                m_file = osp.join(model_dir, file_name)
                self.tokenizer = spm.SentencePieceProcessor(model_file=m_file)
                break
        assert self.tokenizer is not None, 'Can not find .model file'

    def __call__(self, data: str) -> Dict[str, Any]:
        return torch.tensor(self.tokenizer.encode([data]), dtype=torch.long)
