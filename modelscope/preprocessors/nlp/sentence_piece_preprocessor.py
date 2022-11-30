# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp

import sentencepiece as spm
import torch

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sentence_piece)
class SentencePiecePreprocessor(Preprocessor):

    def __init__(self,
                 model_dir: str,
                 mode=ModeKeys.INFERENCE,
                 *args,
                 **kwargs):
        """The preprocessor for the sentence piece tokenizer.

        Args:
            model_dir: The model dir contains the essential files used by the `SentencePieceProcessor`.
            mode: The mode for the preprocessor.
        """

        super().__init__(mode)
        self.tokenizer = None
        for file_name in os.listdir(model_dir):
            if file_name.endswith('.model'):
                m_file = osp.join(model_dir, file_name)
                self.tokenizer = spm.SentencePieceProcessor(model_file=m_file)
                break
        assert self.tokenizer is not None, 'Can not find .model file'

    def __call__(self, data: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode([data]), dtype=torch.long)
