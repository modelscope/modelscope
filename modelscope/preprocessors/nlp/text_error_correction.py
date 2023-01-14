# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from typing import Any, Dict

import torch

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.text_error_correction)
class TextErrorCorrectionPreprocessor(Preprocessor):
    """The preprocessor used in text correction task.
    """

    def __init__(self,
                 model_dir: str,
                 max_length: int = None,
                 *args,
                 **kwargs):
        from fairseq.data import Dictionary
        """preprocess the data via the vocab file from the `model_dir` path

        Args:
            model_dir (str): model path
        """
        super().__init__(*args, **kwargs)
        self.vocab = Dictionary.load(osp.join(model_dir, 'dict.src.txt'))
        self.max_length = max_length + 1 if max_length is not None else 129  # 1 is eos token
        self.padding_value = self.vocab.pad()

    def __call__(self, data: str) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (str): a sentence
                Example:
                    '随着中国经济突飞猛近，建造工业与日俱增'
        Returns:
            Dict[str, Any]: the preprocessed data
            Example:
            {'net_input':
                {'src_tokens':tensor([1,2,3,4]),
                'src_lengths': tensor([4])}
            }
        """

        text = ' '.join([x for x in data])
        inputs = self.vocab.encode_line(
            text, append_eos=True, add_if_not_exist=False)
        lengths = inputs.size()[0]

        padding = torch.tensor([self.padding_value] *  # noqa: W504
                               (self.max_length - lengths))
        inputs = torch.unsqueeze(torch.cat([padding, inputs]), dim=0)
        lengths = torch.tensor([lengths])
        out = {'src_tokens': inputs, 'src_lengths': lengths}

        return out
