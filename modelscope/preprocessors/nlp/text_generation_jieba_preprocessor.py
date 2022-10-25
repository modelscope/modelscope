# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from typing import Any, Dict

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.text_gen_jieba_tokenizer)
class TextGenerationJiebaPreprocessor(Preprocessor):
    """The jieba tokenizer preprocessor used in text generation.
    """

    def __init__(self, model_dir: str, *args, **kwargs):
        from modelscope.models.nlp.gpt3 import JiebaBPETokenizer
        super().__init__(*args, **kwargs)
        self.tokenizer = JiebaBPETokenizer(
            osp.join(model_dir, 'tokenizer.json'))

    def __call__(self, data: str) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (str): a sentence
                Example:
                    '深蓝的天空中挂着一轮金黄的圆月，下面是海边的沙地'
        Returns:
            Dict[str, Any]: the preprocessed data
            Example:
            {'net_input':
                {'src_tokens':tensor([1,2,3,4]),
                'src_lengths': tensor([4])}
            }
        """
        import torch

        return {
            'input_ids':
            torch.tensor(self.tokenizer.tokenize(data)).unsqueeze_(0)
        }
