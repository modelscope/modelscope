# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from .nlp_base import NLPTokenizerPreprocessorBase


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.text_gen_tokenizer)
class TextGenerationPreprocessor(NLPTokenizerPreprocessorBase):
    """The tokenizer preprocessor used in text generation.
    """

    def __init__(self,
                 model_dir: str,
                 tokenizer=None,
                 mode=ModeKeys.INFERENCE,
                 **kwargs):
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get('padding', 'max_length')
        kwargs['return_token_type_ids'] = kwargs.get('return_token_type_ids',
                                                     False)
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        super().__init__(model_dir, mode=mode, **kwargs)

    @staticmethod
    def get_roberta_tokenizer_dir(model_dir: str) -> Optional[str]:
        import os
        for name in os.listdir(model_dir):
            full_name = os.path.join(model_dir, name)
            if 'roberta' in name and os.path.isdir(full_name):
                return full_name

    def build_tokenizer(self, model_dir: str):
        roberta_tokenizer_dir = self.get_roberta_tokenizer_dir(model_dir)
        if roberta_tokenizer_dir:
            from transformers import RobertaTokenizer
            return RobertaTokenizer.from_pretrained(
                roberta_tokenizer_dir, do_lower_case=False)
        return super().build_tokenizer(model_dir)

    def __call__(self, data: Union[Dict, str]) -> Dict[str, Any]:
        if self._mode == ModeKeys.INFERENCE:
            return super().__call__(data)
        src_rst = super().__call__(data['src_txt'])
        src_input_ids = src_rst['input_ids']
        src_attention_mask = src_rst['attention_mask']
        if 'tgt_txt' in data:
            labels = super().__call__(data['tgt_txt'])['input_ids']
        else:
            labels = src_input_ids[1:]
            src_input_ids = src_input_ids[:-1]
            src_attention_mask = src_attention_mask[:-1]

        return {
            'input_ids': src_input_ids,
            'attention_mask': src_attention_mask,
            'labels': labels,
        }
