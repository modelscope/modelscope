# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Union

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from .nlp_base import NLPTokenizerPreprocessorBase


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.text2text_gen_preprocessor)
class Text2TextGenerationPreprocessor(NLPTokenizerPreprocessorBase):
    """The tokenizer preprocessor used in text generation.
    """

    def __init__(self,
                 model_dir: str,
                 tokenizer=None,
                 mode=ModeKeys.INFERENCE,
                 **kwargs):
        kwargs['truncation'] = kwargs.get('truncation', 'do_not_truncate')
        kwargs['padding'] = kwargs.get('padding', False)
        kwargs['return_token_type_ids'] = kwargs.get('return_token_type_ids',
                                                     False)
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        super().__init__(model_dir, mode=mode, **kwargs)

    def __call__(self, data: Union[Dict, str]) -> Dict[str, Any]:
        text_a, _, _ = self.parse_text_and_label(data)

        inputs = self.tokenizer(
            text_a,
            return_tensors='pt' if self._mode == ModeKeys.INFERENCE else None,
            **self.tokenize_kwargs)

        # This is produced by tokenizers but is an invalid generate kwargs
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        return inputs
