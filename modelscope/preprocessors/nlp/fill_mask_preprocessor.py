# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
import re
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.config import Config
from modelscope.utils.constant import Fields, ModeKeys, ModelFile
from modelscope.utils.nlp import import_external_nltk_data
from .nlp_base import NLPTokenizerPreprocessorBase


@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.fill_mask)
@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.feature_extraction)
class NLPPreprocessor(NLPTokenizerPreprocessorBase):
    """The tokenizer preprocessor used in MLM task.
    """

    def __init__(self, model_dir: str, mode=ModeKeys.INFERENCE, **kwargs):
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get('padding', 'max_length')
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        kwargs['return_token_type_ids'] = kwargs.get('return_token_type_ids',
                                                     True)
        super().__init__(model_dir, mode=mode, **kwargs)

    @property
    def mask_id(self):
        return self.tokenizer.mask_token_id

    def decode(self,
               token_ids,
               skip_special_tokens: bool = False,
               clean_up_tokenization_spaces: bool = True,
               **kwargs):
        return self.tokenizer.decode(token_ids, skip_special_tokens,
                                     clean_up_tokenization_spaces, **kwargs)


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.fill_mask_ponet)
class FillMaskPoNetPreprocessor(NLPTokenizerPreprocessorBase):
    """The tokenizer preprocessor used in PoNet model's MLM task.
    """

    def __init__(self, model_dir: str, mode=ModeKeys.INFERENCE, **kwargs):
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get('padding', 'max_length')
        kwargs['max_length'] = kwargs.pop('sequence_length', 512)
        kwargs['return_token_type_ids'] = kwargs.get('return_token_type_ids',
                                                     True)
        super().__init__(model_dir, mode=mode, **kwargs)

        self.cfg = Config.from_file(
            osp.join(model_dir, ModelFile.CONFIGURATION))
        self.language = self.cfg.model.get('language', 'en')
        if self.language == 'en':
            from nltk.tokenize import sent_tokenize
            import_external_nltk_data(
                osp.join(model_dir, 'nltk_data'), 'tokenizers/punkt')
        elif self.language in ['zh', 'cn']:

            def sent_tokenize(para):
                para = re.sub(r'([。！!？\?])([^”’])', r'\1\n\2', para)  # noqa *
                para = re.sub(r'(\.{6})([^”’])', r'\1\n\2', para)  # noqa *
                para = re.sub(r'(\…{2})([^”’])', r'\1\n\2', para)  # noqa *
                para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2',
                              para)  # noqa *
                para = para.rstrip()
                return [_ for _ in para.split('\n') if _]
        else:
            raise NotImplementedError

        self.sent_tokenize = sent_tokenize
        self.max_length = kwargs['max_length']

    def __call__(self, data: Union[str, Tuple, Dict]) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (tuple): [sentence1, sentence2]
                sentence1 (str): a sentence
                    Example:
                        'you are so handsome.'
                sentence2 (str): a sentence
                    Example:
                        'you are so beautiful.'
        Returns:
            Dict[str, Any]: the preprocessed data
        """

        text_a, text_b, labels = self.parse_text_and_label(data)
        output = self.tokenizer(
            text_a,
            text_b,
            return_tensors='pt' if self._mode == ModeKeys.INFERENCE else None,
            **self.tokenize_kwargs)
        max_seq_length = self.max_length

        if text_b is None:
            segment_ids = []
            seg_lens = list(
                map(
                    len,
                    self.tokenizer(
                        self.sent_tokenize(text_a),
                        add_special_tokens=False,
                        truncation=True)['input_ids']))
            segment_id = [0] + sum(
                [[i] * sl for i, sl in enumerate(seg_lens, start=1)], [])
            segment_id = segment_id[:max_seq_length - 1]
            segment_ids.append(segment_id + [segment_id[-1] + 1]
                               * (max_seq_length - len(segment_id)))
            if self.mode == ModeKeys.INFERENCE:
                segment_ids = torch.tensor(segment_ids)
            output['segment_ids'] = segment_ids

        output = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in output.items()
        }

        self.labels_to_id(labels, output)
        return output

    @property
    def mask_id(self):
        return self.tokenizer.mask_token_id

    def decode(self,
               token_ids,
               skip_special_tokens: bool = False,
               clean_up_tokenization_spaces: bool = True,
               **kwargs):
        return self.tokenizer.decode(token_ids, skip_special_tokens,
                                     clean_up_tokenization_spaces, **kwargs)
