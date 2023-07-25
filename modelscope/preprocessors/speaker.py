# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Tuple, Union

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.preprocessors.nlp.text_classification_preprocessor import \
    TextClassificationPreprocessorBase
from modelscope.preprocessors.nlp.transformers_tokenizer import NLPTokenizer
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.hub import get_model_type, parse_label_mapping
from modelscope.utils.logger import get_logger

# from modelscope.preprocessors.nlp.utils import labels_to_id, parse_text_and_label

logger = get_logger()


@PREPROCESSORS.register_module(
    Fields.audio, module_name=Preprocessors.sen_cls_tokenizer)
class SpeakerDiarizationDialogueDetectionPreprocessor(
        TextClassificationPreprocessorBase):

    def _tokenize_text(self, sequence1, sequence2=None, **kwargs):
        if 'return_tensors' not in kwargs:
            kwargs[
                'return_tensors'] = 'pt' if self.mode == ModeKeys.INFERENCE else None
        return self.nlp_tokenizer(sequence1, sequence2, **kwargs)

    def __init__(self,
                 model_dir=None,
                 first_sequence: str = None,
                 second_sequence: str = None,
                 label: Union[str, List] = 'label',
                 label2id: Dict = None,
                 mode: str = ModeKeys.INFERENCE,
                 max_length: int = None,
                 use_fast: bool = None,
                 keep_original_columns=None,
                 **kwargs):
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get('padding', 'max_length')
        kwargs[
            'max_length'] = max_length if max_length is not None else kwargs.get(
                'sequence_length', 128)
        kwargs.pop('sequence_length', None)
        model_type = None
        if model_dir is not None:
            model_type = get_model_type(model_dir)
        self.nlp_tokenizer = NLPTokenizer(
            model_dir, model_type, use_fast=use_fast, tokenize_kwargs=kwargs)
        super().__init__(model_dir, first_sequence, second_sequence, label,
                         label2id, mode, keep_original_columns)
