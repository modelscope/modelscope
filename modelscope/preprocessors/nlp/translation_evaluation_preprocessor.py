# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Union

import torch
from transformers import AutoTokenizer

from modelscope.metainfo import Preprocessors
from modelscope.models.nlp.unite.configuration import InputFormat
from modelscope.models.nlp.unite.translation_evaluation import \
    combine_input_sentences
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from .transformers_tokenizer import NLPTokenizer


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.translation_evaluation)
class TranslationEvaluationTransformersPreprocessor(Preprocessor):
    r"""The tokenizer preprocessor used for translation evaluation.
    """

    def __init__(self,
                 model_dir: str,
                 max_len: int,
                 pad_token_id: int,
                 eos_token_id: int,
                 input_format: InputFormat = InputFormat.SRC_REF,
                 mode=ModeKeys.INFERENCE,
                 *args,
                 **kwargs):
        r"""Preprocessing the data for the model in `model_dir` path

        Args:
            model_dir: A Model instance.
            max_len: Maximum length for input sequence.
            pad_token_id: Token id for padding token.
            eos_token_id: Token id for the ending-of-sequence (eos) token.
            input_format: Input format, choosing one from `"InputFormat.SRC_REF"`,
                `"InputFormat.SRC"`, `"InputFormat.REF"`. Aside from hypothesis, the
                source/reference/source+reference can be presented during evaluation.
            mode: The mode for this preprocessor.
        """
        super().__init__(mode=mode)
        self.tokenizer = NLPTokenizer(
            model_dir=model_dir, use_fast=False, tokenize_kwargs=kwargs)
        self.input_format = input_format

        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        return

    def change_input_format(self, input_format: InputFormat):
        r"""Change the input format for the preprocessor.

        Args:
            input_format: Any choice in InputFormat.SRC_REF, InputFormat.SRC and InputFormat.REF.

        """
        self.input_format = input_format
        return

    def collect_input_ids(self, input_dict: Dict[str, Any]):
        r"""Collect the input ids for the given examples.

        Args:
            input_dict: A dict containing hyp/src/ref sentences.

        Returns:
            The token ids for each example.

        """
        output_sents = [
            self.tokenizer(
                input_dict['hyp'], return_tensors='pt',
                padding=True)['input_ids']
        ]
        if self.input_format == InputFormat.SRC or self.input_format == InputFormat.SRC_REF:
            output_sents += [
                self.tokenizer(
                    input_dict['src'], return_tensors='pt',
                    padding=True)['input_ids']
            ]
        if self.input_format == InputFormat.REF or self.input_format == InputFormat.SRC_REF:
            output_sents += [
                self.tokenizer(
                    input_dict['ref'], return_tensors='pt',
                    padding=True)['input_ids']
            ]

        input_ids = combine_input_sentences(output_sents, self.max_len,
                                            self.pad_token_id,
                                            self.eos_token_id)

        return input_ids

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.input_format == InputFormat.SRC and 'src' not in input_dict.keys(
        ):
            raise ValueError(
                'Source sentences are required for source-only evaluation mode.'
            )
        if self.input_format == InputFormat.REF and 'ref' not in input_dict.keys(
        ):
            raise ValueError(
                'Reference sentences are required for reference-only evaluation mode.'
            )
        if self.input_format == InputFormat.SRC_REF and (
                'src' not in input_dict.keys()
                or 'ref' not in input_dict.keys()):
            raise ValueError(
                'Source and reference sentences are both required for source-reference-combined evaluation mode.'
            )

        if type(input_dict['hyp']) == str:
            input_dict['hyp'] = [input_dict['hyp']]
        if (self.input_format == InputFormat.SRC or self.input_format
                == InputFormat.SRC_REF) and type(input_dict['src']) == str:
            input_dict['src'] = [input_dict['src']]
        if (self.input_format == InputFormat.REF or self.input_format
                == InputFormat.SRC_REF) and type(input_dict['ref']) == str:
            input_dict['ref'] = [input_dict['ref']]

        if (self.input_format == InputFormat.SRC
                or self.input_format == InputFormat.SRC_REF) and (len(
                    input_dict['hyp']) != len(input_dict['src'])):
            raise ValueError(
                'The number of given hyp sentences (%d) is not equal to that of src (%d).'
                % (len(input_dict['hyp']), len(input_dict['src'])))
        if (self.input_format == InputFormat.REF
                or self.input_format == InputFormat.SRC_REF) and (len(
                    input_dict['hyp']) != len(input_dict['ref'])):
            raise ValueError(
                'The number of given hyp sentences (%d) is not equal to that of ref (%d).'
                % (len(input_dict['hyp']), len(input_dict['ref'])))

        output_dict = {'input_ids': self.collect_input_ids(input_dict)}

        if self.mode == ModeKeys.TRAIN or self.mode == ModeKeys.EVAL:
            if 'score' not in input_dict.keys():
                raise KeyError(
                    'During training or evaluating, \'score\' should be provided.'
                )
            if (isinstance(input_dict['score'], List) and len(input_dict['score']) != len(output_dict['input_ids'])) \
                    or (isinstance(input_dict['score'], float) and len(output['input_ids']) != 1):
                raise ValueError(
                    'The number of score is not equal to that of the given examples. '
                    'Required %d, given %d.' %
                    (len(output['input_ids']), len(input_dict['score'])))

            output_dict['score'] = [input_dict['score']] if isinstance(
                input_dict['score'], float) else input_dict['score']

        if self.mode == ModeKeys.EVAL:
            if 'lp' not in input_dict.keys():
                raise ValueError(
                    'Language pair should be provided for evaluation.')

            if 'segment_id' not in input_dict.keys():
                raise ValueError(
                    'Segment id should be provided for evaluation.')

            if 'raw_score' not in input_dict.keys():
                raise ValueError(
                    'Raw scores should be provided for evaluation.')

            output_dict['lp'] = input_dict['lp']
            output_dict['segment_id'] = input_dict['segment_id']
            output_dict['raw_score'] = input_dict['raw_score']

        return output_dict
