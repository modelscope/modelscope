# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Union

from transformers import AutoTokenizer

from modelscope.metainfo import Preprocessors
from modelscope.models.nlp.unite.configuration_unite import EvaluationMode
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from .transformers_tokenizer import NLPTokenizer


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.translation_evaluation)
class TranslationEvaluationPreprocessor(Preprocessor):
    r"""The tokenizer preprocessor used for translation evaluation.
    """

    def __init__(self,
                 model_dir: str,
                 eval_mode: EvaluationMode,
                 mode=ModeKeys.INFERENCE,
                 *args,
                 **kwargs):
        r"""preprocess the data via the vocab file from the `model_dir` path

        Args:
            model_dir: A Model instance.
            eval_mode: Evaluation mode, choosing one from `"EvaluationMode.SRC_REF"`,
                `"EvaluationMode.SRC"`, `"EvaluationMode.REF"`. Aside from hypothesis, the
                source/reference/source+reference can be presented during evaluation.
        """
        super().__init__(mode=mode)
        self.tokenizer = NLPTokenizer(
            model_dir=model_dir, use_fast=False, tokenize_kwargs=kwargs)
        self.eval_mode = eval_mode

        return

    def __call__(self, input_dict: Dict[str, Any]) -> List[List[str]]:
        if self.eval_mode == EvaluationMode.SRC and 'src' not in input_dict.keys(
        ):
            raise ValueError(
                'Source sentences are required for source-only evaluation mode.'
            )
        if self.eval_mode == EvaluationMode.REF and 'ref' not in input_dict.keys(
        ):
            raise ValueError(
                'Reference sentences are required for reference-only evaluation mode.'
            )
        if self.eval_mode == EvaluationMode.SRC_REF and (
                'src' not in input_dict.keys()
                or 'ref' not in input_dict.keys()):
            raise ValueError(
                'Source and reference sentences are both required for source-reference-combined evaluation mode.'
            )

        if type(input_dict['hyp']) == str:
            input_dict['hyp'] = [input_dict['hyp']]
        if (self.eval_mode == EvaluationMode.SRC or self.eval_mode
                == EvaluationMode.SRC_REF) and type(input_dict['src']) == str:
            input_dict['src'] = [input_dict['src']]
        if (self.eval_mode == EvaluationMode.REF or self.eval_mode
                == EvaluationMode.SRC_REF) and type(input_dict['ref']) == str:
            input_dict['ref'] = [input_dict['ref']]

        output_sents = [
            self.tokenizer(
                input_dict['hyp'], return_tensors='pt',
                padding=True)['input_ids']
        ]
        if self.eval_mode == EvaluationMode.SRC or self.eval_mode == EvaluationMode.SRC_REF:
            output_sents += [
                self.tokenizer(
                    input_dict['src'], return_tensors='pt',
                    padding=True)['input_ids']
            ]
        if self.eval_mode == EvaluationMode.REF or self.eval_mode == EvaluationMode.SRC_REF:
            output_sents += [
                self.tokenizer(
                    input_dict['ref'], return_tensors='pt',
                    padding=True)['input_ids']
            ]

        return output_sents
