# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.models.nlp.unite.configuration_unite import EvaluationMode
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import InputModel, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import (Preprocessor,
                                      TranslationEvaluationPreprocessor)
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['TranslationEvaluationPipeline']


@PIPELINES.register_module(
    Tasks.translation_evaluation, module_name=Pipelines.translation_evaluation)
class TranslationEvaluationPipeline(Pipeline):

    def __init__(self,
                 model: InputModel,
                 preprocessor: Optional[Preprocessor] = None,
                 eval_mode: EvaluationMode = EvaluationMode.SRC_REF,
                 device: str = 'gpu',
                 **kwargs):
        r"""Build a translation pipeline with a model dir or a model id in the model hub.

        Args:
            model: A Model instance.
            eval_mode: Evaluation mode, choosing one from `"EvaluationMode.SRC_REF"`,
                `"EvaluationMode.SRC"`, `"EvaluationMode.REF"`. Aside from hypothesis, the
                source/reference/source+reference can be presented during evaluation.
        """
        super().__init__(model=model, preprocessor=preprocessor)

        self.eval_mode = eval_mode
        self.checking_eval_mode()
        assert isinstance(self.model, Model), \
            f'please check whether model config exists in {ModelFile.CONFIGURATION}'

        self.preprocessor = TranslationEvaluationPreprocessor(
            self.model.model_dir,
            self.eval_mode) if preprocessor is None else preprocessor

        self.model.load_checkpoint(
            osp.join(self.model.model_dir, ModelFile.TORCH_MODEL_BIN_FILE),
            self.device)
        self.model.eval()

        return

    def checking_eval_mode(self):
        if self.eval_mode == EvaluationMode.SRC:
            logger.info('Evaluation mode: source-only')
        elif self.eval_mode == EvaluationMode.REF:
            logger.info('Evaluation mode: reference-only')
        elif self.eval_mode == EvaluationMode.SRC_REF:
            logger.info('Evaluation mode: source-reference-combined')
        else:
            raise ValueError(
                'Evaluation mode should be one choice among'
                '\'EvaluationMode.SRC\', \'EvaluationMode.REF\', and'
                '\'EvaluationMode.SRC_REF\'.')

    def change_eval_mode(self,
                         eval_mode: EvaluationMode = EvaluationMode.SRC_REF):
        logger.info('Changing the evaluation mode.')
        self.eval_mode = eval_mode
        self.checking_eval_mode()
        self.preprocessor.eval_mode = eval_mode
        return

    def __call__(self, input: Dict[str, Union[str, List[str]]], **kwargs):
        r"""Implementation of __call__ function.

        Args:
            input: The formatted dict containing the inputted sentences.
            An example of the formatted dict:
                ```
                input = {
                    'hyp': [
                        'This is a sentence.',
                        'This is another sentence.',
                    ],
                    'src': [
                        '这是个句子。',
                        '这是另一个句子。',
                    ],
                    'ref': [
                        'It is a sentence.',
                        'It is another sentence.',
                    ]
                }
                ```
        """
        return super().__call__(input=input, **kwargs)

    def forward(self,
                input_ids: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(input_ids)

    def postprocess(self, output: torch.Tensor) -> Dict[str, Any]:
        result = {OutputKeys.SCORES: output.cpu().tolist()}
        return result
