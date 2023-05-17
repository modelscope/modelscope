# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.models.nlp.unite.configuration import InputFormat
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import InputModel, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
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
                 input_format: InputFormat = InputFormat.SRC_REF,
                 device: str = 'gpu',
                 **kwargs):
        r"""Build a translation evaluation pipeline with a model dir or a model id in the model hub.

        Args:
            model: A Model instance.
            preprocessor: The preprocessor for this pipeline.
            input_format: Input format, choosing one from `"InputFormat.SRC_REF"`,
                `"InputFormat.SRC"`, `"InputFormat.REF"`. Aside from hypothesis, the
                source/reference/source+reference can be presented during evaluation.
            device: Used device for this pipeline.
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            compile=kwargs.pop('compile', False),
            compile_options=kwargs.pop('compile_options', {}))

        self.input_format = input_format
        self.checking_input_format()
        assert isinstance(self.model, Model), \
            f'please check whether model config exists in {ModelFile.CONFIGURATION}'

        self.model.load_checkpoint(
            osp.join(self.model.model_dir, ModelFile.TORCH_MODEL_BIN_FILE),
            device=self.device,
            plm_only=False)
        self.model.eval()

        return

    def checking_input_format(self):
        if self.input_format == InputFormat.SRC:
            logger.info('Evaluation mode: source-only')
        elif self.input_format == InputFormat.REF:
            logger.info('Evaluation mode: reference-only')
        elif self.input_format == InputFormat.SRC_REF:
            logger.info('Evaluation mode: source-reference-combined')
        else:
            raise ValueError('Evaluation mode should be one choice among'
                             '\'InputFormat.SRC\', \'InputFormat.REF\', and'
                             '\'InputFormat.SRC_REF\'.')

    def change_input_format(self,
                            input_format: InputFormat = InputFormat.SRC_REF):
        logger.info('Changing the evaluation mode.')
        self.input_format = input_format
        self.checking_input_format()
        self.preprocessor.change_input_format(input_format)
        return

    def __call__(self, input_dict: Dict[str, Union[str, List[str]]], **kwargs):
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
        return super().__call__(input=input_dict, **kwargs)

    def forward(
            self, input_dict: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(**input_dict)

    def postprocess(self, output: torch.Tensor) -> Dict[str, Any]:
        return output
