# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import NLPPreprocessor, Preprocessor
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks

__all__ = ['FeatureExtractionPipeline']


@PIPELINES.register_module(
    Tasks.feature_extraction, module_name=Pipelines.feature_extraction)
class FeatureExtractionPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 first_sequence='sentence',
                 **kwargs):
        """Use `model` and `preprocessor` to create a nlp feature extraction pipeline for prediction

        Args:
            model (str or Model): Supply either a local model dir which supported feature extraction task, or a
            no-head model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            first_sequence: The key to read the sentence in.
            sequence_length: Max sequence length in the user's custom scenario. 128 will be used as a default value.

            NOTE: Inputs of type 'str' are also supported. In this scenario, the 'first_sequence'
            param will have no effect.

            Example:
            >>> from modelscope.pipelines import pipeline
            >>> pipe_ins = pipeline('feature_extraction', model='damo/nlp_structbert_feature-extraction_english-large')
            >>> input = 'Everything you love is treasure'
            >>> print(pipe_ins(input))


        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

        if preprocessor is None:
            self.preprocessor = NLPPreprocessor(
                self.model.model_dir,
                padding=kwargs.pop('padding', False),
                sequence_length=kwargs.pop('sequence_length', 128))
        self.model.eval()

        self.config = Config.from_file(
            os.path.join(self.model.model_dir, ModelFile.CONFIGURATION))
        self.tokenizer = self.preprocessor.tokenizer

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return self.model(**inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """

        return {
            OutputKeys.TEXT_EMBEDDING:
            inputs[OutputKeys.TEXT_EMBEDDING].tolist()
        }
