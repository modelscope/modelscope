# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import (FillMaskTransformersPreprocessor,
                                      Preprocessor)
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks

__all__ = ['FeatureExtractionPipeline']


@PIPELINES.register_module(
    Tasks.feature_extraction, module_name=Pipelines.feature_extraction)
class FeatureExtractionPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 padding=False,
                 sequence_length=128,
                 **kwargs):
        """Use `model` and `preprocessor` to create a nlp feature extraction pipeline for prediction

        Args:
            model (str or Model): Supply either a local model dir which supported feature extraction task, or a
            no-head model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.

            Example:
            >>> from modelscope.pipelines import pipeline
            >>> pipe_ins = pipeline('feature_extraction', model='damo/nlp_structbert_feature-extraction_english-large')
            >>> input = 'Everything you love is treasure'
            >>> print(pipe_ins(input))


        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate)

        if preprocessor is None:
            self.preprocessor = Preprocessor.from_pretrained(
                self.model.model_dir,
                padding=padding,
                sequence_length=sequence_length,
                **kwargs)
        self.model.eval()

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
