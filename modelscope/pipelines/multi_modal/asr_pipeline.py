# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models.multi_modal import MPlugForAllTasks, OfaForAllTasks
from modelscope.pipelines.base import Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.util import batch_process
from modelscope.preprocessors import (MPlugPreprocessor, OfaPreprocessor,
                                      Preprocessor)
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.auto_speech_recognition, module_name=Pipelines.ofa_asr)
class AutomaticSpeechRecognitionPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        """
        use `model` and `preprocessor` to create an automatic speech recognition pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        assert isinstance(model, str) or isinstance(model, Model), \
            'model must be a single str or OfaForAllTasks'
        if isinstance(model, str):
            pipe_model = Model.from_pretrained(model)
        elif isinstance(model, Model):
            pipe_model = model
        else:
            raise NotImplementedError
        pipe_model.model.eval()
        if preprocessor is None:
            if isinstance(pipe_model, OfaForAllTasks):
                preprocessor = OfaPreprocessor(pipe_model.model_dir)
            elif isinstance(pipe_model, MPlugForAllTasks):
                preprocessor = MPlugPreprocessor(pipe_model.model_dir)
        super().__init__(model=pipe_model, preprocessor=preprocessor, **kwargs)

    def _batch(self, data):
        if isinstance(self.model, OfaForAllTasks):
            return batch_process(self.model, data)
        else:
            return super(AutomaticSpeechRecognitionPipeline, self)._batch(data)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
