# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.multi_modal import MPlugForAllTasks, OfaForAllTasks
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.util import batch_process
from modelscope.preprocessors import (MPlugPreprocessor, OfaPreprocessor,
                                      Preprocessor)
from modelscope.utils.constant import Tasks

__all__ = ['VisualQuestionAnsweringPipeline']


@PIPELINES.register_module(
    Tasks.visual_question_answering,
    module_name=Pipelines.visual_question_answering)
class VisualQuestionAnsweringPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a visual question answering pipeline for prediction

        Args:
            model (MPlugForVisualQuestionAnswering): a model instance
            preprocessor (MPlugVisualQuestionAnsweringPreprocessor): a preprocessor instance
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        if preprocessor is None:
            if isinstance(self.model, OfaForAllTasks):
                self.preprocessor = OfaPreprocessor(self.model.model_dir)
            elif isinstance(self.model, MPlugForAllTasks):
                self.preprocessor = MPlugPreprocessor(self.model.model_dir)
        self.model.eval()

    def _batch(self, data):
        if isinstance(self.model, OfaForAllTasks):
            return batch_process(self.model, data)
        else:
            return super(VisualQuestionAnsweringPipeline, self)._batch(data)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Tensor],
                    **postprocess_params) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """
        return inputs
