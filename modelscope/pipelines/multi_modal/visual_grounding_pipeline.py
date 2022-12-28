# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models.multi_modal import OfaForAllTasks
from modelscope.pipelines.base import Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.util import batch_process
from modelscope.preprocessors import OfaPreprocessor, Preprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.visual_grounding, module_name=Pipelines.visual_grounding)
class VisualGroundingPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        """
        use `model` and `preprocessor` to create a visual grounding pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model.model.eval()
        if preprocessor is None and isinstance(self.model, OfaForAllTasks):
            self.preprocessor = OfaPreprocessor(model_dir=self.model.model_dir)

    def _batch(self, data):
        if isinstance(self.model, OfaForAllTasks):
            return batch_process(self.model, data)
        else:
            return super(VisualGroundingPipeline, self)._batch(data)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
