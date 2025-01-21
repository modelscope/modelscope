# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines, Preprocessors
from modelscope.pipelines.base import Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.util import batch_process
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Fields, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.text_summarization, module_name=Pipelines.text_generation)
class SummarizationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 **kwargs):
        """Use `model` and `preprocessor` to create a Summarization pipeline for prediction.

        Args:
            model (str or Model): Supply either a local model dir which supported the summarization task,
            or a model id from the model hub, or a model instance.
            preprocessor (Preprocessor): An optional preprocessor instance.
            kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate)
        self.model.eval()
        if preprocessor is None:
            if self.model.__class__.__name__ == 'OfaForAllTasks':
                self.preprocessor = Preprocessor.from_pretrained(
                    self.model.model_dir,
                    type=Preprocessors.ofa_tasks_preprocessor,
                    field=Fields.multi_modal)
            else:
                self.preprocessor = Preprocessor.from_pretrained(
                    self.model.model_dir, **kwargs)

    def _batch(self, data):
        if self.model.__class__.__name__ == 'OfaForAllTasks':
            return batch_process(self.model, data)
        else:
            return super(SummarizationPipeline, self)._batch(data)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
