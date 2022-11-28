# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

from modelscope.metainfo import Pipelines
from modelscope.models.multi_modal import OfaForAllTasks
from modelscope.pipelines.base import Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import OfaPreprocessor, Preprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.text_summarization, module_name=Pipelines.text_generation)
class SummarizationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        """Use `model` and `preprocessor` to create a Summarization pipeline for prediction.

        Args:
            model (str or Model): Supply either a local model dir which supported the summarization task,
            or a model id from the model hub, or a model instance.
            preprocessor (Preprocessor): An optional preprocessor instance.
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model.eval()
        if preprocessor is None and isinstance(self.model, OfaForAllTasks):
            self.preprocessor = OfaPreprocessor(model_dir=self.model.model_dir)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
