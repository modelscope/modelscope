# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Union

from modelscope.metainfo import Pipelines
from modelscope.models.multi_modal import OfaForAllTasks
from modelscope.pipelines.base import Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import OfaPreprocessor, Preprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.summarization, module_name=Pipelines.text_generation)
class SummarizationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: [Preprocessor] = None,
                 **kwargs):
        """Use `model` and `preprocessor` to create a Summarization pipeline for prediction.

        Args:
            model (str or Model): Supply either a local model dir which supported the summarization task,
            or a model id from the model hub, or a model instance.
            preprocessor (Preprocessor): An optional preprocessor instance.
        """
        super().__init__(model=model)
        assert isinstance(model, str) or isinstance(model, Model), \
            'model must be a single str or OfaForAllTasks'
        if isinstance(model, str):
            pipe_model = Model.from_pretrained(model)
        elif isinstance(model, Model):
            pipe_model = model
        else:
            raise NotImplementedError
        pipe_model.model.eval()
        if preprocessor is None and isinstance(pipe_model, OfaForAllTasks):
            preprocessor = OfaPreprocessor(model_dir=pipe_model.model_dir)
        super().__init__(model=pipe_model, preprocessor=preprocessor, **kwargs)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
