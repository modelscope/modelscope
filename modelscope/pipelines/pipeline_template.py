# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.models.base.base_model import Model
from modelscope.outputs.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks

__all__ = ['PipelineTemplate']


@PIPELINES.register_module(
    Tasks.task_template, module_name=Pipelines.pipeline_template)
class PipelineTemplate(Pipeline):
    """A pipeline template explain how to define parameters and input and
       output information. As a rule, the first parameter is the input,
       followed by the request parameters. The parameter must add type
       hint information, and set the default value if necessary,
       for the convenience of use.
    """

    def __init__(self, model: Model, **kwargs):
        """A pipeline template to describe input and
        output and parameter processing

        Args:
            model: A Model instance.
        """
        # call base init.
        super().__init__(model=model, **kwargs)

    def preprocess(self,
                   input: Any,
                   max_length: int = 1024,
                   top_p: float = 0.8) -> Any:
        """Pipeline preprocess interface.

        Args:
            input (Any): The pipeline input, ref Tasks.task_template TASK_INPUTS.
            max_length (int, optional): The max_length parameter. Defaults to 1024.
            top_p (float, optional): The top_p parameter. Defaults to 0.8.

        Returns:
            Any: Return result process by forward.
        """
        pass

    def forward(self,
                input: Any,
                max_length: int = 1024,
                top_p: float = 0.8) -> Any:
        """The forward interface.

        Args:
            input (Any): The output of the preprocess.
            max_length (int, optional): max_length. Defaults to 1024.
            top_p (float, optional): top_p. Defaults to 0.8.

        Returns:
            Any: Return result process by postprocess.
        """
        pass

    def postprocess(self,
                    inputs: Any,
                    postprocess_param1: str = None) -> Dict[str, Any]:
        """The postprocess interface.

        Args:
            input (Any): The output of the forward.
            max_length (int, optional): max_length. Defaults to 1024.
            top_p (float, optional): top_p. Defaults to 0.8.

        Returns:
            Any: Return result process by postprocess.
        """
        result = {
            OutputKeys.BOXES: np.zeros(4),
            OutputKeys.OUTPUT_IMG: np.zeros(10, 4),
            OutputKeys.TEXT_EMBEDDING: np.zeros(1, 1000)
        }
        return result
