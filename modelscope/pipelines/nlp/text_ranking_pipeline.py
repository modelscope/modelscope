# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor, TextRankingPreprocessor
from modelscope.utils.constant import Tasks

__all__ = ['TextRankingPipeline']


@PIPELINES.register_module(
    Tasks.text_ranking, module_name=Pipelines.text_ranking)
class TextRankingPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        """Use `model` and `preprocessor` to create a nlp word segment pipeline for prediction.

        Args:
            model (str or Model): Supply either a local model dir which supported the WS task,
            or a model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            sequence_length: Max sequence length in the user's custom scenario. 128 will be used as a default value.
        """
        model = model if isinstance(model,
                                    Model) else Model.from_pretrained(model)

        if preprocessor is None:
            preprocessor = TextRankingPreprocessor(
                model.model_dir if isinstance(model, Model) else model,
                sequence_length=kwargs.pop('sequence_length', 128))
        model.eval()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return {**self.model(inputs, **forward_params)}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """process the prediction results
        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, Any]: the predicted text representation
        """
        pred_list = inputs[OutputKeys.SCORES]

        return {OutputKeys.SCORES: pred_list}
