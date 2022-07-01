# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional

from ...metainfo import Pipelines
from ...models.nlp import SpaceForDialogModeling
from ...preprocessors import DialogModelingPreprocessor
from ...utils.constant import Tasks
from ..base import Pipeline, Tensor
from ..builder import PIPELINES
from ..outputs import OutputKeys

__all__ = ['DialogModelingPipeline']


@PIPELINES.register_module(
    Tasks.dialog_modeling, module_name=Pipelines.dialog_modeling)
class DialogModelingPipeline(Pipeline):

    def __init__(self, model: SpaceForDialogModeling,
                 preprocessor: DialogModelingPreprocessor, **kwargs):
        """use `model` and `preprocessor` to create a nlp text classification pipeline for prediction

        Args:
            model (SequenceClassificationModel): a model instance
            preprocessor (SequenceClassificationPreprocessor): a preprocessor instance
        """

        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model = model
        self.preprocessor = preprocessor

    def postprocess(self, inputs: Dict[str, Tensor]) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """
        sys_rsp = self.preprocessor.text_field.tokenizer.convert_ids_to_tokens(
            inputs['resp'])
        assert len(sys_rsp) > 2
        sys_rsp = sys_rsp[1:len(sys_rsp) - 1]

        inputs[OutputKeys.RESPONSE] = sys_rsp

        return inputs
