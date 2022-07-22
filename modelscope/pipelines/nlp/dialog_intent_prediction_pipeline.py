# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Union

from ...metainfo import Pipelines
from ...models import Model
from ...models.nlp import SpaceForDialogIntent
from ...outputs import OutputKeys
from ...preprocessors import DialogIntentPredictionPreprocessor
from ...utils.constant import Tasks
from ..base import Pipeline
from ..builder import PIPELINES

__all__ = ['DialogIntentPredictionPipeline']


@PIPELINES.register_module(
    Tasks.dialog_intent_prediction,
    module_name=Pipelines.dialog_intent_prediction)
class DialogIntentPredictionPipeline(Pipeline):

    def __init__(self,
                 model: Union[SpaceForDialogIntent, str],
                 preprocessor: DialogIntentPredictionPreprocessor = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a dialog intent prediction pipeline

        Args:
            model (SpaceForDialogIntent): a model instance
            preprocessor (DialogIntentPredictionPreprocessor): a preprocessor instance
        """
        model = model if isinstance(
            model, SpaceForDialogIntent) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = DialogIntentPredictionPreprocessor(model.model_dir)
        self.model = model
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.categories = preprocessor.categories

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """
        import numpy as np
        pred = inputs['pred']
        pos = np.where(pred == np.max(pred))

        result = {
            OutputKeys.PREDICTION: pred,
            OutputKeys.LABEL_POS: pos[0],
            OutputKeys.LABEL: self.categories[pos[0][0]]
        }

        return result
