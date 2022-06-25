from typing import Any, Dict

from ...metainfo import Pipelines
from ...models.nlp import DialogStateTrackingModel
from ...preprocessors import DialogStateTrackingPreprocessor
from ...utils.constant import Tasks
from ..base import Pipeline
from ..builder import PIPELINES

__all__ = ['DialogStateTrackingPipeline']


@PIPELINES.register_module(
    Tasks.dialog_state_tracking, module_name=Pipelines.dialog_state_tracking)
class DialogStateTrackingPipeline(Pipeline):

    def __init__(self, model: DialogStateTrackingModel,
                 preprocessor: DialogStateTrackingPreprocessor, **kwargs):
        """use `model` and `preprocessor` to create a nlp text classification pipeline for prediction

        Args:
            model (SequenceClassificationModel): a model instance
            preprocessor (SequenceClassificationPreprocessor): a preprocessor instance
        """

        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model = model
        # self.tokenizer = preprocessor.tokenizer

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

        result = {'pred': pred, 'label': pos[0]}

        return result
