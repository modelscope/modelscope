from typing import Any, Dict

from ...base import Pipeline
from ...builder import PIPELINES
from ....models.nlp import DialogIntentModel
from ....preprocessors import DialogIntentPredictionPreprocessor
from ....utils.constant import Tasks

__all__ = ['DialogIntentPredictionPipeline']


@PIPELINES.register_module(
    Tasks.dialog_intent_prediction, module_name=r'space-intent')
class DialogIntentPredictionPipeline(Pipeline):

    def __init__(self, model: DialogIntentModel,
                 preprocessor: DialogIntentPredictionPreprocessor, **kwargs):
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
