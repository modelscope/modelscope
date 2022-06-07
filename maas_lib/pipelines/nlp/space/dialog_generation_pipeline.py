from typing import Any, Dict, Optional

from maas_lib.models.nlp import DialogGenerationModel
from maas_lib.preprocessors import DialogGenerationPreprocessor
from maas_lib.utils.constant import Tasks
from ...base import Model, Tensor
from ...builder import PIPELINES

__all__ = ['DialogGenerationPipeline']


@PIPELINES.register_module(Tasks.dialog_generation, module_name=r'space')
class DialogGenerationPipeline(Model):

    def __init__(self, model: DialogGenerationModel,
                 preprocessor: DialogGenerationPreprocessor, **kwargs):
        """use `model` and `preprocessor` to create a nlp text classification pipeline for prediction

        Args:
            model (SequenceClassificationModel): a model instance
            preprocessor (SequenceClassificationPreprocessor): a preprocessor instance
        """

        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        pass

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input (Dict[str, Any]): the preprocessed data

        Returns:
            Dict[str, np.ndarray]: results
                Example:
                    {
                        'predictions': array([1]), # lable 0-negative 1-positive
                        'probabilities': array([[0.11491239, 0.8850876 ]], dtype=float32),
                        'logits': array([[-0.53860897,  1.5029076 ]], dtype=float32) # true value
                    }
        """
        from numpy import array, float32

        return {
            'predictions': array([1]),  # lable 0-negative 1-positive
            'probabilities': array([[0.11491239, 0.8850876]], dtype=float32),
            'logits': array([[-0.53860897, 1.5029076]],
                            dtype=float32)  # true value
        }
