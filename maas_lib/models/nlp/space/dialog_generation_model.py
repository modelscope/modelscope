from typing import Any, Dict, Optional

from maas_lib.utils.constant import Tasks
from ...base import Model, Tensor
from ...builder import MODELS

__all__ = ['DialogGenerationModel']


@MODELS.register_module(Tasks.dialog_generation, module_name=r'space')
class DialogGenerationModel(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the test generation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
            model_cls (Optional[Any], optional): model loader, if None, use the
                default loader to load model weights, by default None.
        """

        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir
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
