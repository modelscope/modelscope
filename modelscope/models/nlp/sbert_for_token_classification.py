from typing import Any, Dict, Union

import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.utils.constant import Tasks
from ..base import Model, Tensor
from ..builder import MODELS

__all__ = ['SbertForTokenClassification']


@MODELS.register_module(Tasks.word_segmentation, module_name=Models.structbert)
class SbertForTokenClassification(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the word segmentation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
            model_cls (Optional[Any], optional): model loader, if None, use the
                default loader to load model weights, by default None.
        """
        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir
        import sofa
        self.model = sofa.SbertForTokenClassification.from_pretrained(
            self.model_dir)
        self.config = sofa.SbertConfig.from_pretrained(self.model_dir)

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()

    def forward(self, input: Dict[str,
                                  Any]) -> Dict[str, Union[str, np.ndarray]]:
        """return the result by the model

        Args:
            input (Dict[str, Any]): the preprocessed data

        Returns:
            Dict[str, Union[str,np.ndarray]]: results
                Example:
                    {
                        'predictions': array([1,4]), # lable 0-negative 1-positive
                        'logits': array([[-0.53860897,  1.5029076 ]], dtype=float32) # true value
                        'text': str(今天),
                    }
        """
        input_ids = torch.tensor(input['input_ids']).unsqueeze(0)
        return {**self.model(input_ids), 'text': input['text']}

    def postprocess(self, input: Dict[str, Tensor],
                    **kwargs) -> Dict[str, Tensor]:
        logits = input['logits']
        pred = torch.argmax(logits[0], dim=-1)
        pred = pred.numpy()
        rst = {'predictions': pred, 'logits': logits, 'text': input['text']}
        return rst
