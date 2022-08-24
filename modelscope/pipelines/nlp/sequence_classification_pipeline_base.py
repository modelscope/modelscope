from typing import Any, Dict, Union

import numpy as np
import torch

from modelscope.models.base import Model
from modelscope.outputs import OutputKeys
from ...preprocessors import Preprocessor
from ..base import Pipeline


class SequenceClassificationPipelineBase(Pipeline):

    def __init__(self, model: Union[Model, str], preprocessor: Preprocessor,
                 **kwargs):
        """This is the base class for all the sequence classification sub-tasks.

        Args:
            model (str or Model): A model instance or a model local dir or a model id in the model hub.
            preprocessor (Preprocessor): a preprocessor instance, must not be None.
        """
        assert isinstance(model, str) or isinstance(model, Model), \
            'model must be a single str or Model'
        model = model if isinstance(model,
                                    Model) else Model.from_pretrained(model)
        assert preprocessor is not None
        model.eval()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.id2label = kwargs.get('id2label')
        if self.id2label is None and hasattr(self.preprocessor, 'id2label'):
            self.id2label = self.preprocessor.id2label
        assert self.id2label is not None, 'Cannot convert id to the original label, please pass in the mapping ' \
                                          'as a parameter or make sure the preprocessor has the attribute.'

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return self.model(inputs, **forward_params)

    def postprocess(self,
                    inputs: Dict[str, Any],
                    topk: int = 5) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_
            topk (int): The topk probs to take
        Returns:
            Dict[str, str]: the prediction results
        """

        probs = inputs[OutputKeys.PROBABILITIES][0]
        num_classes = probs.shape[0]
        topk = min(topk, num_classes)
        top_indices = np.argpartition(probs, -topk)[-topk:]
        cls_ids = top_indices[np.argsort(probs[top_indices])]
        probs = probs[cls_ids].tolist()

        cls_names = [self.id2label[cid] for cid in cls_ids]
        return {OutputKeys.SCORES: probs, OutputKeys.LABELS: cls_names}
