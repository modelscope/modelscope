import uuid
from typing import Any, Dict, Union

import numpy as np
import torch

from ...metainfo import Pipelines
from ...models import Model
from ...models.nlp import SbertForNLI
from ...preprocessors import NLIPreprocessor
from ...utils.constant import Tasks
from ..base import Pipeline
from ..builder import PIPELINES

__all__ = ['NLIPipeline']


@PIPELINES.register_module(Tasks.nli, module_name=Pipelines.nli)
class NLIPipeline(Pipeline):

    def __init__(self,
                 model: Union[SbertForNLI, str],
                 preprocessor: NLIPreprocessor = None,
                 first_sequence='first_sequence',
                 second_sequence='second_sequence',
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp text classification pipeline for prediction

        Args:
            model (SbertForNLI): a model instance
            preprocessor (NLIPreprocessor): a preprocessor instance
        """
        assert isinstance(model, str) or isinstance(model, SbertForNLI), \
            'model must be a single str or SbertForNLI'
        model = model if isinstance(
            model, SbertForNLI) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = NLIPreprocessor(
                model.model_dir,
                first_sequence=first_sequence,
                second_sequence=second_sequence)
        model.eval()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        assert len(model.id2label) > 0

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self,
                    inputs: Dict[str, Any],
                    topk: int = 5) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """

        probs = inputs['probabilities'][0]
        num_classes = probs.shape[0]
        topk = min(topk, num_classes)
        top_indices = np.argpartition(probs, -topk)[-topk:]
        cls_ids = top_indices[np.argsort(probs[top_indices])]
        probs = probs[cls_ids].tolist()

        cls_names = [self.model.id2label[cid] for cid in cls_ids]

        return {'scores': probs, 'labels': cls_names}
