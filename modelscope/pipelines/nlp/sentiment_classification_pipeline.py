from typing import Any, Dict, Union

import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.nlp import SequenceClassificationModel
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import SentimentClassificationPreprocessor
from modelscope.utils.constant import Tasks

__all__ = ['SentimentClassificationPipeline']


@PIPELINES.register_module(
    Tasks.sentiment_classification,
    module_name=Pipelines.sentiment_classification)
class SentimentClassificationPipeline(Pipeline):

    def __init__(self,
                 model: Union[SequenceClassificationModel, str],
                 preprocessor: SentimentClassificationPreprocessor = None,
                 first_sequence='first_sequence',
                 second_sequence='second_sequence',
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp text classification pipeline for prediction

        Args:
            model (SequenceClassificationModel): a model instance
            preprocessor (SentimentClassificationPreprocessor): a preprocessor instance
        """
        assert isinstance(model, str) or isinstance(model, SequenceClassificationModel), \
            'model must be a single str or SentimentClassification'
        model = model if isinstance(
            model,
            SequenceClassificationModel) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = SentimentClassificationPreprocessor(
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
        return {OutputKeys.SCORES: probs, OutputKeys.LABELS: cls_names}
