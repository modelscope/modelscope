from typing import Any, Dict, Union

import numpy as np
import torch

from ...metainfo import Pipelines
from ...models import Model
from ...models.nlp import SbertForSentenceSimilarity
from ...preprocessors import SequenceClassificationPreprocessor
from ...utils.constant import Tasks
from ..base import Input, Pipeline
from ..builder import PIPELINES

__all__ = ['SentenceSimilarityPipeline']


@PIPELINES.register_module(
    Tasks.sentence_similarity, module_name=Pipelines.sentence_similarity)
class SentenceSimilarityPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: SequenceClassificationPreprocessor = None,
                 first_sequence='first_sequence',
                 second_sequence='second_sequence',
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp sentence similarity pipeline for prediction

        Args:
            model (SbertForSentenceSimilarity): a model instance
            preprocessor (SequenceClassificationPreprocessor): a preprocessor instance
        """
        assert isinstance(model, str) or isinstance(model, SbertForSentenceSimilarity), \
            'model must be a single str or SbertForSentenceSimilarity'
        sc_model = model if isinstance(
            model,
            SbertForSentenceSimilarity) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = SequenceClassificationPreprocessor(
                sc_model.model_dir,
                first_sequence=first_sequence,
                second_sequence=second_sequence)
        sc_model.eval()
        super().__init__(model=sc_model, preprocessor=preprocessor, **kwargs)

        assert hasattr(self.model, 'id2label'), \
            'id2label map should be initalizaed in init function.'

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Any],
                    **postprocess_params) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """

        probs = inputs['probabilities'][0]
        num_classes = probs.shape[0]
        top_indices = np.argpartition(probs, -num_classes)[-num_classes:]
        cls_ids = top_indices[np.argsort(-probs[top_indices], axis=-1)]
        probs = probs[cls_ids].tolist()
        cls_names = [self.model.id2label[cid] for cid in cls_ids]
        b = 0
        return {'scores': probs[b], 'labels': cls_names[b]}
