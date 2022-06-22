import os
import uuid
from typing import Any, Dict, Union

import json
import numpy as np

from ...models.nlp import SbertForSentimentClassification
from ...preprocessors import SentimentClassificationPreprocessor
from ...utils.constant import Tasks
from ...models import Model
from ..base import Input, Pipeline
from ..builder import PIPELINES
from ...metainfo import Pipelines

__all__ = ['SentimentClassificationPipeline']


@PIPELINES.register_module(
    Tasks.sentiment_classification,
    module_name=Pipelines.sentiment_classification)
class SentimentClassificationPipeline(Pipeline):

    def __init__(self,
                 model: Union[SbertForSentimentClassification, str],
                 preprocessor: SentimentClassificationPreprocessor = None,
                 first_sequence="first_sequence",
                 second_sequence="second_sequence",
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp text classification pipeline for prediction

        Args:
            model (SbertForSentimentClassification): a model instance
            preprocessor (SentimentClassificationPreprocessor): a preprocessor instance
        """
        assert isinstance(model, str) or isinstance(model, SbertForSentimentClassification), \
            'model must be a single str or SbertForSentimentClassification'
        sc_model = model if isinstance(
            model,
            SbertForSentimentClassification) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = SentimentClassificationPreprocessor(
                sc_model.model_dir,
                first_sequence=first_sequence,
                second_sequence=second_sequence)
        super().__init__(model=sc_model, preprocessor=preprocessor, **kwargs)
        assert len(sc_model.id2label) > 0

    def postprocess(self, inputs: Dict[str, Any], **postprocess_params) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """

        probs = inputs['probabilities']
        logits = inputs['logits']
        predictions = np.argsort(-probs, axis=-1)
        preds = predictions[0]
        b = 0
        new_result = list()
        for pred in preds:
            new_result.append({
                'pred': self.label_id_to_name[pred],
                'prob': float(probs[b][pred]),
                'logit': float(logits[b][pred])
            })
        new_results = list()
        new_results.append({
            'id':
            inputs['id'][b] if 'id' in inputs else str(uuid.uuid4()),
            'output':
            new_result,
            'predictions':
            new_result[0]['pred'],
            'probabilities':
            ','.join([str(t) for t in inputs['probabilities'][b]]),
            'logits':
            ','.join([str(t) for t in inputs['logits'][b]])
        })

        return new_results[0]
