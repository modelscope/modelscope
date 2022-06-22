import os
import uuid
from typing import Any, Dict, Union

import json
import numpy as np
from scipy.special import softmax

from modelscope.models.nlp import SbertForZeroShotClassification
from modelscope.preprocessors import SbertZeroShotClassificationPreprocessor
from modelscope.utils.constant import Tasks
from ...models import Model
from ..base import Input, Pipeline
from ..builder import PIPELINES

__all__ = ['ZeroShotClassificationPipeline']


@PIPELINES.register_module(
    Tasks.zero_shot_classification,
    module_name=r'bert-zero-shot-classification')
class ZeroShotClassificationPipeline(Pipeline):

    def __init__(self,
                 model: Union[SbertForZeroShotClassification, str],
                 preprocessor: SbertZeroShotClassificationPreprocessor = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp text classification pipeline for prediction

        Args:
            model (SbertForSentimentClassification): a model instance
            preprocessor (SentimentClassificationPreprocessor): a preprocessor instance
        """
        assert isinstance(model, str) or isinstance(model, SbertForZeroShotClassification), \
            'model must be a single str or BertForZeroShotClassification'
        sc_model = model if isinstance(
            model,
            SbertForZeroShotClassification) else Model.from_pretrained(model)

        self.entailment_id = 0
        self.contradiction_id = 2

        if preprocessor is None:
            preprocessor = ZeroShotClassificationPreprocessor(
                sc_model.model_dir)
        super().__init__(model=sc_model, preprocessor=preprocessor, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        postprocess_params = {}

        if 'candidate_labels' in kwargs:
            candidate_labels = kwargs.pop('candidate_labels')
            preprocess_params['candidate_labels'] = candidate_labels
            postprocess_params['candidate_labels'] = candidate_labels
        else:
            raise ValueError('You must include at least one label.')
        preprocess_params['hypothesis_template'] = kwargs.pop(
            'hypothesis_template', '{}')

        postprocess_params['multi_label'] = kwargs.pop('multi_label', False)
        return preprocess_params, {}, postprocess_params

    def postprocess(self,
                    inputs: Dict[str, Any],
                    candidate_labels,
                    multi_label=False) -> Dict[str, Any]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, Any]: the prediction results
        """

        logits = inputs['logits']
        if multi_label or len(candidate_labels) == 1:
            logits = logits[..., [self.contradiction_id, self.entailment_id]]
            scores = softmax(logits, axis=-1)[..., 1]
        else:
            logits = logits[..., self.entailment_id]
            scores = softmax(logits, axis=-1)

        reversed_index = list(reversed(scores.argsort()))
        result = {
            'labels': [candidate_labels[i] for i in reversed_index],
            'scores': [scores[i].item() for i in reversed_index],
        }
        return result
