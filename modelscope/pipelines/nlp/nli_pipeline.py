import os
import uuid
from typing import Any, Dict, Union

import json
import numpy as np

from modelscope.models.nlp import SbertForNLI
from modelscope.preprocessors import NLIPreprocessor
from modelscope.utils.constant import Tasks
from ...models import Model
from ..base import Input, Pipeline
from ..builder import PIPELINES

__all__ = ['NLIPipeline']


@PIPELINES.register_module(
    Tasks.nli, module_name=r'nlp_structbert_nli_chinese-base')
class NLIPipeline(Pipeline):

    def __init__(self,
                 model: Union[SbertForNLI, str],
                 preprocessor: NLIPreprocessor = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp text classification pipeline for prediction

        Args:
            model (SbertForNLI): a model instance
            preprocessor (NLIPreprocessor): a preprocessor instance
        """
        assert isinstance(model, str) or isinstance(model, SbertForNLI), \
            'model must be a single str or SbertForNLI'
        sc_model = model if isinstance(model,
                                       SbertForNLI) else SbertForNLI(model)
        if preprocessor is None:
            preprocessor = NLIPreprocessor(
                sc_model.model_dir,
                first_sequence='first_sequence',
                second_sequence='second_sequence')
        super().__init__(model=sc_model, preprocessor=preprocessor, **kwargs)

        self.label_path = os.path.join(sc_model.model_dir,
                                       'label_mapping.json')
        with open(self.label_path) as f:
            self.label_mapping = json.load(f)
        self.label_id_to_name = {
            idx: name
            for name, idx in self.label_mapping.items()
        }

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, str]:
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
