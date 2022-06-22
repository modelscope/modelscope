import uuid
from typing import Any, Dict, Union

import uuid
from typing import Any, Dict, Union

import numpy as np

from ..base import Pipeline
from ..builder import PIPELINES
from ...metainfo import Pipelines
from ...models import Model
from ...models.nlp import SbertForNLI
from ...preprocessors import NLIPreprocessor
from ...utils.constant import Tasks

__all__ = ['NLIPipeline']


@PIPELINES.register_module(
    Tasks.nli, module_name=Pipelines.nli)
class NLIPipeline(Pipeline):

    def __init__(self,
                 model: Union[SbertForNLI, str],
                 preprocessor: NLIPreprocessor = None,
                 first_sequence="first_sequence",
                 second_sequence="second_sequence",
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp text classification pipeline for prediction

        Args:
            model (SbertForNLI): a model instance
            preprocessor (NLIPreprocessor): a preprocessor instance
        """
        assert isinstance(model, str) or isinstance(model, SbertForNLI), \
            'model must be a single str or SbertForNLI'
        sc_model = model if isinstance(
            model, SbertForNLI) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = NLIPreprocessor(
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
