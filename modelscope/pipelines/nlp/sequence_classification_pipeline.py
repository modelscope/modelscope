from typing import Any, Dict, Union

import numpy as np

from modelscope.models.nlp import BertForSequenceClassification
from modelscope.preprocessors import SequenceClassificationPreprocessor
from modelscope.utils.constant import Tasks
from ...models import Model
from ..base import Input, Pipeline
from ..builder import PIPELINES

__all__ = ['SequenceClassificationPipeline']


@PIPELINES.register_module(
    Tasks.text_classification, module_name=r'bert-sentiment-analysis')
class SequenceClassificationPipeline(Pipeline):

    def __init__(self,
                 model: Union[BertForSequenceClassification, str],
                 preprocessor: SequenceClassificationPreprocessor = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp text classification pipeline for prediction

        Args:
            model (BertForSequenceClassification): a model instance
            preprocessor (SequenceClassificationPreprocessor): a preprocessor instance
        """
        assert isinstance(model, str) or isinstance(model, BertForSequenceClassification), \
            'model must be a single str or BertForSequenceClassification'
        sc_model = model if isinstance(
            model,
            BertForSequenceClassification) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = SequenceClassificationPreprocessor(
                sc_model.model_dir,
                first_sequence='sentence',
                second_sequence=None)
        super().__init__(model=sc_model, preprocessor=preprocessor, **kwargs)

        assert hasattr(self.model, 'id2label'), \
            'id2label map should be initalizaed in init function.'

    def postprocess(self,
                    inputs: Dict[str, Any],
                    topk: int = 5) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): input data dict
            topk (int): return topk classification result.

        Returns:
            Dict[str, str]: the prediction results
        """
        # NxC np.ndarray
        probs = inputs['probs'][0]
        num_classes = probs.shape[0]
        topk = min(topk, num_classes)
        top_indices = np.argpartition(probs, -topk)[-topk:]
        cls_ids = top_indices[np.argsort(probs[top_indices])]
        probs = probs[cls_ids].tolist()

        cls_names = [self.model.id2label[cid] for cid in cls_ids]

        return {'scores': probs, 'labels': cls_names}
