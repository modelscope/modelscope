from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.nlp import SbertForTokenClassification
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import TokenClassificationPreprocessor
from modelscope.utils.constant import Tasks

__all__ = ['WordSegmentationPipeline']


@PIPELINES.register_module(
    Tasks.word_segmentation, module_name=Pipelines.word_segmentation)
class WordSegmentationPipeline(Pipeline):

    def __init__(
            self,
            model: Union[SbertForTokenClassification, str],
            preprocessor: Optional[TokenClassificationPreprocessor] = None,
            **kwargs):
        """use `model` and `preprocessor` to create a nlp word segmentation pipeline for prediction

        Args:
            model (StructBertForTokenClassification): a model instance
            preprocessor (TokenClassificationPreprocessor): a preprocessor instance
        """
        model = model if isinstance(
            model,
            SbertForTokenClassification) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = TokenClassificationPreprocessor(model.model_dir)
        model.eval()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.tokenizer = preprocessor.tokenizer
        self.config = model.config
        assert len(self.config.id2label) > 0
        self.id2label = self.config.id2label

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

        pred_list = inputs['predictions']
        labels = []
        for pre in pred_list:
            labels.append(self.id2label[pre])
        labels = labels[1:-1]
        chunks = []
        chunk = ''
        assert len(inputs['text']) == len(labels)
        for token, label in zip(inputs['text'], labels):
            if label[0] == 'B' or label[0] == 'I':
                chunk += token
            else:
                chunk += token
                chunks.append(chunk)
                chunk = ''
        if chunk:
            chunks.append(chunk)
        seg_result = ' '.join(chunks)
        return {OutputKeys.OUTPUT: seg_result}
