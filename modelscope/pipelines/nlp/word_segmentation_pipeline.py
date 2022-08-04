from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import (Preprocessor,
                                      TokenClassificationPreprocessor)
from modelscope.utils.constant import Tasks

__all__ = ['WordSegmentationPipeline']


@PIPELINES.register_module(
    Tasks.word_segmentation, module_name=Pipelines.word_segmentation)
class WordSegmentationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp word segmentation pipeline for prediction

        Args:
            model (Model): a model instance
            preprocessor (Preprocessor): a preprocessor instance
        """
        model = model if isinstance(model,
                                    Model) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = TokenClassificationPreprocessor(
                model.model_dir,
                sequence_length=kwargs.pop('sequence_length', 128))
        model.eval()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.id2label = kwargs.get('id2label')
        if self.id2label is None and hasattr(self.preprocessor, 'id2label'):
            self.id2label = self.preprocessor.id2label
        assert self.id2label is not None, 'Cannot convert id to the original label, please pass in the mapping ' \
                                          'as a parameter or make sure the preprocessor has the attribute.'

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        text = inputs.pop(OutputKeys.TEXT)
        with torch.no_grad():
            return {
                **self.model(inputs, **forward_params), OutputKeys.TEXT: text
            }

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
