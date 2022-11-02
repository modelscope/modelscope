# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import (Preprocessor,
                                      TokenClassificationPreprocessor)
from modelscope.utils.constant import Tasks
from modelscope.utils.tensor_utils import (torch_nested_detach,
                                           torch_nested_numpify)

__all__ = ['WordSegmentationPipeline']


@PIPELINES.register_module(
    Tasks.word_segmentation, module_name=Pipelines.word_segmentation)
class WordSegmentationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        """Use `model` and `preprocessor` to create a nlp word segment pipeline for prediction.

        Args:
            model (str or Model): Supply either a local model dir which supported the WS task,
            or a model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            sequence_length: Max sequence length in the user's custom scenario. 128 will be used as a default value.

            NOTE: The preprocessor will first split the sentence into single characters,
            then feed them into the tokenizer with the parameter is_split_into_words=True.

            Example:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline(task='word-segmentation',
            >>>    model='damo/nlp_structbert_word-segmentation_chinese-base')
            >>> sentence1 = '今天天气不错，适合出去游玩'
            >>> print(pipeline_ins(sentence1))

            To view other examples plese check the tests/pipelines/test_word_segmentation.py.
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
                **self.model(**inputs, **forward_params), OutputKeys.TEXT: text
            }

    def postprocess(self, inputs: Dict[str, Any],
                    **postprocess_params) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): should be tensors from model

        Returns:
            Dict[str, str]: the prediction results
        """
        text = inputs['text']
        logits = inputs[OutputKeys.LOGITS]
        predictions = torch.argmax(logits[0], dim=-1)
        logits = torch_nested_numpify(torch_nested_detach(logits))
        predictions = torch_nested_numpify(torch_nested_detach(predictions))
        offset_mapping = [x.cpu().tolist() for x in inputs['offset_mapping']]

        labels = [self.id2label[x] for x in predictions]
        if len(labels) > len(offset_mapping):
            labels = labels[1:-1]
        chunks = []
        chunk = {}
        for label, offsets in zip(labels, offset_mapping):
            if label[0] in 'BS':
                if chunk:
                    chunk['span'] = text[chunk['start']:chunk['end']]
                    chunks.append(chunk)
                chunk = {
                    'type': label[2:],
                    'start': offsets[0],
                    'end': offsets[1]
                }
            if label[0] in 'IES':
                if chunk:
                    chunk['end'] = offsets[1]

            if label[0] in 'ES':
                if chunk:
                    chunk['span'] = text[chunk['start']:chunk['end']]
                    chunks.append(chunk)
                    chunk = {}

        if chunk:
            chunk['span'] = text[chunk['start']:chunk['end']]
            chunks.append(chunk)

        # for cws outputs
        if len(chunks) > 0 and chunks[0]['type'] == 'cws':
            spans = [
                chunk['span'] for chunk in chunks if chunk['span'].strip()
            ]
            seg_result = ' '.join(spans)
            outputs = {OutputKeys.OUTPUT: seg_result}

        # for ner output
        else:
            outputs = {OutputKeys.OUTPUT: chunks}
        return outputs
