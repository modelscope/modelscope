# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Union

import numpy as np

from modelscope.metainfo import Pipelines, Preprocessors
from modelscope.models.base import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Fields, Tasks


@PIPELINES.register_module(
    Tasks.text_classification, module_name=Pipelines.sentiment_analysis)
@PIPELINES.register_module(Tasks.nli, module_name=Pipelines.nli)
@PIPELINES.register_module(
    Tasks.sentence_similarity, module_name=Pipelines.sentence_similarity)
@PIPELINES.register_module(
    Tasks.text_classification, module_name=Pipelines.text_classification)
@PIPELINES.register_module(
    Tasks.text_classification, module_name=Pipelines.sentiment_classification)
@PIPELINES.register_module(
    Tasks.text_classification, module_name=Pipelines.sentence_similarity)
@PIPELINES.register_module(
    Tasks.sentiment_classification,
    module_name=Pipelines.sentiment_classification)
class TextClassificationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Preprocessor = None,
                 **kwargs):
        """The inference pipeline for all the text classification sub-tasks.

        Args:
            model (`str` or `Model` or module instance): A model instance or a model local dir
                or a model id in the model hub.
            preprocessor (`Preprocessor`, `optional`): A Preprocessor instance.
            first_sequence (`str`, `optional`): The key of the first sentence.
            second_sequence (`str`, `optional`): The key of the second sentence.
            sequence_length (`int`, `optional`): The sequence length.
            id2label (`dict`, `optional`): The id-label mapping.

        Example:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline('text-classification',
                model='damo/nlp_structbert_sentence-similarity_chinese-base')
            >>> input = ('这是个测试', '这也是个测试')
            >>> print(pipeline_ins(input))

        NOTE: Inputs of type 'str' are also supported. In this scenario, the 'first_sequence' and 'second_sequence'
            param will have no affection.
        """
        model = Model.from_pretrained(model) if isinstance(model,
                                                           str) else model

        if preprocessor is None:
            if model.__class__.__name__ == 'OfaForAllTasks':
                preprocessor = Preprocessor.from_pretrained(
                    model_name_or_path=model.model_dir,
                    type=Preprocessors.ofa_tasks_preprocessor,
                    field=Fields.multi_modal)
            else:
                first_sequence = kwargs.pop('first_sequence', 'first_sequence')
                second_sequence = kwargs.pop('second_sequence', None)
                preprocessor = Preprocessor.from_pretrained(
                    model if isinstance(model, str) else model.model_dir,
                    first_sequence=first_sequence,
                    second_sequence=second_sequence,
                    sequence_length=kwargs.pop('sequence_length', 512))

        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.id2label = kwargs.get('id2label')
        if self.id2label is None and hasattr(self.preprocessor, 'id2label'):
            self.id2label = self.preprocessor.id2label

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        if self.model.__class__.__name__ == 'OfaForAllTasks':
            return super().forward(inputs, **forward_params)
        return self.model(**inputs, **forward_params)

    def postprocess(self,
                    inputs: Dict[str, Any],
                    topk: int = 5) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (`Dict[str, Any]` or `TextClassificationModelOutput`): The model output, please check
                the `TextClassificationModelOutput` class for details.
            topk (int): The topk probs to take
        Returns:
            Dict[str, str]: the prediction results.
                scores: The probabilities of each label.
                labels: The real labels.
            Label at index 0 is the smallest probability.
        """
        if self.model.__class__.__name__ == 'OfaForAllTasks':
            return inputs
        else:
            assert self.id2label is not None, 'Cannot convert id to the original label, please pass in the mapping ' \
                                              'as a parameter or make sure the preprocessor has the attribute.'
            logits = inputs[OutputKeys.LOGITS].cpu().numpy()
            if logits.shape[0] == 1:
                logits = logits[0]

            def softmax(logits):
                exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                return exp / exp.sum(axis=-1, keepdims=True)

            probs = softmax(logits)
            num_classes = probs.shape[-1]
            topk = min(topk, num_classes)
            top_indices = np.argpartition(probs, -topk)[-topk:]
            probs = np.take_along_axis(probs, top_indices, axis=-1).tolist()

            def map_to_label(id):
                return self.id2label[id]

            v_func = np.vectorize(map_to_label)
            return {
                OutputKeys.SCORES: probs,
                OutputKeys.LABELS: v_func(top_indices).tolist()
            }
