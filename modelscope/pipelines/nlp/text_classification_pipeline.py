# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Union

import numpy as np
import torch

from modelscope.metainfo import Pipelines, Preprocessors
from modelscope.models.base import Model
from modelscope.outputs import OutputKeys, TextClassificationModelOutput
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.util import batch_process
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Fields, ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


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
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 **kwargs):
        """The inference pipeline for all the text classification sub-tasks.

        Args:
            model (`str` or `Model` or module instance): A model instance or a model local dir
                or a model id in the model hub.
            preprocessor (`Preprocessor`, `optional`): A Preprocessor instance.
            kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.

        Examples:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline('text-classification',
                model='damo/nlp_structbert_sentence-similarity_chinese-base')
            >>> input = ('这是个测试', '这也是个测试')
            >>> print(pipeline_ins(input))
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate)

        assert isinstance(self.model, Model), \
            f'please check whether model config exists in {ModelFile.CONFIGURATION}'

        if preprocessor is None:
            if self.model.__class__.__name__ == 'OfaForAllTasks':
                self.preprocessor = Preprocessor.from_pretrained(
                    model_name_or_path=self.model.model_dir,
                    type=Preprocessors.ofa_tasks_preprocessor,
                    field=Fields.multi_modal,
                    **kwargs)
            else:
                first_sequence = kwargs.pop('first_sequence', 'first_sequence')
                second_sequence = kwargs.pop('second_sequence', None)
                sequence_length = kwargs.pop('sequence_length', 512)
                self.preprocessor = Preprocessor.from_pretrained(
                    self.model.model_dir, **{
                        'first_sequence': first_sequence,
                        'second_sequence': second_sequence,
                        'sequence_length': sequence_length,
                        **kwargs
                    })

        if hasattr(self.preprocessor, 'id2label'):
            self.id2label = self.preprocessor.id2label

    def _batch(self, data):
        if self.model.__class__.__name__ == 'OfaForAllTasks':
            return batch_process(self.model, data)
        else:
            return super(TextClassificationPipeline, self)._batch(data)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        if self.model.__class__.__name__ == 'OfaForAllTasks':
            with torch.no_grad():
                return super().forward(inputs, **forward_params)
        return self.model(**inputs, **forward_params)

    def postprocess(self,
                    inputs: Union[Dict[str, Any],
                                  TextClassificationModelOutput],
                    topk: int = None) -> Dict[str, Any]:
        """Process the prediction results

        Args:
            inputs (`Dict[str, Any]` or `TextClassificationModelOutput`): The model output, please check
                the `TextClassificationModelOutput` class for details.
            topk (int): The topk probs to take
        Returns:
            Dict[str, Any]: the prediction results.
                scores: The probabilities of each label.
                labels: The real labels.
            Label at index 0 is the smallest probability.
        """
        if self.model.__class__.__name__ == 'OfaForAllTasks':
            return inputs
        else:
            if getattr(self, 'id2label', None) is None:
                logger.warning(
                    'The id2label mapping is None, will return original ids.')
            logits = inputs[OutputKeys.LOGITS].cpu().numpy()
            if logits.shape[0] == 1:
                logits = logits[0]

            def softmax(logits):
                exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                return exp / exp.sum(axis=-1, keepdims=True)

            probs = softmax(logits)
            num_classes = probs.shape[-1]
            topk = min(topk, num_classes) if topk is not None else num_classes
            top_indices = np.argpartition(probs, -topk)[-topk:]
            probs = np.take_along_axis(probs, top_indices, axis=-1).tolist()

            def map_to_label(id):
                if getattr(self, 'id2label', None) is not None:
                    if id in self.id2label:
                        return self.id2label[id]
                    elif str(id) in self.id2label:
                        return self.id2label[str(id)]
                    else:
                        raise Exception(
                            f'id {id} not found in id2label: {self.id2label}')
                else:
                    return id

            v_func = np.vectorize(map_to_label)
            top_indices = v_func(top_indices).tolist()
            probs = list(reversed(probs))
            top_indices = list(reversed(top_indices))
            return {OutputKeys.SCORES: probs, OutputKeys.LABELS: top_indices}
