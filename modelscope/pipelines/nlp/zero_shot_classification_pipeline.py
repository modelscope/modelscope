# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Union

import torch
from scipy.special import softmax

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import (Preprocessor,
                                      ZeroShotClassificationPreprocessor)
from modelscope.utils.constant import Tasks

__all__ = ['ZeroShotClassificationPipeline']


@PIPELINES.register_module(
    Tasks.zero_shot_classification,
    module_name=Pipelines.zero_shot_classification)
class ZeroShotClassificationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Preprocessor = None,
                 **kwargs):
        """Use `model` and `preprocessor` to create a nlp zero shot classifiction for prediction.

        A zero-shot classification task is used to classify texts by prompts.
        In a normal classification task, model may produce a positive label by the input text
        like 'The ice cream is made of the high quality milk, it is so delicious'
        In a zero-shot task, the sentence is converted to:
        ['The ice cream is made of the high quality milk, it is so delicious', 'This means it is good']
        And:
        ['The ice cream is made of the high quality milk, it is so delicious', 'This means it is bad']
        Then feed these sentences into the model and turn the task to a NLI task(entailment, contradiction),
        and compare the output logits to give the original classification label.


        Args:
            model (str or Model): Supply either a local model dir which supported the task,
            or a model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            sequence_length: Max sequence length in the user's custom scenario. 512 will be used as a default value.

            Example:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline(task='zero-shot-classification',
            >>>    model='damo/nlp_structbert_zero-shot-classification_chinese-base')
            >>> sentence1 = '全新突破 解放军运20版空中加油机曝光'
            >>> labels = ['文化', '体育', '娱乐', '财经', '家居', '汽车', '教育', '科技', '军事']
            >>> template = '这篇文章的标题是{}'
            >>> print(pipeline_ins(sentence1, candidate_labels=labels, hypothesis_template=template))

            To view other examples plese check the tests/pipelines/test_zero_shot_classification.py.
        """
        assert isinstance(model, str) or isinstance(model, Model), \
            'model must be a single str or Model'
        model = model if isinstance(model,
                                    Model) else Model.from_pretrained(model)
        self.entailment_id = 0
        self.contradiction_id = 2
        if preprocessor is None:
            preprocessor = ZeroShotClassificationPreprocessor(
                model.model_dir,
                sequence_length=kwargs.pop('sequence_length', 512))
        model.eval()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        postprocess_params = {}
        if 'candidate_labels' in kwargs:
            candidate_labels = self._parse_labels(
                kwargs.pop('candidate_labels'))
            preprocess_params['candidate_labels'] = candidate_labels
            postprocess_params['candidate_labels'] = candidate_labels
        else:
            raise ValueError('You must include at least one label.')
        preprocess_params['hypothesis_template'] = kwargs.pop(
            'hypothesis_template', '{}')
        postprocess_params['multi_label'] = kwargs.pop('multi_label', False)
        return preprocess_params, {}, postprocess_params

    def _parse_labels(self, labels):
        if isinstance(labels, str):
            labels = labels.replace('，', ',')  # replace cn comma to en comma
            labels = [
                label.strip() for label in labels.split(',') if label.strip()
            ]
        return labels

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        return self.model(**inputs, **forward_params)

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
        logits = inputs[OutputKeys.LOGITS].cpu().numpy()
        if multi_label or len(candidate_labels) == 1:
            logits = logits[..., [self.contradiction_id, self.entailment_id]]
            scores = softmax(logits, axis=-1)[..., 1]
        else:
            logits = logits[..., self.entailment_id]
            scores = softmax(logits, axis=-1)
        reversed_index = list(reversed(scores.argsort()))
        result = {
            OutputKeys.LABELS: [candidate_labels[i] for i in reversed_index],
            OutputKeys.SCORES: [scores[i].item() for i in reversed_index],
        }
        return result
