# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.nlp import SbertForFaqQuestionAnswering
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import FaqQuestionAnsweringPreprocessor
from modelscope.utils.constant import Tasks

__all__ = ['FaqQuestionAnsweringPipeline']


@PIPELINES.register_module(
    Tasks.faq_question_answering, module_name=Pipelines.faq_question_answering)
class FaqQuestionAnsweringPipeline(Pipeline):

    def __init__(self,
                 model: Union[str, SbertForFaqQuestionAnswering],
                 preprocessor: FaqQuestionAnsweringPreprocessor = None,
                 **kwargs):
        model = model if isinstance(
            model,
            SbertForFaqQuestionAnswering) else Model.from_pretrained(model)
        model.eval()
        if preprocessor is None:
            preprocessor = FaqQuestionAnsweringPreprocessor(
                model.model_dir, **kwargs)
        self.preprocessor = preprocessor
        super(FaqQuestionAnsweringPipeline, self).__init__(
            model=model, preprocessor=preprocessor, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        return pipeline_parameters, pipeline_parameters, pipeline_parameters

    def get_sentence_embedding(self, inputs, max_len=None):
        inputs = self.preprocessor.batch_encode(inputs, max_length=max_len)
        sentence_vecs = self.model.forward_sentence_embedding(inputs)
        sentence_vecs = sentence_vecs.detach().tolist()
        return sentence_vecs

    def forward(self, inputs: [list, Dict[str, Any]],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return self.model(inputs)

    def postprocess(self, inputs: [list, Dict[str, Any]],
                    **postprocess_params) -> Dict[str, Any]:
        scores = inputs['scores']
        labels = []
        for item in scores:
            tmplabels = [
                self.preprocessor.get_label(label_id)
                for label_id in range(len(item))
            ]
            labels.append(tmplabels)

        predictions = []
        for tmp_scores, tmp_labels in zip(scores.tolist(), labels):
            prediction = []
            for score, label in zip(tmp_scores, tmp_labels):
                prediction.append({
                    OutputKeys.LABEL: label,
                    OutputKeys.SCORE: score
                })
            predictions.append(
                list(
                    sorted(
                        prediction,
                        key=lambda d: d[OutputKeys.SCORE],
                        reverse=True)))

        return {OutputKeys.OUTPUT: predictions}
