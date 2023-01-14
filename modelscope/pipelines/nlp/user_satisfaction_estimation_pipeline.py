# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Union

import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import DialogueClassificationUsePreprocessor
from modelscope.utils.constant import Tasks

__all__ = ['UserSatisfactionEstimationPipeline']


@PIPELINES.register_module(
    group_key=Tasks.text_classification,
    module_name=Pipelines.user_satisfaction_estimation)
class UserSatisfactionEstimationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: DialogueClassificationUsePreprocessor = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True):
        """The inference pipeline for the user satisfaction estimation task.

        Args:
            model (str or Model): Supply either a local model dir which supported user satisfaction estimation task, or
            a model id from the model hub, or a torch model instance.
            preprocessor (DialogueClassificationUsePreprocessor): An optional preprocessor instance.
            device (str): device str, should be either cpu, cuda, gpu, gpu:X or cuda:X
            auto_collate (bool): automatically to convert data to tensor or not.

            Example:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline('text-classification',
                model='damo/nlp_user-satisfaction-estimation_chinese')
            >>> input = [('返修退换货咨询|||', '手机有质量问题怎么办|||稍等，我看下', '开不开机了|||',
                       '说话|||谢谢哈')]
            >>> print(pipeline_ins(input))
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate)

        if hasattr(self.preprocessor, 'id2label'):
            self.id2label = self.preprocessor.id2label

        self.model.eval()

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return self.model(**inputs, **forward_params)

    def postprocess(self,
                    inputs: Dict[str, Any],
                    topk: int = None) -> Dict[str, Any]:
        """Process the prediction results

                Args:
                    inputs (`Dict[str, Any]` or `DialogueUseClassificationModelOutput`): The model output, please check
                        the `DialogueUseClassificationModelOutput` class for details.
                    topk (int): The topk probs to take
                Returns:
                    Dict[str, Any]: the prediction results.
                        scores: The probabilities of each label.
                        labels: The real labels.
                    Label at index 0 is the largest probability.
                """
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

        def map_to_label(_id):
            if getattr(self, 'id2label', None) is not None:
                if _id in self.id2label:
                    return self.id2label[_id]
                elif str(_id) in self.id2label:
                    return self.id2label[str(_id)]
                else:
                    raise Exception(
                        f'id {_id} not found in id2label: {self.id2label}')
            else:
                return _id

        v_func = np.vectorize(map_to_label)
        top_indices = v_func(top_indices).tolist()
        probs = list(reversed(probs))
        top_indices = list(reversed(top_indices))

        return {OutputKeys.SCORES: probs, OutputKeys.LABELS: top_indices}
