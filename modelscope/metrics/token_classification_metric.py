# Copyright (c) Alibaba, Inc. and its affiliates.

import importlib
from typing import Dict, List, Optional, Union

import numpy as np

from modelscope.outputs import OutputKeys
from ..metainfo import Metrics
from ..utils.registry import default_group
from ..utils.tensor_utils import torch_nested_detach, torch_nested_numpify
from .base import Metric
from .builder import METRICS, MetricKeys


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.token_cls_metric)
class TokenClassificationMetric(Metric):
    """The metric computation class for token-classification task.

    This metric class uses seqeval to calculate the scores.

    Args:
        label_name(str, `optional`): The key of label column in the 'inputs' arg.
        logit_name(str, `optional`): The key of logits column in the 'inputs' arg.
        return_entity_level_metrics (bool, `optional`):
            Whether to return every label's detail metrics, default False.
        label2id(dict, `optional`): The label2id information to get the token labels.
    """

    def __init__(self,
                 label_name=OutputKeys.LABELS,
                 logit_name=OutputKeys.LOGITS,
                 return_entity_level_metrics=False,
                 label2id=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.return_entity_level_metrics = return_entity_level_metrics
        self.preds = []
        self.labels = []
        self.label2id = label2id
        self.label_name = label_name
        self.logit_name = logit_name

    def add(self, outputs: Dict, inputs: Dict):
        ground_truths = inputs[self.label_name]
        eval_results = outputs[self.logit_name]
        self.preds.append(
            torch_nested_numpify(torch_nested_detach(eval_results)))
        self.labels.append(
            torch_nested_numpify(torch_nested_detach(ground_truths)))

    def evaluate(self):
        label2id = self.label2id
        if label2id is None:
            assert hasattr(self, 'trainer')
            label2id = self.trainer.label2id

        self.id2label = {id: label for label, id in label2id.items()}
        self.preds = np.concatenate(self.preds, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        predictions = np.argmax(self.preds, axis=-1)

        true_predictions = [[
            self.id2label[p] for (p, lb) in zip(prediction, label)
            if lb != -100
        ] for prediction, label in zip(predictions, self.labels)]
        true_labels = [[
            self.id2label[lb] for (p, lb) in zip(prediction, label)
            if lb != -100
        ] for prediction, label in zip(predictions, self.labels)]

        results = self._compute(
            predictions=true_predictions, references=true_labels)
        if self.return_entity_level_metrics:
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f'{key}_{n}'] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                MetricKeys.PRECISION: results[MetricKeys.PRECISION],
                MetricKeys.RECALL: results[MetricKeys.RECALL],
                MetricKeys.F1: results[MetricKeys.F1],
                MetricKeys.ACCURACY: results[MetricKeys.ACCURACY],
            }

    def merge(self, other: 'TokenClassificationMetric'):
        self.preds.extend(other.preds)
        self.labels.extend(other.labels)

    def __getstate__(self):
        return (self.return_entity_level_metrics, self.preds, self.labels,
                self.label2id, self.label_name, self.logit_name)

    def __setstate__(self, state):
        self.__init__()
        (self.return_entity_level_metrics, self.preds, self.labels,
         self.label2id, self.label_name, self.logit_name) = state

    @staticmethod
    def _compute(
        predictions,
        references,
        suffix: bool = False,
        scheme: Optional[str] = None,
        mode: Optional[str] = None,
        sample_weight: Optional[List[int]] = None,
        zero_division: Union[str, int] = 'warn',
    ):
        from seqeval.metrics import accuracy_score, classification_report
        if scheme is not None:
            try:
                scheme_module = importlib.import_module('seqeval.scheme')
                scheme = getattr(scheme_module, scheme)
            except AttributeError:
                raise ValueError(
                    f'Scheme should be one of [IOB1, IOB2, IOE1, IOE2, IOBES, BILOU], got {scheme}'
                )
        report = classification_report(
            y_true=references,
            y_pred=predictions,
            suffix=suffix,
            output_dict=True,
            scheme=scheme,
            mode=mode,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        report.pop('macro avg')
        report.pop('weighted avg')
        overall_score = report.pop('micro avg')

        scores = {
            type_name: {
                MetricKeys.PRECISION: score['precision'],
                MetricKeys.RECALL: score['recall'],
                MetricKeys.F1: score['f1-score'],
                'number': score['support'],
            }
            for type_name, score in report.items()
        }
        scores[MetricKeys.PRECISION] = overall_score['precision']
        scores[MetricKeys.RECALL] = overall_score['recall']
        scores[MetricKeys.F1] = overall_score['f1-score']
        scores[MetricKeys.ACCURACY] = accuracy_score(
            y_true=references, y_pred=predictions)
        return scores
