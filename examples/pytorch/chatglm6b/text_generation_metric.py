# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict, Iterable, List

import jieba
import numpy as np
from nltk.translate.bleu_score import (SmoothingFunction, corpus_bleu,
                                       sentence_bleu)
from rouge import Rouge

from modelscope.metainfo import Metrics
from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS, MetricKeys
from modelscope.utils.chinese_utils import rebuild_chinese_str
from modelscope.utils.registry import default_group


@METRICS.register_module(group_key=default_group, module_name='chatglm')
class TextGenerationMetric(Metric):

    def __init__(self, target_text='tgts', pred_text='preds'):
        self.preds: List[str] = []
        self.tgts: List[str] = []
        self.rouge = Rouge()
        self.target_text = target_text
        self.pred_text = pred_text

    def add(self, outputs: Dict[str, List[str]], inputs: Dict[str, List[str]]):
        ground_truths = inputs[self.target_text]
        eval_results = outputs[self.pred_text]
        for truth in ground_truths:
            self.tgts.append(truth)
        for result in eval_results:
            self.preds.append(result)

    def _check(self, pred: str, tgt: str) -> bool:

        def remove_useless(string: str) -> str:
            return string.replace(' ', '').replace('.', '')

        return len(remove_useless(pred)) != 0 and len(remove_useless(tgt)) != 0

    def evaluate(self):
        preds, labels = self.preds, self.tgts
        if isinstance(preds, tuple):
            preds = preds[0]

        score_dict = {
            'rouge-1': [],
            'rouge-2': [],
            'rouge-l': [],
            'bleu-4': []
        }
        for pred, label in zip(preds, labels):
            hypothesis = list(jieba.cut(pred))
            if len(hypothesis) == 0:
                hypothesis = ['</s>']
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis),
                                      ' '.join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v['f'] * 100, 4))
            bleu_score = sentence_bleu(
                [list(label)],
                list(pred),
                smoothing_function=SmoothingFunction().method3)
            score_dict['bleu-4'].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    def merge(self, other: 'TextGenerationMetric'):
        self.preds.extend(other.preds)
        self.tgts.extend(other.tgts)

    def __getstate__(self):
        return self.preds, self.tgts

    def __setstate__(self, state):
        self.__init__()
        self.preds, self.tgts = state
