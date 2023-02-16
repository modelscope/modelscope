from itertools import zip_longest
from typing import Dict

import sacrebleu

from modelscope.metainfo import Metrics
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS, MetricKeys

EVAL_BLEU_ORDER = 4


@METRICS.register_module(group_key=default_group, module_name=Metrics.BLEU)
class BleuMetric(Metric):
    """The metric computation bleu for text generation classes.

    This metric class calculates accuracy for the whole input batches.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_tokenized_bleu = kwargs.get('eval_tokenized_bleu', False)
        self.hyp_name = kwargs.get('hyp_name', 'hyp')
        self.ref_name = kwargs.get('ref_name', 'ref')
        self.refs = list()
        self.hyps = list()

    def add(self, outputs: Dict, inputs: Dict):
        self.refs.extend(inputs[self.ref_name])
        self.hyps.extend(outputs[self.hyp_name])

    def evaluate(self):
        if self.eval_tokenized_bleu:
            bleu = sacrebleu.corpus_bleu(
                self.hyps, list(zip_longest(*self.refs)), tokenize='none')
        else:
            bleu = sacrebleu.corpus_bleu(self.hyps,
                                         list(zip_longest(*self.refs)))
        return {
            MetricKeys.BLEU_4: bleu.score,
        }

    def merge(self, other: 'BleuMetric'):
        self.refs.extend(other.refs)
        self.hyps.extend(other.hyps)

    def __getstate__(self):
        return self.eval_tokenized_bleu, self.hyp_name, self.ref_name, self.refs, self.hyps

    def __setstate__(self, state):
        self.eval_tokenized_bleu, self.hyp_name, self.ref_name, self.refs, self.hyps = state
