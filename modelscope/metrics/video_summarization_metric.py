# Part of the implementation is borrowed and modified from PGL-SUM,
# publicly available at https://github.com/e-apostolidis/PGL-SUM

from typing import Dict

import numpy as np

from modelscope.metainfo import Metrics
from modelscope.models.cv.video_summarization.summarizer import \
    generate_summary
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS, MetricKeys


def evaluate_summary(predicted_summary, user_summary, eval_method):
    """ Compare the predicted summary with the user defined one(s).

    :param ndarray predicted_summary: The generated summary from our model.
    :param ndarray user_summary: The user defined ground truth summaries (or summary).
    :param str eval_method: The proposed evaluation method; either 'max' (SumMe) or 'avg' (TVSum).
    :return: The reduced fscore based on the eval_method
    """
    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G

        # Compute precision, recall, f-score
        precision = sum(overlapped) / sum(S)
        recall = sum(overlapped) / sum(G)
        if precision + recall == 0:
            f_scores.append(0)
        else:
            f_score = 2 * precision * recall * 100 / (precision + recall)
            f_scores.append(f_score)

    if eval_method == 'max':
        return max(f_scores)
    else:
        return sum(f_scores) / len(f_scores)


def calculate_f_score(outputs: Dict, inputs: Dict):
    scores = outputs['scores']
    scores = scores.squeeze(0).cpu().numpy().tolist()
    user_summary = inputs['user_summary'].cpu().numpy()[0]
    sb = inputs['change_points'].cpu().numpy()[0]
    n_frames = inputs['n_frames'].cpu().numpy()[0]
    positions = inputs['positions'].cpu().numpy()[0]
    summary = generate_summary([sb], [scores], [n_frames], [positions])[0]
    f_score = evaluate_summary(summary, user_summary, 'avg')
    return f_score


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.video_summarization_metric)
class VideoSummarizationMetric(Metric):
    """The metric for video summarization task.
    """

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def add(self, outputs: Dict, inputs: Dict):
        self.outputs.append(outputs)
        self.inputs.append(inputs)

    def evaluate(self):
        f_scores = [
            calculate_f_score(output, input)
            for output, input in zip(self.outputs, self.inputs)
        ]

        return {MetricKeys.FScore: sum(f_scores) / len(f_scores)}

    def merge(self, other: 'VideoSummarizationMetric'):
        self.inputs.extend(other.inputs)
        self.outputs.extend(other.outputs)

    def __getstate__(self):
        return self.inputs, self.outputs

    def __setstate__(self, state):
        self.inputs, self.outputs = state
