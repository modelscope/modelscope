# Copyright (c) Alibaba, Inc. and its affiliates.

import importlib
from typing import Dict, List, Union

from pandas import DataFrame

from modelscope.metainfo import Metrics
from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS, MetricKeys
from modelscope.models.nlp.unite.configuration import InputFormat
from modelscope.utils.logger import get_logger
from modelscope.utils.registry import default_group

logger = get_logger()


@METRICS.register_module(
    group_key=default_group, module_name=Metrics.translation_evaluation_metric)
class TranslationEvaluationMetric(Metric):
    r"""The metric class for translation evaluation.

    """

    def __init__(self, gap_threshold: float = 25.0):
        r"""Build a translation evaluation metric, following the designed
            Kendall's tau correlation from WMT Metrics Shared Task competitions.

            Args:
                gap_threshold: The score gap denoting the available hypothesis pair.

            Returns:
                A metric for translation evaluation.
        """
        self.gap_threshold = gap_threshold

        self.lp = list()
        self.segment_id = list()
        self.raw_score = list()
        self.score = list()
        self.input_format = list()

    def clear(self) -> None:
        r"""Clear all the stored variables.
        """
        self.lp.clear()
        self.segment_id.clear()
        self.raw_score.clear()
        self.input_format.clear()

        self.score.clear()

        return

    def add(self, outputs: Dict[str, List[float]],
            inputs: Dict[str, List[Union[float, int]]]) -> None:
        r"""Collect the related results for processing.

            Args:
                outputs: Dict containing 'scores'
                inputs: Dict containing 'labels' and 'segment_ids'

        """

        self.lp += inputs['lp']
        self.segment_id += inputs['segment_id']
        self.raw_score += inputs['raw_score']
        self.input_format += inputs['input_format']

        self.score += outputs['score']

        return

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        r"""Compute the Kendall's tau correlation.

            Returns:
                A dict denoting Kendall's tau correlation.

        """

        data = {
            'lp': self.lp,
            'segment_id': self.segment_id,
            'raw_score': self.raw_score,
            'input_format': self.input_format,
            'score': self.score
        }
        data = DataFrame(data=data)
        correlation = dict()

        for input_format in data.input_format.unique():
            logger.info('Evaluation results for %s input format'
                        % input_format.value)
            input_format_data = data[data.input_format == input_format]

            temp_correlation = dict()

            for lp in sorted(input_format_data.lp.unique()):
                sub_data = input_format_data[input_format_data.lp == lp]
                temp_correlation[input_format.value + '_'
                                 + lp] = self.compute_kendall_tau(sub_data)
                logger.info(
                    '\t%s: %f' %
                    (lp,
                     temp_correlation[input_format.value + '_' + lp] * 100))

            avg_correlation = sum(
                temp_correlation.values()) / len(temp_correlation)
            correlation[input_format.value + '_avg'] = avg_correlation
            logger.info('Average evaluation result for %s input format: %f' %
                        (input_format.value, avg_correlation))
            logger.info('')
            correlation.update(temp_correlation)

        return correlation

    def merge(self, other: 'TranslationEvaluationMetric') -> None:
        r"""Merge the predictions from other TranslationEvaluationMetric objects.

            Args:
                other: Another TranslationEvaluationMetric object.

        """

        self.lp += other.lp
        self.segment_id += other.segment_ids
        self.raw_score += other.raw_score
        self.input_format += other.input_format

        self.score += other.score

        return

    def compute_kendall_tau(self, csv_data: DataFrame) -> float:
        r"""Compute kendall's tau correlation.

            Args:
                csv_data: The pandas dataframe.

            Returns:
                float: THe kendall's Tau correlation.

        """
        concor = discor = 0

        for segment_id in sorted(csv_data.segment_id.unique()):
            group_csv_data = csv_data[csv_data.segment_id == segment_id]

            examples = group_csv_data.to_dict('records')

            for i in range(0, len(examples)):
                for j in range(i + 1, len(examples)):
                    if self.raw_score[i] - self.raw_score[
                            j] >= self.gap_threshold:
                        if self.score[i] > self.score[j]:
                            concor += 1
                        elif self.score[i] < self.score[j]:
                            discor += 1
                    elif self.raw_score[i] - self.raw_score[
                            j] <= -self.gap_threshold:
                        if self.score[i] < self.score[j]:
                            concor += 1
                        elif self.score[i] > self.score[j]:
                            discor += 1

        if concor + discor == 0:
            logger.warning(
                'We don\'t have available pairs when evaluation. '
                'Marking the kendall tau correlation as the lowest value (-1.0).'
            )
            return -1.0
        else:
            return (concor - discor) / (concor + discor)
