# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from collections.abc import Mapping
from typing import Any, Dict, List, Tuple, Union

import json
import numpy as np
from transformers import AutoTokenizer

from modelscope.metainfo import Models
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.base import Preprocessor
from modelscope.utils.constant import ModeKeys
from modelscope.utils.hub import get_model_type, parse_label_mapping
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['parse_text_and_label', 'labels_to_id']


def parse_text_and_label(data,
                         mode,
                         first_sequence=None,
                         second_sequence=None,
                         label=None):
    """Parse the input and return the sentences and labels.

    When input type is tuple or list and its size is 2:
    If the pair param is False, data will be parsed as the first_sentence and the label,
    else it will be parsed as the first_sentence and the second_sentence.

    Args:
        data: The input data.
        mode: The mode of the preprocessor
        first_sequence: The key of the first sequence
        second_sequence: The key of the second sequence
        label: The key of the label
    Returns:
        The sentences and labels tuple.
    """
    text_a, text_b, labels = None, None, None
    if isinstance(data, str):
        text_a = data
    elif isinstance(data, tuple) or isinstance(data, list):
        if len(data) == 3:
            text_a, text_b, labels = data
        elif len(data) == 2:
            if mode == ModeKeys.INFERENCE:
                text_a, text_b = data
            else:
                text_a, labels = data
    elif isinstance(data, Mapping):
        text_a = data.get(first_sequence)
        text_b = data.get(second_sequence)
        if label is None or isinstance(label, str):
            labels = data.get(label)
        else:
            labels = [data.get(lb) for lb in label]
    return text_a, text_b, labels


def labels_to_id(labels, output, label2id=None):
    """Turn the labels to id with the type int or float.

    If the original label's type is str or int, the label2id mapping will try to convert it to the final label.
    If the original label's type is float, or the label2id mapping does not exist,
    the original label will be returned.

    Args:
        label2id: An extra label2id mapping. If not provided, the label will not be translated to ids.
        labels: The input labels.
        output: The label id.

    Returns:
        The final labels.
    """

    def label_can_be_mapped(label):
        return isinstance(label, str) or isinstance(label, int)

    try:
        if isinstance(labels, (tuple, list)) and all([label_can_be_mapped(label) for label in labels]) \
                and label2id is not None:
            output[OutputKeys.LABELS] = [
                label2id[label] if label in label2id else label2id[str(label)]
                for label in labels
            ]
        elif label_can_be_mapped(labels) and label2id is not None:
            output[OutputKeys.LABELS] = label2id[
                labels] if labels in label2id else label2id[str(labels)]
        elif labels is not None:
            output[OutputKeys.LABELS] = labels
    except KeyError as e:
        logger.error(
            f'Label {labels} cannot be found in the label mapping {label2id},'
            f'which comes from the user input or the configuration files. '
            f'Please consider matching your labels with this mapping.')
        raise e
