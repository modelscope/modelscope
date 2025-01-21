# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np

from modelscope.metainfo import Heads, TaskModels
from modelscope.models.builder import MODELS
from modelscope.models.nlp.task_models.task_model import EncoderModel
from modelscope.utils.constant import Tasks
from modelscope.utils.hub import parse_label_mapping

__all__ = ['ModelForTextRanking']


@MODELS.register_module(
    Tasks.text_ranking, module_name=TaskModels.text_ranking)
class ModelForTextRanking(EncoderModel):
    task = Tasks.text_ranking

    # The default base head type is text-ranking for this head
    head_type = Heads.text_ranking

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the sequence classification model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        # get the num_labels from label_mapping.json
        self.id2label = {}

        # get the num_labels
        num_labels = kwargs.get('num_labels')
        if num_labels is None:
            label2id = parse_label_mapping(model_dir)
            if label2id is not None and len(label2id) > 0:
                num_labels = len(label2id)
                self.id2label = {id: label for label, id in label2id.items()}
            elif label2id is None:
                num_labels = 1
        kwargs['num_labels'] = num_labels
        super().__init__(model_dir, *args, **kwargs)

    def parse_encoder_cfg(self):
        encoder_cfg = super().parse_encoder_cfg()
        encoder_cfg['add_pooling_layer'] = True
        return encoder_cfg

    def parse_head_cfg(self):
        head_cfg = super().parse_head_cfg()
        if hasattr(head_cfg, 'classifier_dropout'):
            head_cfg['classifier_dropout'] = (
                head_cfg.classifier_dropout if head_cfg['classifier_dropout']
                is not None else head_cfg.hidden_dropout_prob)
        else:
            head_cfg['classifier_dropout'] = head_cfg.hidden_dropout_prob
        head_cfg['num_labels'] = self.config.num_labels
        return head_cfg
