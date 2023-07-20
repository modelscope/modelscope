# copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn.functional as F
from torch import nn

from modelscope.metainfo import Heads, Models, TaskModels
from modelscope.models.base import TorchHead
from modelscope.models.builder import HEADS, MODELS
from modelscope.models.nlp.task_models.task_model import EncoderModel
from modelscope.outputs import (AttentionTextClassificationModelOutput,
                                ModelOutputBase, OutputKeys)
from modelscope.utils import logger as logging
from modelscope.utils.constant import Tasks
from modelscope.utils.hub import parse_label_mapping

logger = logging.get_logger()


@HEADS.register_module(
    Tasks.speaker_diarization_dialogue_detection,
    module_name=Heads.text_classification)
class TextClassificationHead(TorchHead):

    def __init__(self,
                 hidden_size=768,
                 classifier_dropout=0.1,
                 num_labels=None,
                 **kwargs):
        super().__init__(
            hidden_size=hidden_size,
            classifier_dropout=classifier_dropout,
            num_labels=num_labels,
        )
        assert num_labels is not None
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self,
                inputs: ModelOutputBase,
                attention_mask=None,
                labels=None,
                **kwargs):
        pooler_output = inputs.pooler_output
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)

        return AttentionTextClassificationModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=inputs.hidden_states,
            attentions=inputs.attentions,
        )

    def compute_loss(self, logits: torch.Tensor, labels) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


@MODELS.register_module(
    Tasks.speaker_diarization_dialogue_detection,
    module_name=TaskModels.text_classification)
class ModelForTextClassification(EncoderModel):
    task = Tasks.text_classification

    head_type = Heads.text_classification

    def __init__(self, model_dir: str, *args, **kwargs):
        # get the num_labels from label_mapping.json
        self.id2label = {}

        # get the num_labels
        num_labels = kwargs.get('num_labels')
        if num_labels is None:
            label2id = parse_label_mapping(model_dir)
            if label2id is not None and len(label2id) > 0:
                num_labels = len(label2id)
            self.id2label = {id: label for label, id in label2id.items()}
        kwargs['num_labels'] = num_labels
        super().__init__(model_dir, *args, **kwargs)

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


@MODELS.register_module(
    Tasks.speaker_diarization_dialogue_detection, module_name=Models.bert)
class BertForSequenceClassification(ModelForTextClassification):
    base_model_type = 'bert'
