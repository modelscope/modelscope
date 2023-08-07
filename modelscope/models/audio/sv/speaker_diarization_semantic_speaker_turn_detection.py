# copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from modelscope.metainfo import Heads, Models, TaskModels
from modelscope.models.base import TorchHead
from modelscope.models.builder import HEADS, MODELS
from modelscope.models.nlp.task_models.task_model import EncoderModel
from modelscope.outputs import (AttentionTokenClassificationModelOutput,
                                ModelOutputBase, OutputKeys)
from modelscope.utils import logger as logging
from modelscope.utils.constant import Tasks
from modelscope.utils.hub import parse_label_mapping

logger = logging.get_logger()


@HEADS.register_module(
    Tasks.speaker_diarization_semantic_speaker_turn_detection,
    module_name=Heads.token_classification)
class TokenClassificationHead(TorchHead):

    def __init__(self,
                 hidden_size=768,
                 classifier_dropout=0.1,
                 num_labels=None,
                 **kwargs):
        super().__init__(
            num_labels=num_labels,
            classifier_dropout=classifier_dropout,
            hidden_size=hidden_size,
        )
        assert num_labels is not None
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self,
                inputs: ModelOutputBase,
                attention_mask=None,
                labels=None,
                **kwargs):
        sequence_output = inputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, attention_mask, labels)

        return AttentionTokenClassificationModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=inputs.hidden_states,
            attentions=inputs.attentions)

    def compute_loss(self, logits: torch.Tensor, attention_mask,
                     labels) -> torch.Tensor:
        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels))
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss


@MODELS.register_module(
    Tasks.speaker_diarization_semantic_speaker_turn_detection,
    module_name=TaskModels.token_classification)
class ModelForTokenClassification(EncoderModel):
    task = Tasks.token_classification
    head_type = Heads.token_classification
    base_model_prefix = 'bert'
    override_base_model_prefix = True

    def __init__(self, model_dir: str, *args, **kwargs):
        self.id2label = {}

        num_labels = kwargs.get('num_labels')
        if num_labels is None:
            label2id = parse_label_mapping(model_dir)
            if label2id is not None and len(label2id) > 0:
                num_labels = len(label2id)
            self.id2label = {id: label for label, id in label2id.items()}
        kwargs['num_labels'] = num_labels
        super(ModelForTokenClassification,
              self).__init__(model_dir, *args, **kwargs)

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

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                offset_mapping=None,
                label_mask=None,
                **kwargs):
        kwargs['offset_mapping'] = offset_mapping
        kwargs['label_mask'] = label_mask
        outputs = super().forward(input_ids, attention_mask, token_type_ids,
                                  position_ids, head_mask, inputs_embeds,
                                  labels, output_attentions,
                                  output_hidden_states, **kwargs)

        outputs.offset_mapping = offset_mapping
        outputs.label_mask = label_mask

        return outputs


@MODELS.register_module(
    Tasks.speaker_diarization_semantic_speaker_turn_detection,
    module_name=Models.bert)
class BertForTokenClassification(ModelForTokenClassification):
    base_model_type = 'bert'
