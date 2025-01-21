# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from modelscope.metainfo import Models
from modelscope.models import Model
from modelscope.models.builder import MODELS
from modelscope.models.nlp.ponet import PoNetConfig
from modelscope.outputs import AttentionTokenClassificationModelOutput
from modelscope.utils.constant import Tasks
from .backbone import BertModel, BertPreTrainedModel
from .configuration import BertConfig

__all__ = ['BertForDocumentSegmentation']


@MODELS.register_module(
    Tasks.document_segmentation, module_name=Models.bert_for_ds)
class BertForDocumentSegmentation(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r'pooler']

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.sentence_pooler_type = None
        self.bert = BertModel(config, add_pooling_layer=False)

        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.class_weights = None
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                sentence_attention_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if self.sentence_pooler_type is not None:
            raise NotImplementedError
        else:
            sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
            if sentence_attention_mask is not None:
                active_loss = sentence_attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return AttentionTokenClassificationModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def _instantiate(cls, model_dir, model_config: Dict[str, Any], **kwargs):
        if model_config['type'] == 'bert':
            config = BertConfig.from_pretrained(model_dir, num_labels=2)
        elif model_config['type'] == 'ponet':
            config = PoNetConfig.from_pretrained(model_dir, num_labels=2)
        else:
            raise ValueError(
                f'Expected config type bert and ponet, which is : {model_config["type"]}'
            )
        model = super(Model, cls).from_pretrained(
            model_dir, from_tf=False, config=config)
        model.model_dir = model_dir
        model.model_cfg = model_config
        return model
