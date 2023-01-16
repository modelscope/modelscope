# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.utils.checkpoint

from modelscope.metainfo import Models
from modelscope.models import Model
from modelscope.models.builder import MODELS
from modelscope.outputs import AttentionTextClassificationModelOutput
from modelscope.utils import logger as logging
from modelscope.utils.constant import Tasks
from .backbone import MGeo, MGeoPreTrainedModel

logger = logging.get_logger()


@MODELS.register_module(Tasks.text_ranking, module_name=Models.mgeo)
class MGeoForTextRanking(MGeoPreTrainedModel):

    def __init__(self,
                 config,
                 finetune_mode: str = 'single-modal',
                 gis_num: int = 1,
                 *args,
                 **kwargs):
        super().__init__(config)
        neg_sample = kwargs.get('neg_sample', 8)
        eval_neg_sample = kwargs.get('eval_neg_sample', 8)
        self.neg_sample = neg_sample
        self.eval_neg_sample = eval_neg_sample
        setattr(
            self, self.base_model_prefix,
            MGeo(self.config, finetune_mode, gis_num, add_pooling_layer=True))
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None
            else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

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
                gis_list=None,
                gis_tp=None,
                *args,
                **kwargs) -> AttentionTextClassificationModelOutput:
        outputs = self.base_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            gis_list=gis_list,
            gis_tp=gis_tp,
        )

        # backbone model should return pooled_output as its second output
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if self.base_model.training:
            scores = logits.view(-1, self.neg_sample + 1)
            batch_size = scores.size(0)
            loss_fct = torch.nn.CrossEntropyLoss()
            target_label = torch.zeros(
                batch_size, dtype=torch.long, device=scores.device)
            loss = loss_fct(scores, target_label)
            return AttentionTextClassificationModelOutput(
                loss=loss,
                logits=logits,
            )
        return AttentionTextClassificationModelOutput(logits=logits, )
