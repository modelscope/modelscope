# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch

from modelscope.metainfo import Heads, Models, TaskModels
from modelscope.models.builder import MODELS
from modelscope.models.nlp.task_models.task_model import EncoderModel
from modelscope.outputs import (AttentionTokenClassificationModelOutput,
                                OutputKeys)
from modelscope.utils.constant import Tasks
from modelscope.utils.hub import parse_label_mapping

__all__ = ['ModelForTokenClassification', 'ModelForTokenClassificationWithCRF']


@MODELS.register_module(
    Tasks.token_classification, module_name=TaskModels.token_classification)
@MODELS.register_module(
    Tasks.part_of_speech, module_name=TaskModels.token_classification)
@MODELS.register_module(
    Tasks.named_entity_recognition,
    module_name=Models.token_classification_for_ner)
class ModelForTokenClassification(EncoderModel):
    task = Tasks.token_classification

    # The default base head type is token-classification for this head
    head_type = Heads.token_classification

    # The default base model prefix for this task is encoder and ignore the base model prefix
    base_model_prefix = 'encoder'
    override_base_model_prefix = True

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


@MODELS.register_module(Tasks.transformer_crf, module_name=Models.tcrf)
@MODELS.register_module(Tasks.token_classification, module_name=Models.tcrf)
@MODELS.register_module(
    Tasks.token_classification, module_name=Models.tcrf_wseg)
@MODELS.register_module(
    Tasks.named_entity_recognition, module_name=Models.tcrf)
@MODELS.register_module(Tasks.part_of_speech, module_name=Models.tcrf)
@MODELS.register_module(Tasks.word_segmentation, module_name=Models.tcrf)
@MODELS.register_module(Tasks.word_segmentation, module_name=Models.tcrf_wseg)
class ModelForTokenClassificationWithCRF(ModelForTokenClassification):
    head_type = Heads.transformer_crf
    base_model_prefix = 'encoder'

    def postprocess(self, inputs, **kwargs):
        predicts = self.head.decode(inputs['logits'], inputs['label_mask'])
        offset_mapping = inputs['offset_mapping']
        mask = inputs['label_mask']

        # revert predicts to original position with respect of label mask
        masked_predict = torch.zeros_like(predicts)
        for i in range(len(mask)):
            masked_lengths = mask[i].sum(-1).long().cpu().item()
            selected_predicts = torch.narrow(
                predicts[i], 0, 0,
                masked_lengths)  # index_select only move loc, not resize
            mask_position = mask[i].bool()
            masked_predict[i][mask_position] = selected_predicts
        predicts = masked_predict

        return AttentionTokenClassificationModelOutput(
            loss=None,
            logits=None,
            hidden_states=None,
            attentions=None,
            label_mask=mask,
            offset_mapping=offset_mapping,
            predictions=predicts,
        )
