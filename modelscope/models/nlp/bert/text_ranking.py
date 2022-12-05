# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.utils.checkpoint

from modelscope.metainfo import Models
from modelscope.models import Model
from modelscope.models.builder import MODELS
from modelscope.outputs import AttentionTextClassificationModelOutput
from modelscope.utils import logger as logging
from modelscope.utils.constant import Tasks
from .backbone import BertModel
from .text_classification import BertForSequenceClassification

logger = logging.get_logger()


@MODELS.register_module(Tasks.text_ranking, module_name=Models.bert)
class BertForTextRanking(BertForSequenceClassification):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        neg_sample = kwargs.get('neg_sample', 8)
        self.neg_sample = neg_sample
        setattr(self, self.base_model_prefix,
                BertModel(self.config, add_pooling_layer=True))

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
            return_dict=return_dict)

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

    @classmethod
    def _instantiate(cls, **kwargs):
        """Instantiate the model.

        Args:
            kwargs: Input args.
                    model_dir: The model dir used to load the checkpoint and the label information.
                    num_labels: An optional arg to tell the model how many classes to initialize.
                                    Method will call utils.parse_label_mapping if num_labels not supplied.
                                    If num_labels is not found, the model will use the default setting (1 classes).

        Returns:
            The loaded model, which is initialized by transformers.PreTrainedModel.from_pretrained
        """
        num_labels = kwargs.get('num_labels', 1)
        neg_sample = kwargs.get('neg_sample', 4)
        model_args = {} if num_labels is None else {'num_labels': num_labels}
        if neg_sample is not None:
            model_args['neg_sample'] = neg_sample

        model_dir = kwargs.get('model_dir')
        model = super(Model, cls).from_pretrained(
            pretrained_model_name_or_path=model_dir, **model_args)
        model.model_dir = model_dir
        return model
