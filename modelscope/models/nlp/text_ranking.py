# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.nlp import SbertForSequenceClassification
from modelscope.models.nlp.structbert import SbertPreTrainedModel
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks

__all__ = ['TextRanking']


@MODELS.register_module(Tasks.text_ranking, module_name=Models.bert)
class TextRanking(SbertForSequenceClassification, SbertPreTrainedModel):
    base_model_prefix: str = 'bert'
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r'position_ids']

    def __init__(self, config, model_dir, *args, **kwargs):
        if hasattr(config, 'base_model_prefix'):
            TextRanking.base_model_prefix = config.base_model_prefix
        super().__init__(config, model_dir)
        self.train_batch_size = kwargs.get('train_batch_size', 4)
        self.register_buffer(
            'target_label',
            torch.zeros(self.train_batch_size, dtype=torch.long))

    def build_base_model(self):
        from .structbert import SbertModel
        return SbertModel(self.config, add_pooling_layer=True)

    def forward(self, input: Dict[str, Any]) -> Dict[str, np.ndarray]:
        outputs = self.base_model.forward(**input)

        # backbone model should return pooled_output as its second output
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if self.base_model.training:
            scores = logits.view(self.train_batch_size, -1)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(scores, self.target_label)
            return {OutputKeys.LOGITS: logits, OutputKeys.LOSS: loss}
        return {OutputKeys.LOGITS: logits}

    def sigmoid(self, logits):
        return np.exp(logits) / (1 + np.exp(logits))

    def postprocess(self, inputs: Dict[str, np.ndarray],
                    **kwargs) -> Dict[str, np.ndarray]:
        logits = inputs['logits'].squeeze(-1).detach().cpu().numpy()
        logits = self.sigmoid(logits).tolist()
        result = {OutputKeys.SCORES: logits}
        return result

    @classmethod
    def _instantiate(cls, **kwargs):
        """Instantiate the model.

        @param kwargs: Input args.
                    model_dir: The model dir used to load the checkpoint and the label information.
                    num_labels: An optional arg to tell the model how many classes to initialize.
                                    Method will call utils.parse_label_mapping if num_labels not supplied.
                                    If num_labels is not found, the model will use the default setting (1 classes).
        @return: The loaded model, which is initialized by transformers.PreTrainedModel.from_pretrained
        """

        num_labels = kwargs.get('num_labels', 1)
        model_args = {} if num_labels is None else {'num_labels': num_labels}

        return super(SbertPreTrainedModel, TextRanking).from_pretrained(
            pretrained_model_name_or_path=kwargs.get('model_dir'),
            model_dir=kwargs.get('model_dir'),
            **model_args)
