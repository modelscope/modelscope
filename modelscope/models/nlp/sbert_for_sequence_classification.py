import os
from typing import Any, Dict

import json
import numpy as np
import torch
from sofa.models.sbert.modeling_sbert import SbertModel, SbertPreTrainedModel
from torch import nn

from modelscope.models import TorchModel


class SbertTextClassfier(SbertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.encoder = SbertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                labels=None,
                **kwargs):
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            return_dict=None,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {'logits': logits, 'loss': loss}
        return {'logits': logits}

    def build(**kwags):
        return SbertTextClassfier.from_pretrained(model_dir, **model_args)


class SbertForSequenceClassificationBase(TorchModel):

    def __init__(self, model_dir: str, model_args=None, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        if model_args is None:
            model_args = {}
        self.model = SbertTextClassfier.from_pretrained(
            model_dir, **model_args)
        self.id2label = {}
        self.label_path = os.path.join(self.model_dir, 'label_mapping.json')
        if os.path.exists(self.label_path):
            with open(self.label_path) as f:
                self.label_mapping = json.load(f)
            self.id2label = {
                idx: name
                for name, idx in self.label_mapping.items()
            }

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()

    def forward(self, input: Dict[str, Any]) -> Dict[str, np.ndarray]:
        input_ids = torch.tensor(input['input_ids'], dtype=torch.long)
        token_type_ids = torch.tensor(
            input['token_type_ids'], dtype=torch.long)
        return self.model.forward(input_ids, token_type_ids)

    def postprocess(self, input, **kwargs):
        logits = input['logits']
        probs = logits.softmax(-1).cpu().numpy()
        pred = logits.argmax(-1).cpu().numpy()
        logits = logits.cpu().numpy()
        res = {'predictions': pred, 'probabilities': probs, 'logits': logits}
        return res
