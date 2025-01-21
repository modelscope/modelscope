# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from transformers import BertModel

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.outputs import DialogueUserSatisfactionEstimationModelOutput
from modelscope.utils.compatible_with_transformers import \
    compatible_position_ids
from modelscope.utils.constant import ModelFile, Tasks
from .transformer import TransformerEncoder

__all__ = ['UserSatisfactionEstimation']


@MODELS.register_module(Tasks.text_classification, module_name=Models.use)
class UserSatisfactionEstimation(TorchModel):

    def __init__(self,
                 model_dir: str,
                 bert_name: str = None,
                 device: str = None,
                 **kwargs):
        """initialize the user satisfaction estimation model from the `model_dir` path. The default preprocessor
        for this task is DialogueClassificationUsePreprocessor.

        Args:
            model_dir: The model dir containing the model.
            bert_name: The pretrained model, default bert-base-chinese
            device: The device of running model, default cpu
        """
        super().__init__(model_dir, **kwargs)
        self.model_dir = model_dir
        self.bert_name = bert_name if bert_name is not None else 'bert-base-chinese'
        self.device = 'cpu'
        if device is not None and torch.cuda.is_available():
            self.device = device
        self.model = self.init_model()
        model_ckpt = os.path.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        stats_dict = torch.load(model_ckpt, map_location=torch.device('cpu'))
        compatible_position_ids(stats_dict,
                                'private.bert.embeddings.position_ids')
        self.model.load_state_dict(stats_dict)

    def init_model(self):
        configs = {
            'bert_name': self.bert_name,
            'cache_dir': self.model_dir,
            'dropout': 0.1
        }
        model = USE(configs)
        return model

    def forward(
        self, input_ids: Tensor
    ) -> Union[DialogueUserSatisfactionEstimationModelOutput, Dict[str,
                                                                   Tensor]]:
        """Compute the logits of satisfaction polarities for a dialogue.

        Args:
           input_ids (Tensor): the preprocessed dialogue input
        Returns:
           output (Dict[str, Any] or DialogueUserSatisfactionEstimationModelOutput): The results of user satisfaction.

        Example:
            >>> {'logits': tensor([[-2.1795,  1.1323,  1.8605]])}
        """
        logits = self.model(input_ids)
        return DialogueUserSatisfactionEstimationModelOutput(logits=logits)


def init_params(model):
    for name, param in model.named_parameters():
        if param.data.dim() > 1:
            xavier_uniform_(param.data)
        else:
            pass


def universal_sentence_embedding(sentences, mask, sqrt=True):
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1),
        mask.float().unsqueeze(-1)).squeeze(-1)
    divisor = (mask.sum(dim=1).view(-1, 1).float())
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums


class BERTBackbone(nn.Module):

    def __init__(self, **config):
        super().__init__()
        bert_name = config.get('bert_name', 'bert-base-chinese')
        cache_dir = config.get('cache_dir')
        self.bert = BertModel.from_pretrained(bert_name, cache_dir=cache_dir)
        self.d_model = 768 * 2

    def forward(self, input_ids):
        attention_mask = input_ids.ne(0).detach()
        outputs = self.bert(input_ids, attention_mask)
        h = universal_sentence_embedding(outputs[0], attention_mask)
        cls = outputs[1]
        out = torch.cat([cls, h], dim=-1)
        return out


class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, din):
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        return dout


class USE(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.drop_out = nn.Dropout(args['dropout'])
        self.private = BERTBackbone(
            bert_name=args['bert_name'], cache_dir=args['cache_dir'])
        d_model = self.private.d_model
        self.encoder = TransformerEncoder(d_model, d_model * 2, 8, 2, 0.1)
        self.content_gru = nn.GRU(
            d_model,
            d_model,
            num_layers=1,
            bidirectional=False,
            batch_first=True)
        self.sat_classifier = nn.Linear(d_model, 3)

        self.U_c = nn.Linear(d_model, d_model)
        self.w_c = nn.Linear(d_model, 1, bias=False)

        init_params(self.encoder)
        init_params(self.sat_classifier)
        init_params(self.U_c)
        init_params(self.w_c)

    def forward(self, input_ids):
        self.content_gru.flatten_parameters()
        batch_size, dialog_len, utt_len = input_ids.size()
        attention_mask = input_ids[:, :, 0].squeeze(-1).ne(0).detach()
        input_ids = input_ids.view(-1, utt_len)

        private_out = self.private(input_ids=input_ids)
        private_out = private_out.view(batch_size, dialog_len, -1)
        H = self.encoder(private_out, attention_mask)
        H = self.drop_out(H)

        H, _ = self.content_gru(H)
        att_c = self.w_c(torch.tanh(self.U_c(H))).squeeze(-1)
        att_c = F.softmax(
            att_c.masked_fill(mask=~attention_mask, value=-np.inf), dim=1)
        hidden = torch.bmm(H.permute(0, 2, 1), att_c.unsqueeze(-1)).squeeze(-1)

        sat_res = self.sat_classifier(hidden)
        return sat_res
