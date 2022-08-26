import math
import os
from collections import namedtuple
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.models.nlp.structbert import SbertConfig, SbertModel
from modelscope.models.nlp.task_models.task_model import BaseTaskModel
from modelscope.utils.config import Config, ConfigFields
from modelscope.utils.constant import ModelFile, Tasks

__all__ = ['SbertForFaqQuestionAnswering']


class SbertForFaqQuestionAnsweringBase(BaseTaskModel):
    """base class for faq models
    """

    def __init__(self, model_dir, *args, **kwargs):
        super(SbertForFaqQuestionAnsweringBase,
              self).__init__(model_dir, *args, **kwargs)

        backbone_cfg = SbertConfig.from_pretrained(model_dir)
        self.bert = SbertModel(backbone_cfg)

        model_config = Config.from_file(
            os.path.join(model_dir,
                         ModelFile.CONFIGURATION)).get(ConfigFields.model, {})

        metric = model_config.get('metric', 'cosine')
        pooling_method = model_config.get('pooling', 'avg')

        Arg = namedtuple('args', [
            'metrics', 'proj_hidden_size', 'hidden_size', 'dropout', 'pooling'
        ])
        args = Arg(
            metrics=metric,
            proj_hidden_size=self.bert.config.hidden_size,
            hidden_size=self.bert.config.hidden_size,
            dropout=0.0,
            pooling=pooling_method)

        self.metrics_layer = MetricsLayer(args)
        self.pooling = PoolingLayer(args)

    def _get_onehot_labels(self, labels, support_size, num_cls):
        labels_ = labels.view(support_size, 1)
        target_oh = torch.zeros(support_size, num_cls).to(labels)
        target_oh.scatter_(dim=1, index=labels_, value=1)
        return target_oh.view(support_size, num_cls).float()

    def forward_sentence_embedding(self, inputs: Dict[str, Tensor]):
        input_ids = inputs['input_ids']
        input_mask = inputs['attention_mask']
        if not isinstance(input_ids, Tensor):
            input_ids = torch.IntTensor(input_ids)
        if not isinstance(input_mask, Tensor):
            input_mask = torch.IntTensor(input_mask)
        rst = self.bert(input_ids, input_mask)
        last_hidden_states = rst.last_hidden_state
        if len(input_mask.shape) == 2:
            input_mask = input_mask.unsqueeze(-1)
        pooled_representation = self.pooling(last_hidden_states, input_mask)
        return pooled_representation


@MODELS.register_module(
    Tasks.faq_question_answering, module_name=Models.structbert)
class SbertForFaqQuestionAnswering(SbertForFaqQuestionAnsweringBase):
    _backbone_prefix = ''

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        assert not self.training
        query = input['query']
        support = input['support']
        if isinstance(query, list):
            query = torch.stack(query)
        if isinstance(support, list):
            support = torch.stack(support)
        n_query = query.shape[0]
        n_support = support.shape[0]
        query_mask = torch.ne(query, 0).view([n_query, -1])
        support_mask = torch.ne(support, 0).view([n_support, -1])

        support_labels = input['support_labels']
        num_cls = torch.max(support_labels) + 1
        onehot_labels = self._get_onehot_labels(support_labels, n_support,
                                                num_cls)

        input_ids = torch.cat([query, support])
        input_mask = torch.cat([query_mask, support_mask], dim=0)
        pooled_representation = self.forward_sentence_embedding({
            'input_ids':
            input_ids,
            'attention_mask':
            input_mask
        })
        z_query = pooled_representation[:n_query]
        z_support = pooled_representation[n_query:]
        cls_n_support = torch.sum(onehot_labels, dim=-2) + 1e-5
        protos = torch.matmul(onehot_labels.transpose(0, 1),
                              z_support) / cls_n_support.unsqueeze(-1)
        scores = self.metrics_layer(z_query, protos).view([n_query, num_cls])
        if self.metrics_layer.name == 'relation':
            scores = torch.sigmoid(scores)
        return {'scores': scores}


activations = {
    'relu': F.relu,
    'tanh': torch.tanh,
    'linear': lambda x: x,
}

activation_coeffs = {
    'relu': math.sqrt(2),
    'tanh': 5 / 3,
    'linear': 1.,
}


class LinearProjection(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 activation='linear',
                 bias=True):
        super().__init__()
        self.activation = activations[activation]
        activation_coeff = activation_coeffs[activation]
        linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.normal_(
            linear.weight, std=math.sqrt(1. / in_features) * activation_coeff)
        if bias:
            nn.init.zeros_(linear.bias)
        self.model = nn.utils.weight_norm(linear)

    def forward(self, x):
        return self.activation(self.model(x))


class RelationModule(nn.Module):

    def __init__(self, args):
        super(RelationModule, self).__init__()
        input_size = args.proj_hidden_size * 4
        self.prediction = torch.nn.Sequential(
            LinearProjection(
                input_size, args.proj_hidden_size * 4, activation='relu'),
            nn.Dropout(args.dropout),
            LinearProjection(args.proj_hidden_size * 4, 1))

    def forward(self, query, protos):
        n_cls = protos.shape[0]
        n_query = query.shape[0]
        protos = protos.unsqueeze(0).repeat(n_query, 1, 1)
        query = query.unsqueeze(1).repeat(1, n_cls, 1)
        input_feat = torch.cat(
            [query, protos, (protos - query).abs(), query * protos], dim=-1)
        dists = self.prediction(input_feat)  # [bsz,n_query,n_cls,1]
        return dists.squeeze(-1)


class MetricsLayer(nn.Module):

    def __init__(self, args):
        super(MetricsLayer, self).__init__()
        self.args = args
        assert args.metrics in ('relation', 'cosine')
        if args.metrics == 'relation':
            self.relation_net = RelationModule(args)

    @property
    def name(self):
        return self.args.metrics

    def forward(self, query, protos):
        """ query : [bsz, n_query, dim]
            support : [bsz, n_query, n_cls, dim] | [bsz, n_cls, dim]
        """
        if self.args.metrics == 'cosine':
            supervised_dists = self.cosine_similarity(query, protos)
            if self.training:
                supervised_dists *= 5
        elif self.args.metrics in ('relation', ):
            supervised_dists = self.relation_net(query, protos)
        else:
            raise NotImplementedError
        return supervised_dists

    def cosine_similarity(self, x, y):
        # x=[bsz, n_query, dim]
        # y=[bsz, n_cls, dim]
        n_query = x.shape[0]
        n_cls = y.shape[0]
        dim = x.shape[-1]
        x = x.unsqueeze(1).expand([n_query, n_cls, dim])
        y = y.unsqueeze(0).expand([n_query, n_cls, dim])
        return F.cosine_similarity(x, y, -1)


class AveragePooling(nn.Module):

    def forward(self, x, mask, dim=1):
        return torch.sum(
            x * mask.float(), dim=dim) / torch.sum(
                mask.float(), dim=dim)


class AttnPooling(nn.Module):

    def __init__(self, input_size, hidden_size=None, output_size=None):
        super().__init__()
        self.input_proj = nn.Sequential(
            LinearProjection(input_size, hidden_size), nn.Tanh(),
            LinearProjection(hidden_size, 1, bias=False))
        self.output_proj = LinearProjection(
            input_size, output_size) if output_size else lambda x: x

    def forward(self, x, mask):
        score = self.input_proj(x)
        score = score * mask.float() + -1e4 * (1. - mask.float())
        score = F.softmax(score, dim=1)
        features = self.output_proj(x)
        return torch.matmul(score.transpose(1, 2), features).squeeze(1)


class PoolingLayer(nn.Module):

    def __init__(self, args):
        super(PoolingLayer, self).__init__()
        if args.pooling == 'attn':
            self.pooling = AttnPooling(args.proj_hidden_size,
                                       args.proj_hidden_size,
                                       args.proj_hidden_size)
        elif args.pooling == 'avg':
            self.pooling = AveragePooling()
        else:
            raise NotImplementedError(args.pooling)

    def forward(self, x, mask):
        return self.pooling(x, mask)
