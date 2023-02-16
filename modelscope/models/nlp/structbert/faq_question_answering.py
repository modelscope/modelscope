# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import os
from collections import namedtuple
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BCEWithLogitsLoss

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.models.nlp.structbert import SbertConfig, SbertModel
from modelscope.models.nlp.task_models.task_model import BaseTaskModel
from modelscope.outputs import FaqQuestionAnsweringOutput
from modelscope.utils.config import Config, ConfigFields
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

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


class Alignment(nn.Module):

    def __init__(self):
        super().__init__()

    def _attention(self, a, b):
        return torch.matmul(a, b.transpose(1, 2))

    def forward(self, a, b, mask_a, mask_b):
        attn = self._attention(a, b)
        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float())
        mask = mask.bool()
        attn.masked_fill_(~mask, -1e4)
        return attn


def _create_args(model_config, hidden_size):
    metric = model_config.get('metric', 'cosine')
    pooling_method = model_config.get('pooling', 'avg')
    Arg = namedtuple(
        'args',
        ['metrics', 'proj_hidden_size', 'hidden_size', 'dropout', 'pooling'])
    args = Arg(
        metrics=metric,
        proj_hidden_size=hidden_size,
        hidden_size=hidden_size,
        dropout=0.0,
        pooling=pooling_method)
    return args


@MODELS.register_module(
    Tasks.faq_question_answering, module_name=Models.structbert)
class SbertForFaqQuestionAnswering(BaseTaskModel):
    _backbone_prefix = ''
    PROTO_NET = 'protonet'
    MGIMN_NET = 'mgimnnet'

    @classmethod
    def _instantiate(cls, **kwargs):
        model_dir = kwargs.pop('model_dir')
        model = cls(model_dir, **kwargs)
        model.load_checkpoint(model_dir)
        return model

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        backbone_cfg = SbertConfig.from_pretrained(model_dir)
        model_config = Config.from_file(
            os.path.join(model_dir,
                         ModelFile.CONFIGURATION)).get(ConfigFields.model, {})
        model_config.update(kwargs)

        network_name = model_config.get('network', self.PROTO_NET)
        if network_name == self.PROTO_NET:
            network = ProtoNet(backbone_cfg, model_config)
        elif network_name == self.MGIMN_NET:
            network = MGIMNNet(backbone_cfg, model_config)
        else:
            raise NotImplementedError(network_name)
        logger.info(f'faq task build {network_name} network')
        self.network = network

    def forward(self, input: Dict[str, Tensor]) -> FaqQuestionAnsweringOutput:
        """
        Args:
            input (Dict[str, Tensor]): the preprocessed data, it contains the following keys:

                - query(:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                    The query to be predicted.
                - support(:obj:`torch.LongTensor` of shape :obj:`(support_size, sequence_length)`):
                    The support set.
                - support_label(:obj:`torch.LongTensor` of shape :obj:`(support_size, )`):
                    The labels of support set.

        Returns:
            Dict[str, Tensor]: result, it contains the following key:

                - scores(:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_cls)`):
                    Predicted scores of all classes for each query.

        Examples:
            >>> from modelscope.hub.snapshot_download import snapshot_download
            >>> from modelscope.preprocessors import FaqQuestionAnsweringTransformersPreprocessor
            >>> from modelscope.models.nlp import SbertForFaqQuestionAnswering
            >>> cache_path = snapshot_download('damo/nlp_structbert_faq-question-answering_chinese-base')
            >>> preprocessor = FaqQuestionAnsweringTransformersPreprocessor.from_pretrained(cache_path)
            >>> model = SbertForFaqQuestionAnswering.from_pretrained(cache_path)
            >>> param = {
            >>>            'query_set': ['如何使用优惠券', '在哪里领券', '在哪里领券'],
            >>>            'support_set': [{
            >>>                    'text': '卖品代金券怎么用',
            >>>                    'label': '6527856'
            >>>               }, {
            >>>                    'text': '怎么使用优惠券',
            >>>                    'label': '6527856'
            >>>                }, {
            >>>                    'text': '这个可以一起领吗',
            >>>                    'label': '1000012000'
            >>>                }, {
            >>>                    'text': '付款时送的优惠券哪里领',
            >>>                    'label': '1000012000'
            >>>                }, {
            >>>                    'text': '购物等级怎么长',
            >>>                    'label': '13421097'
            >>>                }, {
            >>>                    'text': '购物等级二心',
            >>>                    'label': '13421097'
            >>>               }]
            >>>           }
            >>> result = model(preprocessor(param))
        """
        query = input['query']
        support = input['support']
        query_mask = input['query_attention_mask']
        support_mask = input['support_attention_mask']
        support_labels = input['support_labels']
        logits, scores = self.network(query, support, query_mask, support_mask,
                                      support_labels)

        if 'labels' in input:
            query_labels = input['labels']
            num_cls = torch.max(support_labels) + 1
            loss = self._compute_loss(logits, query_labels, num_cls)
            pred_labels = torch.argmax(scores, dim=1)
            return FaqQuestionAnsweringOutput(
                loss=loss, logits=scores, labels=pred_labels).to_dict()
        else:
            return FaqQuestionAnsweringOutput(scores=scores)

    def _compute_loss(self, logits, target, num_cls):
        onehot_labels = get_onehot_labels(target, num_cls)
        loss = BCEWithLogitsLoss(reduction='mean')(logits, onehot_labels)
        return loss

    def forward_sentence_embedding(self, inputs):
        return self.network.sentence_embedding(inputs)

    def load_checkpoint(self,
                        model_local_dir,
                        default_dtype=None,
                        load_state_fn=None,
                        **kwargs):
        ckpt_file = os.path.join(model_local_dir, 'pytorch_model.bin')
        state_dict = torch.load(ckpt_file, map_location='cpu')
        # compatible with the old checkpoints
        new_state_dict = {}
        for var_name, var_value in state_dict.items():
            new_var_name = var_name
            if not str(var_name).startswith('network'):
                new_var_name = f'network.{var_name}'
            new_state_dict[new_var_name] = var_value
        if default_dtype is not None:
            torch.set_default_dtype(default_dtype)

        missing_keys, unexpected_keys, mismatched_keys, error_msgs = self._load_checkpoint(
            new_state_dict,
            load_state_fn=load_state_fn,
            ignore_mismatched_sizes=True,
            _fast_init=True,
        )

        return {
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
            'mismatched_keys': mismatched_keys,
            'error_msgs': error_msgs,
        }


def get_onehot_labels(target, num_cls):
    target = target.view(-1, 1)
    size = target.shape[0]
    target_oh = torch.zeros(size, num_cls).to(target)
    target_oh.scatter_(dim=1, index=target, value=1)
    return target_oh.view(size, num_cls).float()


class ProtoNet(nn.Module):

    def __init__(self, backbone_config, model_config):
        super(ProtoNet, self).__init__()
        self.bert = SbertModel(backbone_config)
        args = _create_args(model_config, self.bert.config.hidden_size)
        self.metrics_layer = MetricsLayer(args)
        self.pooling = PoolingLayer(args)

    def __call__(self, query, support, query_mask, support_mask,
                 support_labels):
        n_query = query.shape[0]

        num_cls = torch.max(support_labels) + 1
        onehot_labels = get_onehot_labels(support_labels, num_cls)

        input_ids = torch.cat([query, support])
        input_mask = torch.cat([query_mask, support_mask], dim=0)
        pooled_representation = self.sentence_embedding({
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
        logits = self.metrics_layer(z_query, protos).view([n_query, num_cls])
        if self.metrics_layer.name == 'relation':
            scores = torch.sigmoid(logits)
        else:
            scores = logits
        return logits, scores

    def sentence_embedding(self, inputs: Dict[str, Tensor]):
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


class MGIMNNet(nn.Module):
    # default use class_level_interaction only
    INSTANCE_LEVEL_INTERACTION = 'instance_level_interaction'
    EPISODE_LEVEL_INTERACTION = 'episode_level_interaction'

    def __init__(self, backbone_config, model_config):
        super(MGIMNNet, self).__init__()
        self.bert = SbertModel(backbone_config)
        self.model_config = model_config
        self.alignment = Alignment()
        hidden_size = self.bert.config.hidden_size
        use_instance_level_interaction = self.safe_get(
            self.INSTANCE_LEVEL_INTERACTION, True)
        use_episode_level_interaction = self.safe_get(
            self.EPISODE_LEVEL_INTERACTION, True)
        output_size = 1 + int(use_instance_level_interaction) + int(
            use_episode_level_interaction)
        logger.info(
            f'faq MGIMN model class-level-interaction:true, instance-level-interaction:{use_instance_level_interaction}, \
            episode-level-interaction:{use_episode_level_interaction}')
        self.fuse_proj = LinearProjection(
            hidden_size + hidden_size * 3 * output_size,
            hidden_size,
            activation='relu')
        args = _create_args(model_config, hidden_size)
        self.pooling = PoolingLayer(args)
        new_args = args._replace(pooling='avg')
        self.avg_pooling = PoolingLayer(new_args)

        self.instance_compare_layer = torch.nn.Sequential(
            LinearProjection(hidden_size * 4, hidden_size, activation='relu'))

        self.prediction = torch.nn.Sequential(
            LinearProjection(hidden_size * 2, hidden_size, activation='relu'),
            nn.Dropout(0), LinearProjection(hidden_size, 1))

    def __call__(self, query, support, query_mask, support_mask,
                 support_labels):
        z_query, z_support = self.context_embedding(query, support, query_mask,
                                                    support_mask)
        n_cls = int(torch.max(support_labels)) + 1
        n_query, sent_len = query.shape
        n_support = support.shape[0]
        k_shot = n_support // n_cls

        q_params, s_params = {
            'n_cls': n_cls,
            'k_shot': k_shot
        }, {
            'n_query': n_query
        }
        if self.safe_get(self.INSTANCE_LEVEL_INTERACTION, True):
            ins_z_query, ins_z_support = self._instance_level_interaction(
                z_query, query_mask, z_support, support_mask)
            q_params['ins_z_query'] = ins_z_query
            s_params['ins_z_support'] = ins_z_support

        cls_z_query, cls_z_support = self._class_level_interaction(
            z_query, query_mask, z_support, support_mask, n_cls)
        q_params['cls_z_query'] = cls_z_query
        s_params['cls_z_support'] = cls_z_support
        if self.safe_get(self.EPISODE_LEVEL_INTERACTION, True):
            eps_z_query, eps_z_support = self._episode_level_interaction(
                z_query, query_mask, z_support, support_mask)
            q_params['eps_z_query'] = eps_z_query
            s_params['eps_z_support'] = eps_z_support
        fused_z_query = self._fuse_query(z_query, **q_params)
        fused_z_support = self._fuse_support(z_support, **s_params)
        query_mask_expanded = query_mask.unsqueeze(1).repeat(
            1, n_support, 1).view(n_query * n_support, sent_len, 1)
        support_mask_expanded = support_mask.unsqueeze(0).repeat(
            n_query, 1, 1).view(n_query * n_support, sent_len, 1)
        Q = self.pooling(fused_z_query, query_mask_expanded)
        S = self.pooling(fused_z_support, support_mask_expanded)
        matching_feature = self._instance_compare(Q, S, n_query, n_cls, k_shot)
        logits = self.prediction(matching_feature)
        logits = logits.view(n_query, n_cls)
        return logits, torch.sigmoid(logits)

    def _instance_compare(self, Q, S, n_query, n_cls, k_shot):
        z_dim = Q.shape[-1]
        S = S.view(n_query, n_cls * k_shot, z_dim)
        Q = Q.view(n_query, k_shot * n_cls, z_dim)
        cat_features = torch.cat([Q, S, Q * S, (Q - S).abs()], dim=-1)
        instance_matching_feature = self.instance_compare_layer(cat_features)
        instance_matching_feature = instance_matching_feature.view(
            n_query, n_cls, k_shot, z_dim)
        cls_matching_feature_mean = instance_matching_feature.mean(2)
        cls_matching_feature_max, _ = instance_matching_feature.max(2)
        cls_matching_feature = torch.cat(
            [cls_matching_feature_mean, cls_matching_feature_max], dim=-1)
        return cls_matching_feature

    def _instance_level_interaction(self, z_query, query_mask, z_support,
                                    support_mask):
        n_query, sent_len, z_dim = z_query.shape
        n_support = z_support.shape[0]
        z_query = z_query.unsqueeze(1).repeat(1, n_support, 1,
                                              1).view(n_query * n_support,
                                                      sent_len, z_dim)
        query_mask = query_mask.unsqueeze(1).repeat(1, n_support, 1).view(
            n_query * n_support, sent_len, 1)
        z_support = z_support.unsqueeze(0).repeat(n_query, 1, 1, 1).view(
            n_query * n_support, sent_len, z_dim)
        support_mask = support_mask.unsqueeze(0).repeat(n_query, 1, 1, 1).view(
            n_query * n_support, sent_len, 1)
        attn = self.alignment(z_query, z_support, query_mask, support_mask)
        attn_a = F.softmax(attn, dim=1)
        attn_b = F.softmax(attn, dim=2)
        ins_support = torch.matmul(attn_a.transpose(1, 2), z_query)
        ins_query = torch.matmul(attn_b, z_support)
        return ins_query, ins_support

    def _class_level_interaction(self, z_query, query_mask, z_support,
                                 support_mask, n_cls):
        z_support_ori = z_support
        support_mask_ori = support_mask

        n_query, sent_len, z_dim = z_query.shape
        n_support = z_support.shape[0]
        k_shot = n_support // n_cls

        # class-based query encoding
        z_query = z_query.unsqueeze(1).repeat(1, n_cls, 1,
                                              1).view(n_query * n_cls,
                                                      sent_len, z_dim)
        query_mask = query_mask.unsqueeze(1).unsqueeze(-1).repeat(
            1, n_cls, 1, 1).view(n_query * n_cls, sent_len, 1)
        z_support = z_support.unsqueeze(0).repeat(n_query, 1, 1, 1).view(
            n_query * n_cls, k_shot * sent_len, z_dim)
        support_mask = support_mask.unsqueeze(0).unsqueeze(-1).repeat(
            n_query, 1, 1, 1).view(n_query * n_cls, k_shot * sent_len, 1)
        attn = self.alignment(z_query, z_support, query_mask, support_mask)
        attn_b = F.softmax(attn, dim=2)
        cls_query = torch.matmul(attn_b, z_support)
        cls_query = cls_query.view(n_query, n_cls, sent_len, z_dim)

        # class-based support encoding
        z_support = z_support_ori.view(n_cls, k_shot * sent_len, z_dim)
        support_mask = support_mask_ori.view(n_cls, k_shot * sent_len, 1)
        attn = self.alignment(z_support, z_support, support_mask, support_mask)
        attn_b = F.softmax(attn, dim=2)
        cls_support = torch.matmul(attn_b, z_support)
        cls_support = cls_support.view(n_cls * k_shot, sent_len, z_dim)
        return cls_query, cls_support

    def _episode_level_interaction(self, z_query, query_mask, z_support,
                                   support_mask):
        z_support_ori = z_support
        support_mask_ori = support_mask

        n_query, sent_len, z_dim = z_query.shape
        n_support = z_support.shape[0]

        # episode-based query encoding
        query_mask = query_mask.view(n_query, sent_len, 1)
        z_support = z_support.unsqueeze(0).repeat(n_query, 1, 1, 1).view(
            n_query, n_support * sent_len, z_dim)
        support_mask = support_mask.unsqueeze(0).unsqueeze(-1).repeat(
            n_query, 1, 1, 1).view(n_query, n_support * sent_len, 1)
        attn = self.alignment(z_query, z_support, query_mask, support_mask)
        attn_b = F.softmax(attn, dim=2)
        eps_query = torch.matmul(attn_b, z_support)

        # episode-based support encoding
        z_support2 = z_support_ori.view(1, n_support * sent_len,
                                        z_dim).repeat(n_support, 1, 1)
        support_mask = support_mask_ori.view(1, n_support * sent_len,
                                             1).repeat(n_support, 1, 1)
        attn = self.alignment(z_support_ori, z_support2,
                              support_mask_ori.unsqueeze(-1), support_mask)
        attn_b = F.softmax(attn, dim=2)
        eps_support = torch.matmul(attn_b, z_support2)
        eps_support = eps_support.view(n_support, sent_len, z_dim)
        return eps_query, eps_support

    def _fuse_query(self,
                    x,
                    n_cls,
                    k_shot,
                    ins_z_query=None,
                    cls_z_query=None,
                    eps_z_query=None):
        n_query, sent_len, z_dim = x.shape
        assert cls_z_query is not None
        cls_features = cls_z_query.unsqueeze(2).repeat(
            1, 1, k_shot, 1, 1).view(n_cls * k_shot * n_query, sent_len, z_dim)
        x = x.unsqueeze(1).repeat(1, n_cls * k_shot, 1,
                                  1).view(n_cls * k_shot * n_query, sent_len,
                                          z_dim)
        features = [
            x, cls_features, x * cls_features, (x - cls_features).abs()
        ]
        if ins_z_query is not None:
            features.extend(
                [ins_z_query, ins_z_query * x, (ins_z_query - x).abs()])
        if eps_z_query is not None:
            eps_z_query = eps_z_query.unsqueeze(1).repeat(
                1, n_cls * k_shot, 1, 1).view(n_cls * k_shot * n_query,
                                              sent_len, z_dim)
            features.extend(
                [eps_z_query, eps_z_query * x, (eps_z_query - x).abs()])
        features = torch.cat(features, dim=-1)
        fusion_feat = self.fuse_proj(features)
        return fusion_feat

    def _fuse_support(self,
                      x,
                      n_query,
                      ins_z_support=None,
                      cls_z_support=None,
                      eps_z_support=None):
        assert cls_z_support is not None
        n_support, sent_len, z_dim = x.shape
        x = x.unsqueeze(0).repeat(n_query, 1, 1,
                                  1).view(n_support * n_query, sent_len, z_dim)
        cls_features = cls_z_support.unsqueeze(0).repeat(
            n_query, 1, 1, 1).view(n_support * n_query, sent_len, z_dim)
        features = [
            x, cls_features, x * cls_features, (x - cls_features).abs()
        ]
        if ins_z_support is not None:
            features.extend(
                [ins_z_support, ins_z_support * x, (ins_z_support - x).abs()])
        if eps_z_support is not None:
            eps_z_support = eps_z_support.unsqueeze(0).repeat(
                n_query, 1, 1, 1).view(n_query * n_support, sent_len, z_dim)
            features.extend(
                [eps_z_support, eps_z_support * x, (eps_z_support - x).abs()])
        features = torch.cat(features, dim=-1)
        fusion_feat = self.fuse_proj(features)
        return fusion_feat

    def context_embedding(self, query, support, query_mask, support_mask):
        n_query = query.shape[0]
        n_support = support.shape[0]
        x = torch.cat([query, support], dim=0)
        x_mask = torch.cat([query_mask, support_mask], dim=0)
        last_hidden_state = self.bert(x, x_mask).last_hidden_state
        z_dim = last_hidden_state.shape[-1]
        sent_len = last_hidden_state.shape[-2]
        z_query = last_hidden_state[:n_query].view([n_query, sent_len, z_dim])
        z_support = last_hidden_state[n_query:].view(
            [n_support, sent_len, z_dim])
        return z_query, z_support

    def sentence_embedding(self, inputs: Dict[str, Tensor]):
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
        pooled_representation = self.avg_pooling(last_hidden_states,
                                                 input_mask)
        return pooled_representation

    def safe_get(self, k, default=None):
        try:
            return self.model_config.get(k, default)
        except Exception as e:
            logger.debug(f'{k} not in model_config, use default:{default}')
            logger.debug(e)
            return default
