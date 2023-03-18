# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision


class Prompt(nn.Module):
    """The implementation of vision prompt tuning method.

    Visual prompt tuning (VPT) is proposed to initialize tunable prompt tokens
    and prepend to the original tokens in the first layer or multiple layers.
    'Visual Prompt Tuning' by Jia et al.(2022)
    See https://arxiv.org/abs/2203.12119

    Attributes:
        dim: An integer indicating the embedding dimension.
        layer_num: An integer indicating number of layers.
        prompt_length: An integer indicating the length of vision prompt tuning.
        prompt_type: A string indicating the type of vision prompt tuning.
    """

    def __init__(self, dim, layer_num, prompt_length=None, prompt_type=None):
        super(Prompt, self).__init__()
        self.dim = dim
        self.layer_num = layer_num
        self.prompt_length = prompt_length
        self.prompt_type = prompt_type

        self.prompt_token = nn.Parameter(torch.zeros(1, prompt_length, dim))
        nn.init.xavier_uniform_(self.prompt_token)

    def forward(self, x):
        B, N, C = x.shape
        prompt_token = self.prompt_token.expand(B, -1, -1)

        if self.layer_num == 0:
            x = torch.cat((x, prompt_token), dim=1)
        else:
            x = torch.cat((x[:, :-self.prompt_length, :], prompt_token), dim=1)
        return x


class Adapter(nn.Module):
    """The implementation of adapter tuning method.

    Adapters project input tokens by an MLP layer.
    'Parameter-Efficient Transfer Learning for NLP' by Houlsby et al.(2019)
    See http://arxiv.org/abs/1902.00751

    Attributes:
        dim: An integer indicating the embedding dimension.
        adapter_length: An integer indicating the length of adapter tuning.
        adapter_type: A string indicating the type of adapter tuning.
    """

    def __init__(
        self,
        dim,
        adapter_length=None,
        adapter_type=None,
        act_layer=nn.GELU,
    ):
        super(Adapter, self).__init__()
        self.dim = dim
        self.adapter_length = adapter_length
        self.adapter_type = adapter_type
        self.ln1 = nn.Linear(dim, adapter_length)
        self.activate = act_layer()
        self.ln2 = nn.Linear(adapter_length, dim)
        self.init_weights()

    def init_weights(self):

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

        self.apply(_init_weights)

    def forward(self, x, identity=None):
        out = self.ln2(self.activate(self.ln1(x)))
        if identity is None:
            identity = x
        out = identity + out
        return out


class LoRA(nn.Module):
    """The implementation of LoRA tuning method.

    LoRA constructs an additional layer with low-rank decomposition matrices of the weights in the network.
    'LoRA: Low-Rank Adaptation of Large Language Models' by Hu et al.(2021)
    See https://arxiv.org/abs/2106.09685

    Attributes:
        dim: An integer indicating the embedding dimension.
        num_heads: An integer indicating number of attention heads.
        lora_length: An integer indicating the length of LoRA tuning.
        lora_type: A string indicating the type of LoRA tuning.
    """

    def __init__(
        self,
        dim,
        num_heads,
        lora_length=None,
        lora_type=None,
    ):
        super(LoRA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.lora_a = nn.Linear(dim, lora_length, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        self.lora_b = nn.Linear(lora_length, dim * 3, bias=False)
        nn.init.zeros_(self.lora_b.weight)

        self.lora_length = lora_length
        self.lora_type = lora_type

    def forward(self, x, q, k, v):
        B, N, C = x.shape
        qkv_delta = self.lora_b(self.lora_a(x))
        qkv_delta = qkv_delta.reshape(B, N, 3, self.num_heads,
                                      C // self.num_heads).permute(
                                          2, 0, 3, 1, 4)
        q_delta, k_delta, v_delta = qkv_delta.unbind(0)
        q, k, v = q + q_delta, k + k_delta, v + v_delta
        return q, k, v


class Prefix(nn.Module):
    """The implementation of prefix tuning method.

    Prefix tuning optimizes the task-specific vector in the multi-head attention layer.
    'Prefix-tuning: Optimizing continuous prompts for generation' by Li & Liang(2021)
    See https://arxiv.org/abs/2101.00190

    Attributes:
        dim: An integer indicating the embedding dimension.
        num_heads: An integer indicating number of attention heads.
        prefix_length: An integer indicating the length of prefix tuning.
        prefix_type: A string indicating the type of prefix tuning.
    """

    def __init__(
        self,
        dim,
        num_heads,
        prefix_length=None,
        prefix_type=None,
    ):
        super(Prefix, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.prefix_length = prefix_length
        self.prefix_type = prefix_type
        self.prefix_key = nn.Parameter(torch.zeros(1, prefix_length, dim))
        self.prefix_value = nn.Parameter(torch.zeros(1, prefix_length, dim))
        nn.init.xavier_uniform_(self.prefix_key)
        nn.init.xavier_uniform_(self.prefix_value)

    def forward(self, x, q, k, v):
        B, N, C = x.shape
        prefix_key = self.prefix_key.expand(B, -1, -1).reshape(
            B, self.prefix_length, self.num_heads,
            self.dim // self.num_heads).permute(0, 2, 1, 3)
        prefix_value = self.prefix_value.expand(B, -1, -1).reshape(
            B, self.prefix_length, self.num_heads,
            self.dim // self.num_heads).permute(0, 2, 1, 3)
        k, v = torch.cat((k, prefix_key), dim=2), torch.cat((v, prefix_value),
                                                            dim=2)
        return q, k, v


class SideTune(nn.Module):
    """The implementation of vision side-tuning method.

    Side-Tuning only needs to train one side network and
    weights the output of pre-trained model and side network.
    'Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks'
    by Zhang et al.(2019)
    See https://arxiv.org/abs/1912.13503

    Attributes:
        sidetune_length: An integer indicating the linear dimension.
        sidetune_type: A string indicating the type of side network.
    """

    def __init__(self, sidetune_length=None, sidetune_type=None):
        super(SideTune, self).__init__()
        self.sidetune_length = sidetune_length
        self.sidetune_type = sidetune_type
        if sidetune_type.lower() == 'fcn4':
            self.side = FCN4(out_dims=self.sidetune_length)
        if sidetune_type.lower() == 'alexnet':
            mm = torchvision.models.alexnet(pretrained=True)
            self.side = nn.Sequential(
                OrderedDict([
                    ('features', mm.features), ('avgpool', mm.avgpool),
                    ('flatten', nn.Flatten()),
                    ('fc', nn.Linear(9216, self.sidetune_length, bias=False))
                ]))
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, x_base):
        alpha_squashed = torch.sigmoid(self.alpha)
        x_side = self.side(x)
        x_out = alpha_squashed * x_base + (1 - alpha_squashed) * x_side
        return x_out


class FCN4(nn.Module):
    """The implementation of simple FCN4 network for side network.
    """

    def __init__(self, out_dims=-1, **kwargs):
        super(FCN4, self).__init__(**kwargs)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                dilation=1), nn.GroupNorm(2, 16), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=False,
                dilation=1), nn.GroupNorm(2, 16), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=False,
                dilation=1), nn.GroupNorm(2, 32), nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
                dilation=1), nn.GroupNorm(2, 64), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if out_dims > 0:
            self.fc = nn.Linear(64, out_dims)
        else:
            self.fc = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        if self.fc is not None:
            x = self.fc(x)
        return x
