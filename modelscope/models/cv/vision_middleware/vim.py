# Part of this code is adopted from PETL-ViT,
# made publicly available under the MIT License at https://github.com/JieShibo/PETL-ViT

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _agg_conv1d(weight_list, bias_list, agg, x):
    """
        weight list: list of conv1d weight ([out, in] * a)
        bias list: list of conv1d bias ([out] * a)
        agg: aggreagtion weights (a)
        x: input tensor (b, in, n)

        return output in (b, n, out)
    """

    weight_list = torch.cat([w.unsqueeze(0) for w in weight_list],
                            dim=0)  # n_ada, out, in
    weight = torch.sum(
        weight_list * rearrange(agg, 'a -> a 1 1'),
        dim=0).unsqueeze(2)  # out, in, 1

    bias_list = torch.cat([w.unsqueeze(0) for w in bias_list],
                          dim=0)  # n_ada, out
    bias = torch.sum(bias_list * rearrange(agg, 'a -> a 1'), dim=0)  # out

    x = F.conv1d(x, weight=weight, bias=bias)

    return x


def _agg_conv2d(weight_list, bias_list, agg, x):
    """
        weight list: list of conv2d weight ([out, in, m, n] * a)
        bias list: list of conv2d bias ([out] * a)
        agg: aggregation weights (a)
        x: input tensor (b, in, p, q)

        return output in (b, out, p, q)
    """

    weight_list = torch.cat([w.unsqueeze(0) for w in weight_list],
                            dim=0)  # n_ada, out, in, m, n
    weight = torch.sum(
        weight_list * rearrange(agg, 'a -> a 1 1 1 1'), dim=0)  # out, in, m, n

    bias_list = torch.cat([w.unsqueeze(0) for w in bias_list],
                          dim=0)  # n_ada, out
    bias = torch.sum(bias_list * rearrange(agg, 'a -> a 1'), dim=0)  # out

    x = F.conv2d(
        x, weight=weight, bias=bias, stride=1, padding=1)  # 1 (b out) p q

    return x


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ViM(nn.Module):

    def __init__(self):
        super().__init__()

        self.act = QuickGELU()

        self.adapter_conv_weight = nn.ParameterList()
        self.adapter_conv_bias = nn.ParameterList()

        self.adapter_up_weight = nn.ParameterList()
        self.adapter_up_bias = nn.ParameterList()

        self.adapter_down_weight = nn.ParameterList()
        self.adapter_down_bias = nn.ParameterList()

        # agg related
        self.num_modules = 0
        self.task_list = []
        self.agg_weights = {}
        self.agg_algos = {}

    def register_ViM(self, vim_list):
        self.num_modules = len(vim_list)
        for state_dict in vim_list:
            self.adapter_conv_weight.append(
                nn.Parameter(state_dict['adapter_conv.weight']))
            self.adapter_conv_bias.append(
                nn.Parameter(state_dict['adapter_conv.bias']))

            self.adapter_up_weight.append(
                nn.Parameter(state_dict['adapter_up.weight']))
            self.adapter_up_bias.append(
                nn.Parameter(state_dict['adapter_up.bias']))

            self.adapter_down_weight.append(
                nn.Parameter(state_dict['adapter_down.weight']))
            self.adapter_down_bias.append(
                nn.Parameter(state_dict['adapter_down.bias']))

    def register_task(self, task_name, agg_weights, agg_algo):
        assert agg_weights.shape[0] == self.num_modules

        self.task_list.append(task_name)
        self.agg_weights[task_name] = agg_weights
        self.agg_algos[task_name] = agg_algo

    def forward(self, x, task_name):
        assert task_name in self.task_list

        agg_algo = self.agg_algos[task_name]
        if agg_algo == 'Ens-MoE':
            return self.forward_ens_moe(x, self.agg_weights[task_name])
        else:
            raise NotImplementedError(
                'Aggregation algorithm [{}] is currently not supported!'.
                format(agg_algo))

    def forward_ens_moe(self, x, agg):

        logits = agg
        k = agg.shape[0]  # MoE-full (k=N)

        top_logits, top_indices = logits.topk(
            min(k + 1, logits.size(0)), dim=0)
        top_k_logits = top_logits[:k]
        top_k_indices = top_indices[:k]
        top_k_gates = F.softmax(top_k_logits, dim=0)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(0, top_k_indices, top_k_gates)

        N, B, C = x.shape
        x = x.permute(1, 2, 0)
        output = None
        for i in range(self.num_modules):

            if gates[i] > 0:

                x_down = F.conv1d(
                    x,
                    weight=self.adapter_down_weight[i].unsqueeze(2),
                    bias=self.adapter_down_bias[i])  # equivalent to 1 * 1 Conv
                x_down = self.act(x_down)

                num_patch_side = int(math.sqrt(x_down.size(2) - 1))
                x_patch = x_down[:, :,
                                 1:].reshape(B, -1, num_patch_side,
                                             num_patch_side)  # b, in, p, p
                x_patch = F.conv2d(
                    x_patch,
                    weight=self.adapter_conv_weight[i],
                    bias=self.adapter_conv_bias[i],
                    stride=1,
                    padding=1)
                x_patch = rearrange(x_patch, 'b o p q -> b o (p q)')

                x_cls = x_down[:, :, :1].reshape(B, -1, 1, 1)
                x_cls = F.conv2d(
                    x_cls,
                    weight=self.adapter_conv_weight[i],
                    bias=self.adapter_conv_bias[i],
                    stride=1,
                    padding=1)
                x_cls = rearrange(x_cls, 'b o 1 1 -> b o 1')

                x_down = torch.cat([x_cls, x_patch], dim=2)

                x_down = self.act(x_down)
                x_up = F.conv1d(
                    x_down,
                    weight=self.adapter_up_weight[i].unsqueeze(2),
                    bias=self.adapter_up_bias[i])  # equivalent to 1 * 1 Conv

                if output is None:
                    output = x_up * gates[i]
                else:
                    output += x_up * gates[i]

        return output.permute(2, 0, 1)
