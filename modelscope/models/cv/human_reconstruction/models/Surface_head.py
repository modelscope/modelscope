# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F


class Surface_Head(nn.Module):
    """
    MLP: aims at learn iso-surface function Implicit function
    """

    def __init__(self,
                 filter_channels,
                 merge_layer=0,
                 res_layers=[],
                 norm='group',
                 last_op=None):
        super(Surface_Head, self).__init__()
        if last_op == 'sigmoid':
            self.last_op = nn.Sigmoid()
        elif last_op == 'tanh':
            self.last_op = nn.Tanh()
        else:
            raise NotImplementedError(
                'only sigmoid/tanh function could be used')

        self.filters = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.merge_layer = merge_layer if merge_layer > 0 else len(
            filter_channels) // 2

        self.res_layers = res_layers
        self.norm = norm

        for i in range(0, len(filter_channels) - 1):
            if i in self.res_layers:
                self.filters.append(
                    nn.Conv1d(filter_channels[i] + filter_channels[0],
                              filter_channels[i + 1], 1))
            else:
                self.filters.append(
                    nn.Conv1d(filter_channels[i], filter_channels[i + 1], 1))
            if i != len(filter_channels) - 2:
                if norm == 'group':
                    self.norms.append(nn.GroupNorm(32, filter_channels[i + 1]))
                elif norm == 'batch':
                    self.norms.append(nn.BatchNorm1d(filter_channels[i + 1]))

    def forward(self, feature):
        """feature may include multiple view inputs
        Parameters:
            feature: [B, C_in, N]
        return:
            prediction: [B, C_out, N] and merge layer features
        """

        y = feature
        tmpy = feature
        phi = None

        for i, f in enumerate(self.filters):
            y = f(y if i not in self.res_layers else torch.cat([y, tmpy], 1))
            if i != len(self.filters) - 1:
                if self.norm not in ['batch', 'group']:
                    y = F.leaky_relu(y)
                else:
                    y = F.leaky_relu(self.norms[i](y))
            if i == self.merge_layer:
                phi = y.clone()

        if self.last_op is not None:
            y = self.last_op(y)
        return y, phi
