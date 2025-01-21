# Copyright (c) Alibaba, Inc. and its affiliates.
# The ZenNAS implementation is also open-sourced by the authors, and available at https://github.com/idstcv/ZenNAS.

import torch
import torch.nn.functional as F
from torch import nn

from . import basic_blocks, plain_net_utils


class PlainNet(plain_net_utils.PlainNet):

    def __init__(self,
                 argv=None,
                 opt=None,
                 num_classes=None,
                 plainnet_struct=None,
                 no_create=False,
                 no_reslink=None,
                 no_BN=None,
                 use_se=None,
                 dropout=None,
                 **kwargs):

        module_opt = None

        if no_BN is None:
            if module_opt is not None:
                no_BN = module_opt.no_BN
            else:
                no_BN = False

        if no_reslink is None:
            if module_opt is not None:
                no_reslink = module_opt.no_reslink
            else:
                no_reslink = False

        if use_se is None:
            if module_opt is not None:
                use_se = module_opt.use_se
            else:
                use_se = False

        if dropout is None:
            if module_opt is not None:
                self.dropout = module_opt.dropout
            else:
                self.dropout = None
        else:
            self.dropout = dropout

        super(PlainNet, self).__init__(
            argv=argv,
            opt=opt,
            num_classes=num_classes,
            plainnet_struct=plainnet_struct,
            no_create=no_create,
            no_reslink=no_reslink,
            no_BN=no_BN,
            use_se=use_se,
            **kwargs)
        self.last_channels = self.block_list[-1].out_channels
        self.fc_linear = basic_blocks.Linear(
            in_channels=self.last_channels,
            out_channels=self.num_classes,
            no_create=no_create)

        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN
        self.use_se = use_se

        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = 1e-3

    def forward(self, x):
        output = x
        for block_id, the_block in enumerate(self.block_list):
            output = the_block(output)
            if self.dropout is not None:
                dropout_p = float(block_id) / len(
                    self.block_list) * self.dropout
                output = F.dropout(
                    output, dropout_p, training=self.training, inplace=True)

        output = F.adaptive_avg_pool2d(output, output_size=1)
        if self.dropout is not None:
            output = F.dropout(
                output, self.dropout, training=self.training, inplace=True)
        output = torch.flatten(output, 1)
        output = self.fc_linear(output)
        return output
