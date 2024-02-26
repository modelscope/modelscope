# Copyright (c) Alibaba, Inc. and its affiliates.
# The ZenNAS implementation is also open-sourced by the authors, and available at https://github.com/idstcv/ZenNAS.

import torch
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES
from torch import nn

from modelscope.models.cv.tinynas_classfication import (basic_blocks,
                                                        plain_net_utils)


@BACKBONES.register_module()
class MasterNet(plain_net_utils.PlainNet):

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
        """
        Any ReLU-CNN Backbone
        Args:
        plainnet_struct: (obj: str):
            Str of network topology structure.
        no_reslink: (obj:bool):
            no use residual structure.
        no_BN: (obj:bool):
            no use BN op.
        no_se: (obj:bool):
            no use se structure.
        no_se: (obj:bool):
            no use se structure.
        """

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

        num_classes = 2048
        super(MasterNet, self).__init__(
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

        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN
        self.use_se = use_se

        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = 1e-3

    def extract_stage_features_and_logit(self, x, target_downsample_ratio=4):
        stage_features_list = []
        image_size = x.shape[2]
        output = x
        block_id = 0
        for block_id, the_block in enumerate(self.block_list):
            output = the_block(output)
            dowsample_ratio = round(image_size / output.shape[2])
            if dowsample_ratio == target_downsample_ratio:
                stage_features_list.append(output)
                target_downsample_ratio *= 2
            pass
        pass

        return stage_features_list

    def forward(self, x):
        """
        Args:
            The input image
        Returns:
            The list of stage-level feature map
        """
        output = self.extract_stage_features_and_logit(x)
        return tuple(output)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=3.26033)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                hyper_para = m.weight.shape[0] + m.weight.shape[1]
                nn.init.normal_(m.weight, 0, 3.26033 * np.sqrt(2 / hyper_para))
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                pass
