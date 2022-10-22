# The implementation here is modified based on nanodet,
# originally Apache 2.0 License and publicly avaialbe at https://github.com/RangiLyu/nanodet

import torch
import torch.nn as nn

from .ghost_pan import GhostPAN
from .nanodet_plus_head import NanoDetPlusHead
from .shufflenetv2 import ShuffleNetV2


class OneStageDetector(nn.Module):

    def __init__(self):
        super(OneStageDetector, self).__init__()
        self.backbone = ShuffleNetV2(
            model_size='1.0x',
            out_stages=(2, 3, 4),
            with_last_conv=False,
            kernal_size=3,
            activation='LeakyReLU',
            pretrain=False)
        self.fpn = GhostPAN(
            in_channels=[116, 232, 464],
            out_channels=96,
            use_depthwise=True,
            kernel_size=5,
            expand=1,
            num_blocks=1,
            use_res=False,
            num_extra_level=1,
            upsample_cfg=dict(scale_factor=2, mode='bilinear'),
            norm_cfg=dict(type='BN'),
            activation='LeakyReLU')
        self.head = NanoDetPlusHead(
            num_classes=3,
            input_channel=96,
            feat_channels=96,
            stacked_convs=2,
            kernel_size=5,
            strides=[8, 16, 32, 64],
            conv_type='DWConv',
            norm_cfg=dict(type='BN'),
            reg_max=7,
            activation='LeakyReLU',
            assigner_cfg=dict(topk=13))
        self.epoch = 0

    def forward(self, x):
        x = self.backbone(x)
        if hasattr(self, 'fpn'):
            x = self.fpn(x)
        if hasattr(self, 'head'):
            x = self.head(x)
        return x

    def inference(self, meta):
        with torch.no_grad():
            preds = self(meta['img'])
            results = self.head.post_process(preds, meta)
        return results
