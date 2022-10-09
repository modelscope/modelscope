# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.cv.skin_retouching.inpainting_model.gconv import \
    GatedConvBNActiv
from ..weights_init import weights_init


class RetouchingNet(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 up_sampling_node='nearest',
                 init_weights=True):
        super(RetouchingNet, self).__init__()

        self.freeze_ec_bn = False
        self.up_sampling_node = up_sampling_node

        self.ec_images_1 = GatedConvBNActiv(
            in_channels, 64, bn=False, sample='down-3')
        self.ec_images_2 = GatedConvBNActiv(64, 128, sample='down-3')
        self.ec_images_3 = GatedConvBNActiv(128, 256, sample='down-3')
        self.ec_images_4 = GatedConvBNActiv(256, 512, sample='down-3')
        self.ec_images_5 = GatedConvBNActiv(512, 512, sample='down-3')
        self.ec_images_6 = GatedConvBNActiv(512, 512, sample='down-3')

        self.dc_images_6 = GatedConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_images_5 = GatedConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_images_4 = GatedConvBNActiv(512 + 256, 256, activ='leaky')
        self.dc_images_3 = GatedConvBNActiv(256 + 128, 128, activ='leaky')
        self.dc_images_2 = GatedConvBNActiv(128 + 64, 64, activ='leaky')
        self.dc_images_1 = GatedConvBNActiv(
            64 + in_channels,
            out_channels,
            bn=False,
            sample='none-3',
            activ=None,
            bias=True)

        self.tanh = nn.Tanh()

        if init_weights:
            self.apply(weights_init())

    def forward(self, input_images, input_masks):

        ec_images = {}

        ec_images['ec_images_0'] = torch.cat((input_images, input_masks),
                                             dim=1)
        ec_images['ec_images_1'] = self.ec_images_1(ec_images['ec_images_0'])
        ec_images['ec_images_2'] = self.ec_images_2(ec_images['ec_images_1'])
        ec_images['ec_images_3'] = self.ec_images_3(ec_images['ec_images_2'])

        ec_images['ec_images_4'] = self.ec_images_4(ec_images['ec_images_3'])
        ec_images['ec_images_5'] = self.ec_images_5(ec_images['ec_images_4'])
        ec_images['ec_images_6'] = self.ec_images_6(ec_images['ec_images_5'])

        # --------------
        # images decoder
        # --------------
        dc_images = ec_images['ec_images_6']
        for _ in range(6, 0, -1):
            ec_images_skip = 'ec_images_{:d}'.format(_ - 1)
            dc_conv = 'dc_images_{:d}'.format(_)

            dc_images = F.interpolate(
                dc_images, scale_factor=2, mode=self.up_sampling_node)
            dc_images = torch.cat((dc_images, ec_images[ec_images_skip]),
                                  dim=1)

            dc_images = getattr(self, dc_conv)(dc_images)

        outputs = self.tanh(dc_images)

        return outputs

    def train(self, mode=True):

        super().train(mode)

        if self.freeze_ec_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
