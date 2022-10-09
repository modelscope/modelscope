# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..weights_init import weights_init
from .detection_module import ConvBNActiv


class DetectionUNet(nn.Module):

    def __init__(self,
                 n_channels,
                 n_classes,
                 up_sampling_node='nearest',
                 init_weights=True):
        super(DetectionUNet, self).__init__()

        self.n_classes = n_classes
        self.up_sampling_node = up_sampling_node

        self.ec_images_1 = ConvBNActiv(
            n_channels, 64, bn=False, sample='down-3')
        self.ec_images_2 = ConvBNActiv(64, 128, sample='down-3')
        self.ec_images_3 = ConvBNActiv(128, 256, sample='down-3')
        self.ec_images_4 = ConvBNActiv(256, 512, sample='down-3')
        self.ec_images_5 = ConvBNActiv(512, 512, sample='down-3')
        self.ec_images_6 = ConvBNActiv(512, 512, sample='down-3')

        self.dc_images_6 = ConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_images_5 = ConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_images_4 = ConvBNActiv(512 + 256, 256, activ='leaky')
        self.dc_images_3 = ConvBNActiv(256 + 128, 128, activ='leaky')
        self.dc_images_2 = ConvBNActiv(128 + 64, 64, activ='leaky')
        self.dc_images_1 = nn.Conv2d(64 + n_channels, n_classes, kernel_size=1)

        if init_weights:
            self.apply(weights_init())

    def forward(self, input_images):

        ec_images = {}

        ec_images['ec_images_0'] = input_images
        ec_images['ec_images_1'] = self.ec_images_1(input_images)
        ec_images['ec_images_2'] = self.ec_images_2(ec_images['ec_images_1'])
        ec_images['ec_images_3'] = self.ec_images_3(ec_images['ec_images_2'])
        ec_images['ec_images_4'] = self.ec_images_4(ec_images['ec_images_3'])
        ec_images['ec_images_5'] = self.ec_images_5(ec_images['ec_images_4'])
        ec_images['ec_images_6'] = self.ec_images_6(ec_images['ec_images_5'])
        # --------------
        # images decoder
        # --------------
        logits = ec_images['ec_images_6']

        for _ in range(6, 0, -1):

            ec_images_skip = 'ec_images_{:d}'.format(_ - 1)
            dc_conv = 'dc_images_{:d}'.format(_)

            logits = F.interpolate(
                logits, scale_factor=2, mode=self.up_sampling_node)
            logits = torch.cat((logits, ec_images[ec_images_skip]), dim=1)

            logits = getattr(self, dc_conv)(logits)

        return logits
