# The implementation is modified from ErenBalatkan/Bts-PyTorch
# made publicly available under the MIT license
# https://github.com/ErenBalatkan/Bts-PyTorch/blob/master/BTS.py

import torch
import torch.nn as nn

from .utils import (MAX_DEPTH, ASSPBlock, LPGBlock, Reduction, UpscaleBlock,
                    UpscaleLayer, UpscaleNetwork, activation_fn)


class Decoder(nn.Module):

    def __init__(self, dataset='kitti'):
        super(Decoder, self).__init__()
        self.UpscaleNet = UpscaleNetwork()
        self.DenseASSPNet = ASSPBlock()

        self.upscale_block3 = UpscaleBlock(64, 96, 128)  # H4
        self.upscale_block4 = UpscaleBlock(128, 96, 128)  # H2

        self.LPGBlock8 = LPGBlock(8, 128)
        self.LPGBlock4 = LPGBlock(4, 64)  # 64 Filter
        self.LPGBlock2 = LPGBlock(2, 64)  # 64 Filter

        self.upconv_h4 = UpscaleLayer(128, 64)
        self.upconv_h2 = UpscaleLayer(64, 32)  # 64 Filter
        self.upconv_h = UpscaleLayer(64, 32)  # 32 filter

        self.conv_h4 = nn.Conv2d(161, 64, 3, 1, 1, bias=True)  # 64 Filter
        self.conv_h2 = nn.Conv2d(129, 64, 3, 1, 1, bias=True)  # 64 Filter
        self.conv_h1 = nn.Conv2d(36, 32, 3, 1, 1, bias=True)

        self.reduction1x1 = Reduction(1, 32, True)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1, bias=True)

        self.dataset = dataset

    def forward(self, joint_input, focal):
        (dense_features, dense_op_h2, dense_op_h4, dense_op_h8,
         dense_op_h16) = joint_input
        upscaled_out = self.UpscaleNet(joint_input)

        dense_assp_out = self.DenseASSPNet(upscaled_out)

        upconv_h4 = self.upconv_h4(dense_assp_out)
        depth_8x8 = self.LPGBlock8(dense_assp_out) / MAX_DEPTH
        depth_8x8_ds = nn.functional.interpolate(
            depth_8x8, scale_factor=1 / 4, mode='nearest')
        depth_concat_4x4 = torch.cat((depth_8x8_ds, dense_op_h4, upconv_h4), 1)

        conv_h4 = activation_fn(self.conv_h4(depth_concat_4x4))
        upconv_h2 = self.upconv_h2(conv_h4)
        depth_4x4 = self.LPGBlock4(conv_h4) / MAX_DEPTH

        depth_4x4_ds = nn.functional.interpolate(
            depth_4x4, scale_factor=1 / 2, mode='nearest')
        depth_concat_2x2 = torch.cat((depth_4x4_ds, dense_op_h2, upconv_h2), 1)

        conv_h2 = activation_fn(self.conv_h2(depth_concat_2x2))
        upconv_h = self.upconv_h(conv_h2)
        depth_1x1 = self.reduction1x1(upconv_h)
        depth_2x2 = self.LPGBlock2(conv_h2) / MAX_DEPTH

        depth_concat = torch.cat(
            (upconv_h, depth_1x1, depth_2x2, depth_4x4, depth_8x8), 1)
        depth = activation_fn(self.conv_h1(depth_concat))
        depth = self.final_conv(depth).sigmoid() * MAX_DEPTH + 0.1
        if self.dataset == 'kitti':
            depth *= focal.view(-1, 1, 1, 1) / 715.0873
        return depth
