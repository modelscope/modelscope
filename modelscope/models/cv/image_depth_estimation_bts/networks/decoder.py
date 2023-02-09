# The implementation is modified from cleinc / bts
# made publicly available under the GPL-3.0-or-later
# https://github.com/cleinc/bts/blob/master/pytorch/bts.py
import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func

from .tools import AtrousConv, LocalPlanarGuidance, Reduction1x1, UpConv


class Decoder(nn.Module):

    def __init__(self,
                 feat_out_channels,
                 max_depth=10,
                 dataset='nyu',
                 num_features=512):
        super(Decoder, self).__init__()
        self.max_depth = max_depth
        self.dataset = dataset

        self.upconv5 = UpConv(feat_out_channels[4], num_features)
        self.bn5 = nn.BatchNorm2d(
            num_features, momentum=0.01, affine=True, eps=1.1e-5)

        self.conv5 = torch.nn.Sequential(
            nn.Conv2d(
                num_features + feat_out_channels[3],
                num_features,
                3,
                1,
                1,
                bias=False), nn.ELU())
        self.upconv4 = UpConv(num_features, num_features // 2)
        self.bn4 = nn.BatchNorm2d(
            num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv4 = torch.nn.Sequential(
            nn.Conv2d(
                num_features // 2 + feat_out_channels[2],
                num_features // 2,
                3,
                1,
                1,
                bias=False), nn.ELU())
        self.bn4_2 = nn.BatchNorm2d(
            num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)

        self.daspp_3 = AtrousConv(
            num_features // 2, num_features // 4, 3, apply_bn_first=False)
        self.daspp_6 = AtrousConv(
            num_features // 2 + num_features // 4 + feat_out_channels[2],
            num_features // 4, 6)
        self.daspp_12 = AtrousConv(num_features + feat_out_channels[2],
                                   num_features // 4, 12)
        self.daspp_18 = AtrousConv(
            num_features + num_features // 4 + feat_out_channels[2],
            num_features // 4, 18)
        self.daspp_24 = AtrousConv(
            num_features + num_features // 2 + feat_out_channels[2],
            num_features // 4, 24)
        self.daspp_conv = torch.nn.Sequential(
            nn.Conv2d(
                num_features + num_features // 2 + num_features // 4,
                num_features // 4,
                3,
                1,
                1,
                bias=False), nn.ELU())
        self.reduc8x8 = Reduction1x1(num_features // 4, num_features // 4,
                                     self.max_depth)
        self.lpg8x8 = LocalPlanarGuidance(8)

        self.upconv3 = UpConv(num_features // 4, num_features // 4)
        self.bn3 = nn.BatchNorm2d(
            num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(
                num_features // 4 + feat_out_channels[1] + 1,
                num_features // 4,
                3,
                1,
                1,
                bias=False), nn.ELU())
        self.reduc4x4 = Reduction1x1(num_features // 4, num_features // 8,
                                     self.max_depth)
        self.lpg4x4 = LocalPlanarGuidance(4)

        self.upconv2 = UpConv(num_features // 4, num_features // 8)
        self.bn2 = nn.BatchNorm2d(
            num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(
                num_features // 8 + feat_out_channels[0] + 1,
                num_features // 8,
                3,
                1,
                1,
                bias=False), nn.ELU())

        self.reduc2x2 = Reduction1x1(num_features // 8, num_features // 16,
                                     self.max_depth)
        self.lpg2x2 = LocalPlanarGuidance(2)

        self.upconv1 = UpConv(num_features // 8, num_features // 16)
        self.reduc1x1 = Reduction1x1(
            num_features // 16,
            num_features // 32,
            self.max_depth,
            is_final=True)
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(
                num_features // 16 + 4,
                num_features // 16,
                3,
                1,
                1,
                bias=False), nn.ELU())
        self.get_depth = torch.nn.Sequential(
            nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False),
            nn.Sigmoid())

    def forward(self, features, focal):
        skip0, skip1, skip2, skip3 = features[1], features[2], features[
            3], features[4]
        dense_features = torch.nn.ReLU()(features[5])
        upconv5 = self.upconv5(dense_features)  # H/16
        upconv5 = self.bn5(upconv5)
        concat5 = torch.cat([upconv5, skip3], dim=1)
        iconv5 = self.conv5(concat5)

        upconv4 = self.upconv4(iconv5)  # H/8
        upconv4 = self.bn4(upconv4)
        concat4 = torch.cat([upconv4, skip2], dim=1)
        iconv4 = self.conv4(concat4)
        iconv4 = self.bn4_2(iconv4)

        daspp_3 = self.daspp_3(iconv4)
        concat4_2 = torch.cat([concat4, daspp_3], dim=1)
        daspp_6 = self.daspp_6(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        daspp_12 = self.daspp_12(concat4_3)
        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.daspp_18(concat4_4)
        concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
        daspp_24 = self.daspp_24(concat4_5)
        concat4_daspp = torch.cat(
            [iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        daspp_feat = self.daspp_conv(concat4_daspp)

        reduc8x8 = self.reduc8x8(daspp_feat)
        plane_normal_8x8 = reduc8x8[:, :3, :, :]
        plane_normal_8x8 = torch_nn_func.normalize(plane_normal_8x8, 2, 1)
        plane_dist_8x8 = reduc8x8[:, 3, :, :]
        plane_eq_8x8 = torch.cat(
            [plane_normal_8x8, plane_dist_8x8.unsqueeze(1)], 1)
        depth_8x8 = self.lpg8x8(plane_eq_8x8, focal)
        depth_8x8_scaled = depth_8x8.unsqueeze(1) / self.max_depth
        depth_8x8_scaled_ds = torch_nn_func.interpolate(
            depth_8x8_scaled, scale_factor=0.25, mode='nearest')

        upconv3 = self.upconv3(daspp_feat)  # H/4
        upconv3 = self.bn3(upconv3)
        concat3 = torch.cat([upconv3, skip1, depth_8x8_scaled_ds], dim=1)
        iconv3 = self.conv3(concat3)

        reduc4x4 = self.reduc4x4(iconv3)
        plane_normal_4x4 = reduc4x4[:, :3, :, :]
        plane_normal_4x4 = torch_nn_func.normalize(plane_normal_4x4, 2, 1)
        plane_dist_4x4 = reduc4x4[:, 3, :, :]
        plane_eq_4x4 = torch.cat(
            [plane_normal_4x4, plane_dist_4x4.unsqueeze(1)], 1)
        depth_4x4 = self.lpg4x4(plane_eq_4x4, focal)
        depth_4x4_scaled = depth_4x4.unsqueeze(1) / self.max_depth
        depth_4x4_scaled_ds = torch_nn_func.interpolate(
            depth_4x4_scaled, scale_factor=0.5, mode='nearest')

        upconv2 = self.upconv2(iconv3)  # H/2
        upconv2 = self.bn2(upconv2)
        concat2 = torch.cat([upconv2, skip0, depth_4x4_scaled_ds], dim=1)
        iconv2 = self.conv2(concat2)

        reduc2x2 = self.reduc2x2(iconv2)
        plane_normal_2x2 = reduc2x2[:, :3, :, :]
        plane_normal_2x2 = torch_nn_func.normalize(plane_normal_2x2, 2, 1)
        plane_dist_2x2 = reduc2x2[:, 3, :, :]
        plane_eq_2x2 = torch.cat(
            [plane_normal_2x2, plane_dist_2x2.unsqueeze(1)], 1)
        depth_2x2 = self.lpg2x2(plane_eq_2x2, focal)
        depth_2x2_scaled = depth_2x2.unsqueeze(1) / self.max_depth

        upconv1 = self.upconv1(iconv2)
        reduc1x1 = self.reduc1x1(upconv1)
        concat1_list = [
            upconv1, reduc1x1, depth_2x2_scaled, depth_4x4_scaled,
            depth_8x8_scaled
        ]
        concat1 = torch.cat(concat1_list, dim=1)
        iconv1 = self.conv1(concat1)
        final_depth = self.max_depth * self.get_depth(iconv1)
        if self.dataset == 'kitti':
            final_depth = final_depth * focal.view(-1, 1, 1,
                                                   1).float() / 715.0873

        return final_depth
