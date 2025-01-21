# @Description: Some sub-modules for the network.
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07
# @https://github.com/doublez0108/geomvsnet

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """FPN aligncorners downsample 4x"""

    def __init__(self, base_channels, gn=False):
        super(FPN, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels, base_channels, 3, 1, padding=1, gn=gn),
        )

        self.conv1 = nn.Sequential(
            Conv2d(
                base_channels,
                base_channels * 2,
                5,
                stride=2,
                padding=2,
                gn=gn),
            Conv2d(
                base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
            Conv2d(
                base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
        )

        self.conv2 = nn.Sequential(
            Conv2d(
                base_channels * 2,
                base_channels * 4,
                5,
                stride=2,
                padding=2,
                gn=gn),
            Conv2d(
                base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
            Conv2d(
                base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
        )

        self.conv3 = nn.Sequential(
            Conv2d(
                base_channels * 4,
                base_channels * 8,
                5,
                stride=2,
                padding=2,
                gn=gn),
            Conv2d(
                base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
            Conv2d(
                base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
        )

        self.out_channels = [8 * base_channels]
        final_chs = base_channels * 8

        self.inner1 = nn.Conv2d(base_channels * 4, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner3 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out1 = nn.Conv2d(final_chs, base_channels * 8, 1, bias=False)
        self.out2 = nn.Conv2d(
            final_chs, base_channels * 4, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(
            final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.out4 = nn.Conv2d(
            final_chs, base_channels, 3, padding=1, bias=False)

        self.out_channels.append(base_channels * 4)
        self.out_channels.append(base_channels * 2)
        self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        intra_feat = conv3
        outputs = {}
        out1 = self.out1(intra_feat)

        intra_feat = F.interpolate(
            intra_feat, scale_factor=2, mode='bilinear',
            align_corners=True) + self.inner1(conv2)
        out2 = self.out2(intra_feat)

        intra_feat = F.interpolate(
            intra_feat, scale_factor=2, mode='bilinear',
            align_corners=True) + self.inner2(conv1)
        out3 = self.out3(intra_feat)

        intra_feat = F.interpolate(
            intra_feat, scale_factor=2, mode='bilinear',
            align_corners=True) + self.inner3(conv0)
        out4 = self.out4(intra_feat)

        outputs['stage1'] = out1
        outputs['stage2'] = out2
        outputs['stage3'] = out3
        outputs['stage4'] = out4

        return outputs


class Reg2d(nn.Module):

    def __init__(self, input_channel=128, base_channel=32):
        super(Reg2d, self).__init__()

        self.conv0 = ConvBnReLU3D(
            input_channel, base_channel, kernel_size=(1, 3, 3), pad=(0, 1, 1))
        self.conv1 = ConvBnReLU3D(
            base_channel,
            base_channel * 2,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            pad=(0, 1, 1))
        self.conv2 = ConvBnReLU3D(base_channel * 2, base_channel * 2)

        self.conv3 = ConvBnReLU3D(
            base_channel * 2,
            base_channel * 4,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            pad=(0, 1, 1))
        self.conv4 = ConvBnReLU3D(base_channel * 4, base_channel * 4)

        self.conv5 = ConvBnReLU3D(
            base_channel * 4,
            base_channel * 8,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            pad=(0, 1, 1))
        self.conv6 = ConvBnReLU3D(base_channel * 8, base_channel * 8)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channel * 8,
                base_channel * 4,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1),
                stride=(1, 2, 2),
                bias=False), nn.BatchNorm3d(base_channel * 4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channel * 4,
                base_channel * 2,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1),
                stride=(1, 2, 2),
                bias=False), nn.BatchNorm3d(base_channel * 2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channel * 2,
                base_channel,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1),
                stride=(1, 2, 2),
                bias=False), nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 1, stride=1, padding=0)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        return x.squeeze(1)


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    C = src_fea.shape[1]
    Hs, Ws = src_fea.shape[-2:]
    B, num_depth, Hr, Wr = depth_values.shape

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([
            torch.arange(0, Hr, dtype=torch.float32, device=src_fea.device),
            torch.arange(0, Wr, dtype=torch.float32, device=src_fea.device)
        ])
        y = y.reshape(Hr * Wr)
        x = x.reshape(Hr * Wr)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(
            1, 1, num_depth, 1) * depth_values.reshape(
                B, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.reshape(B, 3, 1,
                                                 1)  # [B, 3, Ndepth, H*W]
        # FIXME divide 0
        temp = proj_xyz[:, 2:3, :, :]
        temp[temp == 0] = 1e-9
        proj_xy = proj_xyz[:, :2, :, :] / temp  # [B, 2, Ndepth, H*W]
        # proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]

        proj_x_normalized = proj_xy[:, 0, :, :] / ((Ws - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((Hs - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized),
                              dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
    if len(src_fea.shape) == 4:
        warped_src_fea = F.grid_sample(
            src_fea,
            grid.reshape(B, num_depth * Hr, Wr, 2),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True)
        warped_src_fea = warped_src_fea.reshape(B, C, num_depth, Hr, Wr)
    elif len(src_fea.shape) == 5:
        warped_src_fea = []
        for d in range(src_fea.shape[2]):
            warped_src_fea.append(
                F.grid_sample(
                    src_fea[:, :, d],
                    grid.reshape(B, num_depth, Hr, Wr, 2)[:, d],
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True))
        warped_src_fea = torch.stack(warped_src_fea, dim=2)

    return warped_src_fea


def init_inverse_range(cur_depth, ndepths, device, dtype, H, W):
    inverse_depth_min = 1. / cur_depth[:, 0]  # (B,)
    inverse_depth_max = 1. / cur_depth[:, -1]
    itv = torch.arange(
        0, ndepths, device=device, dtype=dtype, requires_grad=False).reshape(
            1, -1, 1, 1).repeat(1, 1, H, W) / (ndepths - 1)  # 1 D H W
    inverse_depth_hypo = inverse_depth_max[:, None, None, None] + (
        inverse_depth_min - inverse_depth_max)[:, None, None, None] * itv

    return 1. / inverse_depth_hypo


def schedule_inverse_range(inverse_min_depth, inverse_max_depth, ndepths, H,
                           W):
    # cur_depth_min, (B, H, W)
    # cur_depth_max: (B, H, W)
    itv = torch.arange(
        0,
        ndepths,
        device=inverse_min_depth.device,
        dtype=inverse_min_depth.dtype,
        requires_grad=False).reshape(1, -1, 1, 1).repeat(
            1, 1, H // 2, W // 2) / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = inverse_max_depth[:, None, :, :] + (
        inverse_min_depth - inverse_max_depth)[:, None, :, :] * itv  # B D H W
    inverse_depth_hypo = F.interpolate(
        inverse_depth_hypo.unsqueeze(1), [ndepths, H, W],
        mode='trilinear',
        align_corners=True).squeeze(1)
    return 1. / inverse_depth_hypo


# --------------------------------------------------------------


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == 'kaiming':
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == 'xavier':
            nn.init.xavier_uniform_(module.weight)
    return


class ConvBnReLU3D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Conv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 relu=True,
                 bn_momentum=0.1,
                 init_method='xavier',
                 gn=False,
                 group_channel=8,
                 **kwargs):
        super(Conv2d, self).__init__()
        bn = not gn
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=(not bn),
            **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(
            out_channels, momentum=bn_momentum) if bn else None
        self.gn = nn.GroupNorm(
            int(max(1, out_channels
                    / group_channel)), out_channels) if gn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        else:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)
