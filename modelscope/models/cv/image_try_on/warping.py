# The implementation here is modified based on flow-style-vton,
# originally Apache 2.0 License and publicly available at https://github.com/SenHe/Flow-Style-VTON

from collections import OrderedDict
from math import sqrt

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid(
        [torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)
    grid_list = [
        grid.float().unsqueeze(0) + offset[:, dim, ...]
        for dim, grid in enumerate(grid_list)
    ]
    grid_list = [
        grid / ((size - 1.0) / 2.0) - 1.0
        for grid, size in zip(grid_list, reversed(sizes))
    ]

    return torch.stack(grid_list, dim=-1)


class Conv2dBlock(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 stride,
                 padding=0,
                 norm='batch',
                 activation='prelu',
                 pad_type='zero',
                 bias=True):
        super().__init__()
        self.use_bias = bias
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, 'Unsupported padding type: {}'.format(pad_type)

        norm_dim = input_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(
            input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        x = self.conv(self.pad(x))
        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.PReLU(),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=1,
                bias=False), nn.BatchNorm2d(in_channels), nn.PReLU(),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=1,
                bias=False))

    def forward(self, x):
        return self.block(x) + x


class ResBlock_2(nn.Module):

    def __init__(self, in_channels):
        super(ResBlock_2, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.PReLU(),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=1,
                bias=False), nn.BatchNorm2d(in_channels), nn.PReLU(),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=1,
                bias=False))

    def forward(self, x1, x2):
        return self.block(x1) + x2


class DownSample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.PReLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False))

    def forward(self, x):
        return self.block(x)


class UpSample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.PReLU(),
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2,
                bias=False))

    def forward(self, x):
        return self.block(x)

    def upsample(self, F, scale):
        """[2x nearest neighbor upsampling]
        Arguments:
            F {[torch tensor]} -- [tensor to be upsampled, (B, C, H, W)]
        """
        upsample = torch.nn.Upsample(scale_factor=scale, mode='nearest')
        return upsample(F)


class FeatureEncoder(nn.Module):

    def __init__(self, in_channels, chns=[64, 128, 256, 256, 256]):
        # in_channels = 3 for images, and is larger (e.g., 17+1+1) for agnositc representation
        super(FeatureEncoder, self).__init__()
        self.encoders = []
        for i, out_chns in enumerate(chns):
            if i == 0:
                encoder = nn.Sequential(
                    DownSample(in_channels, out_chns), ResBlock(out_chns),
                    ResBlock(out_chns))
            else:
                encoder = nn.Sequential(
                    DownSample(chns[i - 1], out_chns), ResBlock(out_chns),
                    ResBlock(out_chns))

            self.encoders.append(encoder)

        self.encoders = nn.ModuleList(self.encoders)

    def forward(self, x):
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        return encoder_features


class RefinePyramid(nn.Module):

    def __init__(self, chns=[64, 128, 256, 256, 256], fpn_dim=256):
        super(RefinePyramid, self).__init__()
        self.chns = chns

        # adaptive
        self.adaptive = []
        for in_chns in list(reversed(chns)):
            adaptive_layer = nn.Conv2d(in_chns, fpn_dim, kernel_size=1)
            self.adaptive.append(adaptive_layer)
        self.adaptive = nn.ModuleList(self.adaptive)
        # output conv
        self.smooth = []
        for i in range(len(chns)):
            smooth_layer = nn.Conv2d(
                fpn_dim, fpn_dim, kernel_size=3, padding=1)
            self.smooth.append(smooth_layer)
        self.smooth = nn.ModuleList(self.smooth)

    def forward(self, x):
        conv_ftr_list = x

        feature_list = []
        last_feature = None
        for i, conv_ftr in enumerate(list(reversed(conv_ftr_list))):
            # adaptive
            feature = self.adaptive[i](conv_ftr)
            # fuse
            if last_feature is not None:
                feature = feature + F.interpolate(
                    last_feature, scale_factor=2, mode='nearest')
            # smooth
            feature = self.smooth[i](feature)
            last_feature = feature
            feature_list.append(feature)

        return tuple(reversed(feature_list))


def Round(x):
    ''' x: tensor '''
    return torch.round(x) - x.detach() + x


def Camp(x, mi=0, ma=192):
    ''' x: tensor '''
    if x < mi:
        x = (mi - x.detach()) + x
    elif x > ma:
        x = x - (x.detach() - ma)
    return x


class CorrelationLayer(nn.Module):

    def __init__(self, init_scale=5):
        super(CorrelationLayer, self).__init__()
        self.init_scale = init_scale
        self.softmax3 = nn.Softmax(dim=3)
        self.sig = nn.Sigmoid()
        self.reduce1 = Conv2dBlock(
            64, 4, 1, 1, 0, activation='prelu', norm='none')
        self.reduce2 = Conv2dBlock(
            64, 4, 1, 1, 0, activation='prelu', norm='none')

        init_h, init_w = 192 / self.init_scale, 256 / self.init_scale
        self.step_h, self.step_w = [], []
        step_h_half = [i + 1 for i in range(0, int(init_h + 1), 2)]
        step_h_half_re = list(reversed(step_h_half[1:]))
        self.step_h.extend(step_h_half_re)
        self.step_h.extend(step_h_half)
        step_w_half = [i + 1 for i in range(0, int(init_w + 1), 2)]
        step_w_half_re = list(reversed(step_w_half[1:]))
        self.step_w.extend(step_w_half_re)
        self.step_w.extend(step_w_half)
        self.mask_h, self.mask_w = len(self.step_h), len(self.step_w)

    def forward(self, location, fea_c, fea_p, scale_param, H, W, c_landmark,
                p_landmark):
        init_h, init_w = torch.tensor(
            W / self.init_scale, device=fea_c.device), torch.tensor(
                H / self.init_scale, device=fea_c.device)
        N, C, fea_H = fea_c.shape[0], fea_c.shape[1], fea_c.shape[2]
        downsample_ratio = H / fea_H
        landmark_flow = -1 * torch.ones((N, 64, H, W), device=fea_c.device)
        mask_h, mask_w = self.mask_h, self.mask_w

        fea_cn = self.upsample(fea_c, scale=downsample_ratio)
        fea_pn = self.upsample(fea_p, scale=downsample_ratio)

        fea_cn = self.reduce1(fea_cn)
        fea_pn = self.reduce2(fea_pn)
        C = fea_cn.shape[1]

        src_box_h0 = torch.tensor(
            self.step_h,
            device=fea_cn.device).unsqueeze(dim=0).unsqueeze(dim=0).repeat(
                N, 32, 1)
        src_box_w0 = torch.tensor(
            self.step_w,
            device=fea_cn.device).unsqueeze(dim=0).unsqueeze(dim=0).repeat(
                N, 32, 1)
        tar_box_h0 = torch.tensor(
            self.step_h,
            device=fea_cn.device).unsqueeze(dim=0).unsqueeze(dim=0).repeat(
                N, 32, 1)
        tar_box_w0 = torch.tensor(
            self.step_w,
            device=fea_cn.device).unsqueeze(dim=0).unsqueeze(dim=0).repeat(
                N, 32, 1)
        adj_cw, adj_ch, adj_pw, adj_ph = scale_param[:, :, 0, 0], scale_param[:, :, 0, 1],\
            scale_param[:, :, 1, 0], scale_param[:, :, 1, 1]
        c_h, c_w = (init_h * self.sig(adj_ch)).unsqueeze(
            dim=2), (init_w * self.sig(adj_cw)).unsqueeze(dim=2)
        p_h, p_w = (init_h * self.sig(adj_ph)).unsqueeze(
            dim=2), (init_w * self.sig(adj_pw)).unsqueeze(dim=2)
        src_box_h = self.sig((c_h - src_box_h0) * 2)
        src_box_w = self.sig((c_w - src_box_w0) * 2)
        tar_box_h = self.sig((p_h - tar_box_h0) * 2)
        tar_box_w = self.sig((p_w - tar_box_w0) * 2)

        src_box_h, src_box_w = src_box_h.unsqueeze(dim=3), src_box_w.unsqueeze(
            dim=2)
        tar_box_h, tar_box_w = tar_box_h.unsqueeze(dim=3), tar_box_w.unsqueeze(
            dim=2)
        src_mask = torch.matmul(src_box_h, src_box_w)
        tar_mask = torch.matmul(tar_box_h, tar_box_w)

        cloth_patch_all = torch.zeros((N, 32 * C, mask_h, mask_w),
                                      device=fea_cn.device)
        person_patch_all = torch.zeros((N, 32 * C, mask_h, mask_w),
                                       device=fea_cn.device)

        location_patch_all = -1 * torch.ones(
            (N, 64, mask_h, mask_w), device=fea_cn.device)
        one_flow_all = -1 * torch.ones(
            (N, 64, mask_h, mask_w), device=fea_cn.device)
        coord_info = []
        mask = torch.zeros((N, 32, H, W), device=fea_cn.device)

        for b in range(N):
            for i in range(32):
                c_center_x, c_center_y = c_landmark[b, i, 0], c_landmark[b, i,
                                                                         1]
                p_center_x, p_center_y = p_landmark[b, i, 0], p_landmark[b, i,
                                                                         1]
                if (c_center_x == 0
                        and c_center_y == 0) or (p_center_x == 0
                                                 and p_center_y == 0):
                    p_new_top_y, p_new_bottom_y, p_new_left_x, p_new_right_x, p_new_h, p_new_w = 0, 0, 0, 0, 0, 0
                else:
                    c_left_x = torch.floor(c_center_x.int() - mask_w / 2)
                    c_right_x = torch.floor(c_center_x.int() + mask_w / 2)
                    delta_x1 = int(torch.clamp(0 - c_left_x, min=0))
                    delta_x2 = int(torch.clamp(c_right_x - W, min=0))
                    c_left_x, c_right_x = int(c_left_x), int(c_right_x)
                    c_top_y = torch.floor(c_center_y.int() - mask_h / 2)
                    c_bottom_y = torch.floor(c_center_y.int() + mask_h / 2)
                    delta_y1 = int(torch.clamp(0 - c_top_y, min=0))
                    delta_y2 = int(torch.clamp(c_bottom_y - H, min=0))
                    c_top_y, c_bottom_y = int(c_top_y), int(c_bottom_y)

                    p_left_x = torch.floor(p_center_x.int() - mask_w / 2)
                    p_right_x = torch.floor(p_center_x.int() + mask_w / 2)
                    delta_x3 = int(torch.clamp(0 - p_left_x, min=0))
                    delta_x4 = int(torch.clamp(p_right_x - W, min=0))
                    p_left_x, p_right_x = int(p_left_x), int(p_right_x)
                    p_top_y = torch.floor(p_center_y.int() - mask_h / 2)
                    p_bottom_y = torch.floor(p_center_y.int() + mask_h / 2)
                    delta_y3 = int(torch.clamp(0 - p_top_y, min=0))
                    delta_y4 = int(torch.clamp(p_bottom_y - H, min=0))
                    p_top_y, p_bottom_y = int(p_top_y), int(p_bottom_y)

                    # Deformable Patch
                    c_new_top_y, c_new_bottom_y, c_new_left_x, c_new_right_x = \
                        c_top_y + delta_y1, c_bottom_y - delta_y2, c_left_x + delta_x1, c_right_x - delta_x2
                    p_new_top_y, p_new_bottom_y, p_new_left_x, p_new_right_x = \
                        p_top_y + delta_y3, p_bottom_y - delta_y4, p_left_x + delta_x3, p_right_x - delta_x4
                    c_new_h, c_new_w = c_new_bottom_y - c_new_top_y, c_new_right_x - c_new_left_x
                    p_new_h, p_new_w = p_new_bottom_y - p_new_top_y, p_new_right_x - p_new_left_x

                    cloth_patch_all[b, i * C:(i + 1) * C, : c_new_h, : c_new_w] = \
                        fea_cn[b, :, c_new_top_y: c_new_bottom_y, c_new_left_x: c_new_right_x] * \
                        src_mask[b, i, 0 + delta_y1: mask_h - delta_y2, 0 + delta_x1: mask_w - delta_x2]
                    person_patch_all[b, i * C: (i + 1) * C, : p_new_h, : p_new_w] = \
                        fea_pn[b, :, p_new_top_y: p_new_bottom_y, p_new_left_x: p_new_right_x] * \
                        tar_mask[b, i, 0 + delta_y3: mask_h - delta_y4, 0 + delta_x3: mask_w - delta_x4]
                    location_patch_all[b, i * 2:i * 2
                                       + 2, :c_new_h, :c_new_w] = location[
                                           b, i * 2:i * 2 + 2,
                                           c_new_top_y:c_new_bottom_y,
                                           c_new_left_x:c_new_right_x]

                coord_info.append([
                    p_new_top_y, p_new_bottom_y, p_new_left_x, p_new_right_x,
                    p_new_h, p_new_w
                ])

        Q = person_patch_all.view(N, 32, C, mask_h,
                                  mask_w).view(N, 32, C,
                                               -1).permute(0, 1, 3,
                                                           2).contiguous()
        K = cloth_patch_all.view(N, 32, C, mask_h, mask_w).view(N, 32, C, -1)
        correlation = self.softmax3(torch.matmul(Q, K))
        V = location_patch_all.view(N, 32, 2, mask_h,
                                    mask_w).view(N, 32, 2,
                                                 -1).permute(0, 1, 3,
                                                             2).contiguous()
        one_flow_all = torch.matmul(correlation,
                                    V).permute(0, 1, 3, 2).contiguous().view(
                                        N, 32 * 2, mask_h, mask_w)

        for b in range(N):
            for i in range(32):
                p_new_top_y, p_new_bottom_y, p_new_left_x, p_new_right_x, p_new_h, p_new_w = coord_info[
                    b * 32 + i]
                if p_new_h != 0 and p_new_w != 0:
                    landmark_flow[b, i * 2: i * 2 + 2, p_new_top_y: p_new_bottom_y, p_new_left_x: p_new_right_x] = \
                        one_flow_all[b, i * 2: i * 2 + 2, : p_new_h, : p_new_w]
                    mask[b, i, p_new_top_y:p_new_bottom_y,
                         p_new_left_x:p_new_right_x] = torch.tensor(
                             1, device=fea_cn.device)

        return landmark_flow, mask

    def upsample(self, F, scale):
        """[2x nearest neighbor upsampling]
        Arguments:
            F {[torch tensor]} -- [tensor to be upsampled, (B, C, H, W)]
        """
        upsample = torch.nn.Upsample(scale_factor=scale, mode='nearest')
        return upsample(F)


class LocalFlow(nn.Module):

    def __init__(self, fpn_dim=256, use_num=3, init_scale=6):
        super(LocalFlow, self).__init__()
        self.use_num = use_num
        self.fusers = []
        self.fusers_refine = []
        self.location_preds = []
        self.patch_preds = []
        self.map_preds = []
        self.map_refine = []
        self.reduc = []

        for i in range(use_num):
            in_chns = fpn_dim
            map_refine = nn.Conv2d(
                32, 32, kernel_size=3, stride=1, padding=1, groups=32)

            reduc = nn.Sequential(
                Conv2dBlock(
                    in_chns, 64, 3, 1, 1, activation='prelu', bias=False))
            fusers = nn.Sequential(
                Conv2dBlock(
                    fpn_dim * 2,
                    fpn_dim,
                    3,
                    1,
                    1,
                    activation='prelu',
                    bias=False),
                Conv2dBlock(
                    fpn_dim, 32, 3, 1, 1, activation='prelu', bias=False))
            fusers_refine_layer = [
                nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True),
                ResBlock(32)
            ] * (
                i + 1)
            fusers_refine = nn.Sequential(*fusers_refine_layer)
            location_preds = nn.Sequential(
                Conv2dBlock(32, 64, 3, 1, 1, activation='prelu', norm='none'),
                Conv2dBlock(64, 64, 3, 1, 1, activation='prelu', norm='none'),
                Conv2dBlock(64, 64, 3, 1, 1, activation='prelu', norm='none'),
            )
            # patch
            patch_layer = [DownSample(32, 32), ResBlock(32)] * (use_num + 2)
            patch_layer.append(nn.AdaptiveAvgPool2d((2, 2)))
            patch_preds = nn.Sequential(*patch_layer)
            # attention map
            map_layer = []
            map_layer = [
                ResBlock(32),
                Conv2dBlock(32, 32, 3, 1, 1, activation='prelu', norm='none')
            ]
            map_preds = nn.Sequential(*map_layer)

            self.reduc.append(reduc)
            self.fusers.append(fusers)
            self.fusers_refine.append(fusers_refine)
            self.location_preds.append(location_preds)
            self.patch_preds.append(patch_preds)
            self.map_preds.append(map_preds)
            self.map_refine.append(map_refine)

        self.reduc = nn.ModuleList(self.reduc)

        self.fusers = nn.ModuleList(self.fusers)
        self.fusers_refine = nn.ModuleList(self.fusers_refine)
        self.location_preds = nn.ModuleList(self.location_preds)
        self.patch_preds = nn.ModuleList(self.patch_preds)
        self.map_preds = nn.ModuleList(self.map_preds)
        self.map_refine = nn.ModuleList(self.map_refine)
        self.CorrelationLayer = CorrelationLayer(init_scale=init_scale)

    def forward(self, cloth, fea_c, fea_p, c_landmark, p_landmark):
        flow_list, map_list = [], []
        N, _, H, W = cloth.shape
        for i in range(self.use_num):
            fuse = self.fusers[i](torch.cat((fea_c[i], fea_p[i]), dim=1))
            fuse = self.fusers_refine[i](fuse)
            location = self.location_preds[i](fuse)
            patch = self.patch_preds[i](fuse)
            att_map = self.map_preds[i](fuse)
            fea_c_reduc = self.reduc[i](fea_c[i])
            fea_p_reduc = self.reduc[i](fea_p[i])
            flow, mask = self.CorrelationLayer(location, fea_c_reduc,
                                               fea_p_reduc, patch, H, W,
                                               c_landmark, p_landmark)
            if i != 0:
                flow = (flow + last_flow) / 2
                add_map = torch.add(last_att_map, att_map)
                att_map = self.map_refine[i](add_map)

            last_flow = flow
            flow_list.append(last_flow)
            last_att_map = att_map * mask
            map_list.append(last_att_map)

        return flow_list, map_list


class EqualLR:

    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ModulatedConv2d(nn.Module):

    def __init__(self,
                 fin,
                 fout,
                 kernel_size,
                 padding_type='zero',
                 upsample=False,
                 downsample=False,
                 latent_dim=512,
                 normalize_mlp=False):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = fin
        self.out_channels = fout
        self.kernel_size = kernel_size
        padding_size = kernel_size // 2

        if kernel_size == 1:
            self.demudulate = False
        else:
            self.demudulate = True

        self.weight = nn.Parameter(
            torch.Tensor(fout, fin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(1, fout, 1, 1))

        if normalize_mlp:
            self.mlp_class_std = nn.Sequential(
                EqualLinear(latent_dim, fin), PixelNorm())
        else:
            self.mlp_class_std = EqualLinear(latent_dim, fin)

        if padding_type == 'reflect':
            self.padding = nn.ReflectionPad2d(padding_size)
        else:
            self.padding = nn.ZeroPad2d(padding_size)

        self.weight.data.normal_()
        self.bias.data.zero_()

    def forward(self, input, latent):
        fan_in = self.weight.data.size(1) * self.weight.data[0][0].numel()
        weight = self.weight * sqrt(2 / fan_in)
        weight = weight.view(1, self.out_channels, self.in_channels,
                             self.kernel_size, self.kernel_size)

        s = self.mlp_class_std(latent).view(-1, 1, self.in_channels, 1, 1)
        weight = s * weight
        if self.demudulate:
            d = torch.rsqrt((weight**2).sum(4).sum(3).sum(2) + 1e-5).view(
                -1, self.out_channels, 1, 1, 1)
            weight = (d * weight).view(-1, self.in_channels, self.kernel_size,
                                       self.kernel_size)
        else:
            weight = weight.view(-1, self.in_channels, self.kernel_size,
                                 self.kernel_size)

        batch, _, height, width = input.shape

        input = input.view(1, -1, height, width)
        input = self.padding(input)
        out = F.conv2d(
            input, weight, groups=batch).view(batch, self.out_channels, height,
                                              width) + self.bias

        return out


class StyledConvBlock(nn.Module):

    def __init__(self,
                 fin,
                 fout,
                 latent_dim=256,
                 padding='zero',
                 actvn='lrelu',
                 normalize_affine_output=False,
                 modulated_conv=False):
        super(StyledConvBlock, self).__init__()
        if not modulated_conv:
            if padding == 'reflect':
                padding_layer = nn.ReflectionPad2d
            else:
                padding_layer = nn.ZeroPad2d

        if modulated_conv:
            conv2d = ModulatedConv2d
        else:
            conv2d = EqualConv2d

        if modulated_conv:
            self.actvn_gain = sqrt(2)
        else:
            self.actvn_gain = 1.0

        self.modulated_conv = modulated_conv

        if actvn == 'relu':
            activation = nn.ReLU(True)
        else:
            activation = nn.LeakyReLU(0.2, True)

        if self.modulated_conv:
            self.conv0 = conv2d(
                fin,
                fout,
                kernel_size=3,
                padding_type=padding,
                upsample=False,
                latent_dim=latent_dim,
                normalize_mlp=normalize_affine_output)
        else:
            conv0 = conv2d(fin, fout, kernel_size=3)

            seq0 = [padding_layer(1), conv0]
            self.conv0 = nn.Sequential(*seq0)

        self.actvn0 = activation

        if self.modulated_conv:
            self.conv1 = conv2d(
                fout,
                fout,
                kernel_size=3,
                padding_type=padding,
                downsample=False,
                latent_dim=latent_dim,
                normalize_mlp=normalize_affine_output)
        else:
            conv1 = conv2d(fout, fout, kernel_size=3)
            seq1 = [padding_layer(1), conv1]
            self.conv1 = nn.Sequential(*seq1)

        self.actvn1 = activation

    def forward(self, input, latent=None):
        if self.modulated_conv:
            out = self.conv0(input, latent)
        else:
            out = self.conv0(input)

        out = self.actvn0(out) * self.actvn_gain

        if self.modulated_conv:
            out = self.conv1(out, latent)
        else:
            out = self.conv1(out)

        out = self.actvn1(out) * self.actvn_gain

        return out


class Styled_F_ConvBlock(nn.Module):

    def __init__(self,
                 fin,
                 fout,
                 latent_dim=256,
                 padding='zero',
                 actvn='prelu',
                 normalize_affine_output=False,
                 modulated_conv=False):
        super(Styled_F_ConvBlock, self).__init__()
        if not modulated_conv:
            if padding == 'reflect':
                padding_layer = nn.ReflectionPad2d
            else:
                padding_layer = nn.ZeroPad2d

        if modulated_conv:
            conv2d = ModulatedConv2d
        else:
            conv2d = EqualConv2d

        if modulated_conv:
            self.actvn_gain = sqrt(2)
        else:
            self.actvn_gain = 1.0

        self.modulated_conv = modulated_conv

        if actvn == 'relu':
            activation = nn.ReLU(True)
        else:
            activation = nn.LeakyReLU(0.2, True)

        if self.modulated_conv:
            self.conv0 = conv2d(
                fin,
                128,
                kernel_size=3,
                padding_type=padding,
                upsample=False,
                latent_dim=latent_dim,
                normalize_mlp=normalize_affine_output)
        else:
            conv0 = conv2d(fin, 128, kernel_size=3)

            seq0 = [padding_layer(1), conv0]
            self.conv0 = nn.Sequential(*seq0)

        self.actvn0 = activation

        if self.modulated_conv:
            self.conv1 = conv2d(
                128,
                fout,
                kernel_size=3,
                padding_type=padding,
                downsample=False,
                latent_dim=latent_dim,
                normalize_mlp=normalize_affine_output)
        else:
            conv1 = conv2d(128, fout, kernel_size=3)
            seq1 = [padding_layer(1), conv1]
            self.conv1 = nn.Sequential(*seq1)

    def forward(self, input, latent=None):
        if self.modulated_conv:
            out = self.conv0(input, latent)
        else:
            out = self.conv0(input)

        out = self.actvn0(out) * self.actvn_gain

        if self.modulated_conv:
            out = self.conv1(out, latent)
        else:
            out = self.conv1(out)

        return out


class AFlowNet(nn.Module):

    def __init__(self, num_pyramid, fpn_dim=256, use_num=3):
        super(AFlowNet, self).__init__()

        padding_type = 'zero'
        actvn = 'lrelu'
        normalize_mlp = False
        modulated_conv = True
        self.use_num = use_num

        self.netRefine = []
        self.netStyle = []
        self.netF = []
        self.map_preds = []
        self.map_refine = []

        for i in range(num_pyramid):

            netRefine_layer = torch.nn.Sequential(
                torch.nn.Conv2d(
                    2 * fpn_dim,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1), nn.PReLU(), ResBlock(128), ResBlock(128),
                torch.nn.Conv2d(
                    in_channels=128,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1), nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1), nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=32,
                    out_channels=12,
                    kernel_size=3,
                    stride=1,
                    padding=1), nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=12,
                    out_channels=2,
                    kernel_size=3,
                    stride=1,
                    padding=1))

            style_block = torch.nn.Sequential(
                torch.nn.Conv2d(
                    2 * fpn_dim,
                    out_channels=fpn_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1), nn.PReLU(),
                torch.nn.Conv2d(
                    fpn_dim,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1), nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=128,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1), nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1), nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=32,
                    out_channels=2,
                    kernel_size=3,
                    stride=1,
                    padding=1))

            style_F_block = Styled_F_ConvBlock(
                49,
                2,
                latent_dim=256,
                padding=padding_type,
                actvn=actvn,
                normalize_affine_output=normalize_mlp,
                modulated_conv=modulated_conv)

            map_preds = torch.nn.Sequential(
                torch.nn.Conv2d(
                    2 * fpn_dim,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1), nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=128,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1), nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1), nn.PReLU(),
                torch.nn.Conv2d(
                    in_channels=32,
                    out_channels=2,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            if i == 0:
                map_refine = nn.Sequential(
                    Conv2dBlock(
                        2, 2, 3, 1, 1, activation='prelu', norm='none'),
                    Conv2dBlock(
                        2, 2, 3, 1, 1, activation='prelu', norm='none'))
            elif i >= 2:
                map_refine = nn.Sequential(
                    Conv2dBlock(
                        36, 16, 3, 1, 1, activation='prelu', norm='none'),
                    Conv2dBlock(
                        16, 4, 3, 1, 1, activation='prelu', norm='none'),
                    Conv2dBlock(
                        4, 2, 3, 1, 1, activation='prelu', norm='none'))
            else:
                map_refine = nn.Sequential(
                    Conv2dBlock(
                        4, 2, 3, 1, 1, activation='prelu', norm='none'),
                    Conv2dBlock(
                        2, 2, 3, 1, 1, activation='prelu', norm='none'))

            self.netRefine.append(netRefine_layer)
            self.netStyle.append(style_block)
            self.netF.append(style_F_block)
            self.map_preds.append(map_preds)
            self.map_refine.append(map_refine)

        self.netRefine = nn.ModuleList(self.netRefine)
        self.netStyle = nn.ModuleList(self.netStyle)
        self.netF = nn.ModuleList(self.netF)
        self.map_preds = nn.ModuleList(self.map_preds)
        self.map_refine = nn.ModuleList(self.map_refine)

        self.cond_style = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=(8, 6), stride=1, padding=0),
            nn.PReLU())
        self.image_style = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=(8, 6), stride=1, padding=0),
            nn.PReLU())

    def forward(self, global_flow_input, device, warp_feature=True):

        x = global_flow_input[0]
        x_edge = global_flow_input[1]
        x_warps = global_flow_input[2]
        x_conds = global_flow_input[3]
        localmap_list = global_flow_input[5]
        last_flow = None
        last_gmap = None
        gmap_all = []
        last_flow_all = []
        delta_list = []
        x_all = []
        x_edge_all = []
        cond_fea_all = []
        delta_x_all = []
        delta_y_all = []
        filter_x = [[0, 0, 0], [1, -2, 1], [0, 0, 0]]
        filter_y = [[0, 1, 0], [0, -2, 0], [0, 1, 0]]
        filter_diag1 = [[1, 0, 0], [0, -2, 0], [0, 0, 1]]
        filter_diag2 = [[0, 0, 1], [0, -2, 0], [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2

        weight_array = torch.FloatTensor(weight_array).permute(3, 2, 0,
                                                               1).to(device)
        self.weight = nn.Parameter(data=weight_array, requires_grad=False)

        fea_num = len(x_warps)

        for i in range(fea_num):
            x_warp = x_warps[fea_num - 1 - i]
            x_cond = x_conds[fea_num - 1 - i]
            cond_fea_all.append(x_cond)

            if last_flow is not None and warp_feature:
                x_warp_after = F.grid_sample(
                    x_warp,
                    last_flow.detach().permute(0, 2, 3, 1),
                    mode='bilinear',
                    padding_mode='border')
            else:
                x_warp_after = x_warp

            flow = self.netStyle[i](torch.cat([x_warp_after, x_cond], 1))
            delta_list.append(flow)
            flow = apply_offset(flow)
            if last_flow is not None:
                flow = F.grid_sample(
                    last_flow, flow, mode='bilinear', padding_mode='border')
            else:
                flow = flow.permute(0, 3, 1, 2)

            last_flow = flow
            x_warp = F.grid_sample(
                x_warp,
                flow.permute(0, 2, 3, 1),
                mode='bilinear',
                padding_mode='border')
            concat = torch.cat([x_warp, x_cond], 1)
            flow = self.netRefine[i](concat)

            g_map = self.map_preds[i](concat)
            if i >= fea_num - self.use_num:
                upsample_tmp = 0.5**(fea_num - i)
                g_map = self.map_refine[i](
                    torch.cat([
                        g_map, last_gmap,
                        self.upsample(
                            localmap_list[i - (fea_num - self.use_num)],
                            upsample_tmp)
                    ], 1))
            elif last_gmap is not None:
                g_map = self.map_refine[i](torch.cat([g_map, last_gmap], 1))
            elif i == 0:
                g_map = self.map_refine[i](g_map)
            last_gmap = self.upsample(g_map, 2)
            gmap_all.append(last_gmap)

            delta_list.append(flow)
            flow = apply_offset(flow)
            flow = F.grid_sample(
                last_flow, flow, mode='bilinear', padding_mode='border')

            last_flow = F.interpolate(flow, scale_factor=2, mode='bilinear')
            last_flow_all.append(last_flow)
            cur_x = F.interpolate(
                x, scale_factor=0.5**(len(x_warps) - 1 - i), mode='bilinear')
            cur_x_warp = F.grid_sample(
                cur_x,
                last_flow.permute(0, 2, 3, 1),
                mode='bilinear',
                padding_mode='border')

            x_all.append(cur_x_warp)
            cur_x_edge = F.interpolate(
                x_edge,
                scale_factor=0.5**(len(x_warps) - 1 - i),
                mode='bilinear')
            cur_x_warp_edge = F.grid_sample(
                cur_x_edge,
                last_flow.permute(0, 2, 3, 1),
                mode='bilinear',
                padding_mode='zeros')
            x_edge_all.append(cur_x_warp_edge)
            flow_x, flow_y = torch.split(last_flow, 1, dim=1)
            delta_x = F.conv2d(flow_x, self.weight)
            delta_y = F.conv2d(flow_y, self.weight)
            delta_x_all.append(delta_x)
            delta_y_all.append(delta_y)

        x_warp = F.grid_sample(
            x,
            last_flow.permute(0, 2, 3, 1),
            mode='bilinear',
            padding_mode='border')
        return x_warp, last_flow, cond_fea_all, last_flow_all, delta_list, x_all, \
            x_edge_all, delta_x_all, delta_y_all, gmap_all

    def upsample(self, F, scale):
        """[2x nearest neighbor upsampling]
        Arguments:
            F {[torch tensor]} -- [tensor to be upsampled, (B, C, H, W)]
        """
        upsample = torch.nn.Upsample(scale_factor=scale, mode='nearest')
        return upsample(F)


class Warping(nn.Module):
    """initialize the try on warping model
    """

    def __init__(self, input_nc=32 + 3, use_num=3):
        super(Warping, self).__init__()
        num_filters = [64, 128, 256, 256, 256]
        self.num = len(num_filters) - 1
        self.cloth_input = nn.Conv2d(
            3, 32, kernel_size=3, padding=1, bias=False)
        self.person_input = nn.Conv2d(
            3, 32, kernel_size=3, padding=1, bias=False)
        self.c_input_refine = Conv2dBlock(
            64, 35, 3, 1, 1, activation='prelu', bias=False)
        self.p_input_refine = Conv2dBlock(
            64, 35, 3, 1, 1, activation='prelu', bias=False)

        self.src_features = FeatureEncoder(input_nc, num_filters)
        self.tar_features = FeatureEncoder(input_nc, num_filters)
        self.src_FPN = RefinePyramid(num_filters)
        self.tar_FPN = RefinePyramid(num_filters)
        self.global_flow = AFlowNet(len(num_filters), use_num=use_num)
        self.local_flow = LocalFlow(use_num=use_num, init_scale=5)

        self.softmax = nn.Softmax(dim=1)

        self.input_scale = 4

    def forward(self, warping_input):

        cloth = warping_input[0]
        person = warping_input[1]
        c_heatmap = warping_input[2]
        p_heatmap = warping_input[3]
        c_landmark = warping_input[4]
        p_landmark = warping_input[5]
        x_edge = warping_input[6]
        org_cloth = warping_input[7]
        device = warping_input[8]

        N, _, H, W = cloth.shape
        cf = self.cloth_input(cloth)
        pf = self.person_input(person)
        src_input = self.c_input_refine(torch.cat((cf, c_heatmap), dim=1))
        tar_input = self.p_input_refine(torch.cat((pf, p_heatmap), dim=1))

        src_en_fea = self.src_features(src_input)
        tar_en_fea = self.tar_features(tar_input)
        src_c_fea = self.src_FPN(src_en_fea)
        tar_c_fea = self.tar_FPN(tar_en_fea)
        src_fuse_fea = src_c_fea
        tar_fuse_fea = tar_c_fea

        localflow_list, localmap_list = self.local_flow(
            cloth, src_fuse_fea, tar_fuse_fea, c_landmark, p_landmark)
        global_flow_input = [
            cloth, x_edge, src_fuse_fea, tar_fuse_fea, localflow_list,
            localmap_list
        ]
        x_warp, last_flow, last_flow_all, flow_all, delta_list, x_all, x_edge_all, \
            delta_x_all, delta_y_all, gmap_all = self.global_flow(global_flow_input, device)

        local_warped_cloth_list = []
        for i in range(len(localflow_list)):
            localmap = self.softmax(localmap_list[i])
            warped_cloth = torch.zeros_like(cloth)
            for j in range(32):
                once_warped_cloth = F.grid_sample(
                    cloth,
                    localflow_list[i][:, j * 2:(j + 1) * 2, :, :].permute(
                        0, 2, 3, 1),
                    mode='bilinear',
                    padding_mode='border')
                warped_cloth += once_warped_cloth * localmap[:,
                                                             j, :, :].unsqueeze(
                                                                 dim=1)
            local_warped_cloth_list.append(warped_cloth)
        globalmap = self.softmax(gmap_all[-1])
        fuse_cloth = globalmap[:, 0, :, :].unsqueeze(dim=1) * x_warp + \
            globalmap[:, 1, :, :].unsqueeze(dim=1) * local_warped_cloth_list[-1]

        # global local fused
        # global
        _, _, h, w = org_cloth.shape
        up_last_flow = F.interpolate(last_flow, size=(h, w), mode='bicubic')
        up_warped_gcloth = F.grid_sample(
            org_cloth,
            up_last_flow.permute(0, 2, 3, 1),
            mode='bicubic',
            padding_mode='border')
        # local
        localmap = self.softmax(localmap_list[-1])
        up_map = F.interpolate(localmap, size=(h, w), mode='bicubic')
        up_flow = F.interpolate(
            localflow_list[-1], size=(h, w), mode='bicubic')
        up_warped_lcloth = torch.zeros_like(org_cloth)
        for j in range(32):
            once_up_flow = F.interpolate(
                up_flow[:, j * 2:(j + 1) * 2, :, :],
                size=(h, w),
                mode='bicubic')
            once_warped_cloth = F.grid_sample(
                org_cloth,
                once_up_flow.permute(0, 2, 3, 1),
                mode='bilinear',
                padding_mode='border')
            up_warped_lcloth += once_warped_cloth * up_map[:,
                                                           j, :, :].unsqueeze(
                                                               dim=1)
        up_globalmap = F.interpolate(globalmap, size=(h, w), mode='bicubic')
        up_fuse_cloth = up_globalmap[:, 0, :, :].unsqueeze(
            dim=1) * up_warped_gcloth + up_globalmap[:, 1, :, :].unsqueeze(
                dim=1) * up_warped_lcloth
        up_cloth = torch.cat(
            [up_warped_gcloth, up_warped_lcloth, up_fuse_cloth], 1)

        return x_warp, last_flow, last_flow_all, flow_all, delta_list, x_all, x_edge_all, \
            delta_x_all, delta_y_all, local_warped_cloth_list, fuse_cloth, gmap_all, up_cloth

    def warp(self, x, flo, device):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        import torch.autograd as autograd
        from torch.autograd import Variable
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:,
              0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:,
              1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask
