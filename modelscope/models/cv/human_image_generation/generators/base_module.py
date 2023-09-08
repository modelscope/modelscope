import collections
import functools
import math
from tkinter.ttk import Style

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .base_function import *
from .flow_module import MaskStyle, StyleFlow
from .tps import TPS


# adding flow version
class Encoder_wiflow(nn.Module):

    def __init__(
        self,
        size,
        input_dim,
        channels,
        num_labels=None,
        match_kernels=None,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()
        self.first = EncoderLayer_flow(input_dim, channels[size], 1)
        self.convs = nn.ModuleList()
        self.num_labels = num_labels
        self.match_kernels = match_kernels

        log_size = int(math.log(size, 2))
        self.log_size = log_size

        in_channel = channels[size]
        for i in range(log_size - 1, 3, -1):
            out_channel = channels[2**i]
            num_label = num_labels[2**i] if num_labels is not None else None
            match_kernel = match_kernels[
                2**i] if match_kernels is not None else None
            use_extraction = num_label and match_kernel
            conv = EncoderLayer_flow(
                in_channel,
                out_channel,
                kernel_size=3,
                downsample=True,
                blur_kernel=blur_kernel,
                use_extraction=use_extraction,
                num_label=num_label,
                match_kernel=match_kernel)

            self.convs.append(conv)
            in_channel = out_channel

    def forward(self, input, recoder=None, out_list=None):
        out = self.first(input)
        for layer in self.convs:
            out = layer(out, recoder)
            if out_list is not None:
                out_list.append(out)
        return out


class Decoder_wiflow_wavelet_fuse25(nn.Module):

    def __init__(
        self,
        size,
        channels,
        num_labels,
        match_kernels,
        blur_kernel=[1, 3, 3, 1],
        wavelet_down_levels={'16': 3},
        window_size=8,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        # input at resolution 16*16
        in_channel = channels[16]
        self.log_size = int(math.log(size, 2))
        self.conv_mask_dict = nn.ModuleDict()
        self.conv_mask_fuse_dict = nn.ModuleDict()

        flow_fusion = False

        for i in range(4, self.log_size + 1):
            out_channel = channels[2**i]
            num_label, match_kernel = num_labels[2**i], match_kernels[2**i]
            use_distribution = num_label and match_kernel
            upsample = (i != 4)
            wavelet_down_level = wavelet_down_levels[(2**i)]
            base_layer = functools.partial(
                DecoderLayer_flow_wavelet_fuse24,
                out_channel=out_channel,
                kernel_size=3,
                blur_kernel=blur_kernel,
                use_distribution=use_distribution,
                num_label=num_label,
                match_kernel=match_kernel,
                wavelet_down_level=wavelet_down_level,
                window_size=window_size)
            # mask head for fusion
            if use_distribution:
                conv_mask = [
                    EqualConv2d(
                        2 * out_channel,
                        3,
                        3,
                        stride=1,
                        padding=3 // 2,
                        bias=False),
                    nn.Sigmoid()
                ]
                conv_mask = nn.Sequential(*conv_mask)
                self.conv_mask_dict[str(2**i)] = conv_mask

                if not i == 4:
                    conv_mask_fuse = nn.Sequential(*[
                        EqualConv2d(
                            2, 1, 3, stride=1, padding=3 // 2, bias=False),
                        nn.Sigmoid()
                    ])
                    self.conv_mask_fuse_dict[str(2**i)] = conv_mask_fuse

                if not flow_fusion:
                    self.conv_flow_fusion = nn.Sequential(
                        EqualConv2d(
                            2 * out_channel,
                            1,
                            kernel_size=7,
                            stride=1,
                            padding=3,
                            bias=False), nn.Sigmoid())
                    flow_fusion = True

            up = nn.Module()
            up.conv0 = base_layer(in_channel=in_channel, upsample=upsample)
            up.conv1 = base_layer(in_channel=out_channel, upsample=False)
            up.to_rgb = ToRGB(out_channel, upsample=upsample)
            self.convs.append(up)
            in_channel = out_channel

        style_in_channels = channels[16]
        self.style_out_channel = 128
        self.cond_style = nn.Sequential(
            nn.Conv2d(
                style_in_channels,
                self.style_out_channel,
                kernel_size=3,
                stride=1,
                padding=1), nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.AdaptiveAvgPool2d(1))
        self.image_style = nn.Sequential(
            nn.Conv2d(
                style_in_channels,
                self.style_out_channel,
                kernel_size=3,
                stride=1,
                padding=1), nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.AdaptiveAvgPool2d(1))
        self.flow_model = StyleFlow(
            channels, self.log_size, style_in=2 * self.style_out_channel)

        self.num_labels, self.match_kernels = num_labels, match_kernels

        # for mask prediction
        self.mask_style = MaskStyle(
            channels,
            self.log_size,
            style_in=2 * self.style_out_channel,
            channels_multiplier=1)

        # tps transformation
        self.tps = TPS()

    def forward(self,
                input,
                neural_textures,
                skeleton_features,
                source_features,
                kp_skeleton,
                recoder,
                add_nted=True):
        source_features = source_features[::-1]
        skeleton_features = skeleton_features[::-1]

        counter = 0
        out, skip = input, None

        last_flow = None
        mask_all_h, mask_all_l = [], []
        delta_list = []
        delta_x_all = []
        delta_y_all = []
        last_flow_all = []
        filter_x = [[0, 0, 0], [1, -2, 1], [0, 0, 0]]
        filter_y = [[0, 1, 0], [0, -2, 0], [0, 1, 0]]
        filter_diag1 = [[1, 0, 0], [0, -2, 0], [0, 0, 1]]
        filter_diag2 = [[0, 0, 1], [0, -2, 0], [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2
        weight_array = torch.FloatTensor(weight_array).permute(3, 2, 0, 1).to(
            input.device)
        self.weight = nn.Parameter(data=weight_array, requires_grad=False)

        B = source_features[0].shape[0]
        source_style = self.cond_style(source_features[0]).view(B, -1)
        target_style = self.image_style(skeleton_features[0]).view(B, -1)
        style = torch.cat([source_style, target_style], 1)

        for i, up in enumerate(self.convs):
            use_distribution = (
                self.num_labels[2**(i + 4)] and self.match_kernels[2**(i + 4)])
            if use_distribution:
                # warp features with styleflow
                source_feature = source_features[i]
                skeleton_feature = skeleton_features[i]
                if last_flow is not None:
                    last_flow = F.interpolate(
                        last_flow, scale_factor=2, mode='bilinear')
                    s_warp_after = F.grid_sample(
                        source_feature,
                        last_flow.detach().permute(0, 2, 3, 1),
                        mode='bilinear',
                        padding_mode='border')
                else:
                    s_warp_after = source_feature
                scale = str(2**(i + 4))

                # use tps transformation to estimate flow at the very beginning
                if last_flow is not None:
                    style_map = self.flow_model.netStyle[scale](s_warp_after,
                                                                style)
                    flow = self.flow_model.netF[scale](style_map, style)
                    flow = apply_offset(flow)

                else:
                    style_map = self.flow_model.netStyle[scale](s_warp_after,
                                                                style)
                    flow = self.flow_model.netF[scale](style_map, style)
                    flow_dense = apply_offset(flow)
                    flow_tps = self.tps(source_feature, kp_skeleton)
                    warped_dense = F.grid_sample(
                        source_feature,
                        flow_dense,
                        mode='bilinear',
                        padding_mode='border')
                    warped_tps = F.grid_sample(
                        source_feature,
                        flow_tps,
                        mode='bilinear',
                        padding_mode='border')
                    contribution_map = self.conv_flow_fusion(
                        torch.cat([warped_dense, warped_tps], 1))
                    flow = contribution_map * flow_tps.permute(0, 3, 1, 2) + (
                        1 - contribution_map) * flow_dense.permute(0, 3, 1, 2)
                    flow = flow.permute(0, 2, 3, 1).contiguous()

                if last_flow is not None:
                    # update flow according to the last scale flow
                    flow = F.grid_sample(
                        last_flow,
                        flow,
                        mode='bilinear',
                        padding_mode='border')
                else:
                    flow = flow.permute(0, 3, 1, 2)

                last_flow = flow
                s_warp = F.grid_sample(
                    source_feature,
                    flow.permute(0, 2, 3, 1),
                    mode='bilinear',
                    padding_mode='border')

                # refine flow according to the original flow
                flow = self.flow_model.netRefine[scale](
                    torch.cat([s_warp, skeleton_feature], 1))

                delta_list.append(flow)
                flow = apply_offset(flow)
                flow = F.grid_sample(
                    last_flow, flow, mode='bilinear', padding_mode='border')
                last_flow_all.append(flow)

                last_flow = flow
                flow_x, flow_y = torch.split(last_flow, 1, dim=1)
                delta_x = F.conv2d(flow_x, self.weight)
                delta_y = F.conv2d(flow_y, self.weight)
                delta_x_all.append(delta_x)
                delta_y_all.append(delta_y)

                s_warp = F.grid_sample(
                    source_feature,
                    last_flow.permute(0, 2, 3, 1),
                    mode='bilinear',
                    padding_mode='border')

                # nted attention
                neural_texture_conv0 = neural_textures[counter]
                neural_texture_conv1 = neural_textures[counter + 1]
                counter += 2

                if not add_nted:  # turn off the nted attention
                    neural_texture_conv0, neural_texture_conv1 = None, None
            else:
                neural_texture_conv0, neural_texture_conv1 = None, None
                s_warp = None

            mask_style_net = self.mask_style.netM[
                scale] if use_distribution else None
            out, mask_h, mask_l = up.conv0(
                out,
                neural_texture=neural_texture_conv0,
                recoder=recoder,
                warped_texture=s_warp,
                style_net=mask_style_net,
                gstyle=style)
            out, mask_h, mask_l = up.conv1(
                out,
                neural_texture=neural_texture_conv1,
                recoder=recoder,
                warped_texture=s_warp,
                style_net=mask_style_net,
                gstyle=style)
            if use_distribution:
                if mask_h is not None:
                    mask_all_h.append(mask_h)
                if mask_l is not None:
                    mask_all_l.append(mask_l)
            skip = up.to_rgb(out, skip)

        image = skip
        return image, delta_x_all, delta_y_all, delta_list, last_flow_all, mask_all_h, mask_all_l


def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid(
        [torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)
    # apply offset
    grid_list = [
        grid.float().unsqueeze(0) + offset[:, dim, ...]
        for dim, grid in enumerate(grid_list)
    ]
    # normalize
    grid_list = [
        grid / ((size - 1.0) / 2.0) - 1.0
        for grid, size in zip(grid_list, reversed(sizes))
    ]

    return torch.stack(grid_list, dim=-1)
