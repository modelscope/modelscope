# Part of the implementation is borrowed and modified from RIFE,
# publicly available at https://github.com/megvii-research/ECCV2022-RIFE

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from modelscope.models.cv.video_frame_interpolation.interp_model.transformer_layers import (
    RTFL, PatchEmbed, PatchUnEmbed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(
            -1.0, 1.0, tenFlow.shape[3], device=device).view(
                1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1,
                                                  tenFlow.shape[2], -1)
        tenVertical = torch.linspace(
            -1.0, 1.0, tenFlow.shape[2],
            device=device).view(1, 1, tenFlow.shape[2],
                                1).expand(tenFlow.shape[0], -1, -1,
                                          tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical],
                                        1).to(device)

    tmp1 = tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0)
    tmp2 = tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)
    tenFlow = torch.cat([tmp1, tmp2], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=torch.clamp(g, -1, 1),
        mode='bilinear',
        padding_mode='border',
        align_corners=True)


def conv_wo_act(in_planes,
                out_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True), )


def conv(in_planes,
         out_planes,
         kernel_size=3,
         stride=1,
         padding=1,
         dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True), nn.PReLU(out_planes))


def conv_bn(in_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False), nn.BatchNorm2d(out_planes), nn.PReLU(out_planes))


class TransModel(nn.Module):

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 embed_dim=64,
                 depths=[[3, 3]],
                 num_heads=[[2, 2]],
                 window_size=4,
                 mlp_ratio=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 resi_connection='1conv',
                 use_crossattn=[[[False, False, False, False],
                                 [True, True, True, True]]]):
        super(TransModel, self).__init__()
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr0 = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[0]))
        ]  # stochastic depth decay rule

        self.layers0 = nn.ModuleList()
        num_layers = len(depths[0])
        for i_layer in range(num_layers):
            layer = RTFL(
                dim=embed_dim,
                input_resolution=(patches_resolution[0],
                                  patches_resolution[1]),
                depth=depths[0][i_layer],
                num_heads=num_heads[0][i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr0[sum(depths[0][:i_layer]):sum(depths[0][:i_layer
                                                                      + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=(img_size[0], img_size[1]),
                patch_size=patch_size,
                resi_connection=resi_connection,
                use_crossattn=use_crossattn[0][i_layer])
            self.layers0.append(layer)

        self.norm = norm_layer(self.num_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, layers):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        if isinstance(layers, nn.ModuleList):
            for layer in layers:
                x = layer(x, x_size)
        else:
            x = layers(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        out = self.forward_features(x, self.layers0)
        return out


class IFBlock(nn.Module):

    def __init__(self, in_planes, scale=1, c=64):
        super(IFBlock, self).__init__()
        self.scale = scale
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
            conv(c, c, 3, 1, 1),
        )

        self.trans = TransModel(
            img_size=(128 // scale, 128 // scale),
            patch_size=1,
            embed_dim=c,
            depths=[[3, 3]],
            num_heads=[[2, 2]])

        self.conv1 = nn.Sequential(
            conv(c, c, 3, 1, 1),
            conv(c, c, 3, 1, 1),
        )

        self.up = nn.ConvTranspose2d(c, 4, 4, 2, 1)

        self.conv2 = nn.Conv2d(4, 4, 3, 1, 1)

    def forward(self, x, flow0, flow1):
        if self.scale != 1:
            x = F.interpolate(
                x,
                scale_factor=1. / self.scale,
                mode='bilinear',
                align_corners=False)
            flow0 = F.interpolate(
                flow0,
                scale_factor=1. / self.scale,
                mode='bilinear',
                align_corners=False) * (1. / self.scale)
            flow1 = F.interpolate(
                flow1,
                scale_factor=1. / self.scale,
                mode='bilinear',
                align_corners=False) * (1. / self.scale)

        x = torch.cat((x, flow0, flow1), 1)

        x = self.conv0(x)
        x = self.trans(x)
        x = self.conv1(x) + x

        # upsample 2.0
        x = self.up(x)

        # upsample 2.0
        x = self.conv2(x)
        flow = F.interpolate(
            x, scale_factor=2.0, mode='bilinear', align_corners=False) * 2.0

        if self.scale != 1:
            flow = F.interpolate(
                flow,
                scale_factor=self.scale,
                mode='bilinear',
                align_corners=False) * self.scale

        flow0 = flow[:, :2, :, :]
        flow1 = flow[:, 2:, :, :]

        return flow0, flow1


class IFBlock_wo_Swin(nn.Module):

    def __init__(self, in_planes, scale=1, c=64):
        super(IFBlock_wo_Swin, self).__init__()
        self.scale = scale
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )

        self.convblock1 = nn.Sequential(conv(c, c), conv(c, c), conv(c, c))
        self.convblock2 = nn.Sequential(conv(c, c), conv(c, c), conv(c, c))

        self.up = nn.ConvTranspose2d(c, 4, 4, 2, 1)

        self.conv2 = nn.Conv2d(4, 4, 3, 1, 1)

    def forward(self, x, flow0, flow1):
        if self.scale != 1:
            x = F.interpolate(
                x,
                scale_factor=1. / self.scale,
                mode='bilinear',
                align_corners=False)
            flow0 = F.interpolate(
                flow0,
                scale_factor=1. / self.scale,
                mode='bilinear',
                align_corners=False) * (1. / self.scale)
            flow1 = F.interpolate(
                flow1,
                scale_factor=1. / self.scale,
                mode='bilinear',
                align_corners=False) * (1. / self.scale)

        x = torch.cat((x, flow0, flow1), 1)

        x = self.conv0(x)
        x = self.convblock1(x) + x
        x = self.convblock2(x) + x
        # upsample 2.0
        x = self.up(x)

        # upsample 2.0
        x = self.conv2(x)
        flow = F.interpolate(
            x, scale_factor=2.0, mode='bilinear', align_corners=False) * 2.0

        if self.scale != 1:
            flow = F.interpolate(
                flow,
                scale_factor=self.scale,
                mode='bilinear',
                align_corners=False) * self.scale

        flow0 = flow[:, :2, :, :]
        flow1 = flow[:, 2:, :, :]

        return flow0, flow1


class IFNet(nn.Module):

    def __init__(self):
        super(IFNet, self).__init__()
        self.block1 = IFBlock_wo_Swin(16, scale=4, c=128)
        self.block2 = IFBlock(16, scale=2, c=64)
        self.block3 = IFBlock(16, scale=1, c=32)

    # flow0: flow from img0 to img1
    # flow1: flow from img1 to img0
    def forward(self, img0, img1, flow0, flow1, sc_mode=2):

        if sc_mode == 0:
            sc = 0.25
        elif sc_mode == 1:
            sc = 0.5
        else:
            sc = 1

        if sc != 1:
            img0_sc = F.interpolate(
                img0, scale_factor=sc, mode='bilinear', align_corners=False)
            img1_sc = F.interpolate(
                img1, scale_factor=sc, mode='bilinear', align_corners=False)
            flow0_sc = F.interpolate(
                flow0, scale_factor=sc, mode='bilinear',
                align_corners=False) * sc
            flow1_sc = F.interpolate(
                flow1, scale_factor=sc, mode='bilinear',
                align_corners=False) * sc
        else:
            img0_sc = img0
            img1_sc = img1
            flow0_sc = flow0
            flow1_sc = flow1

        warped_img0 = warp(img1_sc, flow0_sc)  # -> img0
        warped_img1 = warp(img0_sc, flow1_sc)  # -> img1
        flow0_1, flow1_1 = self.block1(
            torch.cat((img0_sc, img1_sc, warped_img0, warped_img1), 1),
            flow0_sc, flow1_sc)
        F0_2 = (flow0_sc + flow0_1)
        F1_2 = (flow1_sc + flow1_1)

        warped_img0 = warp(img1_sc, F0_2)  # -> img0
        warped_img1 = warp(img0_sc, F1_2)  # -> img1
        flow0_2, flow1_2 = self.block2(
            torch.cat((img0_sc, img1_sc, warped_img0, warped_img1), 1), F0_2,
            F1_2)
        F0_3 = (F0_2 + flow0_2)
        F1_3 = (F1_2 + flow1_2)

        warped_img0 = warp(img1_sc, F0_3)  # -> img0
        warped_img1 = warp(img0_sc, F1_3)  # -> img1
        flow0_3, flow1_3 = self.block3(
            torch.cat((img0_sc, img1_sc, warped_img0, warped_img1), dim=1),
            F0_3, F1_3)
        flow_res_0 = flow0_1 + flow0_2 + flow0_3
        flow_res_1 = flow1_1 + flow1_2 + flow1_3

        if sc != 1:
            flow_res_0 = F.interpolate(
                flow_res_0,
                scale_factor=1 / sc,
                mode='bilinear',
                align_corners=False) / sc
            flow_res_1 = F.interpolate(
                flow_res_1,
                scale_factor=1 / sc,
                mode='bilinear',
                align_corners=False) / sc

        F0_4 = flow0 + flow_res_0
        F1_4 = flow1 + flow_res_1

        return F0_4, F1_4
