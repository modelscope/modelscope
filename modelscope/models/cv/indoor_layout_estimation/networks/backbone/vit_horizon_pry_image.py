# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 stride_size=16,
                 padding=[0, 1],
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride_size,
            padding=[0, 1])
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class ViTHorizonPryImage(nn.Module):

    def __init__(self, backbone, fourier, embedding):
        super(ViTHorizonPryImage, self).__init__()
        embed_dim = 192
        F_lens = [256, 128, 64, 32, 512]
        position_encode = np.sum(np.array(F_lens))
        self.embedding = embedding
        if fourier is False:
            in_chans = 3
        else:
            in_chans = 9
        self.pre_image = PatchEmbed([512, 1024], [32, 32], [32, 32],
                                    in_chans=in_chans,
                                    embed_dim=embed_dim)
        self.pre_net = nn.ModuleList([
            PatchEmbed([128, 256], [128, 3], [128, 1],
                       padding=[0, 1],
                       in_chans=64,
                       embed_dim=embed_dim),
            PatchEmbed([64, 128], [64, 3], [64, 1],
                       padding=[0, 1],
                       in_chans=128,
                       embed_dim=embed_dim),
            PatchEmbed([32, 64], [32, 3], [32, 1],
                       padding=[0, 1],
                       in_chans=256,
                       embed_dim=embed_dim),
            PatchEmbed([16, 32], [16, 3], [16, 1],
                       padding=[0, 1],
                       in_chans=512,
                       embed_dim=embed_dim)
        ])

        self.encoder = timm.create_model(backbone, pretrained=False)
        del self.encoder.patch_embed, self.encoder.head

        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, position_encode + 1, embed_dim))

        def EfficientConvCompressH(in_c, out_c, down_h):

            net1 = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

            net2 = nn.Sequential(
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, (down_h, 1), groups=out_c, bias=False),
            )
            return net1, net2

        self.ECH1, self.ECH2 = EfficientConvCompressH(embed_dim, 2 * embed_dim,
                                                      4)
        self.scales = [1, 2, 4, 8]
        # self.L = nn.Linear(454,1024)
        if self.embedding == 'sin':
            import math
            max_len, d_model = position_encode, embed_dim
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = \
                torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pos', pe.T[None].contiguous())

        elif self.embedding == 'recurrent':
            import math
            d_model = embed_dim
            index = torch.randint(0, F_lens[0], [1])
            for i, max_len in enumerate(F_lens):
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(
                    0, max_len, dtype=torch.float).unsqueeze(1)
                if i < len(F_lens) - 1:
                    index = torch.div(index, 2, rounding_mode='floor')**i
                    position = (index + position) % max_len
                position = position + np.sum(np.array(F_lens[:i]))
                div_term = \
                    torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                if i == 0:
                    pe_re = pe
                else:
                    pe_re = torch.cat((pe_re, pe), dim=0)
            self.register_buffer('pos', pe_re.T[None].contiguous())

    def forward(self, img, x):
        for i, feat in enumerate(x):
            pre = self.pre_net[i](feat)
            if i == 0:
                inputs = pre
            else:
                inputs = torch.cat((inputs, pre), 1)
        pre = self.pre_image(img)
        inputs = torch.cat((inputs, pre), 1)
        if self.embedding == 'learnable':
            inputs = torch.cat(
                (self.dist_token.expand(inputs.shape[0], -1, -1), inputs),
                dim=1)
            inputs = inputs + self.pos_embed
        if self.embedding == 'sin':
            inputs = inputs + self.pos.permute(0, 2, 1)
        if self.embedding == 'recurrent':
            inputs = inputs + self.pos.permute(0, 2, 1)

        x = self.encoder.pos_drop(inputs)
        for i in range(12):
            x = self.encoder.blocks[i](x)
        x = x.permute(0, 2, 1)
        a1 = x[:, :, :256].reshape(x.shape[0], x.shape[1], 1, 256)
        a1 = F.interpolate(
            a1, scale_factor=(1, 4), mode='bilinear', align_corners=False)
        a2 = x[:, :, 256:384].reshape(x.shape[0], x.shape[1], 1, 128)
        a2 = F.interpolate(
            a2, scale_factor=(1, 8), mode='bilinear', align_corners=False)
        a3 = x[:, :, 384:448].reshape(x.shape[0], x.shape[1], 1, 64)
        a3 = F.interpolate(
            a3, scale_factor=(1, 16), mode='bilinear', align_corners=False)
        a4 = x[:, :, 448:480].reshape(x.shape[0], x.shape[1], 1, 32)
        a4 = F.interpolate(
            a4, scale_factor=(1, 32), mode='bilinear', align_corners=False)
        a = torch.cat((a1, a2, a3, a4), dim=2)
        a = self.ECH1(a)
        a = self.ECH2(a).flatten(2)

        feat = {}
        feat['1D'] = a
        return feat
