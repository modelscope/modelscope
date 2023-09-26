# Copyright (c) Alibaba, Inc. and its affiliates.
import math
from functools import partial

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .networks_stylegan2 import FullyConnectedLayer
from .superresolution import SuperresolutionHybrid8XDC
from .volumetric_rendering.ray_sampler import RaySampler
from .volumetric_rendering.renderer import ImportanceRenderer


class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(
                                        2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class Encoder_low(nn.Module):

    def __init__(self,
                 img_size=64,
                 depth=5,
                 in_chans=256,
                 embed_dims=1024,
                 num_head=4,
                 mlp_ratio=2,
                 sr_ratio=1,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.depth = depth

        self.deeplabnet = smp.DeepLabV3(
            encoder_name='resnet34',
            encoder_depth=5,
            encoder_weights=None,
            decoder_channels=256,
            in_channels=5,
            classes=1)

        self.deeplabnet.encoder.conv1 = nn.Conv2d(
            5,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False)
        self.deeplabnet.segmentation_head = nn.Sequential()
        self.deeplabnet.encoder.bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer1[0].bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer1[0].bn2 = nn.Sequential()
        self.deeplabnet.encoder.layer1[1].bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer1[1].bn2 = nn.Sequential()
        self.deeplabnet.encoder.layer1[2].bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer1[2].bn2 = nn.Sequential()

        self.deeplabnet.encoder.layer2[0].bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer2[0].bn2 = nn.Sequential()
        self.deeplabnet.encoder.layer2[0].downsample[1] = nn.Sequential()
        self.deeplabnet.encoder.layer2[1].bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer2[1].bn2 = nn.Sequential()
        self.deeplabnet.encoder.layer2[2].bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer2[2].bn2 = nn.Sequential()
        self.deeplabnet.encoder.layer2[3].bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer2[3].bn2 = nn.Sequential()

        self.deeplabnet.encoder.layer3[0].bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer3[0].bn2 = nn.Sequential()
        self.deeplabnet.encoder.layer3[0].downsample[1] = nn.Sequential()
        self.deeplabnet.encoder.layer3[1].bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer3[1].bn2 = nn.Sequential()
        self.deeplabnet.encoder.layer3[2].bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer3[2].bn2 = nn.Sequential()
        self.deeplabnet.encoder.layer3[3].bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer3[3].bn2 = nn.Sequential()
        self.deeplabnet.encoder.layer3[4].bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer3[4].bn2 = nn.Sequential()
        self.deeplabnet.encoder.layer3[5].bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer3[5].bn2 = nn.Sequential()

        self.deeplabnet.encoder.layer4[0].bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer4[0].bn2 = nn.Sequential()
        self.deeplabnet.encoder.layer4[0].downsample[1] = nn.Sequential()
        self.deeplabnet.encoder.layer4[1].bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer4[1].bn2 = nn.Sequential()
        self.deeplabnet.encoder.layer4[2].bn1 = nn.Sequential()
        self.deeplabnet.encoder.layer4[2].bn2 = nn.Sequential()

        self.deeplabnet.decoder[0].convs[0][1] = nn.Sequential()
        self.deeplabnet.decoder[0].convs[1][1] = nn.Sequential()
        self.deeplabnet.decoder[0].convs[2][1] = nn.Sequential()
        self.deeplabnet.decoder[0].convs[3][1] = nn.Sequential()
        self.deeplabnet.decoder[0].convs[4][2] = nn.Sequential()
        self.deeplabnet.decoder[0].project[1] = nn.Sequential()
        self.deeplabnet.decoder[2] = nn.Sequential()

        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=3,
            stride=2,
            in_chans=in_chans,
            embed_dim=embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        cur = 0
        self.vit_block = nn.ModuleList([
            Block(
                dim=embed_dims,
                num_heads=num_head,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratio) for i in range(depth)
        ])
        self.norm1 = norm_layer(embed_dims)
        self.ps = nn.PixelShuffle(2)

        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, input):
        B = input.shape[0]

        f_low = self.deeplabnet(input)
        x, H, W = self.patch_embed(f_low)

        for i, blk in enumerate(self.vit_block):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.ps(x)

        x = self.relu1(self.conv1(self.upsample1(x)))
        x = self.relu2(self.conv2(self.upsample2(x)))
        x = self.conv3(x)

        return x


class Encoder_high(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.LeakyReLU(0.01)
        self.conv2 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(0.01)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU(0.01)
        self.conv4 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.LeakyReLU(0.01)
        self.conv5 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.LeakyReLU(0.01)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))

        return x


class MixFeature(nn.Module):

    def __init__(self,
                 img_size=256,
                 depth=1,
                 in_chans=128,
                 embed_dims=1024,
                 num_head=2,
                 mlp_ratio=2,
                 sr_ratio=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv1 = nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(0.01)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(0.01)

        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=3,
            stride=2,
            in_chans=in_chans,
            embed_dim=embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        cur = 0
        self.vit_block = nn.ModuleList([
            Block(
                dim=embed_dims,
                num_heads=num_head,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratio) for i in range(depth)
        ])
        self.norm1 = norm_layer(embed_dims)
        self.ps = nn.PixelShuffle(2)

        self.conv3 = nn.Conv2d(352, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU(0.01)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.LeakyReLU(0.01)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.LeakyReLU(0.01)
        self.conv6 = nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x_low, x_high):
        x = torch.cat((x_low, x_high), 1)
        B = x.shape[0]

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))

        x, H, W = self.patch_embed(x)

        for i, blk in enumerate(self.vit_block):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.ps(x)

        x = torch.cat((x, x_low), 1)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.conv6(x)

        return x


class OSGDecoder(torch.nn.Module):

    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(
                n_features,
                self.hidden_dim,
                lr_multiplier=options['decoder_lr_mul']), torch.nn.Softplus(),
            FullyConnectedLayer(
                self.hidden_dim,
                1 + options['decoder_output_dim'],
                lr_multiplier=options['decoder_lr_mul']))

    def forward(self, sampled_features, ray_directions):
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N * M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:]) * (1 + 2 * 0.001) - 0.001
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}


class TriplaneEncoder(nn.Module):

    def __init__(self,
                 img_resolution,
                 sr_num_fp16_res=0,
                 rendering_kwargs={},
                 sr_kwargs={}):
        super().__init__()
        self.encoder_low = Encoder_low(
            img_size=64,
            depth=5,
            in_chans=256,
            embed_dims=1024,
            num_head=4,
            mlp_ratio=2,
            sr_ratio=1,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.encoder_high = Encoder_high()
        self.mix = MixFeature(
            img_size=256,
            depth=1,
            in_chans=128,
            embed_dims=1024,
            num_head=2,
            mlp_ratio=2,
            sr_ratio=2,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.superresolution = SuperresolutionHybrid8XDC(
            channels=32,
            img_resolution=img_resolution,
            sr_num_fp16_res=sr_num_fp16_res,
            sr_antialias=rendering_kwargs['sr_antialias'],
            **sr_kwargs)
        self.decoder = OSGDecoder(
            32, {
                'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                'decoder_output_dim': 32
            })
        self.neural_rendering_resolution = 128
        self.rendering_kwargs = rendering_kwargs

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def gen_interfeats(self, ws, planes, camera_params):
        planes = planes.view(
            len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        cam2world_matrix = camera_params[:, :16].view(-1, 4, 4)
        intrinsics = camera_params[:, 16:25].view(-1, 3, 3)
        H = W = self.neural_rendering_resolution
        ray_origins, ray_directions = self.ray_sampler(
            cam2world_matrix, intrinsics, self.neural_rendering_resolution)
        N, M, _ = ray_origins.shape
        feature_samples, depth_samples, weights_samples = self.renderer(
            planes, self.decoder, ray_origins, ray_directions,
            self.rendering_kwargs)
        feature_image = feature_samples.permute(0, 2, 1).reshape(
            N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(
            rgb_image, feature_image, ws, noise_mode='const')

        return depth_image, feature_image, rgb_image, sr_image

    def sample(self, coordinates, directions, planes):
        planes = planes.view(
            len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates,
                                       directions, self.rendering_kwargs)

    def forward(self, ws, x, camera_ref, camera_mv):
        f = self.encoder_low(x)
        f_high = self.encoder_high(x)
        planes = self.mix(f, f_high)

        depth_ref, feature_ref, rgb_ref, sr_ref = self.gen_interfeats(
            ws, planes, camera_ref)
        if camera_mv is not None:
            depth_mv, feature_mv, rgb_mv, sr_mv = self.gen_interfeats(
                ws, planes, camera_mv)
        else:
            depth_mv = feature_mv = rgb_mv = sr_mv = None

        return planes, depth_ref, feature_ref, rgb_ref, sr_ref, depth_mv, feature_mv, rgb_mv, sr_mv


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
