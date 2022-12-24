# Adopted from https://github.com/Limingxing00/RDE-VOS-CVPR2022
# under MIT License

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from modelscope.models.cv.video_object_segmentation import cbam, mod_resnet


class ResBlock(nn.Module):

    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim is None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(
                indim, outdim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class FeatureFusionBlock(nn.Module):

    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = cbam.CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)

        return x


# Single object version, used only in static image pretraining
# See model.py (load_network) for the modification procedure
class ValueEncoderSO(nn.Module):

    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet18(pretrained=False, extra_chan=1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4, 64
        self.layer2 = resnet.layer2  # 1/8, 128
        self.layer3 = resnet.layer3  # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512)

    def forward(self, image, key_f16, mask):
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)  # 1/4, 64
        x = self.layer2(x)  # 1/8, 128
        x = self.layer3(x)  # 1/16, 256

        x = self.fuser(x, key_f16)

        return x


# Multiple objects version, used in other times
class ValueEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet18(pretrained=False, extra_chan=2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4, 64
        self.layer2 = resnet.layer2  # 1/8, 128
        self.layer3 = resnet.layer3  # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512)

    def forward(self, image, key_f16, mask, other_masks):
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask, other_masks], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)  # 1/4, 64
        x = self.layer2(x)  # 1/8, 128
        x = self.layer3(x)  # 1/16, 256
        # x = torch.cat([x, x], dim=1)
        x = self.fuser(x, key_f16)

        return x


# from retrying import retry
# @retry(stop_max_attempt_number=5)
class KeyEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=False)
        # if torch.distributed.get_rank() == 0:
        #     torch.distributed.barrier()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.layer2 = resnet.layer2  # 1/8, 512
        self.layer3 = resnet.layer3  # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)  # 1/4, 256
        f8 = self.layer2(f4)  # 1/8, 512
        f16 = self.layer3(f8)  # 1/16, 1024

        return f16, f8, f4


class UpsampleBlock(nn.Module):

    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(
            up_f,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False)
        x = self.out_conv(x)
        return x


class KeyProjection(nn.Module):

    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x):
        return self.key_proj(x)


class _NonLocalBlockND(nn.Module):

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=3,
                 sub_sample=True,
                 bn_layer=False):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.InstanceNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0), bn(self.in_channels))
            # nn.init.constant_(self.W[1].weight, 0)
            # nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0)
            # nn.init.constant_(self.W.weight, 0)
            # nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0)

        self.phi = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=True):
        super(NONLocalBlock1D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=1,
            sub_sample=sub_sample,
            bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=False):
        super(NONLocalBlock2D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=2,
            sub_sample=sub_sample,
            bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=True):
        super(NONLocalBlock3D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=3,
            sub_sample=sub_sample,
            bn_layer=bn_layer)


class _ASPPModule3D(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule3D, self).__init__()
        self.atrous_conv = nn.Conv3d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)

    def forward(self, x):
        x = self.atrous_conv(x)
        return F.relu(x, inplace=True)


class ASPP3D(nn.Module):

    def __init__(self, in_plane, out_plane, reduction=4):
        super().__init__()
        dilations = [1, 2, 4, 6]
        mid_plane = out_plane // reduction
        self.aspp1 = _ASPPModule3D(
            in_plane, mid_plane, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule3D(
            in_plane,
            mid_plane, (1, 3, 3),
            padding=(0, dilations[1], dilations[1]),
            dilation=(1, dilations[1], dilations[1]))
        self.aspp3 = _ASPPModule3D(
            in_plane,
            mid_plane, (1, 3, 3),
            padding=(0, dilations[2], dilations[2]),
            dilation=(1, dilations[2], dilations[2]))
        self.aspp4 = _ASPPModule3D(
            in_plane,
            mid_plane, (1, 3, 3),
            padding=(0, dilations[3], dilations[3]),
            dilation=(1, dilations[3], dilations[3]))
        self.conv1 = nn.Conv3d(
            mid_plane * 4,
            out_plane,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            bias=False)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.relu(self.conv1(x), inplace=True)
        return x


class SELayerS(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayerS, self).__init__()
        channel = channel * 2  # 2 is time axis
        self.avg_pool = nn.AdaptiveAvgPool3d((2, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, 2 * c)
        y = self.fc(y).view(b, c, 2, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    # https://github.com/moskomule/senet.pytorch

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.in1 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.in2 = nn.InstanceNorm3d(planes)
        self.ses = SELayerS(planes, reduction)
        self.in3 = nn.InstanceNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)

        out = self.ses(out)
        out = self.in3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SAM(nn.Module):
    """
    Spatio-temporal aggregation module (SAM)
    """

    def __init__(self, indim, outdim=None, repeat=0, norm=False):
        super(SAM, self).__init__()
        self.indim = indim
        self.repeat = repeat
        if outdim is None:
            outdim = indim
        if repeat > 0:
            self.se_block = self.seRepeat(repeat)
        self.conv1 = ASPP3D(indim, outdim, reduction=4)  # norm is 4

        # self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
        self.non_local = NONLocalBlock3D(indim, bn_layer=False)

    def seRepeat(self, repeat=2):
        return nn.Sequential(*nn.ModuleList(
            [SEBasicBlock(self.indim, self.indim) for _ in range(repeat)]))

    def forward(self, x):
        x = self.non_local(x)

        if self.repeat > 0:
            x = self.se_block(x)
        r = x
        x = self.conv1(x) + r
        return x


class MemCrompress(nn.Module):

    def __init__(self, repeat=0, norm=True):
        super().__init__()

        self.key_encoder = SAM(64, 64, repeat=repeat, norm=norm)
        self.value_encoder = SAM(512, 512, repeat=repeat, norm=norm)

        self.compress_key = nn.Conv3d(
            64, 64, kernel_size=(2, 3, 3), padding=(0, 1, 1))
        self.compress_value = nn.Conv3d(
            512, 512, kernel_size=(2, 3, 3), padding=(0, 1, 1))
        # self.temporal_shuffle = temporal_shuffle

    def forward(self, key, value):
        # key    N, C, T, H, W      [4, 64, 2, 24, 24]
        # value  N, O, C, T, H, W
        # return
        # key    N, C, 1, H, W
        # value  N, O, C, 1, H, W
        N, O, C, T, H, W = value.shape
        value = value.flatten(
            start_dim=0, end_dim=1)  # N*O, C, T, H, W [8, 512, 2, 24, 24]

        k = self.compress_key(self.key_encoder(key))
        v = self.compress_value(self.value_encoder(value))
        v = v.view(N, O, C, 1, H, W)
        return k, v
