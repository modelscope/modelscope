# @Description: Geometric Prior Guided Feature Fusion & Probability Volume Geometry Embedding (Sec 3.1 in the paper).
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07
# @https://github.com/doublez0108/geomvsnet

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .submodules import ConvBnReLU3D


class GeoFeatureFusion(nn.Module):

    def __init__(self,
                 convolutional_layer_encoding='z',
                 mask_type='basic',
                 add_origin_feat_flag=True):
        super(GeoFeatureFusion, self).__init__()

        self.convolutional_layer_encoding = convolutional_layer_encoding  # std / uv / z / xyz
        self.mask_type = mask_type  # basic / mean
        self.add_origin_feat_flag = add_origin_feat_flag  # True / False

        if self.convolutional_layer_encoding == 'std':
            self.geoplanes = 0
        elif self.convolutional_layer_encoding == 'uv':
            self.geoplanes = 2
        elif self.convolutional_layer_encoding == 'z':
            self.geoplanes = 1
        elif self.convolutional_layer_encoding == 'xyz':
            self.geoplanes = 3
            self.geofeature = GeometryFeature()

        # rgb encoder
        self.rgb_conv_init = convbnrelu(
            in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=2)

        self.rgb_encoder_layer1 = BasicBlockGeo(
            inplanes=8, planes=16, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer2 = BasicBlockGeo(
            inplanes=16, planes=32, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer3 = BasicBlockGeo(
            inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer4 = BasicBlockGeo(
            inplanes=64, planes=128, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer5 = BasicBlockGeo(
            inplanes=128, planes=256, stride=2, geoplanes=self.geoplanes)

        self.rgb_decoder_layer4 = deconvbnrelu(
            in_channels=256,
            out_channels=128,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1)
        self.rgb_decoder_layer2 = deconvbnrelu(
            in_channels=128,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1)
        self.rgb_decoder_layer0 = deconvbnrelu(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=0)
        self.rgb_decoder_layer = deconvbnrelu(
            in_channels=16,
            out_channels=8,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1)
        self.rgb_decoder_output = deconvbnrelu(
            in_channels=8,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=0)

        # depth encoder
        self.depth_conv_init = convbnrelu(
            in_channels=2, out_channels=8, kernel_size=5, stride=1, padding=2)

        self.depth_layer1 = BasicBlockGeo(
            inplanes=8, planes=16, stride=2, geoplanes=self.geoplanes)
        self.depth_layer2 = BasicBlockGeo(
            inplanes=16, planes=32, stride=1, geoplanes=self.geoplanes)
        self.depth_layer3 = BasicBlockGeo(
            inplanes=64, planes=64, stride=2, geoplanes=self.geoplanes)
        self.depth_layer4 = BasicBlockGeo(
            inplanes=64, planes=128, stride=1, geoplanes=self.geoplanes)
        self.depth_layer5 = BasicBlockGeo(
            inplanes=256, planes=256, stride=2, geoplanes=self.geoplanes)

        self.decoder_layer3 = deconvbnrelu(
            in_channels=256,
            out_channels=128,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1)
        self.decoder_layer4 = deconvbnrelu(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=0)
        self.decoder_layer5 = deconvbnrelu(
            in_channels=64,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1)
        self.decoder_layer6 = deconvbnrelu(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=0)
        self.decoder_layer7 = deconvbnrelu(
            in_channels=16,
            out_channels=8,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1)

        # output
        self.rgbdepth_decoder_stage1 = deconvbnrelu(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1)
        self.rgbdepth_decoder_stage2 = deconvbnrelu(
            in_channels=16,
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1)
        self.rgbdepth_decoder_stage3 = deconvbnrelu(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=0)

        self.final_decoder_stage1 = deconvbnrelu(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=0)
        self.final_decoder_stage2 = deconvbnrelu(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=0)
        self.final_decoder_stage3 = deconvbnrelu(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=0)

        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)

        weights_init(self)

    def forward(self, rgb, depth, confidence, depth_values, stage_idx,
                origin_feat, intrinsics_matrices_stage):

        rgb = rgb
        depth_min, depth_max = depth_values[:, 0, None, None,
                                            None], depth_values[:, -1, None,
                                                                None, None]
        d = (depth - depth_min) / (depth_max - depth_min)

        if self.mask_type == 'basic':
            valid_mask = torch.where(d > 0, torch.full_like(d, 1.0),
                                     torch.full_like(d, 0.0))
        elif self.mask_type == 'mean':
            valid_mask = torch.where(
                torch.logical_and(d > 0, confidence > confidence.mean()),
                torch.full_like(d, 1.0), torch.full_like(d, 0.0))

        # pre-data preparation
        if self.convolutional_layer_encoding in ['uv', 'xyz']:
            B, _, W, H = rgb.shape
            position = AddCoordsNp(H, W)
            position = position.call()
            position = torch.from_numpy(position).to(rgb.device).repeat(
                B, 1, 1, 1).transpose(-1, 1)
            unorm = position[:, 0:1, :, :]
            vnorm = position[:, 1:2, :, :]

            vnorm_s2 = self.pooling(vnorm)
            vnorm_s3 = self.pooling(vnorm_s2)
            vnorm_s4 = self.pooling(vnorm_s3)

            unorm_s2 = self.pooling(unorm)
            unorm_s3 = self.pooling(unorm_s2)
            unorm_s4 = self.pooling(unorm_s3)

        if self.convolutional_layer_encoding in ['z', 'xyz']:
            d_s2, vm_s2 = self.sparsepooling(d, valid_mask)
            d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
            d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)

        if self.convolutional_layer_encoding == 'xyz':
            K = intrinsics_matrices_stage
            f352 = K[:, 1, 1]
            f352 = f352.unsqueeze(1)
            f352 = f352.unsqueeze(2)
            f352 = f352.unsqueeze(3)
            c352 = K[:, 1, 2]
            c352 = c352.unsqueeze(1)
            c352 = c352.unsqueeze(2)
            c352 = c352.unsqueeze(3)
            f1216 = K[:, 0, 0]
            f1216 = f1216.unsqueeze(1)
            f1216 = f1216.unsqueeze(2)
            f1216 = f1216.unsqueeze(3)
            c1216 = K[:, 0, 2]
            c1216 = c1216.unsqueeze(1)
            c1216 = c1216.unsqueeze(2)
            c1216 = c1216.unsqueeze(3)

        # geometric info
        if self.convolutional_layer_encoding == 'std':
            geo_s1 = None
            geo_s2 = None
            geo_s3 = None
            geo_s4 = None
        elif self.convolutional_layer_encoding == 'uv':
            geo_s1 = torch.cat((vnorm, unorm), dim=1)
            geo_s2 = torch.cat((vnorm_s2, unorm_s2), dim=1)
            geo_s3 = torch.cat((vnorm_s3, unorm_s3), dim=1)
            geo_s4 = torch.cat((vnorm_s4, unorm_s4), dim=1)
        elif self.convolutional_layer_encoding == 'z':
            geo_s1 = d
            geo_s2 = d_s2
            geo_s3 = d_s3
            geo_s4 = d_s4
        elif self.convolutional_layer_encoding == 'xyz':
            geo_s1 = self.geofeature(d, vnorm, unorm, H, W, c352, c1216, f352,
                                     f1216)
            geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, H / 2, W / 2,
                                     c352, c1216, f352, f1216)
            geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, H / 4, W / 4,
                                     c352, c1216, f352, f1216)
            geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, H / 8, W / 8,
                                     c352, c1216, f352, f1216)

        # -----------------------------------------------------------------------------------------

        # 128*160 -> 256*320 -> 512*640
        rgb_feature = self.rgb_conv_init(torch.cat((rgb, d), dim=1))  # b 8 h w
        rgb_feature1 = self.rgb_encoder_layer1(rgb_feature, geo_s1,
                                               geo_s2)  # b 16 h/2 w/2
        rgb_feature2 = self.rgb_encoder_layer2(rgb_feature1, geo_s2,
                                               geo_s2)  # b 32 h/2 w/2
        rgb_feature3 = self.rgb_encoder_layer3(rgb_feature2, geo_s2,
                                               geo_s3)  # b 64 h/4 w/4
        rgb_feature4 = self.rgb_encoder_layer4(rgb_feature3, geo_s3,
                                               geo_s3)  # b 128 h/4 w/4
        rgb_feature5 = self.rgb_encoder_layer5(rgb_feature4, geo_s3,
                                               geo_s4)  # b 256 h/8 w/8

        rgb_feature_decoder4 = self.rgb_decoder_layer4(rgb_feature5)
        rgb_feature4_plus = rgb_feature_decoder4 + rgb_feature4  # b 128 h/4 w/4

        rgb_feature_decoder2 = self.rgb_decoder_layer2(rgb_feature4_plus)
        rgb_feature2_plus = rgb_feature_decoder2 + rgb_feature2  # b 32 h/2 w/2

        rgb_feature_decoder0 = self.rgb_decoder_layer0(rgb_feature2_plus)
        rgb_feature0_plus = rgb_feature_decoder0 + rgb_feature1  # b 16 h/2 w/2

        rgb_feature_decoder = self.rgb_decoder_layer(rgb_feature0_plus)
        rgb_feature_plus = rgb_feature_decoder + rgb_feature  # b 8 h w

        rgb_output = self.rgb_decoder_output(rgb_feature_plus)  # b 2 h w

        rgb_depth = rgb_output[:, 0:1, :, :]
        # rgb_conf = rgb_output[:, 1:2, :, :]

        # -----------------------------------------------------------------------------------------

        sparsed_feature = self.depth_conv_init(
            torch.cat((d, rgb_depth), dim=1))  # b 8 h w
        sparsed_feature1 = self.depth_layer1(sparsed_feature, geo_s1,
                                             geo_s2)  # b 16 h/2 w/2
        sparsed_feature2 = self.depth_layer2(sparsed_feature1, geo_s2,
                                             geo_s2)  # b 32 h/2 w/2

        sparsed_feature2_plus = torch.cat(
            [rgb_feature2_plus, sparsed_feature2], 1)
        sparsed_feature3 = self.depth_layer3(sparsed_feature2_plus, geo_s2,
                                             geo_s3)  # b 64 h/4 w/4
        sparsed_feature4 = self.depth_layer4(sparsed_feature3, geo_s3,
                                             geo_s3)  # b 128 h/4 w/4

        sparsed_feature4_plus = torch.cat(
            [rgb_feature4_plus, sparsed_feature4], 1)
        sparsed_feature5 = self.depth_layer5(sparsed_feature4_plus, geo_s3,
                                             geo_s4)  # b 256 h/8 w/8

        # -----------------------------------------------------------------------------------------

        fusion3 = rgb_feature5 + sparsed_feature5
        decoder_feature3 = self.decoder_layer3(fusion3)  # b 128 h/4 w/4

        fusion4 = sparsed_feature4 + decoder_feature3
        decoder_feature4 = self.decoder_layer4(fusion4)  # b 64 h/4 w/4

        if stage_idx >= 1:
            decoder_feature5 = self.decoder_layer5(decoder_feature4)
            fusion5 = sparsed_feature2 + decoder_feature5  # b 32 h/2 w/2
            if stage_idx == 1:
                rgbdepth_feature = self.rgbdepth_decoder_stage1(fusion5)
                if self.add_origin_feat_flag:
                    final_feature = self.final_decoder_stage1(rgbdepth_feature
                                                              + origin_feat)
                else:
                    final_feature = self.final_decoder_stage1(rgbdepth_feature)

        if stage_idx >= 2:
            decoder_feature6 = self.decoder_layer6(decoder_feature5)
            fusion6 = sparsed_feature1 + decoder_feature6  # b 16 h/2 w/2
            if stage_idx == 2:
                rgbdepth_feature = self.rgbdepth_decoder_stage2(fusion6)
                if self.add_origin_feat_flag:
                    final_feature = self.final_decoder_stage2(rgbdepth_feature
                                                              + origin_feat)
                else:
                    final_feature = self.final_decoder_stage2(rgbdepth_feature)

        if stage_idx >= 3:
            decoder_feature7 = self.decoder_layer7(decoder_feature6)
            fusion7 = sparsed_feature + decoder_feature7  # b 8 h w
            if stage_idx == 3:
                rgbdepth_feature = self.rgbdepth_decoder_stage3(fusion7)
                if self.add_origin_feat_flag:
                    final_feature = self.final_decoder_stage3(rgbdepth_feature
                                                              + origin_feat)
                else:
                    final_feature = self.final_decoder_stage3(rgbdepth_feature)

        return final_feature


class GeoRegNet2d(nn.Module):

    def __init__(self,
                 input_channel=128,
                 base_channel=32,
                 convolutional_layer_encoding='std'):
        super(GeoRegNet2d, self).__init__()

        self.convolutional_layer_encoding = convolutional_layer_encoding  # std / uv / z / xyz
        self.mask_type = 'basic'  # basic / mean

        if self.convolutional_layer_encoding == 'std':
            self.geoplanes = 0
        elif self.convolutional_layer_encoding == 'z':
            self.geoplanes = 1

        self.conv_init = ConvBnReLU3D(
            input_channel,
            out_channels=8,
            kernel_size=(1, 3, 3),
            pad=(0, 1, 1))
        self.encoder_layer1 = Reg_BasicBlockGeo(
            inplanes=8,
            planes=16,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            geoplanes=self.geoplanes)
        self.encoder_layer2 = Reg_BasicBlockGeo(
            inplanes=16,
            planes=32,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
            geoplanes=self.geoplanes)
        self.encoder_layer3 = Reg_BasicBlockGeo(
            inplanes=32,
            planes=64,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            geoplanes=self.geoplanes)
        self.encoder_layer4 = Reg_BasicBlockGeo(
            inplanes=64,
            planes=128,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
            geoplanes=self.geoplanes)
        self.encoder_layer5 = Reg_BasicBlockGeo(
            inplanes=128,
            planes=256,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            geoplanes=self.geoplanes)

        self.decoder_layer4 = reg_deconvbnrelu(
            in_channels=256,
            out_channels=128,
            kernel_size=(1, 5, 5),
            stride=(1, 2, 2),
            padding=(0, 2, 2),
            output_padding=(0, 1, 1))
        self.decoder_layer3 = reg_deconvbnrelu(
            in_channels=128,
            out_channels=64,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
            output_padding=0)
        self.decoder_layer2 = reg_deconvbnrelu(
            in_channels=64,
            out_channels=32,
            kernel_size=(1, 5, 5),
            stride=(1, 2, 2),
            padding=(0, 2, 2),
            output_padding=(0, 1, 1))
        self.decoder_layer1 = reg_deconvbnrelu(
            in_channels=32,
            out_channels=16,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
            output_padding=0)
        self.decoder_layer = reg_deconvbnrelu(
            in_channels=16,
            out_channels=8,
            kernel_size=(1, 5, 5),
            stride=(1, 2, 2),
            padding=(0, 2, 2),
            output_padding=(0, 1, 1))

        self.prob = reg_deconvbnrelu(
            in_channels=8,
            out_channels=1,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
            output_padding=0)

        self.depthpooling = nn.MaxPool3d((2, 1, 1), (2, 1, 1))
        self.basicpooling = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        weights_init(self)

    def forward(self, x, stage_idx, geo_reg_data=None):

        B, C, D, W, H = x.shape

        if stage_idx >= 1 and self.convolutional_layer_encoding == 'z':
            prob_volume = geo_reg_data['prob_volume_last'].unsqueeze(
                1)  # B 1 D H W
        else:
            assert self.convolutional_layer_encoding == 'std'

        # geometric info
        if self.convolutional_layer_encoding == 'std':
            geo_s1 = None
            geo_s2 = None
            geo_s3 = None
            # geo_s4 = None
        elif self.convolutional_layer_encoding == 'z':
            if stage_idx == 2:
                geo_s1 = self.depthpooling(prob_volume)
            else:
                geo_s1 = prob_volume  # B 1 D H W
            geo_s2 = self.basicpooling(geo_s1)
            geo_s3 = self.basicpooling(geo_s2)

        feature = self.conv_init(x)  # B 8 D H W
        feature1 = self.encoder_layer1(feature, geo_s1,
                                       geo_s1)  # B  16 D H/2 W/2
        feature2 = self.encoder_layer2(feature1, geo_s2,
                                       geo_s2)  # B  32 D H/2 W/2
        feature3 = self.encoder_layer3(feature2, geo_s2,
                                       geo_s2)  # B  64 D H/4 W/4
        feature4 = self.encoder_layer4(feature3, geo_s3,
                                       geo_s3)  # B 128 D H/4 W/4
        feature5 = self.encoder_layer5(feature4, geo_s3,
                                       geo_s3)  # B 256 D H/8 W/8

        feature_decoder4 = self.decoder_layer4(feature5)
        feature4_plus = feature_decoder4 + feature4  # B 128 D H/4 W/4

        feature_decoder3 = self.decoder_layer3(feature4_plus)
        feature3_plus = feature_decoder3 + feature3  # B 64 D H/4 W/4

        feature_decoder2 = self.decoder_layer2(feature3_plus)
        feature2_plus = feature_decoder2 + feature2  # B 32 D H/2 W/2

        feature_decoder1 = self.decoder_layer1(feature2_plus)
        feature1_plus = feature_decoder1 + feature1  # B 16 D H/2 W/2

        feature_decoder = self.decoder_layer(feature1_plus)
        feature_plus = feature_decoder + feature  # B  8 D H W

        x = self.prob(feature_plus)

        return x.squeeze(1)


# --------------------------------------------------------------


class BasicBlockGeo(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 geoplanes=3):
        super(BasicBlockGeo, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')

        self.conv1 = conv3x3(inplanes + geoplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes + geoplanes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes + geoplanes, planes, stride),
                norm_layer(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, g1=None, g2=None):
        identity = x
        if g1 is not None:
            x = torch.cat((x, g1), 1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if g2 is not None:
            out = torch.cat((g2, out), 1)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class GeometryFeature(nn.Module):

    def __init__(self):
        super(GeometryFeature, self).__init__()

    def forward(self, z, vnorm, unorm, h, w, ch, cw, fh, fw):
        x = z * (0.5 * h * (vnorm + 1) - ch) / fh
        y = z * (0.5 * w * (unorm + 1) - cw) / fw
        return torch.cat((x, y, z), 1)


class SparseDownSampleClose(nn.Module):

    def __init__(self, stride):
        super(SparseDownSampleClose, self).__init__()
        self.pooling = nn.MaxPool2d(stride, stride)
        self.large_number = 600

    def forward(self, d, mask):
        encode_d = -(1 - mask) * self.large_number - d

        d = -self.pooling(encode_d)
        mask_result = self.pooling(mask)
        d_result = d - (1 - mask_result) * self.large_number

        return d_result, mask_result


def convbnrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


def deconvbnrelu(in_channels,
                 out_channels,
                 kernel_size=5,
                 stride=2,
                 padding=2,
                 output_padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


def weights_init(m):
    """Initialize filters with Gaussian random weights"""
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def conv3x3(in_planes,
            out_planes,
            stride=1,
            groups=1,
            dilation=1,
            bias=False,
            padding=1):
    """3x3 convolution with padding"""
    if padding >= 1:
        padding = dilation
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias)


class AddCoordsNp():
    """Add coords to a tensor"""

    def __init__(self, x_dim=64, y_dim=64, with_r=False):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r

    def call(self):
        """
        input_tensor: (batch, x_dim, y_dim, c)
        """
        xx_ones = np.ones([self.x_dim], dtype=np.int32)
        xx_ones = np.expand_dims(xx_ones, 1)

        xx_range = np.expand_dims(np.arange(self.y_dim), 0)

        xx_channel = np.matmul(xx_ones, xx_range)
        xx_channel = np.expand_dims(xx_channel, -1)

        yy_ones = np.ones([self.y_dim], dtype=np.int32)
        yy_ones = np.expand_dims(yy_ones, 0)

        yy_range = np.expand_dims(np.arange(self.x_dim), 1)

        yy_channel = np.matmul(yy_range, yy_ones)
        yy_channel = np.expand_dims(yy_channel, -1)

        xx_channel = xx_channel.astype('float32') / (self.y_dim - 1)
        yy_channel = yy_channel.astype('float32') / (self.x_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        ret = np.concatenate([xx_channel, yy_channel], axis=-1)

        if self.with_r:
            rr = np.sqrt(
                np.square(xx_channel - 0.5) + np.square(yy_channel - 0.5))
            ret = np.concatenate([ret, rr], axis=-1)

        return ret


# --------------------------------------------------------------


class Reg_BasicBlockGeo(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size,
                 stride,
                 padding,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=nn.BatchNorm3d,
                 geoplanes=3):
        super(Reg_BasicBlockGeo, self).__init__()

        self.conv1 = regconv3D(
            inplanes + geoplanes,
            planes,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1))
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = regconv3D(planes + geoplanes, planes, kernel_size, stride,
                               padding)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                regconv1x1(inplanes + geoplanes, planes, kernel_size, stride,
                           padding),
                norm_layer(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, g1=None, g2=None):
        identity = x
        if g1 is not None:
            x = torch.cat((x, g1), 1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if g2 is not None:
            out = torch.cat((g2, out), 1)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def regconv3D(in_planes,
              out_planes,
              kernel_size,
              stride,
              padding,
              groups=1,
              dilation=1,
              bias=False):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation)


def regconv1x1(in_planes,
               out_planes,
               kernel_size,
               stride,
               padding,
               groups=1,
               bias=False):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias)


def reg_deconvbnrelu(in_channels, out_channels, kernel_size, stride, padding,
                     output_padding):
    return nn.Sequential(
        nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False), nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True))
