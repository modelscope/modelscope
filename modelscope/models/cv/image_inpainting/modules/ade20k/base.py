"""
Part of the implementation is borrowed and modified from LaMa, publicly available at
https://github.com/saic-mdal/lama
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import BatchNorm2d

from . import resnet

NUM_CLASS = 150


# Model Builder
class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    @staticmethod
    def build_encoder(arch='resnet50dilated',
                      fc_dim=512,
                      weights='',
                      model_dir=''):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](
                pretrained=pretrained, model_dir=model_dir)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](
                pretrained=pretrained, model_dir=model_dir)
            net_encoder = Resnet(orig_resnet)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage),
                strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='ppm_deepsup',
                      fc_dim=512,
                      num_class=NUM_CLASS,
                      weights='',
                      use_softmax=False,
                      drop_last_conv=False):
        arch = arch.lower()
        if arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                drop_last_conv=drop_last_conv)
        elif arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                drop_last_conv=drop_last_conv)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage),
                strict=False)
        return net_decoder

    @staticmethod
    def get_decoder(weights_path, arch_encoder, arch_decoder, fc_dim,
                    drop_last_conv, *arts, **kwargs):
        path = os.path.join(
            weights_path, 'ade20k',
            f'ade20k-{arch_encoder}-{arch_decoder}/decoder_epoch_20.pth')
        return ModelBuilder.build_decoder(
            arch=arch_decoder,
            fc_dim=fc_dim,
            weights=path,
            use_softmax=True,
            drop_last_conv=drop_last_conv)

    @staticmethod
    def get_encoder(weights_path, arch_encoder, arch_decoder, fc_dim,
                    segmentation, *arts, **kwargs):
        if segmentation:
            path = os.path.join(
                weights_path, 'ade20k',
                f'ade20k-{arch_encoder}-{arch_decoder}/encoder_epoch_20.pth')
        else:
            path = ''
        return ModelBuilder.build_encoder(
            arch=arch_encoder,
            fc_dim=fc_dim,
            weights=path,
            model_dir=weights_path)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False),
        BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):

    def __init__(self,
                 num_class=NUM_CLASS,
                 fc_dim=4096,
                 use_softmax=False,
                 pool_scales=(1, 2, 3, 6),
                 drop_last_conv=False):
        super().__init__()
        self.use_softmax = use_softmax
        self.drop_last_conv = drop_last_conv

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                    BatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(
                fc_dim + len(pool_scales) * 512,
                512,
                kernel_size=3,
                padding=1,
                bias=False), BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1), nn.Conv2d(512, num_class, kernel_size=1))
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(
                nn.functional.interpolate(
                    pool_scale(conv5), (input_size[2], input_size[3]),
                    mode='bilinear',
                    align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        if self.drop_last_conv:
            return ppm_out
        else:
            x = self.conv_last(ppm_out)

            if self.use_softmax:  # is True during inference
                x = nn.functional.interpolate(
                    x, size=segSize, mode='bilinear', align_corners=False)
                x = nn.functional.softmax(x, dim=1)
                return x

            # deep sup
            conv4 = conv_out[-2]
            _ = self.cbr_deepsup(conv4)
            _ = self.dropout_deepsup(_)
            _ = self.conv_last_deepsup(_)

            x = nn.functional.log_softmax(x, dim=1)
            _ = nn.functional.log_softmax(_, dim=1)

            return (x, _)


class Resnet(nn.Module):

    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


# Resnet Dilated
class ResnetDilated(nn.Module):

    def __init__(self, orig_resnet, dilate_scale=8):
        super().__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


# last conv, deep supervision
class C1DeepSup(nn.Module):

    def __init__(self,
                 num_class=150,
                 fc_dim=2048,
                 use_softmax=False,
                 drop_last_conv=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax
        self.drop_last_conv = drop_last_conv

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)

        if self.drop_last_conv:
            return x
        else:
            x = self.conv_last(x)

            if self.use_softmax:  # is True during inference
                x = nn.functional.interpolate(
                    x, size=segSize, mode='bilinear', align_corners=False)
                x = nn.functional.softmax(x, dim=1)
                return x

            # deep sup
            conv4 = conv_out[-2]
            _ = self.cbr_deepsup(conv4)
            _ = self.conv_last_deepsup(_)

            x = nn.functional.log_softmax(x, dim=1)
            _ = nn.functional.log_softmax(_, dim=1)

            return (x, _)


# last conv
class C1(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x
