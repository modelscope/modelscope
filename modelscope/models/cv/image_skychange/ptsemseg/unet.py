# Copyright 2021-2022 The Alibaba Vision Team Authors. All rights reserved.
import torch
import torch.nn as nn

from .BlockModules import ASPP


class Conv2DBatchNormRelu(nn.Module):

    def __init__(self,
                 in_channels,
                 n_filters,
                 k_size,
                 stride,
                 padding,
                 bias=True,
                 dilation=1,
                 with_bn=True,
                 with_relu=True):
        super(Conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if with_bn:
            if with_relu:
                self.cbr_unit = nn.Sequential(conv_mod,
                                              nn.BatchNorm2d(int(n_filters)),
                                              nn.ReLU(inplace=True))
            else:
                self.cbr_unit = nn.Sequential(conv_mod,
                                              nn.BatchNorm2d(int(n_filters)))
        else:
            if with_relu:
                self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))
            else:
                self.cbr_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class SegnetDown2(nn.Module):

    def __init__(self, in_size, out_size):
        super(SegnetDown2, self).__init__()
        self.conv1 = Conv2DBatchNormRelu(
            in_size, out_size, k_size=3, stride=1, padding=1)
        self.conv2 = Conv2DBatchNormRelu(
            out_size, out_size, k_size=3, stride=1, padding=1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class SegnetDown3(nn.Module):

    def __init__(self, in_size, out_size):
        super(SegnetDown3, self).__init__()
        self.conv1 = Conv2DBatchNormRelu(
            in_size, out_size, k_size=3, stride=1, padding=1)
        self.conv2 = Conv2DBatchNormRelu(
            out_size, out_size, k_size=3, stride=1, padding=1)
        self.conv3 = Conv2DBatchNormRelu(
            out_size, out_size, k_size=3, stride=1, padding=1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class SegnetUp1(nn.Module):

    def __init__(self, in_size, out_size):
        super(SegnetUp1, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv = Conv2DBatchNormRelu(
            in_size, out_size, k_size=5, stride=1, padding=2, with_relu=False)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(
            input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv(outputs)
        return outputs


class Unet(nn.Module):

    def __init__(self,
                 n_classes=2,
                 in_channels=4,
                 is_unpooling=True,
                 pretrain=True,
                 **kwargs):
        super(Unet, self).__init__()
        print('Load Unet')
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling
        self.pretrain = pretrain
        self.is_contain_aspp = True if 'aspp' in kwargs else False

        if self.is_contain_aspp:
            aspp_param = kwargs['aspp']
            self.aspp_layer = ASPP(
                inplanes=128,
                outplanes=aspp_param['outplanes'],
                dilations=aspp_param['dilations'],
                drop_rate=aspp_param['drop_rate'])
            self.aspp_channels = aspp_param['outplanes']

        self.down1 = SegnetDown2(self.in_channels, 64)
        self.down2 = SegnetDown2(64, 128)
        self.down3 = SegnetDown3(128, 256)
        self.down4 = SegnetDown3(256, 512)
        self.down5 = SegnetDown3(512, 512)

        self.up5 = SegnetUp1(512, 512)
        self.up4 = SegnetUp1(512, 256)
        self.up3 = SegnetUp1(256, 128)

        if self.is_contain_aspp:
            self.conv_1x1_aspp = Conv2DBatchNormRelu(
                128 + self.aspp_channels,
                128,
                k_size=1,
                stride=1,
                padding=0,
                with_relu=False)

        self.up2 = SegnetUp1(128, 64)
        self.up1 = SegnetUp1(64, n_classes)
        self.sigmoid = nn.Sigmoid()

        if self.pretrain:
            import torchvision.models as models
            vgg16 = models.vgg16()
            self.init_vgg16_params(vgg16)

    def forward(self, inputs):  # [1, 4, 1346, 1152] [2, 4, 1280, 1280]
        # inputs: [N, 4, 320, 320]
        # outputs, indices, unpooled_shape
        down1, indices_1, unpool_shape1 = self.down1(
            inputs)  # [1, 64, 673, 576]  [2, 64, 640, 640]
        down2, indices_2, unpool_shape2 = self.down2(
            down1)  # [1, 128, 336, 288]  [2, 128, 320, 320]
        down3, indices_3, unpool_shape3 = self.down3(
            down2)  # [1, 256, 168, 144]  [2, 256, 160, 160]
        torch.cuda.empty_cache()
        if self.is_contain_aspp:  # batchsize can not be 1
            aspp_output = self.aspp_layer(down2)

        down4, indices_4, unpool_shape4 = self.down4(
            down3)  # [1, 512, 84, 72]    [2, 512, 80, 80]
        down5, indices_5, unpool_shape5 = self.down5(
            down4)  # [1, 512, 42, 36]    [2, 512, 80, 80]
        torch.cuda.empty_cache()
        up5 = self.up5(down5, indices_5,
                       unpool_shape5)  # [1, 512, 84, 72]  [2, 512, 80, 80]
        up4 = self.up4(up5, indices_4,
                       unpool_shape4)  # [1, 256, 168, 144]  [2, 256, 160, 160]
        torch.cuda.empty_cache()
        up3 = self.up3(
            up4, indices_3,
            unpool_shape3)  # [1, 128, 336, 288]     [2, 128, 320, 320]
        if self.is_contain_aspp:
            up3 = torch.cat([up3, aspp_output], 1)  # [2, 256, 320, 320]
            up3 = self.conv_1x1_aspp(up3)  # [2, 128, 320, 320]

        up2 = self.up2(
            up3, indices_2,
            unpool_shape2)  # [1, 64, 673, 576]  indices_2: [2, 128, 320, 320]
        up1 = self.up1(up2, indices_1, unpool_shape1)  # [1, 1, 1346, 1152]

        x = torch.squeeze(up1, dim=1)  # [N, 1, 320, 320] -> [N, 320, 320]
        x = self.sigmoid(x)

        return x  # [2, 1280, 1280]

    def init_vgg16_params(self, vgg16):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                if l1.weight.size() == l2.weight.size() and l1.bias.size(
                ) == l2.bias.size():
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
