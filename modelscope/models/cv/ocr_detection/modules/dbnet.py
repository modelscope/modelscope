# ------------------------------------------------------------------------------
# Part of implementation is adopted from ViLT,
# made publicly available under the Apache License 2.0 at https://github.com/dandelin/ViLT.
# ------------------------------------------------------------------------------
import math
import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn

from .proxyless import CompactDetBackbone

BatchNorm2d = nn.BatchNorm2d


def constant_init(module, constant, bias=0):
    nn.init.constant_(module.weight, constant)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class DwPwConv(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DwPwConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size,
            stride,
            padding,
            groups=in_planes,
            bias=bias)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.pointwise = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.pointwise(out)

        return out


class DwPwConvTranspose(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(DwPwConvTranspose, self).__init__()
        self.depthwise = nn.ConvTranspose2d(
            in_planes, in_planes, kernel_size, stride, groups=in_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.pointwise = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.pointwise(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(BasicBlock, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        # self.conv2 = conv3x3(planes, planes)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, padding=1, bias=False)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                from assets.ops.dcn import DeformConv
                conv_op = DeformConv
                offset_channels = 18
            else:
                from assets.ops.dcn import ModulatedDeformConv
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                padding=1)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                padding=1,
                deformable_groups=deformable_groups,
                bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(Bottleneck, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                from assets.ops.dcn import DeformConv
                conv_op = DeformConv
                offset_channels = 18
            else:
                from assets.ops.dcn import ModulatedDeformConv
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                padding=1)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                padding=1,
                stride=stride,
                deformable_groups=deformable_groups,
                bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dcn = dcn
        self.with_dcn = dcn is not None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False)):
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dcn=dcn)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dcn=dcn)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dcn=dcn)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # self.smooth = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.dcn is not None:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                    if hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x2, x3, x4, x5


class LightSegDetector(nn.Module):

    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256,
                 k=10,
                 bias=False,
                 adaptive=False,
                 smooth=False,
                 serial=False,
                 dw_kernel_size=3,
                 dw_padding=1,
                 *args,
                 **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(LightSegDetector, self).__init__()
        self.k = k
        self.serial = serial
        self.inner_channels = inner_channels
        self.bias = bias
        self.dw_kernel_size = dw_kernel_size
        self.dw_padding = dw_padding

        self.up5 = nn.Upsample(scale_factor=8, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        # DwPM
        self.binarize = nn.Sequential(
            DwPwConv(
                inner_channels,
                inner_channels // 4,
                kernel_size=self.dw_kernel_size,
                padding=self.dw_padding,
                bias=bias), BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            DwPwConvTranspose(inner_channels // 4, inner_channels // 4, 2, 2),
            BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            DwPwConvTranspose(inner_channels // 4, 1, 2, 2), nn.Sigmoid())

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                inner_channels, serial=serial, smooth=smooth, bias=bias)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

    def _init_thresh(self,
                     inner_channels,
                     serial=False,
                     smooth=False,
                     bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(
                inner_channels,
                inner_channels // 4,
                self.dw_kernel_size,
                padding=self.dw_padding,
                bias=bias), BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(
                inner_channels // 4,
                inner_channels // 4,
                smooth=smooth,
                bias=bias), BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(
                inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())

        return self.thresh

    def _init_upsample(self,
                       in_channels,
                       out_channels,
                       smooth=False,
                       bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)
            ]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=1,
                        bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        p5 = self.up5(self.in5(c5))
        p4 = self.up4(self.in4(c4))
        p3 = self.up3(self.in3(c3))
        p2 = self.in2(c2)

        fuse = p5 + p4 + p3 + p2

        # this is the pred module, not binarization module;
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                    (fuse, nn.functional.interpolate(binary, fuse.shape[2:])),
                    1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class SegDetector(nn.Module):

    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256,
                 k=10,
                 bias=False,
                 adaptive=False,
                 smooth=False,
                 serial=False,
                 *args,
                 **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(SegDetector, self).__init__()
        self.k = k
        self.serial = serial
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.out5 = nn.Sequential(
            nn.Conv2d(
                inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(
                inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(
                inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels // 4, 3, padding=1, bias=bias)

        self.binarize = nn.Sequential(
            nn.Conv2d(
                inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2), nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self,
                     inner_channels,
                     serial=False,
                     smooth=False,
                     bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(
                in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            self._init_upsample(
                inner_channels // 4,
                inner_channels // 4,
                smooth=smooth,
                bias=bias), BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(
                inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels,
                       out_channels,
                       smooth=False,
                       bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)
            ]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=1,
                        bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        # this is the pred module, not binarization module;
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                    (fuse, nn.functional.interpolate(binary, fuse.shape[2:])),
                    1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class BasicModel(nn.Module):

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)

        self.backbone = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        self.decoder = SegDetector(
            in_channels=[64, 128, 256, 512], adaptive=True, k=50, **kwargs)

    def forward(self, data, *args, **kwargs):
        return self.decoder(self.backbone(data), *args, **kwargs)


def parallelize(model, distributed, local_rank):
    if distributed:
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=[local_rank],
            find_unused_parameters=True)
    else:
        return nn.DataParallel(model)


class VLPTModel(nn.Module):

    def __init__(self, *args, **kwargs):
        """
        VLPT-STD pretrained DBNet-resnet50 model,
        paper reference: https://arxiv.org/pdf/2204.13867.pdf
        """
        super(VLPTModel, self).__init__()
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        self.decoder = SegDetector(
            in_channels=[256, 512, 1024, 2048], adaptive=True, k=50, **kwargs)

    def forward(self, x):
        return self.decoder(self.backbone(x))


class DBNasModel(nn.Module):

    def __init__(self, *args, **kwargs):
        """
        DB-NAS model
        """
        super(DBNasModel, self).__init__()
        self.backbone = CompactDetBackbone(
            width_stages=[32, 64, 96, 128], input_channel=32, **kwargs)
        self.decoder = LightSegDetector(
            in_channels=[32, 64, 96, 128],
            adaptive=True,
            k=50,
            inner_channels=64,
            dw_kernel_size=5,
            dw_padding=2,
            **kwargs)

    def forward(self, x):
        return self.decoder(self.backbone(x))


class DBModel(nn.Module):

    def __init__(self, *args, **kwargs):
        """
        DBNet-resnet18 model without deformable conv,
        paper reference: https://arxiv.org/pdf/1911.08947.pdf
        """
        super(DBModel, self).__init__()
        self.backbone = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        self.decoder = SegDetector(
            in_channels=[64, 128, 256, 512], adaptive=True, k=50, **kwargs)

    def forward(self, x):
        return self.decoder(self.backbone(x))


class DBModel_v2(nn.Module):

    def __init__(self,
                 device,
                 distributed: bool = False,
                 local_rank: int = 0,
                 *args,
                 **kwargs):
        """
        DBNet-resnet18 model without deformable conv,
        paper reference: https://arxiv.org/pdf/1911.08947.pdf
        """
        super(DBModel_v2, self).__init__()
        from .seg_detector_loss import L1BalanceCELoss

        self.model = BasicModel(*args, **kwargs)
        self.model = parallelize(self.model, distributed, local_rank)
        self.criterion = L1BalanceCELoss()
        self.criterion = parallelize(self.criterion, distributed, local_rank)
        self.device = device
        self.to(self.device)

    def forward(self, batch, training=False):
        if isinstance(batch, dict):
            data = batch['image'].to(self.device)
        else:
            data = batch.to(self.device)
        data = data.float()
        pred = self.model(data, training=self.training)

        if self.training:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred
