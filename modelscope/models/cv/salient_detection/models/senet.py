# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import res2net50_v1b_26w_4s as res2net
from .modules import AMFusion, AreaLayer, EdgeLayer, StructureE
from .utils import ASPP, CBAM, ConvBNReLU


class SENet(nn.Module):

    def __init__(self, backbone_path=None, pretrained=False):
        super(SENet, self).__init__()
        resnet50 = res2net(backbone_path, pretrained)
        self.layer0_1 = nn.Sequential(resnet50.conv1, resnet50.bn1,
                                      resnet50.relu)
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.aspp3 = ASPP(1024, 256)
        self.aspp4 = ASPP(2048, 256)
        self.cbblock3 = CBAM(inchs=256, kernel_size=5)
        self.cbblock4 = CBAM(inchs=256, kernel_size=5)
        self.up = nn.Upsample(
            mode='bilinear', scale_factor=2, align_corners=False)
        self.conv_up = ConvBNReLU(512, 512, 1)
        self.aux_edge = EdgeLayer(512, 256)
        self.aux_area = AreaLayer(512, 256)
        self.layer1_enhance = StructureE(256, 128, 128)
        self.layer2_enhance = StructureE(512, 256, 128)
        self.layer3_decoder = AMFusion(512, 256, 128)
        self.layer2_decoder = AMFusion(256, 128, 128)
        self.out_conv_8 = nn.Conv2d(256, 1, 1)
        self.out_conv_4 = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        layer0 = self.layer0_1(x)
        layer0s = self.maxpool(layer0)
        layer1 = self.layer1(layer0s)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer3_eh = self.cbblock3(self.aspp3(layer3))
        layer4_eh = self.cbblock4(self.aspp4(layer4))
        layer34 = self.conv_up(
            torch.cat((self.up(layer4_eh), layer3_eh), dim=1))
        edge_atten = self.aux_edge(layer1, layer34)
        area_atten = self.aux_area(layer1, layer34)
        edge_atten_ = torch.sigmoid(edge_atten)
        layer1_eh = self.layer1_enhance(layer1, edge_atten_)
        layer2_eh = self.layer2_enhance(layer2, edge_atten_)
        layer2_fu = self.layer3_decoder(layer2_eh, layer34, area_atten)
        out_8 = self.out_conv_8(layer2_fu)
        layer1_fu = self.layer2_decoder(layer1_eh, layer2_fu, out_8)
        out_4 = self.out_conv_4(layer1_fu)
        out_16 = F.interpolate(
            area_atten,
            size=x.size()[2:],
            mode='bilinear',
            align_corners=False)
        out_8 = F.interpolate(
            out_8, size=x.size()[2:], mode='bilinear', align_corners=False)
        out_4 = F.interpolate(
            out_4, size=x.size()[2:], mode='bilinear', align_corners=False)
        edge_out = F.interpolate(
            edge_atten_,
            size=x.size()[2:],
            mode='bilinear',
            align_corners=False)

        return out_4.sigmoid(), out_8.sigmoid(), out_16.sigmoid(), edge_out
