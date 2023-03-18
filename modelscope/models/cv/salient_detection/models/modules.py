# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ConvBNReLU


class AreaLayer(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(AreaLayer, self).__init__()
        self.lbody = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.hbody = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1), nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))
        self.body = nn.Sequential(
            nn.Conv2d(2 * out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, 1, 1))

    def forward(self, xl, xh):
        xl1 = self.lbody(xl)
        xl1 = F.interpolate(
            xl1, size=xh.size()[2:], mode='bilinear', align_corners=True)
        xh1 = self.hbody(xh)
        x = torch.cat((xl1, xh1), dim=1)
        x_out = self.body(x)
        return x_out


class EdgeLayer(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(EdgeLayer, self).__init__()
        self.lbody = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.hbody = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1), nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))
        self.bodye = nn.Sequential(
            nn.Conv2d(2 * out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, 1, 1))

    def forward(self, xl, xh):
        xl1 = self.lbody(xl)
        xh1 = self.hbody(xh)
        xh1 = F.interpolate(
            xh1, size=xl.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((xl1, xh1), dim=1)
        x_out = self.bodye(x)
        return x_out


class EBlock(nn.Module):

    def __init__(self, inchs, outchs):
        super(EBlock, self).__init__()
        self.elayer = nn.Sequential(
            ConvBNReLU(inchs + 1, outchs, kernel_size=3, padding=1, stride=1),
            ConvBNReLU(outchs, outchs, 1))
        self.salayer = nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1, momentum=0.01), nn.Sigmoid())

    def forward(self, x, edgeAtten):
        x = torch.cat((x, edgeAtten), dim=1)
        ex = self.elayer(x)
        ex_max = torch.max(ex, 1, keepdim=True)[0]
        ex_mean = torch.mean(ex, dim=1, keepdim=True)
        xei_compress = torch.cat((ex_max, ex_mean), dim=1)

        scale = self.salayer(xei_compress)
        x_out = ex * scale
        return x_out


class StructureE(nn.Module):

    def __init__(self, inchs, outchs, EM):
        super(StructureE, self).__init__()
        self.ne_modules = int(inchs / EM)
        NM = int(outchs / self.ne_modules)
        elayes = []
        for i in range(self.ne_modules):
            emblock = EBlock(EM, NM)
            elayes.append(emblock)
        self.emlayes = nn.ModuleList(elayes)
        self.body = nn.Sequential(
            ConvBNReLU(outchs, outchs, 3, 1, 1), ConvBNReLU(outchs, outchs, 1))

    def forward(self, x, edgeAtten):
        if edgeAtten.size() != x.size():
            edgeAtten = F.interpolate(
                edgeAtten, x.size()[2:], mode='bilinear', align_corners=False)
        xx = torch.chunk(x, self.ne_modules, dim=1)
        efeas = []
        for i in range(self.ne_modules):
            xei = self.emlayes[i](xx[i], edgeAtten)
            efeas.append(xei)
        efeas = torch.cat(efeas, dim=1)
        x_out = self.body(efeas)
        return x_out


class ABlock(nn.Module):

    def __init__(self, inchs, outchs, k):
        super(ABlock, self).__init__()
        self.alayer = nn.Sequential(
            ConvBNReLU(inchs, outchs, k, 1, k // 2),
            ConvBNReLU(outchs, outchs, 1))
        self.arlayer = nn.Sequential(
            ConvBNReLU(inchs, outchs, k, 1, k // 2),
            ConvBNReLU(outchs, outchs, 1))
        self.fusion = ConvBNReLU(2 * outchs, outchs, 1)

    def forward(self, x, areaAtten):
        xa = x * areaAtten
        xra = x * (1 - areaAtten)
        xout = self.fusion(torch.cat((xa, xra), dim=1))
        return xout


class AMFusion(nn.Module):

    def __init__(self, inchs, outchs, AM):
        super(AMFusion, self).__init__()
        self.k = [3, 3, 5, 5]
        self.conv_up = ConvBNReLU(inchs, outchs, 3, 1, 1)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.na_modules = int(outchs / AM)
        alayers = []
        for i in range(self.na_modules):
            layer = ABlock(AM, AM, self.k[i])
            alayers.append(layer)
        self.alayers = nn.ModuleList(alayers)
        self.fusion_0 = ConvBNReLU(outchs, outchs, 3, 1, 1)
        self.fusion_e = nn.Sequential(
            nn.Conv2d(
                outchs, outchs, kernel_size=(3, 1), padding=(1, 0),
                bias=False), nn.BatchNorm2d(outchs), nn.ReLU(inplace=True),
            nn.Conv2d(
                outchs, outchs, kernel_size=(1, 3), padding=(0, 1),
                bias=False), nn.BatchNorm2d(outchs), nn.ReLU(inplace=True))
        self.fusion_e1 = nn.Sequential(
            nn.Conv2d(
                outchs, outchs, kernel_size=(5, 1), padding=(2, 0),
                bias=False), nn.BatchNorm2d(outchs), nn.ReLU(inplace=True),
            nn.Conv2d(
                outchs, outchs, kernel_size=(1, 5), padding=(0, 2),
                bias=False), nn.BatchNorm2d(outchs), nn.ReLU(inplace=True))
        self.fusion = ConvBNReLU(3 * outchs, outchs, 1)

    def forward(self, xl, xh, xhm):
        xh1 = self.up(self.conv_up(xh))
        x = xh1 + xl
        xm = self.up(torch.sigmoid(xhm))
        xx = torch.chunk(x, self.na_modules, dim=1)
        xxmids = []
        for i in range(self.na_modules):
            xi = self.alayers[i](xx[i], xm)
            xxmids.append(xi)
        xfea = torch.cat(xxmids, dim=1)
        x0 = self.fusion_0(xfea)
        x1 = self.fusion_e(xfea)
        x2 = self.fusion_e1(xfea)
        x_out = self.fusion(torch.cat((x0, x1, x2), dim=1))
        return x_out
