"""
Part of the implementation is borrowed and modified from Deeplab v3
publicly available at <https://arxiv.org/abs/1706.05587v3>
"""
import torch
from torch import nn


class ASP_OC_Module(nn.Module):

    def __init__(self, features, out_features=96, dilations=(2, 4, 8)):
        super(ASP_OC_Module, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                features,
                out_features,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False), nn.BatchNorm2d(out_features))
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                features,
                out_features,
                kernel_size=3,
                padding=dilations[0],
                dilation=dilations[0],
                bias=False), nn.BatchNorm2d(out_features))
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                features,
                out_features,
                kernel_size=3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False), nn.BatchNorm2d(out_features))
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                features,
                out_features,
                kernel_size=3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False), nn.BatchNorm2d(out_features))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(
                out_features * 4,
                out_features * 2,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False), nn.InstanceNorm2d(out_features * 2),
            nn.Dropout2d(0.05))

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert (len(feat1) == len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(
                torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]),
                          1))
        return z

    def forward(self, x):
        _, _, h, w = x.size()
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat2, feat3, feat4, feat5), 1)
        output = self.conv_bn_dropout(out)
        return output


class LRASPP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aspp = ASP_OC_Module(in_channels, out_channels)

    def forward_single_frame(self, x):
        return self.aspp(x)

    def forward_time_series(self, x):
        B, T = x.shape[:2]
        x = self.forward_single_frame(x.flatten(0, 1)).unflatten(0, (B, T))
        return x

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
