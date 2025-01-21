# The implementation is adopted from ASPP made publicly available under the MIT License License
# at https://github.com/jfzhang95/pytorch-deeplab-xception
import torch
import torch.nn.functional as F
from torch import nn

BatchNorm2d = nn.BatchNorm2d


class ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# this aspp is official version
# copy from :https://github.com/jfzhang95/pytorch-deeplab-xception
class ASPP(nn.Module):

    def __init__(self, inplanes, outplanes, dilations, drop_rate=0.1):
        super(ASPP, self).__init__()

        self.aspp1 = ASPPModule(
            inplanes,
            outplanes,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm2d)
        self.aspp2 = ASPPModule(
            inplanes,
            outplanes,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm2d)
        self.aspp3 = ASPPModule(
            inplanes,
            outplanes,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm2d)
        self.aspp4 = ASPPModule(
            inplanes,
            outplanes,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm2d)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, outplanes, 1, stride=1, bias=False),
            BatchNorm2d(outplanes), nn.ReLU())
        self.conv1 = nn.Conv2d(outplanes * 5, outplanes, 1, bias=False)
        self.bn1 = BatchNorm2d(outplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)
        self._init_weight()

    def forward(self, x):  # [1, 256, 320, 320]
        x1 = self.aspp1(x)  # [1, 128, 160, 160]
        x2 = self.aspp2(x)  # [1, 128, 160, 160]
        x3 = self.aspp3(x)  # [1, 128, 160, 160]
        x4 = self.aspp4(x)  # [1, 128, 160, 160]
        x5 = self.global_avg_pool(x)  # b,c,h,w [1, 128, 1, 1]
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear',
            align_corners=True)  # [1, 128, 160, 160]
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # [1, 640, 160, 160]

        x = self.conv1(x)  # [1, 640, 160, 160]
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
