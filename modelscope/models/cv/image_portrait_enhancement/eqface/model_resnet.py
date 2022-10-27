# The implementation is adopted from FaceQuality, made publicly available under the MIT License
# at https://github.com/deepcam-cn/FaceQuality/blob/master/models/model_resnet.py
import torch
from torch import nn


class BottleNeck_IR(nn.Module):

    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR, self).__init__()
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(out_channel), nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(out_channel))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=(1, 1),
                    stride=stride,
                    bias=False), nn.BatchNorm2d(out_channel))

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res


channel_list = [64, 64, 128, 256, 512]


def get_layers(num_layers):
    if num_layers == 34:
        return [3, 4, 6, 3]
    if num_layers == 50:
        return [3, 4, 14, 3]
    elif num_layers == 100:
        return [3, 13, 30, 3]
    elif num_layers == 152:
        return [3, 8, 36, 3]


class ResNet(nn.Module):

    def __init__(self,
                 num_layers=100,
                 feature_dim=512,
                 drop_ratio=0.4,
                 channel_list=channel_list):
        super(ResNet, self).__init__()
        assert num_layers in [34, 50, 100, 152]
        layers = get_layers(num_layers)
        block = BottleNeck_IR

        self.input_layer = nn.Sequential(
            nn.Conv2d(
                3, channel_list[0], (3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel_list[0]), nn.PReLU(channel_list[0]))
        self.layer1 = self._make_layer(
            block, channel_list[0], channel_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(
            block, channel_list[1], channel_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, channel_list[2], channel_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, channel_list[3], channel_list[4], layers[3], stride=2)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512), nn.Dropout(drop_ratio), nn.Flatten())
        self.feature_layer = nn.Sequential(
            nn.Linear(512 * 7 * 7, feature_dim), nn.BatchNorm1d(feature_dim))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(
                    m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_channel, out_channel, blocks, stride):
        layers = []
        layers.append(block(in_channel, out_channel, stride, False))
        for i in range(1, blocks):
            layers.append(block(out_channel, out_channel, 1, True))
        return nn.Sequential(*layers)

    def forward(self, x, fc=False):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)
        feature = self.feature_layer(x)
        if fc:
            return feature, x
        return feature


class FaceQuality(nn.Module):

    def __init__(self, feature_dim):
        super(FaceQuality, self).__init__()
        self.qualtiy = nn.Sequential(
            nn.Linear(feature_dim, 512, bias=False), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Linear(512, 2, bias=False),
            nn.Softmax(dim=1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(
                    m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.qualtiy(x)
        return x[:, 0:1]
