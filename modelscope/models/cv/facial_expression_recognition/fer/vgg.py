# The implementation is based on Facial-Expression-Recognition, available at
# https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modelscope.utils.config import Config


class VGG(nn.Module):

    def __init__(self, vgg_name, cfg_path):
        super(VGG, self).__init__()
        model_cfg = Config.from_file(cfg_path)['models']
        self.features = self._make_layers(model_cfg[vgg_name])
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
