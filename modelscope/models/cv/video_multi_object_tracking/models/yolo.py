# The implementation is adopted from FairMOT,
# made publicly available under the MIT License at https://github.com/ifzhang/FairMOT
import math
from copy import deepcopy

import torch.nn as nn

from modelscope.models.base import TorchModel
from modelscope.utils.logger import get_logger
from .common import C3, SPP, Concat, Conv, Focus

logger = get_logger()

backbone_param = {
    'nc':
    80,
    'depth_multiple':
    0.33,
    'width_multiple':
    0.5,
    'backbone': [[-1, 1, 'Focus', [64, 3]], [-1, 1, 'Conv', [128, 3, 2]],
                 [-1, 3, 'C3', [128]], [-1, 1, 'Conv', [256, 3, 2]],
                 [-1, 9, 'C3', [256]], [-1, 1, 'Conv', [512, 3, 2]],
                 [-1, 9, 'C3', [512]], [-1, 1, 'Conv', [1024, 3, 2]],
                 [-1, 1, 'SPP', [1024, [5, 9, 13]]],
                 [-1, 3, 'C3', [1024, False]], [-1, 1, 'Conv', [512, 1, 1]],
                 [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
                 [[-1, 6], 1, 'Concat', [1]], [-1, 3, 'C3', [512, False]],
                 [-1, 1, 'Conv', [256, 1, 1]],
                 [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
                 [[-1, 4], 1, 'Concat', [1]], [-1, 3, 'C3', [256, False]],
                 [-1, 1, 'Conv', [128, 1, 1]],
                 [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
                 [[-1, 2], 1, 'Concat', [1]], [-1, 3, 'C3', [128, False]]]
}


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Model(nn.Module):

    def __init__(self, config=backbone_param, ch=3, nc=None, anchors=None):
        super(Model, self).__init__()
        self.yaml = config  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]
        self.inplace = self.yaml.get('inplace', True)

    def forward(self, x, augment=False, profile=False):
        return self.forward_once(x, profile)

    def forward_once(self, x, profile=False):
        y = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(
                    m.f, int) else [x if j == -1 else y[j] for j in m.f]

            x = m(x)  # run
            y.append(x if m.i in self.save else None)

        return x


def parse_model(d, ch):
    gd, gw = d['depth_multiple'], d['width_multiple']

    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except Exception:
                pass

        n = max(round(n * gd), 1) if n > 1 else n
        if m in [Conv, SPP, Focus, C3]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args)
                             for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        save.extend(x % i for x in ([f] if isinstance(f, int) else f)
                    if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class PoseYOLO(TorchModel):

    def __init__(self, heads):
        self.heads = heads
        super(PoseYOLO, self).__init__()
        self.backbone = Model()
        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                nn.SiLU(),
                nn.Conv2d(64, num_output, kernel_size=1, stride=1, padding=0))
            self.__setattr__(head, fc)
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)

    def forward(self, x):
        x = self.backbone(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]


def get_pose_net(num_layers, heads, head_conv):
    model = PoseYOLO(heads)
    return model


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor
