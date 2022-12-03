"""
Part of the implementation is borrowed and modified from EfficientNetV2
publicly available at <https://arxiv.org/abs/2104.00298>
"""

import torch
import torch.nn.functional


class SiLU(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1710.05941.pdf]
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.silu = torch.nn.SiLU(inplace=inplace)

    def forward(self, x):
        return self.silu(x)


class Conv(torch.nn.Module):

    def __init__(self, in_ch, out_ch, activation, k=1, s=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_ch, out_ch, k, s, k // 2, 1, g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.01)
        self.silu = activation

    def forward(self, x):
        return self.silu(self.norm(self.conv(x)))


class SE(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1709.01507.pdf]
    """

    def __init__(self, ch, r):
        super().__init__()
        self.se = torch.nn.Sequential(
            torch.nn.Conv2d(ch, ch // (4 * r), 1), torch.nn.SiLU(),
            torch.nn.Conv2d(ch // (4 * r), ch, 1), torch.nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x.mean((2, 3), keepdim=True))


class Residual(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1801.04381.pdf]
    """

    def __init__(self, in_ch, out_ch, s, r, fused=True):
        super().__init__()
        identity = torch.nn.Identity()
        if fused:
            if r == 1:
                features = [Conv(in_ch, r * in_ch, torch.nn.SiLU(), 3, s)]
            else:
                features = [
                    Conv(in_ch, r * in_ch, torch.nn.SiLU(), 3, s),
                    Conv(r * in_ch, out_ch, identity)
                ]
        else:
            if r == 1:
                features = [
                    Conv(r * in_ch, r * in_ch, torch.nn.SiLU(), 3, s,
                         r * in_ch),
                    SE(r * in_ch, r),
                    Conv(r * in_ch, out_ch, identity)
                ]
            else:
                features = [
                    Conv(in_ch, r * in_ch, torch.nn.SiLU()),
                    Conv(r * in_ch, r * in_ch, torch.nn.SiLU(), 3, s,
                         r * in_ch),
                    SE(r * in_ch, r),
                    Conv(r * in_ch, out_ch, identity)
                ]
        self.add = s == 1 and in_ch == out_ch
        self.res = torch.nn.Sequential(*features)

    def forward(self, x):
        return x + self.res(x) if self.add else self.res(x)


class EfficientNet(torch.nn.Module):

    def __init__(self, pretrained: bool = False):
        super().__init__()
        gate_fn = [True, False]
        filters = [24, 48, 64, 128, 160, 256]
        feature = [Conv(3, filters[0], torch.nn.SiLU(), 3, 2)]
        for i in range(2):
            if i == 0:
                feature.append(
                    Residual(filters[0], filters[0], 1, 1, gate_fn[0]))
            else:
                feature.append(
                    Residual(filters[0], filters[0], 1, 1, gate_fn[0]))

        for i in range(4):
            if i == 0:
                feature.append(
                    Residual(filters[0], filters[1], 2, 4, gate_fn[0]))
            else:
                feature.append(
                    Residual(filters[1], filters[1], 1, 4, gate_fn[0]))

        for i in range(4):
            if i == 0:
                feature.append(
                    Residual(filters[1], filters[2], 2, 4, gate_fn[0]))
            else:
                feature.append(
                    Residual(filters[2], filters[2], 1, 4, gate_fn[0]))

        for i in range(6):
            if i == 0:
                feature.append(
                    Residual(filters[2], filters[3], 2, 4, gate_fn[1]))
            else:
                feature.append(
                    Residual(filters[3], filters[3], 1, 4, gate_fn[1]))

        for i in range(9):
            if i == 0:
                feature.append(
                    Residual(filters[3], filters[4], 1, 6, gate_fn[1]))
            else:
                feature.append(
                    Residual(filters[4], filters[4], 1, 6, gate_fn[1]))

        self.feature = torch.nn.Sequential(*feature)

    def forward_single_frame(self, x):
        x = self.feature[0](x)
        x = self.feature[1](x)
        x = self.feature[2](x)
        f1 = x  # 1/2 24
        for i in range(4):
            x = self.feature[i + 3](x)
        f2 = x  # 1/4 48
        for i in range(4):
            x = self.feature[i + 7](x)
        f3 = x  # 1/8 64
        for i in range(6):
            x = self.feature[i + 11](x)
        for i in range(9):
            x = self.feature[i + 17](x)
        f5 = x  # 1/16 160
        return [f1, f2, f3, f5]

    def forward_time_series(self, x):
        B, T = x.shape[:2]
        features = self.forward_single_frame(x.flatten(0, 1))
        features = [f.unflatten(0, (B, T)) for f in features]
        return features

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)

    def export(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'silu'):
                if isinstance(m.silu, torch.nn.SiLU):
                    m.silu = SiLU()
            if type(m) is SE:
                if isinstance(m.se[1], torch.nn.SiLU):
                    m.se[1] = SiLU()
        return self
