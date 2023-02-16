# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn

from .ops import SpectralNorm


class AADLayer(nn.Module):

    def __init__(self, c_x, attr_c, c_id=256):
        super(AADLayer, self).__init__()
        self.attr_c = attr_c
        self.c_id = c_id
        self.c_x = c_x

        ks = 3
        pw = ks // 2

        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(attr_c, nhidden, kernel_size=ks, padding=0), nn.ReLU())
        self.pad = nn.ReflectionPad2d(pw)
        self.conv1 = nn.Conv2d(
            nhidden, c_x, kernel_size=ks, stride=1, padding=0)
        self.conv2 = nn.Conv2d(
            nhidden, c_x, kernel_size=ks, stride=1, padding=0)
        self.fc1 = nn.Linear(c_id, c_x)
        self.fc2 = nn.Linear(c_id, c_x)

        self.norm = PositionalNorm2d

        self.pad_h = nn.ReflectionPad2d(pw)
        self.conv_h = nn.Conv2d(c_x, 1, kernel_size=ks, stride=1, padding=0)

    def forward(self, h_in, z_attr, z_id):

        h = self.norm(h_in)
        actv = self.mlp_shared(z_attr)
        gamma_attr = self.conv1(self.pad(actv))
        beta_attr = self.conv2(self.pad(actv))

        gamma_id = self.fc1(z_id)
        beta_id = self.fc2(z_id)
        A = gamma_attr * h + beta_attr
        gamma_id = gamma_id.reshape(h.shape[0], self.c_x, 1, 1).expand_as(h)
        beta_id = beta_id.reshape(h.shape[0], self.c_x, 1, 1).expand_as(h)
        B = gamma_id * h + beta_id

        M = torch.sigmoid(self.conv_h(self.pad_h(h)))

        out = (torch.ones_like(M).to(M.device) - M) * A + M * B

        return out


def PositionalNorm2d(x, epsilon=1e-5):
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output


class AAD_ResBlk(nn.Module):

    def __init__(self, cin, cout, c_attr, c_id=256):
        super(AAD_ResBlk, self).__init__()
        self.cin = cin
        self.cout = cout
        self.learned_shortcut = (self.cin != self.cout)
        fmiddle = min(self.cin, self.cout)

        self.AAD1 = AADLayer(cin, c_attr, c_id)
        self.AAD2 = AADLayer(fmiddle, c_attr, c_id)
        self.pad = nn.ReflectionPad2d(1)
        self.conv1 = SpectralNorm(
            nn.Conv2d(cin, fmiddle, kernel_size=3, stride=1, padding=0))
        self.conv2 = SpectralNorm(
            nn.Conv2d(fmiddle, cout, kernel_size=3, stride=1, padding=0))

        self.relu1 = nn.LeakyReLU(2e-1)
        self.relu2 = nn.LeakyReLU(2e-1)

        if self.learned_shortcut:
            self.AAD3 = AADLayer(cin, c_attr, c_id)
            self.conv3 = SpectralNorm(
                nn.Conv2d(cin, cout, kernel_size=1, bias=False))

    def forward(self, h, z_attr, z_id):
        x = self.conv1(self.pad(self.relu1(self.AAD1(h, z_attr, z_id))))
        x = self.conv2(self.pad(self.relu2(self.AAD2(x, z_attr, z_id))))

        if self.learned_shortcut:
            h = self.conv3(self.AAD3(h, z_attr, z_id))

        x = x + h

        return x
