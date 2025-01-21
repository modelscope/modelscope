"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import os.path as osp

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        elif self.layer_type == "timepreserve":
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == "half":
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError("Got unexpected donwsampletype %s, expected is [none, timepreserve, half]" % self.layer_type)


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        elif self.layer_type == "timepreserve":
            return F.interpolate(x, scale_factor=(2, 1), mode="nearest")
        elif self.layer_type == "half":
            return F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            raise RuntimeError("Got unexpected upsampletype %s, expected is [none, timepreserve, half]" % self.layer_type)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, out_for_onnx=False, downsample="none"):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.learned_sc = dim_in != dim_out
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in)
            self.norm2 = nn.InstanceNorm2d(dim_in)
            if out_for_onnx:
                self.norm1.training = False
                self.norm2.training = False
            # self.norm1 = AdaIN(dim_in,dim_in)
            # self.norm2 = AdaIN(dim_in,dim_in)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features, out_for_onnx=False, device=None):
        super().__init__()

        self.norm = nn.InstanceNorm2d(num_features)
        if out_for_onnx:
            self.norm.training = False
        self.fc = nn.Linear(style_dim, num_features * 2)
        self.emb = torch.nn.Linear(192, style_dim)
        self.spk_emb = torch.nn.Parameter(torch.randn([1, 1000, style_dim]))

    def forward(self, x, s: torch.Tensor):
        s = self.emb(s)
        s = s.unsqueeze(1)
        score = torch.sum(s * self.spk_emb, dim=-1)
        score = torch.softmax(score, dim=-1).unsqueeze(-1)
        value = torch.sum(self.spk_emb * score, dim=1)

        h = self.fc(value)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        # print(x.shape)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0, actv=nn.LeakyReLU(0.2), upsample="none", out_for_onnx=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = UpSample(upsample)
        # self.norm=norm
        self.learned_sc = dim_in != dim_out
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in, out_for_onnx)
        self.norm2 = AdaIN(style_dim, dim_out, out_for_onnx)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1], [-1, 8.0, -1], [-1, -1, -1]]) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=48 * 8, out_for_onnx=False):
        super().__init__()
        self.out_for_onnx = out_for_onnx
        self.stem = nn.Conv2d(1, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_out = nn.Sequential(nn.InstanceNorm2d(dim_in, affine=True), nn.LeakyReLU(0.2), nn.Conv2d(dim_in, 1, 1, 1, 0))
        if out_for_onnx:
            for m in self.to_out.modules():
                if isinstance(m, torch.nn.InstanceNorm2d):
                    m.eval()
            # self.to_out.training=False

        # down/up-sampling blocks
        # self.spk_embedding=torch.nn.Embedding(num_spk,style_dim)
        repeat_num = 4  # int(np.log2(img_size)) - 4

        for lid in range(repeat_num):
            if lid in [1, 3]:
                _downtype = "timepreserve"
            else:
                _downtype = "half"

            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(ResBlk(dim_in, dim_out, normalize=True, downsample=_downtype, out_for_onnx=out_for_onnx))
            self.decode.insert(0, AdainResBlk(dim_out, dim_in, style_dim, w_hpf=1, upsample=_downtype, out_for_onnx=out_for_onnx))  # stack-like
            dim_in = dim_out

        # bottleneck blocks (encoder)
        for _ in range(2):
            self.encode.append(ResBlk(dim_out, dim_out, normalize=True, out_for_onnx=out_for_onnx))

        # bottleneck blocks (decoder)
        for _ in range(2):
            self.decode.insert(0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=1, out_for_onnx=out_for_onnx))

    def forward(self, x: torch.Tensor, c):

        x = self.stem(x)

        for block in self.encode:

            x = block(x)

        for block in self.decode:
            x = block(x, c)

        out = self.to_out(x)

        return out


class Generator2(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=48 * 8, num_spk=1883, w_hpf=1, F0_channel=0, out_for_onnx=False):
        super().__init__()
        self.out_for_onnx = out_for_onnx
        self.stem = nn.Conv2d(1, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_out = nn.Sequential(nn.InstanceNorm2d(dim_in, affine=True), nn.LeakyReLU(0.2), nn.Conv2d(dim_in, 1, 1, 1, 0))
        self.F0_channel = F0_channel
        # down/up-sampling blocks
        self.spk_embedding = torch.nn.Embedding(num_spk, style_dim)
        repeat_num = 4  # int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1

        for lid in range(repeat_num):
            if lid in [1, 3]:
                _downtype = "timepreserve"
            else:
                _downtype = "half"

            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(ResBlk(dim_in, dim_out, normalize=False, downsample=_downtype))
            self.decode.insert(0, AdainResBlk(dim_out, dim_in, style_dim, w_hpf=w_hpf, upsample=_downtype, norm=False))  # stack-like
            dim_in = dim_out

        # bottleneck blocks (encoder)
        for _ in range(2):
            self.encode.append(ResBlk(dim_out, dim_out, normalize=True))

        # F0 blocks

        # bottleneck blocks (decoder)
        for _ in range(2):
            self.decode.insert(0, AdainResBlk(dim_out + int(F0_channel / 2), dim_out + int(F0_channel / 2), style_dim, w_hpf=w_hpf, norm=False))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hpf = HighPass(w_hpf, device)

    def forward(self, x, c):

        if self.out_for_onnx:
            x = x.permute(0, 3, 1, 2)
        x = self.stem(x)
        for block in self.encode:
            x = block(x)
        s = self.spk_embedding(c)
        for block in self.decode:
            x = block(x, s)

        out = self.to_out(x)
        if self.out_for_onnx:
            out = out.squeeze(dim=1)

        return out


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=48, num_domains=2, hidden_dim=384):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, hidden_dim)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, style_dim),
                )
            ]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, num_domains=2, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample="half")]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)

        h = h.view(h.size(0), -1)
        out = []

        for layer in self.unshared:
            out += [layer(h)]

        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    def __init__(self, dim_in=48, num_domains=2, max_conv_dim=384, repeat_num=4):
        super().__init__()

        # real/fake discriminator
        self.dis = Discriminator2d(dim_in=dim_in, num_domains=num_domains, max_conv_dim=max_conv_dim, repeat_num=repeat_num)
        # adversarial classifier
        self.cls = Discriminator2d(dim_in=dim_in, num_domains=num_domains, max_conv_dim=max_conv_dim, repeat_num=repeat_num)
        self.num_domains = num_domains

    def forward(self, x, y):
        return self.dis(x, y)

    def classifier(self, x):
        return self.cls.get_feature(x)


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class Discriminator2d(nn.Module):
    def __init__(self, dim_in=48, num_domains=2, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        for lid in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample="half")]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out

    def forward(self, x, y):
        out = self.get_feature(x)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out


def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))


def build_model(args, F0_model, ASR_model):
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    discriminator = Discriminator(args.dim_in, args.num_domains, args.max_conv_dim, args.n_repeat)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)
    print(generator, "generator")
    print(mapping_network, "mapping_network")
    print(style_encoder, "style_encoder")
    nets = Munch(generator=generator, mapping_network=mapping_network, style_encoder=style_encoder, discriminator=discriminator, f0_model=F0_model, asr_model=ASR_model)

    nets_ema = Munch(generator=generator_ema, mapping_network=mapping_network_ema, style_encoder=style_encoder_ema)

    return nets, nets_ema


if __name__ == "__main__":
    generator = Generator(48, 48, 256, w_hpf=1, F0_channel=0)
    a = torch.randn([1, 1, 256 + 32, 80])
    c = torch.randint(0, 1883, [1])
    b = generator(a, c)
    print(b.shape)
