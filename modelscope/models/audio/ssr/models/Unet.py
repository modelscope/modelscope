"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSample(nn.Module):

    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            return F.avg_pool2d(x, 2)
        else:
            raise


class UpSample(nn.Module):

    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise f'unknown upsample type: {self.layer_type}'


class ResBlk(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 actv=nn.LeakyReLU(0.2),
                 normalize=False,
                 style_dim=256,
                 downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.learned_sc = dim_in != dim_out
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            # self.norm1=nn.InstanceNorm2d(dim_in)
            # self.norm2=nn.InstanceNorm2d(dim_in)

            self.norm1 = AdaIN(style_dim, dim_in)
            self.norm2 = AdaIN(style_dim, dim_in)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x, s=None):
        if self.normalize:
            x = self.norm1(x, s)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample(x)
        if self.normalize:
            x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s=None):
        x = self._shortcut(x) + self._residual(x, s)
        return x / math.sqrt(2)  # unit variance


class ResBlk1D(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 actv=nn.LeakyReLU(0.2),
                 normalize=False,
                 out_for_onnx=False,
                 downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.learned_sc = dim_in != dim_out
        self.conv1 = nn.Conv1d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv1d(dim_in, dim_out, 3, 1, 1)

        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in)
            self.norm2 = nn.InstanceNorm1d(dim_in)

        if self.learned_sc:
            self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)

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

    def __init__(self, style_dim, num_features):
        super().__init__()

        self.norm = nn.InstanceNorm2d(num_features)

        self.fc = nn.Linear(style_dim, num_features * 2)
        # self.emb=torch.nn.Linear(num_features,style_dim)
        self.spk_emb = torch.nn.Parameter(torch.randn([1, 1000, style_dim]))
        self.mha = torch.nn.MultiheadAttention(
            style_dim, 4, bias=False, batch_first=True)

    def forward(self, x, s: torch.Tensor):

        s = s.unsqueeze(1)
        B = s.size(0)
        key = self.spk_emb.repeat(B, 1, 1)
        value, _ = self.mha(s, key, key)

        h = self.fc(value).squeeze(dim=1)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)

        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 style_dim=256,
                 w_hpf=0,
                 actv=nn.LeakyReLU(0.2),
                 upsample='none'):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = UpSample(upsample)
        # self.norm=norm
        self.learned_sc = dim_in != dim_out
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
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
        self.filter = torch.tensor([[-1, -1, -1], [-1, 8., -1], [-1, -1, -1]
                                    ]) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(
            x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class UnetMapping(nn.Module):

    def __init__(self,
                 dim_in=48,
                 style_dim=48,
                 max_conv_dim=48 * 8,
                 repeat_num=4):
        super().__init__()
        self.stem = nn.Conv2d(1, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_out = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True), nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 1, 1, 1, 0))

        for lid in range(repeat_num):
            if lid in [1, 3]:
                _downtype = 'timepreserve'
            else:
                _downtype = 'half'

            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResBlk(
                    dim_in,
                    dim_out,
                    style_dim=style_dim,
                    normalize=True,
                    downsample=_downtype))
            self.decode.insert(0,
                               AdainResBlk(
                                   dim_out,
                                   dim_in,
                                   style_dim,
                                   w_hpf=0,
                                   upsample=_downtype))  # stack-like
            dim_in = dim_out

        # bottleneck blocks (encoder)
        for _ in range(repeat_num):
            self.encode.append(
                ResBlk(dim_out, dim_out, style_dim=style_dim, normalize=True))

        # bottleneck blocks (decoder)
        for _ in range(repeat_num):
            self.decode.insert(0, AdainResBlk(dim_out, dim_out, style_dim))
        # self.proj = nn.Conv1d(80, 80 * 2, 1)
        self.style_extractor = StyleEncoder(dim_in, style_dim, num_domains=8)
        self.flow = FlowBlocks(256, style_dim, 5, 1, 4)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        s = self.style_extractor(c)
        x = self.stem(x)

        for block in self.encode:

            x = block(x, s)

        for block in self.decode:
            x = block(x, s)

        out = self.to_out(x).squeeze(dim=1)
        out = self.flow(out, reverse=True)

        return out


class MaskMapping(nn.Module):

    def __init__(self,
                 dim_in=48,
                 style_dim=48,
                 max_conv_dim=48 * 8,
                 repeat_num=4):
        super().__init__()
        self.stem = nn.Conv2d(1, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_out = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True), nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 1, 1, 1, 0))

        for lid in range(repeat_num):
            if lid in [1, 3]:
                _downtype = 'timepreserve'
            else:
                _downtype = 'half'

            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResBlk(
                    dim_in,
                    dim_out,
                    style_dim=style_dim,
                    normalize=True,
                    downsample=_downtype))
            self.decode.insert(0,
                               AdainResBlk(
                                   dim_out,
                                   dim_in,
                                   style_dim,
                                   w_hpf=0,
                                   upsample=_downtype))  # stack-like
            dim_in = dim_out

        # bottleneck blocks (encoder)
        for _ in range(repeat_num):
            self.encode.append(
                ResBlk(dim_out, dim_out, style_dim=style_dim, normalize=True))

        # bottleneck blocks (decoder)
        for _ in range(repeat_num):
            self.decode.insert(0, AdainResBlk(dim_out, dim_out, style_dim))
        # self.proj = nn.Conv1d(80, 80 * 2, 1)
        self.style_extractor = StyleEncoder(dim_in, style_dim, num_domains=8)
        self.flow = FlowBlocks(256, style_dim, 5, 1, 4)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        s = self.style_extractor(c)
        t = c.size(-1)
        x = torch.cat((c.unsqueeze(1), x), dim=-1)
        x = self.stem(x)

        for block in self.encode:

            x = block(x, s)

        for block in self.decode:
            x = block(x, s)

        out = self.to_out(x).squeeze(dim=1)
        out = self.flow(out, reverse=True)
        out = out[:, :, t:]
        return out


class StyleEncoder(nn.Module):

    def __init__(self,
                 dim_in=48,
                 style_dim=48,
                 num_domains=4,
                 max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [nn.Conv1d(256, dim_in, 3, 1, 1)]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk1D(dim_in, dim_out, downsample='none')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv1d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.AdaptiveAvgPool1d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim // num_domains)]

    def forward(self, x):
        h = self.shared(x)

        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.cat(out, dim=-1)  # (batch, num_domains, style_dim)
        return out


class ResidualCouplingLayer(nn.Module):

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, 'channels should be divisible by 2'
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels,
                              self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.enc(h)
        stats = self.post(h)
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
            # print(m)
            # print(logs)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs)
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs)
            x = torch.cat([x0, x1], 1)
            return x


def fused_add_tanh_sigmoid_multiply(input_a, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WN(nn.Module):

    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size, )
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        cond_layer = nn.Conv1d(hidden_channels, 2 * hidden_channels * n_layers,
                               1)
        self.cond_layer = cond_layer

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )

            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)

            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)

            acts = fused_add_tanh_sigmoid_multiply(x_in, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts)
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output


class Discriminator(nn.Module):

    def __init__(self,
                 dim_in=48,
                 num_domains=2,
                 max_conv_dim=384,
                 repeat_num=4):
        super().__init__()

        # real/fake discriminator
        self.dis = Discriminator2d(
            dim_in=dim_in,
            num_domains=num_domains,
            max_conv_dim=max_conv_dim,
            repeat_num=repeat_num)
        # adversarial classifier
        self.cls = Discriminator2d(
            dim_in=dim_in,
            num_domains=num_domains,
            max_conv_dim=max_conv_dim,
            repeat_num=repeat_num)
        self.num_domains = num_domains

    def forward(self, x, y):
        return self.dis(x, y)

    def classifier(self, x):
        return self.cls.get_feature(x)


class LinearNorm(torch.nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class Discriminator2d(nn.Module):

    def __init__(self,
                 dim_in=48,
                 num_domains=2,
                 max_conv_dim=384,
                 repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        for lid in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
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

    def forward(self, x):
        out = self.get_feature(x)

        return out


class FlowBlocks(nn.Module):

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=False,
                ))
            self.flows.append(Flip())

    def forward(self, x, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, log = flow(x, reverse=reverse)
            return x, log
        else:
            for flow in reversed(self.flows):
                x = flow(x, reverse=reverse)
            return x


class Flip(nn.Module):

    def forward(self, x, *args, reverse=False, **kwargs):

        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


def print_network(model):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print('The number of parameters: {}'.format(num_params))


if __name__ == '__main__':
    generator = UnetMapping(48, 256)
    a = torch.randn([1, 1, 256, 224])
    c = torch.randn([1, 256, 1000])
    b = generator(a, c)

    print(b.shape)

    print_network(generator)
