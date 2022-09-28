# The implementation is adopted from https://github.com/facebookresearch/ConvNeXt,
# made pubicly available under the MIT License at https://github.com/facebookresearch/ConvNeXt
# Copyright 2021-2022 The Alibaba FVI Team Authors. All rights reserved.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py.
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py.
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class TadaConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self, cfg
        #  in_chans=3, num_classes=1000,
        #  depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
        #  layer_scale_init_value=1e-6, head_init_scale=1.,
    ):
        super().__init__()
        in_chans = cfg.VIDEO.BACKBONE.NUM_INPUT_CHANNELS
        dims = cfg.VIDEO.BACKBONE.NUM_FILTERS
        drop_path_rate = cfg.VIDEO.BACKBONE.DROP_PATH
        depths = cfg.VIDEO.BACKBONE.DEPTH
        layer_scale_init_value = cfg.VIDEO.BACKBONE.LARGE_SCALE_INIT_VALUE
        stem_t_kernel_size = cfg.VIDEO.BACKBONE.STEM.T_KERNEL_SIZE if hasattr(
            cfg.VIDEO.BACKBONE.STEM, 'T_KERNEL_SIZE') else 2
        t_stride = cfg.VIDEO.BACKBONE.STEM.T_STRIDE if hasattr(
            cfg.VIDEO.BACKBONE.STEM, 'T_STRIDE') else 2

        self.downsample_layers = nn.ModuleList(
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(
                in_chans,
                dims[0],
                kernel_size=(stem_t_kernel_size, 4, 4),
                stride=(t_stride, 4, 4),
                padding=((stem_t_kernel_size - 1) // 2, 0, 0)),
            LayerNorm(dims[0], eps=1e-6, data_format='channels_first'))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv3d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2)),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList(
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                TAdaConvNeXtBlock(
                    cfg,
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value)
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean(
            [-3, -2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        if isinstance(x, dict):
            x = x['video']
        x = self.forward_features(x)
        return x

    def get_num_layers(self):
        return 12, 0


class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, cfg, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(
            dim, dim, kernel_size=(1, 7, 7), padding=(0, 3, 3),
            groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim,
            4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, T, H, W) -> (N, T, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, T, H, W, C) -> (N, C, T, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None,
                                                                 None]
            return x


class TAdaConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_fi rst) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, cfg, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        layer_scale_init_value = float(layer_scale_init_value)
        self.dwconv = TAdaConv2d(
            dim,
            dim,
            kernel_size=(1, 7, 7),
            padding=(0, 3, 3),
            groups=dim,
            cal_dim='cout')
        route_func_type = cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_TYPE
        if route_func_type == 'normal':
            self.dwconv_rf = RouteFuncMLP(
                c_in=dim,
                ratio=cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_R,
                kernels=cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_K,
                with_bias_cal=self.dwconv.bias is not None)
        elif route_func_type == 'normal_lngelu':
            self.dwconv_rf = RouteFuncMLPLnGelu(
                c_in=dim,
                ratio=cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_R,
                kernels=cfg.VIDEO.BACKBONE.BRANCH.ROUTE_FUNC_K,
                with_bias_cal=self.dwconv.bias is not None)
        else:
            raise ValueError(
                'Unknown route_func_type: {}'.format(route_func_type))
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim,
            4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x, self.dwconv_rf(x))
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, T, H, W) -> (N, T, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, T, H, W, C) -> (N, C, T, H, W)

        x = input + self.drop_path(x)
        return x


class RouteFuncMLPLnGelu(nn.Module):
    """
    The routing function for generating the calibration weights.
    """

    def __init__(self,
                 c_in,
                 ratio,
                 kernels,
                 with_bias_cal=False,
                 bn_eps=1e-5,
                 bn_mmt=0.1):
        """
        Args:
            c_in (int): number of input channels.
            ratio (int): reduction ratio for the routing function.
            kernels (list): temporal kernel size of the stacked 1D convolutions
        """
        super(RouteFuncMLPLnGelu, self).__init__()
        self.c_in = c_in
        self.with_bias_cal = with_bias_cal
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in // ratio),
            kernel_size=[kernels[0], 1, 1],
            padding=[kernels[0] // 2, 0, 0],
        )
        # self.bn = nn.BatchNorm3d(int(c_in//ratio), eps=bn_eps, momentum=bn_mmt)
        self.ln = LayerNorm(
            int(c_in // ratio), eps=1e-6, data_format='channels_first')
        self.gelu = nn.GELU()
        # self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in // ratio),
            out_channels=c_in,
            kernel_size=[kernels[1], 1, 1],
            padding=[kernels[1] // 2, 0, 0],
            bias=False)
        self.b.skip_init = True
        self.b.weight.data.zero_()  # to make sure the initial values
        # for the output is 1.
        if with_bias_cal:
            self.b_bias = nn.Conv3d(
                in_channels=int(c_in // ratio),
                out_channels=c_in,
                kernel_size=[kernels[1], 1, 1],
                padding=[kernels[1] // 2, 0, 0],
                bias=False)
            self.b_bias.skip_init = True
            self.b_bias.weight.data.zero_()  # to make sure the initial values
            # for the output is 1.

    def forward(self, x):
        g = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(g))
        # x = self.bn(x)
        # x = self.relu(x)
        x = self.ln(x)
        x = self.gelu(x)
        if self.with_bias_cal:
            return [self.b(x) + 1, self.b_bias(x) + 1]
        else:
            return self.b(x) + 1


class TAdaConv2d(nn.Module):
    """
    Performs temporally adaptive 2D convolution.
    Currently, only application on 5D tensors is supported, which makes TAdaConv2d
        essentially a 3D convolution with temporal kernel size of 1.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 cal_dim='cin'):
        super(TAdaConv2d, self).__init__()
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (list): kernel size of TAdaConv2d.
            stride (list): stride for the convolution in TAdaConv2d.
             padding (list): padding for the convolution in TAdaConv2d.
            dilation (list): dilation of the convolution in TAdaConv2d.
            groups (int): number of groups for TAdaConv2d.
            bias (bool): whether to use bias in TAdaConv2d.
            calibration_mode (str): calibrated dimension in TAdaConv2d.
                Supported input "cin", "cout".
        """

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1
        assert cal_dim in ['cin', 'cout']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cal_dim = cal_dim

        # base weights (W_b)
        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups,
                         kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, alpha):
        """
        Args:
            x (tensor): feature to perform convolution on.
            alpha (tensor): calibration weight for the base weights.
                W_t = alpha_t * W_b
        """
        if isinstance(alpha, list):
            w_alpha, b_alpha = alpha[0], alpha[1]
        else:
            w_alpha = alpha
            b_alpha = None
        _, _, c_out, c_in, kh, kw = self.weight.size()
        b, c_in, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(1, -1, h, w)

        if self.cal_dim == 'cin':
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, 1, C, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (w_alpha.permute(0, 2, 1, 3, 4).unsqueeze(2)
                      * self.weight).reshape(-1, c_in // self.groups, kh, kw)
        elif self.cal_dim == 'cout':
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, C, 1, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (w_alpha.permute(0, 2, 1, 3, 4).unsqueeze(3)
                      * self.weight).reshape(-1, c_in // self.groups, kh, kw)

        bias = None
        if self.bias is not None:
            if b_alpha is not None:
                # b_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, C
                bias = (b_alpha.permute(0, 2, 1, 3, 4).squeeze()
                        * self.bias).reshape(-1)
            else:
                bias = self.bias.repeat(b, t, 1).reshape(-1)
        output = F.conv2d(
            x,
            weight=weight,
            bias=bias,
            stride=self.stride[1:],
            padding=self.padding[1:],
            dilation=self.dilation[1:],
            groups=self.groups * b * t)

        output = output.view(b, t, c_out, output.size(-2),
                             output.size(-1)).permute(0, 2, 1, 3, 4)

        return output

    def __repr__(self):
        return f'TAdaConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, ' +\
            f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, cal_dim=\"{self.cal_dim}\")"
