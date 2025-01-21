# The implementation here is modified based on timm,
# originally Apache 2.0 License and publicly available at
# https://github.com/naver-ai/vidt/blob/vidt-plus/methods/vidt/fpn_fusion.py

import torch.nn as nn


class FPNFusionModule(nn.Module):
    """ This is a fpn-style cross-scale feature fusion module" """

    def __init__(self, embed_dims, fuse_dim=256, n_block=4, use_bn=False):
        super().__init__()
        """ Initializes the model.
        Args:
            embed_dims: the list of channel dim for different scale feature maps (i.e., the input)
            fuse_dim: the channel dim of the fused feature map (i.e., the output)
            n_block: the number of multi-scale features (default=4)
            use_bn: whether to use bn
        """

        self.embed_dims = embed_dims
        self.fuse_dim = fuse_dim
        self.n_block = n_block

        # cross-scale fusion layers
        self.multi_scaler = _make_multi_scale_layers(
            embed_dims, fuse_dim, use_bn=use_bn, n_block=n_block)

    def forward(self, x_blocks):

        x_blocks = x_blocks

        # preparation: channel reduction and normalization
        for idx in range(self.n_block - 1, -1, -1):
            x_blocks[idx] = getattr(self.multi_scaler, f'layer_{idx}_rn')(
                x_blocks[idx])
            x_blocks[idx] = getattr(self.multi_scaler, f'p_norm_{idx}')(
                x_blocks[idx])

        # cross-scale fusion
        refined_embeds = []
        for idx in range(self.n_block - 1, -1, -1):
            if idx == self.n_block - 1:
                path = getattr(self.multi_scaler,
                               f'refinenet_{idx}')([x_blocks[idx]], None)
            else:
                path = getattr(self.multi_scaler,
                               f'refinenet_{idx}')([path, x_blocks[idx]],
                                                   x_blocks[idx].size()[2:])
            refined_embeds.append(path)

        return refined_embeds


def _make_multi_scale_layers(in_shape,
                             out_shape,
                             n_block=4,
                             groups=1,
                             use_bn=False):

    out_shapes = [out_shape for _ in range(n_block)]
    multi_scaler = nn.Module()

    for idx in range(n_block - 1, -1, -1):
        """
          1 x 1 conv for dim reduction -> group norm
        """
        layer_name = f'layer_{(idx)}_rn'
        multi_scaler.add_module(
            layer_name,
            nn.Conv2d(in_shape[idx], out_shapes[idx], kernel_size=1))

        layer_name = f'p_norm_{(idx)}'
        multi_scaler.add_module(layer_name, nn.GroupNorm(32, out_shapes[idx]))

        layer_name = f'refinenet_{idx}'
        multi_scaler.add_module(layer_name,
                                _make_fusion_block(out_shape, use_bn))

        # initialize for the 1x1 conv
        nn.init.xavier_uniform_(
            getattr(multi_scaler, f'layer_{idx}_rn').weight, gain=1)
        nn.init.constant_(getattr(multi_scaler, f'layer_{idx}_rn').bias, 0)

    return multi_scaler


def _make_fusion_block(features, use_bn):
    """ We use a resnet bottleneck structure for fpn """

    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class FeatureFusionBlock(nn.Module):
    """ Feature fusion block """

    def __init__(self,
                 features,
                 activation,
                 bn=False,
                 expand=False,
                 align_corners=True):
        """Init.
        Args:
            features (int): channel dim of the input feature
            activation: activation function to use
            bn: whether to use bn
            expand: whether to expand feature or not
            align_corners: whether to use align_corners for interpolation
        """

        super(FeatureFusionBlock, self).__init__()
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features

        if self.expand is True:
            out_features = features // 2

        self.smoothing = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, xs, up_size):
        """ Forward pass.
        Args
            xs: xs[0]: the feature refined from the previous step, xs[1]: the next scale features to fuse
            up_size: the size for upsampling; xs[0] is upsampled before merging with xs[1]
        Returns:
            output: the fused feature, which is fed to the next fusion step as an input
        """

        output = xs[0]
        if len(xs) == 2:
            # upsampling
            output = nn.functional.interpolate(
                output,
                size=up_size,
                mode='bilinear',
                align_corners=self.align_corners)
            # feature smoothing since the upsampled feature is coarse-grain
            output = self.smoothing(output)

            # refine the next scale feature before fusion
            res = self.resConfUnit1(xs[1])

            # fusion
            output = self.skip_add.add(output, res)

        # post refine after fusion
        output = self.resConfUnit2(output)

        return output


class ResidualConvUnit(nn.Module):
    """ Residual convolution module. """

    def __init__(self, features, activation, bn):
        """Init.
        Args:
            features (int): channel dim of the input
            activation: activation function
            bn: whether to use bn
        """

        super().__init__()

        self.bn = bn
        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            64,
            kernel_size=1,
            stride=1,
            bias=not self.bn,
            groups=self.groups,
        )
        self.conv2 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )
        self.conv3 = nn.Conv2d(
            64,
            features,
            kernel_size=1,
            stride=1,
            bias=not self.bn,
            groups=self.groups,
        )
        if self.bn is True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
            self.bn3 = nn.BatchNorm2d(features)

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """ Forward pass

        Args:
            x (tensor): input feature

        Returns:
            tensor: output feature
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn is True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn is True:
            out = self.bn2(out)

        out = self.activation(out)
        out = self.conv3(out)
        if self.bn is True:
            out = self.bn3(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)
