# The implementation here is modified based on https://github.com/xy-guo/MVSNet_pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == 'kaiming':
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == 'xavier':
            nn.init.xavier_uniform_(module.weight)
    return


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 relu=True,
                 bn=True,
                 bn_momentum=0.1,
                 init_method='xavier',
                 **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=(not bn),
            **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(
            out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 relu=True,
                 bn=True,
                 bn_momentum=0.1,
                 init_method='xavier',
                 **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=(not bn),
            **kwargs)
        self.bn = nn.BatchNorm2d(
            out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 relu=True,
                 bn=True,
                 bn_momentum=0.1,
                 init_method='xavier',
                 **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=(not bn),
            **kwargs)
        self.bn = nn.BatchNorm3d(
            out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 relu=True,
                 bn=True,
                 bn_momentum=0.1,
                 init_method='xavier',
                 **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=(not bn),
            **kwargs)
        self.bn = nn.BatchNorm3d(
            out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class ConvBnReLU(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    """
        src_fea: [B, C, H, W]
        src_proj: [B, 4, 4]
        ref_proj: [B, 4, 4]
        depth_values: [B, Ndepth] o [B, Ndepth, H, W]
        out: [B, C, Ndepth, H, W]
    """
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([
            torch.arange(
                0, height, dtype=torch.float32, device=src_fea.device),
            torch.arange(0, width, dtype=torch.float32, device=src_fea.device)
        ])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(
            1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                    -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1,
                                              1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :
                           2, :, :] / proj_xyz[:, 2:
                                               3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized),
                              dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(
        src_fea,
        grid.view(batch, num_depth * height, width, 2),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height,
                                         width)

    return warped_src_fea


class DeConv2dFuse(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 relu=True,
                 bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding=1,
            output_padding=1,
            bn=True,
            relu=relu,
            bn_momentum=bn_momentum)

        self.conv = Conv2d(
            2 * out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=1,
            bn=bn,
            relu=relu,
            bn_momentum=bn_momentum)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class FeatureNet(nn.Module):

    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode='unet'):
        super(FeatureNet, self).__init__()
        assert arch_mode in [
            'unet', 'fpn'
        ], f"mode must be in 'unet' or 'fpn', but get:{arch_mode}"
        self.arch_mode = arch_mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(
                base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(
            base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]

        if self.arch_mode == 'unet':
            if num_stage == 3:
                self.deconv1 = DeConv2dFuse(base_channels * 4,
                                            base_channels * 2, 3)
                self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels,
                                            3)

                self.out2 = nn.Conv2d(
                    base_channels * 2, base_channels * 2, 1, bias=False)
                self.out3 = nn.Conv2d(
                    base_channels, base_channels, 1, bias=False)
                self.out_channels.append(2 * base_channels)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.deconv1 = DeConv2dFuse(base_channels * 4,
                                            base_channels * 2, 3)

                self.out2 = nn.Conv2d(
                    base_channels * 2, base_channels * 2, 1, bias=False)
                self.out_channels.append(2 * base_channels)
        elif self.arch_mode == 'fpn':
            final_chs = base_channels * 4
            if num_stage == 3:
                self.inner1 = nn.Conv2d(
                    base_channels * 2, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(
                    base_channels * 1, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(
                    final_chs, base_channels * 2, 3, padding=1, bias=False)
                self.out3 = nn.Conv2d(
                    final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels * 2)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.inner1 = nn.Conv2d(
                    base_channels * 2, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(
                    final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs['stage1'] = out
        if self.arch_mode == 'unet':
            if self.num_stage == 3:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs['stage2'] = out

                intra_feat = self.deconv2(conv0, intra_feat)
                out = self.out3(intra_feat)
                outputs['stage3'] = out

            elif self.num_stage == 2:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs['stage2'] = out

        elif self.arch_mode == 'fpn':
            if self.num_stage == 3:
                intra_feat = F.interpolate(
                    intra_feat, scale_factor=2,
                    mode='nearest') + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs['stage2'] = out

                intra_feat = F.interpolate(
                    intra_feat, scale_factor=2,
                    mode='nearest') + self.inner2(conv0)
                out = self.out3(intra_feat)
                outputs['stage3'] = out

            elif self.num_stage == 2:
                intra_feat = F.interpolate(
                    intra_feat, scale_factor=2,
                    mode='nearest') + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs['stage2'] = out

        return outputs


class CostRegNet(nn.Module):

    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(
            base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(
            base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(
            base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(
            base_channels * 8,
            base_channels * 4,
            stride=2,
            padding=1,
            output_padding=1)

        self.conv9 = Deconv3d(
            base_channels * 4,
            base_channels * 2,
            stride=2,
            padding=1,
            output_padding=1)

        self.conv11 = Deconv3d(
            base_channels * 2,
            base_channels * 1,
            stride=2,
            padding=1,
            output_padding=1)

        self.prob = nn.Conv3d(
            base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


class RefineNet(nn.Module):

    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth


def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get('dlossw', None)

    total_loss = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=mask_ms['stage1'].device,
        requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys()
                                      if 'stage' in k]:
        depth_est = stage_inputs['depth']
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(
            depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace('stage', '')) - 1
            total_loss += depth_loss_weights[stage_idx] * depth_loss
        else:
            total_loss += 1.0 * depth_loss

    return total_loss, depth_loss


def get_cur_depth_range_samples(cur_depth,
                                ndepth,
                                depth_inteval_pixel,
                                shape,
                                max_depth=192.0,
                                min_depth=0.0):
    """
        shape, (B, H, W)
        cur_depth: (B, H, W)
        return depth_range_values: (B, D, H, W)
    """
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)

    assert cur_depth.shape == torch.Size(
        shape), 'cur_depth:{}, input shape:{}'.format(cur_depth.shape, shape)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (
        torch.arange(
            0,
            ndepth,
            device=cur_depth.device,
            dtype=cur_depth.dtype,
            requires_grad=False).reshape(1, -1, 1, 1)
        * new_interval.unsqueeze(1))

    return depth_range_samples


def get_depth_range_samples(cur_depth,
                            ndepth,
                            depth_inteval_pixel,
                            device,
                            dtype,
                            shape,
                            max_depth=192.0,
                            min_depth=0.0):
    """
        shape: (B, H, W)
        cur_depth: (B, H, W) or (B, D)
        return depth_range_samples: (B, D, H, W)
    """
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )

        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(
            0, ndepth, device=device, dtype=dtype,
            requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)
                                                            )  # noqa  # (B, D)

        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(
            -1).repeat(1, 1, shape[1], shape[2])  # (B, D, H, W)

    else:

        depth_range_samples = get_cur_depth_range_samples(
            cur_depth, ndepth, depth_inteval_pixel, shape, max_depth,
            min_depth)

    return depth_range_samples
