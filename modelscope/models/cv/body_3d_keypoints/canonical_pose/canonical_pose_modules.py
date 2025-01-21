# The implementation is based on VideoPose3D, available at https://github.com/facebookresearch/VideoPose3D
import torch
import torch.nn as nn


class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = self._forward_blocks(x)

        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)

        return x


class TemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self,
                 num_joints_in,
                 in_features,
                 num_joints_out,
                 filter_widths,
                 causal=False,
                 dropout=0.25,
                 channels=1024,
                 dense=False):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out,
                         filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv1d(
            num_joints_in * in_features,
            channels,
            filter_widths[0],
            bias=False)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2
                                      * next_dilation) if causal else 0)

            layers_conv.append(
                nn.Conv1d(
                    channels,
                    channels,
                    filter_widths[i] if not dense else (2 * self.pad[-1] + 1),
                    dilation=next_dilation if not dense else 1,
                    bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(
                nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            res = x[:, :, pad + shift:x.shape[2] - pad + shift]
            x = self.drop(
                self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(
                self.relu(self.layers_bn[2 * i + 1](
                    self.layers_conv[2 * i + 1](x))))

        x = self.shrink(x)
        return x


# regression of the trajectory
class TransCan3Dkeys(nn.Module):

    def __init__(self,
                 in_channels=74,
                 num_features=256,
                 out_channels=44,
                 time_window=10,
                 num_blocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.time_window = time_window

        self.expand_bn = nn.BatchNorm1d(self.num_features, momentum=0.1)
        self.conv1 = nn.Sequential(
            nn.ReplicationPad1d(1),
            nn.Conv1d(
                self.in_channels, self.num_features, kernel_size=3,
                bias=False), self.expand_bn, nn.ReLU(inplace=True),
            nn.Dropout(p=0.25))
        self._make_blocks()
        self.pad = nn.ReplicationPad1d(4)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.25)
        self.reduce = nn.Conv1d(
            self.num_features, self.num_features, kernel_size=self.time_window)
        self.embedding_3d_1 = nn.Linear(in_channels // 2 * 3, 500)
        self.embedding_3d_2 = nn.Linear(500, 500)
        self.LReLU1 = nn.LeakyReLU()
        self.LReLU2 = nn.LeakyReLU()
        self.LReLU3 = nn.LeakyReLU()
        self.out1 = nn.Linear(self.num_features + 500, self.num_features)
        self.out2 = nn.Linear(self.num_features, self.out_channels)

    def _make_blocks(self):
        layers_conv = []
        layers_bn = []
        for i in range(self.num_blocks):
            layers_conv.append(
                nn.Conv1d(
                    self.num_features,
                    self.num_features,
                    kernel_size=5,
                    bias=False,
                    dilation=2))
            layers_bn.append(nn.BatchNorm1d(self.num_features))
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def forward(self, p2ds, p3d):
        """
        Args:
        x - (B x T x J x C)
        """
        B, T, C = p2ds.shape
        x = p2ds.permute((0, 2, 1))
        x = self.conv1(x)
        for i in range(self.num_blocks):
            pre = x
            x = self.pad(x)
            x = self.layers_conv[i](x)
            x = self.layers_bn[i](x)
            x = self.drop(self.relu(x))
            x = pre + x
        x_2d = self.relu(self.reduce(x))
        x_2d = x_2d.view(B, -1)
        x_3d = self.LReLU1(self.embedding_3d_1(p3d))
        x = torch.cat((x_2d, x_3d), 1)
        x = self.LReLU3(self.out1(x))
        x = self.out2(x)
        return x
