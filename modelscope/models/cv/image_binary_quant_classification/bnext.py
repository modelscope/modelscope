# Part of the implementation is borrowed and modified from BNext,
# publicly available at https://github.com/hpi-xnor/BNext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# stage ratio: 1:1:3:1
stage_out_channel_tiny = [32] + [
    64
] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2

# stage ratio 1:1:3:1
stage_out_channel_small = [48] + [
    96
] + [192] * 2 + [384] * 2 + [768] * 6 + [1536] * 2

# stage ratio 2:2:4:2
stage_out_channel_middle = [48] + [
    96
] + [192] * 4 + [384] * 4 + [768] * 8 + [1536] * 4

# stage ratio 2:2:8:2
stage_out_channel_large = [64] + [
    128
] + [256] * 4 + [512] * 4 + [1024] * 16 + [2048] * 4


def conv3x3(in_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            groups=1,
            dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        dilation=dilation,
        groups=groups,
        bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class HardSigmoid(nn.Module):

    def __init__(self, ):
        super(HardSigmoid, self).__init__()

    def forward(self, x):
        return F.relu6(x + 3) / 6


class firstconv3x3(nn.Module):

    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)
        self.prelu = nn.PReLU(oup, oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)

        return out


class LearnableBias(nn.Module):

    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(
            torch.zeros(1, out_chn, 1, 1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class HardSign(nn.Module):

    def __init__(self, range=[-1, 1], progressive=False):
        super(HardSign, self).__init__()
        self.range = range
        self.progressive = progressive
        self.register_buffer('temperature', torch.ones(1))

    def adjust(self, x, scale=0.1):
        self.temperature.mul_(scale)

    def forward(self, x):
        replace = x.clamp(self.range[0], self.range[1])
        x = x.div(self.temperature.clamp(min=1e-8)).clamp(-1, 1)
        if not self.progressive:
            sign = x.sign()
        else:
            sign = x
        return (sign - replace).detach() + replace


class HardBinaryConv(nn.Module):

    def __init__(self,
                 in_chn,
                 out_chn,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups
        self.number_of_weights = in_chn // groups * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(
            torch.randn((self.shape)) * 0.001, requires_grad=True)

        self.register_buffer('temperature', torch.ones(1))

    def forward(self, x):
        if self.training:
            self.weight.data.clamp_(-1.5, 1.5)

        real_weights = self.weight

        if self.temperature < 1e-7:
            binary_weights_no_grad = real_weights.sign()
        else:
            binary_weights_no_grad = (
                real_weights / self.temperature.clamp(min=1e-8)).clamp(-1, 1)
        cliped_weights = real_weights

        if self.training:
            binary_weights = binary_weights_no_grad.detach(
            ) - cliped_weights.detach() + cliped_weights
        else:
            binary_weights = binary_weights_no_grad

        y = F.conv2d(
            x,
            binary_weights,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups)

        return y


class SqueezeAndExpand(nn.Module):

    def __init__(self,
                 channels,
                 planes,
                 ratio=8,
                 attention_mode='hard_sigmoid'):
        super(SqueezeAndExpand, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, channels // ratio, kernel_size=1, padding=0),
            nn.ReLU(channels // ratio),
            nn.Conv2d(channels // ratio, planes, kernel_size=1, padding=0),
        )

        if attention_mode == 'sigmoid':
            self.attention = nn.Sigmoid()

        elif attention_mode == 'hard_sigmoid':
            self.attention = HardSigmoid()

        else:
            self.attention = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.se(x)
        x = self.attention(x)
        return x


class Attention(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 drop_rate=0.1,
                 infor_recoupling=True,
                 groups=1):
        super(Attention, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.infor_recoupling = infor_recoupling

        self.move = LearnableBias(inplanes)
        self.binary_activation = HardSign(range=[-1.5, 1.5])
        self.binary_conv = HardBinaryConv(
            inplanes, planes, kernel_size=3, stride=stride, groups=groups)

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        self.activation1 = nn.PReLU(inplanes)
        self.activation2 = nn.PReLU(planes)

        self.downsample = downsample
        self.stride = stride
        if stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)

        if self.infor_recoupling:
            self.se = SqueezeAndExpand(
                planes, planes, attention_mode='sigmoid')
            self.scale = nn.Parameter(torch.ones(1, planes, 1, 1) * 0.5)

    def forward(self, input):

        residual = self.activation1(input)

        if self.stride == 2:
            residual = self.pooling(residual)

        x = self.move(input)
        x = self.binary_activation(x)
        x = self.binary_conv(x)
        x = self.norm1(x)
        x = self.activation2(x)

        if self.infor_recoupling:
            if self.training:
                self.scale.data.clamp_(0, 1)
            if self.stride == 2:
                input = self.pooling(input)
            mix = self.scale * input + x * (1 - self.scale)
            x = self.se(mix) * x
        else:
            pass
        x = x * residual
        x = self.norm2(x)
        x = x + residual

        return x


class FFN_3x3(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 drop_rate=0.1,
                 infor_recoupling=True,
                 groups=1):
        super(FFN_3x3, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.infor_recoupling = infor_recoupling

        self.move = LearnableBias(inplanes)
        self.binary_activation = HardSign(range=[-1.5, 1.5])
        self.binary_conv = HardBinaryConv(
            inplanes, planes, kernel_size=3, stride=stride, groups=groups)

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        self.activation1 = nn.PReLU(inplanes)
        self.activation2 = nn.PReLU(planes)

        if stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)

        if self.infor_recoupling:
            self.se = SqueezeAndExpand(
                inplanes, planes, attention_mode='sigmoid')
            self.scale = nn.Parameter(torch.ones(1, planes, 1, 1) * 0.5)

    def forward(self, input):

        residual = input

        if self.stride == 2:
            residual = self.pooling(residual)

        x = self.move(input)
        x = self.binary_activation(x)
        x = self.binary_conv(x)
        x = self.norm1(x)
        x = self.activation2(x)

        if self.infor_recoupling:
            if self.training:
                self.scale.data.clamp_(0, 1)
            if self.stride == 2:
                input = self.pooling(input)
            mix = self.scale * input + (1 - self.scale) * x
            x = self.se(mix) * x
            x = self.norm2(x)
        else:
            pass

        x = x + residual

        return x


class FFN_1x1(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 attention=True,
                 drop_rate=0.1,
                 infor_recoupling=True):
        super(FFN_1x1, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.infor_recoupling = infor_recoupling

        self.move = LearnableBias(inplanes)
        self.binary_activation = HardSign(range=[-1.5, 1.5])
        self.binary_conv = HardBinaryConv(
            inplanes, planes, kernel_size=1, stride=stride, padding=0)

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        self.activation1 = nn.PReLU(inplanes)
        self.activation2 = nn.PReLU(planes)

        if stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)

        if self.infor_recoupling:
            self.se = SqueezeAndExpand(
                inplanes, planes, attention_mode='sigmoid')
            self.scale = nn.Parameter(torch.ones(1, planes, 1, 1) * 0.5)

    def forward(self, input):

        residual = input

        if self.stride == 2:
            residual = self.pooling(residual)

        x = self.move(input)
        x = self.binary_activation(x)
        x = self.binary_conv(x)
        x = self.norm1(x)
        x = self.activation2(x)
        if self.infor_recoupling:
            self.scale.data.clamp_(0, 1)
            mix = self.scale * input + (1 - self.scale) * x
            x = self.se(mix) * x
            x = self.norm2(x)
        else:
            pass

        x = x + residual

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 drop_rate=0.1,
                 mode='scale'):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes

        if mode == 'scale':
            self.Attention = Attention(
                inplanes,
                inplanes,
                stride,
                None,
                drop_rate=drop_rate,
                groups=1)
        else:
            self.Attention = FFN_3x3(
                inplanes,
                inplanes,
                stride,
                None,
                drop_rate=drop_rate,
                groups=1)

        if inplanes == planes:
            self.FFN = FFN_1x1(inplanes, inplanes, drop_rate=drop_rate)

        else:
            self.FFN_1 = FFN_1x1(inplanes, inplanes, drop_rate=drop_rate)

            self.FFN_2 = FFN_1x1(inplanes, inplanes, drop_rate=drop_rate)

    def forward(self, input):
        x = self.Attention(input)

        if self.inplanes == self.planes:
            y = self.FFN(x)

        else:
            y_1 = self.FFN_1(x)
            y_2 = self.FFN_2(x)
            y = torch.cat((y_1, y_2), dim=1)

        return y


class BasicBlock_No_ELM_Attention(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 drop_rate=0.1,
                 mode='scale'):
        super(BasicBlock_No_ELM_Attention, self).__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.FFN_3x3 = FFN_3x3(
            inplanes, inplanes, stride, None, drop_rate=drop_rate, groups=1)

        if self.inplanes == self.planes:
            self.FFN = FFN_1x1(inplanes, inplanes, drop_rate=drop_rate)
        else:
            self.FFN_1 = FFN_1x1(inplanes, inplanes, drop_rate=drop_rate)
            self.FFN_2 = FFN_1x1(inplanes, inplanes, drop_rate=drop_rate)

    def forward(self, input):
        x = self.FFN_3x3(input)
        if self.inplanes == self.planes:
            y = self.FFN(x)
        else:
            y_1 = self.FFN_1(x)
            y_2 = self.FFN_2(x)
            y = torch.cat((y_1, y_2), dim=1)

        return y


class BasicBlock_No_Infor_Recoupling(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 drop_rate=0.1,
                 mode='scale'):
        super(BasicBlock_No_Infor_Recoupling, self).__init__()
        self.inplanes = inplanes
        self.planes = planes

        if mode == 'scale':
            self.Attention = Attention(
                inplanes,
                inplanes,
                stride,
                None,
                drop_rate,
                infor_recoupling=False,
                groups=1)
        else:
            self.Attention = FFN_3x3(
                inplanes,
                inplanes,
                stride,
                None,
                drop_rate=drop_rate,
                infor_recoupling=False,
                groups=1)

        if self.inplanes == self.planes:
            self.FFN = FFN_1x1(
                inplanes,
                inplanes,
                drop_rate=drop_rate,
                infor_recoupling=False)
        else:
            self.FFN_1 = FFN_1x1(
                inplanes,
                inplanes,
                drop_rate=drop_rate,
                infor_recoupling=False)
            self.FFN_2 = FFN_1x1(
                inplanes,
                inplanes,
                drop_rate=drop_rate,
                infor_recoupling=False)

    def forward(self, input):
        x = self.Attention(input)
        if self.inplanes == self.planes:
            y = self.FFN(x)
        else:
            y_1 = self.FFN_1(x)
            y_2 = self.FFN_2(x)
            y = torch.cat((y_1, y_2), dim=1)

        return y


class BasicBlock_No_Extra_Design(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 drop_rate=0.1,
                 mode='scale'):
        super(BasicBlock_No_Extra_Design, self).__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.FFN_3x3 = FFN_3x3(
            inplanes,
            inplanes,
            stride,
            None,
            drop_rate,
            infor_recoupling=False,
            groups=1)
        if self.inplanes == self.planes:
            self.FFN = FFN_1x1(
                inplanes,
                inplanes,
                drop_rate=drop_rate,
                infor_recoupling=False)
        else:
            self.FFN_1 = FFN_1x1(
                inplanes,
                inplanes,
                drop_rate=drop_rate,
                infor_recoupling=False)
            self.FFN_2 = FFN_1x1(
                inplanes,
                inplanes,
                drop_rate=drop_rate,
                infor_recoupling=False)

    def forward(self, input):
        x = self.FFN_3x3(input)
        if self.inplanes == self.planes:
            y = self.FFN(x)
        else:
            y_1 = self.FFN_1(x)
            y_2 = self.FFN_2(x)
            y = torch.cat((y_1, y_2), dim=1)

        return y


class BNext(nn.Module):

    def __init__(self,
                 num_classes=1000,
                 size='small',
                 ELM_Attention=True,
                 Infor_Recoupling=True):
        super(BNext, self).__init__()
        drop_rate = 0.2 if num_classes == 100 else 0.0

        if size == 'tiny':
            stage_out_channel = stage_out_channel_tiny
        elif size == 'small':
            stage_out_channel = stage_out_channel_small
        elif size == 'middle':
            stage_out_channel = stage_out_channel_middle
        elif size == 'large':
            stage_out_channel = stage_out_channel_large
        else:
            raise ValueError('The size is not defined!')

        if ELM_Attention and Infor_Recoupling:
            basicblock = BasicBlock
            print('Model with ELM Attention and Infor-Recoupling')
        elif (ELM_Attention and not Infor_Recoupling):
            basicblock = BasicBlock_No_Infor_Recoupling
            print('Model with ELM Attention, No Infor-Recoupling')
        elif (not ELM_Attention and Infor_Recoupling):
            basicblock = BasicBlock_No_ELM_Attention
            print('Model with Infor-Recoupling, No ELM Attention')
        else:
            basicblock = BasicBlock_No_Extra_Design
            print('Model with no Extra Design')

        self.feature = nn.ModuleList()
        drop_rates = [
            x.item()
            for x in torch.linspace(0, drop_rate, (len(stage_out_channel)))
        ]

        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(
                    firstconv3x3(3, stage_out_channel[i],
                                 1 if num_classes != 1000 else 2))
            elif i == 1:
                self.feature.append((basicblock(
                    stage_out_channel[i - 1],
                    stage_out_channel[i],
                    1,
                    drop_rate=drop_rates[i],
                    mode='bias')))
            elif stage_out_channel[i - 1] != stage_out_channel[
                    i] and stage_out_channel[i] != stage_out_channel[1]:
                self.feature.append(
                    basicblock(
                        stage_out_channel[i - 1],
                        stage_out_channel[i],
                        2,
                        drop_rate=drop_rates[i],
                        mode='scale' if i % 2 == 0 else 'bias'))
            else:
                self.feature.append(
                    basicblock(
                        stage_out_channel[i - 1],
                        stage_out_channel[i],
                        1,
                        drop_rate=drop_rates[i],
                        mode='scale' if i % 2 == 0 else 'bias'))

        self.prelu = nn.PReLU(stage_out_channel[-1])
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_out_channel[-1], num_classes)

    def forward(self, img, return_loss=False, img_metas=None):
        x = img
        for i, block in enumerate(self.feature):
            x = block(x)
        x = self.prelu(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = list(x.detach().cpu().numpy())
        return x
