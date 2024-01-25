import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet


def init_weights(m):
    """init_weights"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def conv_bn_relu(in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bn=True,
                 relu=True):
    """conv_bn_relu"""
    bias = not bn
    layers = []
    layers.append(
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers


def convt_bn_relu(in_channels,
                  out_channels,
                  kernel_size,
                  stride=1,
                  padding=0,
                  output_padding=0,
                  bn=True,
                  relu=True):
    """convt_bn_relu"""
    bias = not bn
    layers = []
    layers.append(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers


class DepthCompletionNet(nn.Module):
    """DepthCompletionNet"""

    def __init__(self, args):
        assert (
            args.layers in [18, 34, 50, 101, 152]
        ), f'Only layers 18, 34, 50, 101, and 152 are defined, but got {layers}'.format(
            layers)
        super(DepthCompletionNet, self).__init__()
        self.modality = args.input

        if 'd' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_d = conv_bn_relu(
                1, channels, kernel_size=3, stride=1, padding=1)
        if 'rgb' in self.modality:
            channels = 64 * 3 // len(self.modality)
            self.conv1_img = conv_bn_relu(
                3, channels, kernel_size=3, stride=1, padding=1)
        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_img = conv_bn_relu(
                1, channels, kernel_size=3, stride=1, padding=1)

        pretrained_model = resnet.__dict__['resnet{}'.format(args.layers)](
            pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model.apply(init_weights)
        # self.maxpool = pretrained_model._modules['maxpool']
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory

        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.conv6 = conv_bn_relu(
            num_channels, 512, kernel_size=3, stride=2, padding=1)

        # decoding layers
        kernel_size = 3
        stride = 2
        self.convt5 = convt_bn_relu(
            in_channels=512,
            out_channels=256,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            output_padding=1)
        self.convt4 = convt_bn_relu(
            in_channels=768,
            out_channels=128,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            output_padding=1)
        self.convt3 = convt_bn_relu(
            in_channels=(256 + 128),
            out_channels=64,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            output_padding=1)
        self.convt2 = convt_bn_relu(
            in_channels=(128 + 64),
            out_channels=64,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            output_padding=1)
        self.convt1 = convt_bn_relu(
            in_channels=128,
            out_channels=64,
            kernel_size=kernel_size,
            stride=1,
            padding=1)
        self.convtf = conv_bn_relu(
            in_channels=128,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bn=False,
            relu=False)

    def forward(self, x):
        """forward"""
        # first layer
        if 'd' in self.modality:
            conv1_d = self.conv1_d(x['d'])
        if 'rgb' in self.modality:
            conv1_img = self.conv1_img(x['rgb'])
        elif 'g' in self.modality:
            conv1_img = self.conv1_img(x['g'])

        if self.modality == 'rgbd' or self.modality == 'gd':
            conv1 = torch.cat((conv1_d, conv1_img), 1)
        else:
            conv1 = conv1_d if (self.modality == 'd') else conv1_img

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)  # batchsize * ? * 176 * 608
        conv4 = self.conv4(conv3)  # batchsize * ? * 88 * 304
        conv5 = self.conv5(conv4)  # batchsize * ? * 44 * 152
        conv6 = self.conv6(conv5)  # batchsize * ? * 22 * 76

        # decoder
        convt5 = self.convt5(conv6)
        y = torch.cat((convt5, conv5), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.convt1(y)
        y = torch.cat((convt1, conv1), 1)

        y = self.convtf(y)

        if self.training:
            return 100 * y
        else:
            min_distance = 0.9
            return F.relu(
                100 * y - min_distance
            ) + min_distance  # the minimum range of Velodyne is around 3 feet ~= 0.9m
