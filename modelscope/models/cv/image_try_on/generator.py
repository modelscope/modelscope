# The implementation here is modified based on spade,
# originally Apache 2.0 License and publicly available at https://github.com/NVlabs/SPADE

import functools
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torchvision import models


class ResidualBlock(nn.Module):

    def __init__(self, in_features=64, norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.relu = nn.PReLU()
        if norm_layer is None:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                nn.PReLU(),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features), nn.PReLU(),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class ResUnetSkipConnectionBlock(nn.Module):

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 input_nc=None,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(
            input_nc,
            inner_nc,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=use_bias)
        # add two resblock
        res_downconv = [
            ResidualBlock(inner_nc, norm_layer),
            ResidualBlock(inner_nc, norm_layer),
            ResidualBlock(inner_nc, norm_layer)
        ]
        res_upconv = [
            ResidualBlock(outer_nc, norm_layer),
            ResidualBlock(outer_nc, norm_layer),
            ResidualBlock(outer_nc, norm_layer)
        ]

        downrelu = nn.PReLU()
        uprelu = nn.PReLU()
        if norm_layer is not None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            up = [upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(
                inner_nc,
                outer_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            if norm_layer is None:
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                up = [upsample, upconv, upnorm, uprelu] + res_upconv
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias)
            if norm_layer is None:
                down = [downconv, downrelu] + res_downconv
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                down = [downconv, downnorm, downrelu] + res_downconv
                up = [upsample, upconv, upnorm, uprelu] + res_upconv

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class LandmarkNorm(nn.Module):

    def __init__(self, param_free_norm_type, norm_nc, label_nc):
        super().__init__()

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(
                norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(
                '%s is not a recognized param-free norm type in LandmarkNorm'
                % param_free_norm_type)

        nhidden = 128
        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU())
        self.mlp_gamma = nn.Conv2d(
            nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class LandmarkNormResnetBlock(nn.Module):

    def __init__(self, fin, fout):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        landmarknorm_config_str = 'batch'
        semantic_nc = 32
        self.norm_0 = LandmarkNorm(landmarknorm_config_str, fin, semantic_nc)
        self.norm_1 = LandmarkNorm(landmarknorm_config_str, fmiddle,
                                   semantic_nc)
        if self.learned_shortcut:
            self.norm_s = LandmarkNorm(landmarknorm_config_str, fin,
                                       semantic_nc)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class VTONGenerator(nn.Module):
    """ initialize the try on generator model
    """

    def __init__(self,
                 input_nc,
                 output_nc,
                 num_downs,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(VTONGenerator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        ngf_list = [ngf * 1, ngf * 2, ngf * 4, ngf * 8, ngf * 8]
        self.num_downs = num_downs
        self.Encoder = []
        self.Decoder = []
        self.LMnorm = []

        for i in range(num_downs):
            # Encoder
            if i == 0:
                in_nc = input_nc
                inner_nc = ngf_list[i]
            else:
                in_nc, inner_nc = ngf_list[i - 1], ngf_list[i]

            downconv = nn.Conv2d(
                in_nc,
                inner_nc,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=use_bias)
            downnorm = norm_layer(inner_nc)
            downrelu = nn.PReLU()
            res_downconv = [
                ResidualBlock(inner_nc, norm_layer),
                ResidualBlock(inner_nc, norm_layer),
                ResidualBlock(inner_nc, norm_layer)
            ]

            # Decoder
            if i == (num_downs - 1):
                outer_nc = ngf // 2
                inner_nc = 2 * ngf_list[0]
            elif i == 0:
                inner_nc, outer_nc = ngf_list[num_downs - i
                                              - 1], ngf_list[num_downs - i - 1]
            else:
                inner_nc, outer_nc = 2 * ngf_list[num_downs - i
                                                  - 1], ngf_list[num_downs - i
                                                                 - 2]

            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(
                inner_nc,
                outer_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias)
            upnorm = norm_layer(outer_nc)
            uprelu = nn.PReLU()
            res_upconv = [
                ResidualBlock(outer_nc, norm_layer),
                ResidualBlock(outer_nc, norm_layer),
                ResidualBlock(outer_nc, norm_layer)
            ]

            if i == 0:
                encoderLayer = [downconv, downrelu] + res_downconv
                decoderLayer = [upsample, upconv, upnorm, uprelu] + res_upconv
            elif i == (num_downs - 1):
                encoderLayer = [downconv, downrelu] + res_downconv
                decoderLayer = [upsample, upconv]
            else:
                encoderLayer = [downconv, downnorm, downrelu] + res_downconv
                decoderLayer = [upsample, upconv, upnorm, uprelu] + res_upconv

            encoderLayer = nn.Sequential(*encoderLayer)
            decoderLayer = nn.Sequential(*decoderLayer)
            self.Encoder.append(encoderLayer)
            self.Decoder.append(decoderLayer)

            LMnorm = LandmarkNormResnetBlock(outer_nc, outer_nc)
            self.LMnorm.append(LMnorm)

        self.Encoder = nn.ModuleList(self.Encoder)
        self.Decoder = nn.ModuleList(self.Decoder)
        self.LMnorm = nn.ModuleList(self.LMnorm)

        self.conv_img = nn.Conv2d(ngf // 2, 3, kernel_size=3, padding=1)
        self.act = nn.PReLU()
        self.tanh = nn.Tanh()

    def forward(self, inputs, p_point_heatmap):
        en_fea = []
        x = inputs
        for i in range(self.num_downs):
            x = self.Encoder[i](x)
            if i < (self.num_downs - 1):
                en_fea.append(x)

        for i in range(self.num_downs):
            if i != 0:
                x = torch.cat([en_fea[-i], x], 1)
            x = self.Decoder[i](x)
            x = self.LMnorm[i](x, p_point_heatmap)

        x = self.conv_img(self.act(x))
        x = self.tanh(x)
        return x


class ResUnetGenerator(nn.Module):

    def __init__(self,
                 input_nc,
                 output_nc,
                 num_downs,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(ResUnetGenerator, self).__init__()
        # construct unet structure
        unet_block = ResUnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True)

        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout)
        unet_block = ResUnetSkipConnectionBlock(
            ngf * 4,
            ngf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(
            ngf * 2,
            ngf * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(
            ngf,
            ngf * 2,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class Vgg19(nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=False)
        # for torchvision >= 0.4.0 or torch >= 1.2.0
        for x in vgg_pretrained_features.modules():
            if isinstance(x, nn.MaxPool2d) or isinstance(
                    x, nn.AdaptiveAvgPool2d):
                x.ceil_mode = True
        vgg_pretrained_features.load_state_dict(torch.load(vgg_path))
        vgg_pretrained_features = vgg_pretrained_features.features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):

    def __init__(self, layids=None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i],
                                                     y_vgg[i].detach())
        return loss


def load_checkpoint_parallel(model, checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        checkpoint_new[param] = checkpoint[param]
    model.load_state_dict(checkpoint_new)
