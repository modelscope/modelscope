# Adapted from https://github.com/isl-org/lang-seg.
# Originally MIT License, Copyright (c) 2021 Intelligent Systems Lab Org.

import numpy as np
import torch
import torch.nn as nn

from . import clip
from .lseg_blocks import (FeatureFusionBlock, FeatureFusionBlock_custom,
                          Interpolate, _make_encoder, forward_vit)
from .simple_tokenizer import SimpleTokenizer


class depthwise_clipseg_conv(nn.Module):

    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def depthwise_clipseg(self, x, channels):
        x = torch.cat(
            [self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)],
            dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out


class depthwise_conv(nn.Module):

    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(
            1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):

    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):

    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x


class BaseModel(torch.nn.Module):

    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device('cpu'))

        if 'optimizer' in parameters:
            parameters = parameters['model']

        self.load_state_dict(parameters)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class LSeg(BaseModel):

    def __init__(
        self,
        features=256,
        backbone='clip_vitl16_384',
        readout='project',
        use_bn=True,
        model_dir=None,
    ):
        super(LSeg, self).__init__()
        hooks = {
            'clip_vitl16_384': [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.logit_scale = nn.Parameter(torch.ones([])
                                        * np.log(1 / 0.07)).exp()
        self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.scratch.output_conv = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear', align_corners=True), )

        self.tau = 0.07
        self.model_dir = model_dir
        self.tokenizer = SimpleTokenizer(model_dir
                                         + '/bpe_simple_vocab_16e6.txt.gz')

    def forward(self, x, labelset=''):
        text = clip.tokenize(self.tokenizer, labelset)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        text = text.to(x.device)
        text_features = self.clip_pretrained.encode_text(text)

        image_features = self.scratch.head1(path_1)

        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3,
                                                1).reshape(-1, self.out_c)

        # normalized features
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True)

        logits_per_image = image_features @ text_features.t() / self.tau

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3],
                                            -1).permute(0, 3, 1, 2)

        out = self.scratch.output_conv(out)

        return out
