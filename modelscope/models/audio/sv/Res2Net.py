# Copyright (c) Alibaba, Inc. and its affiliates.
""" Res2Net implementation is adapted from https://github.com/Res2Net/Res2Net-PretrainedModels.
    Res2Net is an advanced neural network architecture that enhances the capabilities of standard ResNets
    by incorporating hierarchical residual-like connections. This innovative structure improves
    performance across various computer vision tasks, such as image classification and object
    detection, without significant computational overhead.
    Reference: https://arxiv.org/pdf/1904.01169.pdf
    Some modifications from the original architecture:
    1. Smaller kernel size for the input layer
    2. Smaller expansion in BasicBlockRes2Net
"""
import math
import os
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi

import modelscope.models.audio.sv.pooling_layers as pooling_layers
from modelscope.metainfo import Models
from modelscope.models import MODELS, TorchModel
from modelscope.utils.constant import Tasks
from modelscope.utils.device import create_device


class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


class BasicBlockRes2Net(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, baseWidth=32, scale=2):
        super(BasicBlockRes2Net, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(
            in_planes, width * scale, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self,
                 block=BasicBlockRes2Net,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=32,
                 feat_dim=80,
                 embedding_size=192,
                 pooling_func='TSTP',
                 two_emb_layer=False):
        super(Res2Net, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embedding_size = embedding_size
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(
            block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(
            block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            block, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(
            block, m_channels * 8, num_blocks[3], stride=2)

        self.n_stats = 1 if pooling_func == 'TAP' or pooling_func == 'TSDP' else 2
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim * block.expansion)
        self.seg_1 = nn.Linear(self.stats_dim * block.expansion * self.n_stats,
                               embedding_size)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embedding_size, affine=False)
            self.seg_2 = nn.Linear(embedding_size, embedding_size)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)

        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        stats = self.pool(out)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a


@MODELS.register_module(
    Tasks.speaker_verification, module_name=Models.res2net_sv)
class SpeakerVerificationResNet(TorchModel):
    r"""
    Args:
        model_dir: A model dir.
        model_config: The model config.
    """

    def __init__(self, model_dir, model_config: Dict[str, Any], *args,
                 **kwargs):
        super().__init__(model_dir, model_config, *args, **kwargs)
        self.model_config = model_config
        self.embed_dim = self.model_config['embed_dim']
        self.m_channels = self.model_config['channels']
        self.other_config = kwargs
        self.feature_dim = 80
        self.device = create_device(self.other_config['device'])

        self.embedding_model = Res2Net(
            embedding_size=self.embed_dim, m_channels=self.m_channels)

        pretrained_model_name = kwargs['pretrained_model']
        self.__load_check_point(pretrained_model_name)

        self.embedding_model.to(self.device)
        self.embedding_model.eval()

    def forward(self, audio):
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        assert len(
            audio.shape
        ) == 2, 'modelscope error: the shape of input audio to model needs to be [N, T]'
        # audio shape: [N, T]
        feature = self.__extract_feature(audio)
        embedding = self.embedding_model(feature.to(self.device))

        return embedding.detach().cpu()

    def __extract_feature(self, audio):
        feature = Kaldi.fbank(audio, num_mel_bins=self.feature_dim)
        feature = feature - feature.mean(dim=0, keepdim=True)
        feature = feature.unsqueeze(0)
        return feature

    def __load_check_point(self, pretrained_model_name, device=None):
        if not device:
            device = torch.device('cpu')
        self.embedding_model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, pretrained_model_name),
                map_location=device),
            strict=True)
