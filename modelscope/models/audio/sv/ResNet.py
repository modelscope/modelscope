# Copyright (c) Alibaba, Inc. and its affiliates.
""" ResNet implementation is adapted from https://github.com/wenet-e2e/wespeaker.
    ResNet, or Residual Neural Network, is notable for its optimization ease
    and depth-induced accuracy gains. It utilizes skip connections within its residual
    blocks to counteract the vanishing gradient problem in deep networks.
    Reference: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self,
                 block=BasicBlock,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=32,
                 feat_dim=80,
                 embedding_size=128,
                 pooling_func='TSTP',
                 two_emb_layer=True):
        super(ResNet, self).__init__()
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
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = self.layer4(out3)
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
    Tasks.speaker_verification, module_name=Models.resnet_sv)
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

        self.embedding_model = ResNet(
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
