# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from collections import OrderedDict
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi

from modelscope.metainfo import Models
from modelscope.models import MODELS, TorchModel
from modelscope.models.audio.sv.DTDNN_layers import (BasicResBlock,
                                                     CAMDenseTDNNBlock,
                                                     DenseLayer, StatsPool,
                                                     TDNNLayer, TransitLayer,
                                                     get_nonlinear)
from modelscope.utils.constant import Tasks
from modelscope.utils.device import create_device


class FCM(nn.Module):

    def __init__(self,
                 block=BasicResBlock,
                 num_blocks=[2, 2],
                 m_channels=32,
                 feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(
            block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(
            block, m_channels, num_blocks[0], stride=2)

        self.conv2 = nn.Conv2d(
            m_channels,
            m_channels,
            kernel_size=3,
            stride=(2, 1),
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class CAMPPlus(nn.Module):

    def __init__(self,
                 feat_dim=80,
                 embedding_size=512,
                 growth_rate=32,
                 bn_size=4,
                 init_channels=128,
                 config_str='batchnorm-relu',
                 memory_efficient=True,
                 output_level='segment'):
        super(CAMPPlus, self).__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.output_level = output_level

        self.xvector = nn.Sequential(
            OrderedDict([
                ('tdnn',
                 TDNNLayer(
                     channels,
                     init_channels,
                     5,
                     stride=2,
                     dilation=1,
                     padding=-1,
                     config_str=config_str)),
            ]))
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
                zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient)
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                'transit%d' % (i + 1),
                TransitLayer(
                    channels, channels // 2, bias=False,
                    config_str=config_str))
            channels //= 2

        self.xvector.add_module('out_nonlinear',
                                get_nonlinear(config_str, channels))

        if self.output_level == 'segment':
            self.xvector.add_module('stats', StatsPool())
            self.xvector.add_module(
                'dense',
                DenseLayer(
                    channels * 2, embedding_size, config_str='batchnorm_'))
        else:
            assert self.output_level == 'frame', '`output_level` should be set to \'segment\' or \'frame\'. '

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        if self.output_level == 'frame':
            x = x.transpose(1, 2)
        return x


@MODELS.register_module(
    Tasks.speaker_verification, module_name=Models.campplus_sv)
class SpeakerVerificationCAMPPlus(TorchModel):
    r"""A fast and efficient speaker embedding model, using a 2-dimensional convolution residual network as the head
    and a densely connected time delay neural network as the backbone.
    Args:
        model_dir: A model dir.
        model_config: The model config.
    """

    def __init__(self, model_dir, model_config: Dict[str, Any], *args,
                 **kwargs):
        super().__init__(model_dir, model_config, *args, **kwargs)
        self.model_config = model_config
        self.other_config = kwargs

        self.feature_dim = self.model_config['fbank_dim']
        self.emb_size = self.model_config['emb_size']
        self.device = create_device(self.other_config['device'])

        self.embedding_model = CAMPPlus(self.feature_dim, self.emb_size)
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
        features = []
        for au in audio:
            feature = Kaldi.fbank(
                au.unsqueeze(0), num_mel_bins=self.feature_dim)
            feature = feature - feature.mean(dim=0, keepdim=True)
            features.append(feature.unsqueeze(0))
        features = torch.cat(features)
        return features

    def __load_check_point(self, pretrained_model_name):
        self.embedding_model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, pretrained_model_name),
                map_location=torch.device('cpu')),
            strict=True)
