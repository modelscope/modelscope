from collections import OrderedDict

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi

from .layers import (BasicResBlock, CAMDenseTDNNBlock, DenseLayer, StatsPool,
                     TDNNLayer, TransitLayer, get_nonlinear)


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
                 memory_efficient=True):
        super(CAMPPlus, self).__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

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
                memory_efficient=memory_efficient,
            )
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

        self.xvector.add_module('stats', StatsPool())
        self.xvector.add_module(
            'dense',
            DenseLayer(channels * 2, embedding_size, config_str='batchnorm_'))

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        return x


class SpeakerVerificationCamplus:
    r"""Enhanced Res2Net_aug architecture with local and global feature fusion.
    ERes2Net_aug is an upgraded version of ERes2Net that uses a larger
    parameters to achieve better recognition performance.
    Args:
        model_dir: A model dir.
        model_config: The model config.
    """

    def __init__(self, pretrained_model_name, device='cpu', *args, **kwargs):
        super().__init__()

        self.feature_dim = 80
        self.device = torch.device(device)
        self.embedding_model = CAMPPlus(embedding_size=192)

        self.__load_check_point(pretrained_model_name)

        self.embedding_model.to(self.device)
        self.embedding_model.eval()

    def forward(self, audio):
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        elif isinstance(audio, str):
            audio = librosa.load(audio, sr=16000)[0]
            audio = torch.from_numpy(audio)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        elif len(audio.shape) == 3:
            audio = audio.squeeze(1)
        assert len(
            audio.shape
        ) == 2, 'modelscope error: the shape of input audio to model needs to'
        # audio shape: [N, T]
        feature = self.__extract_feature(audio)
        embedding = self.embedding_model(feature.to(self.device))

        return embedding

    def inference(self, feature):
        feature = feature - feature.mean(dim=1, keepdim=True)
        embedding = self.embedding_model(feature.to(self.device))

        return embedding

    def __extract_feature(self, audio):
        B = audio.size(0)

        feature = Kaldi.fbank(
            audio.flatten().unsqueeze(0), num_mel_bins=self.feature_dim)
        # print(feature.shape)

        feature = feature - feature.mean(dim=0, keepdim=True)
        pad = torch.zeros([2, self.feature_dim], device=feature.device)
        feature = torch.cat([feature, pad], dim=0)
        feature = feature.reshape([B, -1, self.feature_dim])
        return feature

    def __load_check_point(self, pretrained_model_name, device=None):
        if not device:
            device = torch.device('cpu')
        self.embedding_model.load_state_dict(
            torch.load(pretrained_model_name, map_location=device),
            strict=True)
