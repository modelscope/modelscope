# Copyright (c) Alibaba, Inc. and its affiliates.
"""
    This TDNN implementation is adapted from https://github.com/wenet-e2e/wespeaker.
    TDNN replaces i-vectors for text-independent speaker verification with embeddings
    extracted from a feedforward deep neural network. The specific structure can be
    referred to in https://www.danielpovey.com/files/2017_interspeech_embeddings.pdf.
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


class TdnnLayer(nn.Module):

    def __init__(self, in_dim, out_dim, context_size, dilation=1, padding=0):
        """Define the TDNN layer, essentially 1-D convolution

        Args:
            in_dim (int): input dimension
            out_dim (int): output channels
            context_size (int): context size, essentially the filter size
            dilation (int, optional):  Defaults to 1.
            padding (int, optional):  Defaults to 0.
        """
        super(TdnnLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.context_size = context_size
        self.dilation = dilation
        self.padding = padding
        self.conv_1d = nn.Conv1d(
            self.in_dim,
            self.out_dim,
            self.context_size,
            dilation=self.dilation,
            padding=self.padding)

        # Set Affine=false to be compatible with the original kaldi version
        self.bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        out = self.conv_1d(x)
        out = F.relu(out)
        out = self.bn(out)
        return out


class XVEC(nn.Module):

    def __init__(self,
                 feat_dim=40,
                 hid_dim=512,
                 stats_dim=1500,
                 embed_dim=512,
                 pooling_func='TSTP'):
        """
        Implementation of Kaldi style xvec, as described in
        X-VECTORS: ROBUST DNN EMBEDDINGS FOR SPEAKER RECOGNITION
        """
        super(XVEC, self).__init__()
        self.feat_dim = feat_dim
        self.stats_dim = stats_dim
        self.embed_dim = embed_dim

        self.frame_1 = TdnnLayer(feat_dim, hid_dim, context_size=5, dilation=1)
        self.frame_2 = TdnnLayer(hid_dim, hid_dim, context_size=3, dilation=2)
        self.frame_3 = TdnnLayer(hid_dim, hid_dim, context_size=3, dilation=3)
        self.frame_4 = TdnnLayer(hid_dim, hid_dim, context_size=1, dilation=1)
        self.frame_5 = TdnnLayer(
            hid_dim, stats_dim, context_size=1, dilation=1)
        self.n_stats = 1 if pooling_func == 'TAP' or pooling_func == 'TSDP' else 2
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim)
        self.seg_1 = nn.Linear(self.stats_dim * self.n_stats, embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) -> (B,F,T)

        out = self.frame_1(x)
        out = self.frame_2(out)
        out = self.frame_3(out)
        out = self.frame_4(out)
        out = self.frame_5(out)

        stats = self.pool(out)
        embed_a = self.seg_1(stats)
        return embed_a


@MODELS.register_module(Tasks.speaker_verification, module_name=Models.tdnn_sv)
class SpeakerVerificationTDNN(TorchModel):

    def __init__(self, model_dir, model_config: Dict[str, Any], *args,
                 **kwargs):
        super().__init__(model_dir, model_config, *args, **kwargs)
        self.model_config = model_config
        self.other_config = kwargs

        self.feature_dim = 80
        self.embed_dim = 512
        self.device = create_device(self.other_config['device'])
        print(self.device)

        self.embedding_model = XVEC(
            feat_dim=self.feature_dim, embed_dim=self.embed_dim)
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
