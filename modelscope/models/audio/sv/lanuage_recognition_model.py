# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as Kaldi

from modelscope.metainfo import Models
from modelscope.models import MODELS, TorchModel
from modelscope.models.audio.sv.DTDNN import CAMPPlus
from modelscope.models.audio.sv.DTDNN_layers import DenseLayer
from modelscope.utils.constant import Tasks
from modelscope.utils.device import create_device


class LinearClassifier(nn.Module):

    def __init__(
        self,
        input_dim,
        num_blocks=0,
        inter_dim=512,
        out_neurons=1000,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        self.nonlinear = nn.ReLU(inplace=True)
        for _ in range(num_blocks):
            self.blocks.append(DenseLayer(input_dim, inter_dim, bias=True))
            input_dim = inter_dim

        self.linear = nn.Linear(input_dim, out_neurons, bias=True)

    def forward(self, x):
        # x: [B, dim]
        x = self.nonlinear(x)
        for layer in self.blocks:
            x = layer(x)
        x = self.linear(x)
        return x


@MODELS.register_module(
    Tasks.speech_language_recognition, module_name=Models.campplus_lre)
class LanguageRecognitionCAMPPlus(TorchModel):
    r"""A speech language recognition model using the CAM++ architecture as the backbone.
    Args:
        model_dir: A model dir.
        model_config: The model config.
    """

    def __init__(self, model_dir, model_config: Dict[str, Any], *args,
                 **kwargs):
        super().__init__(model_dir, model_config, *args, **kwargs)
        self.model_config = model_config

        self.emb_size = self.model_config['emb_size']
        self.feature_dim = self.model_config['fbank_dim']
        self.sample_rate = self.model_config['sample_rate']
        self.device = create_device(kwargs['device'])

        self.encoder = CAMPPlus(self.feature_dim, self.emb_size)
        self.backend = LinearClassifier(
            input_dim=self.emb_size,
            out_neurons=len(self.model_config['languages']))

        pretrained_encoder = kwargs['pretrained_encoder']
        pretrained_backend = kwargs['pretrained_backend']

        self._load_check_point(pretrained_encoder, pretrained_backend)

        self.encoder.to(self.device)
        self.backend.to(self.device)
        self.encoder.eval()
        self.backend.eval()

    def forward(self, audio):
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        assert len(audio.shape) == 2, \
            'modelscope error: the shape of input audio to model needs to be [N, T]'
        # audio shape: [N, T]
        feature = self._extract_feature(audio)
        embs = self.encoder(feature.to(self.device))
        scores = self.backend(embs).detach()
        output = scores.cpu().argmax(-1)
        return scores, output

    def _extract_feature(self, audio):
        features = []
        for au in audio:
            feature = Kaldi.fbank(
                au.unsqueeze(0),
                num_mel_bins=self.feature_dim,
                sample_frequency=self.sample_rate)
            feature = feature - feature.mean(dim=0, keepdim=True)
            features.append(feature.unsqueeze(0))
        features = torch.cat(features)
        return features

    def _load_check_point(self, pretrained_encoder, pretrained_backend):
        self.encoder.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, pretrained_encoder),
                map_location=torch.device('cpu')))

        self.backend.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, pretrained_backend),
                map_location=torch.device('cpu')))
