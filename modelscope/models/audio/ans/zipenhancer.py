#!/usr/bin/env python3
#
# Copyright (c) Alibaba, Inc. and its affiliates.


import random

import os
from typing import Dict


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .zipenhancer_layers.generator import DenseEncoder, MappingDecoder, PhaseDecoder
from .zipenhancer_layers.zipenhancer_layer import Zipformer2DualPathEncoder
from .zipenhancer_layers.scaling import (
    ScheduledFloat,
)

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks


@MODELS.register_module(
    Tasks.acoustic_noise_suppression,
    module_name=Models.speech_zipenhancer_ans_multiloss_16k_base)
class ZipenhancerDecorator(TorchModel):
    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)

        h = dict(
            num_tsconformers = kwargs['num_tsconformers'],
            dense_channel = kwargs['dense_channel'],
            former_conf = kwargs['former_conf'],
            batch_first = kwargs['batch_first'],
            model_num_spks = kwargs['model_num_spks'],

        )
        # num_tsconformers, dense_channel, former_name, former_conf, batch_first, model_num_spks
        h = AttrDict(h)
        self.model = ZipEnhancer(h)
        model_bin_file = os.path.join(model_dir,
                                      ModelFile.TORCH_MODEL_BIN_FILE)
        if os.path.exists(model_bin_file):
            checkpoint = torch.load(
                model_bin_file, map_location=torch.device('cpu'))
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # the new trained model by user is based on ZipenhancerDecorator
                self.load_state_dict(checkpoint['state_dict'])
            else:
                # The released model on Modelscope is based on Zipenhancer
                # self.model.load_state_dict(checkpoint, strict=False)
                self.model.load_state_dict(checkpoint['generator'])
                # print(checkpoint['generator'].keys())

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        n_fft = 400
        hop_size = 100
        win_size = 400
        noisy_wav = inputs['noisy']
        norm_factor = torch.sqrt(noisy_wav.shape[1] / torch.sum(noisy_wav ** 2.0))
        noisy_audio = (noisy_wav * norm_factor)

        mag, pha, com = mag_pha_stft(noisy_audio, n_fft, hop_size, win_size, compress_factor=0.3, center=True)
        amp_g, pha_g, com_g, _, others = self.model.forward(mag, pha)
        wav = mag_pha_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor=0.3, center=True)

        wav = wav / norm_factor

        output = {
            'wav_l2': wav,
        }

        return output

class ZipEnhancer(nn.Module):
    def __init__(self, h):
        """
        Initialize the ZipEnhancer module.

        Args:
        h (object): Configuration object containing various hyperparameters and settings.
        having num_tsconformers, former_name, former_conf, mask_decoder_type, ...
        """
        super(ZipEnhancer, self).__init__()
        self.h = h

        num_tsconformers = h.num_tsconformers
        self.num_tscblocks = num_tsconformers
        self.dense_encoder = DenseEncoder(h, in_channel=2)

        self.TSConformer = Zipformer2DualPathEncoder(
            output_downsampling_factor=1,
            dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
            **h.former_conf
        )


        self.mask_decoder = MappingDecoder(h, out_channel=h.model_num_spks)
        self.phase_decoder = PhaseDecoder(h, out_channel=h.model_num_spks)

    def forward(self, noisy_mag, noisy_pha):  # [B, F, T]
        """
        Forward pass of the ZipEnhancer module.

        Args:
        noisy_mag (Tensor): Noisy magnitude input tensor of shape [B, F, T].
        noisy_pha (Tensor): Noisy phase input tensor of shape [B, F, T].

        Returns:
        Tuple: denoised magnitude, denoised phase, denoised complex representation,
               (optional) predicted noise components, and other auxiliary information.
        """
        others = dict()

        noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)


        # [B, C, T, F]
        x = self.TSConformer(x)

        pred_mag = self.mask_decoder(x)
        pred_pha = self.phase_decoder(x)
        # b, c, t, f -> b, 1, t, f -> b, f, t, 1 -> b, f, t
        denoised_mag = pred_mag[:, 0, :, :].unsqueeze(1).permute(0, 3, 2, 1).squeeze(-1)

        # b, t, f
        denoised_pha = pred_pha[:, 0, :, :].unsqueeze(1).permute(0, 3, 2, 1).squeeze(-1)
        # b, t, f
        denoised_com = torch.stack((denoised_mag * torch.cos(denoised_pha),
                                    denoised_mag * torch.sin(denoised_pha)), dim=-1)

        return denoised_mag, denoised_pha, denoised_com, None, others

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    hann_window = torch.hann_window(win_size, device=y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + (1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1], stft_spec[:, :, :, 0] + (1e-5))
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)

    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0 / compress_factor))
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size, device=com.device)

    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)
    return wav