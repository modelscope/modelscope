# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Dict
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from .models.hifigan import HiFiGANGenerator
from .models.Unet import MaskMapping


@MODELS.register_module(Tasks.speech_super_resolution, module_name=Models.hifissr)
class HifiSSR(TorchModel):
    r"""A decorator of FRCRN for integrating into modelscope framework"""

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the frcrn model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        self.device=kwargs.get('device', 'cpu')
        self.front = Spectrogram(512, 512, int(48000 * 0.01)).to(self.device)
        self.vocoder = HiFiGANGenerator(
            input_channels=256, upsample_rates=[5, 4, 4, 3, 2], upsample_kernel_sizes=[10, 8, 8, 6, 4], weight_norm=False, upsample_initial_channel=1024
        ).to(self.device)
        self.mapping = MaskMapping(32, 256).to(self.device)
        model_bin_file = os.path.join(model_dir, "checkpoint.pt")
        if os.path.exists(model_bin_file):
            checkpoint = torch.load(model_bin_file, map_location=self.device)
            self.vocoder.load_state_dict(checkpoint["voc_state_dict"])
            self.vocoder.eval()
            self.mapping.load_state_dict(checkpoint["unet_state_dict"])
            self.mapping.eval()

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        ref_fp = inputs["ref_wav"]
        source_fp = inputs["source_wav"]
        out_fp = inputs["out_wav"]
        sr = 48000
        wav = librosa.load(source_fp, sr=sr)[0]
        source_mel = self.front(torch.FloatTensor(wav).unsqueeze(0).to(self.device))[:, :-1]
        source_mel = torch.log10(source_mel + 1e-6)
        source_mel = source_mel.unsqueeze(0)
        ref_wav = librosa.load(ref_fp, sr=sr)[0]
        ref_mel = self.front(torch.FloatTensor(ref_wav).unsqueeze(0).to(self.device))[:, :-1]
        ref_mel = torch.log10(ref_mel + 1e-6)
        with torch.no_grad():
            g_out = self.mapping(source_mel, ref_mel)
            g_out_wav = self.vocoder(g_out)
            g_out_wav = g_out_wav.flatten()
        sf.write(out_fp, g_out_wav.cpu().data.numpy(), sr)
        return g_out_wav.cpu().data.numpy()
