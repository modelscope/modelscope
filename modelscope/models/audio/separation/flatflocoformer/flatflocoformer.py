# Copyright (c) Alibaba, Inc. and its affiliates.

import os

import torch

from modelscope.metainfo import Models
from modelscope.models import MODELS, TorchModel
from modelscope.utils.constant import ModelFile, Tasks
from .locoformer_block import FLATFLocoformerSeparator


@MODELS.register_module(
    Tasks.speech_separation,
    module_name=Models.
    speech_flatflocoformer_separation_timefrequency_8k_middle_libri2mix360)
class FLATFLocoformer(TorchModel):
    """FLA-TF-Locoformer speech separation model.

    arxiv: https://arxiv.org/abs/2508.19528
    INTERSPEECH 2025: https://www.isca-archive.org/interspeech_2025/wang25j_interspeech.html

    Wraps the STFT front end, the separator and the iSTFT back end so the model
    consumes and produces waveforms.

    Args:
        model_dir (str): the model path.
        separator_conf (dict): arguments for :class:`FLATFLocoformerSeparator`.
        n_fft (int): STFT window size.
        hop_length (int): STFT hop size.
        normalize_variance (bool): divide the mixture by its standard deviation
            before the front end and restore the scale afterwards.
    """

    def __init__(self,
                 model_dir: str,
                 separator_conf: dict,
                 n_fft: int = 128,
                 hop_length: int = 64,
                 normalize_variance: bool = True,
                 *args,
                 **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.normalize_variance = normalize_variance
        self.separator = FLATFLocoformerSeparator(**separator_conf)
        self.num_spks = self.separator.num_spk

    def forward(self, input):
        """Separate a mixture waveform.

        Args:
            input (torch.Tensor): mixture waveform, [B, L].

        Returns:
            torch.Tensor: separated waveforms, [B, L, num_spks].
        """
        n_samples = input.size(-1)

        if self.normalize_variance:
            mix_std = torch.std(input, dim=1, keepdim=True)
            input = input / mix_std

        window = torch.hann_window(
            self.win_length, dtype=input.dtype, device=input.device)
        spec = torch.stft(
            input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            return_complex=True).transpose(1, 2)  # [B, T, F]

        specs = self.separator(spec)

        waves = [
            torch.istft(
                s.transpose(1, 2),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=True,
                normalized=False,
                onesided=True,
                length=n_samples,
                return_complex=False) for s in specs
        ]

        if self.normalize_variance:
            waves = [w * mix_std for w in waves]

        return torch.stack(waves, dim=-1)

    def load_check_point(self, load_path=None, device=None):
        if not load_path:
            load_path = self.model_dir
        if not device:
            device = torch.device('cpu')
        state_dict = torch.load(
            os.path.join(load_path, ModelFile.TORCH_MODEL_FILE),
            map_location=device,
            weights_only=True)
        self.load_state_dict(state_dict, strict=True)
