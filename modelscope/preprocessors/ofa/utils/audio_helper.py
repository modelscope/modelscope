# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Optional, Tuple, Union

import numpy as np
import torch


def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization

    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """
    try:
        import torchaudio.sox_effects as ta_sox
    except ImportError:
        raise ImportError('Please install torchaudio: pip install torchaudio')

    effects = []
    if normalize_volume:
        effects.append(['gain', '-n'])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(['rate', f'{to_sample_rate}'])
    if to_mono and waveform.shape[0] > 1:
        effects.append(['channels', '1'])
    if len(effects) > 0:
        is_np_input = isinstance(waveform, np.ndarray)
        _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(
            _waveform, sample_rate, effects)
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate
    return waveform, sample_rate


def _get_kaldi_fbank(waveform: np.ndarray,
                     sample_rate: int,
                     n_bins=80) -> Optional[np.ndarray]:
    """Get mel-filter bank features via PyKaldi."""
    try:
        from kaldi.feat.fbank import Fbank, FbankOptions
        from kaldi.feat.mel import MelBanksOptions
        from kaldi.feat.window import FrameExtractionOptions
        from kaldi.matrix import Vector

        mel_opts = MelBanksOptions()
        mel_opts.num_bins = n_bins
        frame_opts = FrameExtractionOptions()
        frame_opts.samp_freq = sample_rate
        opts = FbankOptions()
        opts.mel_opts = mel_opts
        opts.frame_opts = frame_opts
        fbank = Fbank(opts=opts)
        features = fbank.compute(Vector(waveform.squeeze()), 1.0).numpy()
        return features
    except ImportError:
        return None


def _get_torchaudio_fbank(waveform: np.ndarray,
                          sample_rate,
                          n_bins=80) -> Optional[np.ndarray]:
    """Get mel-filter bank features via TorchAudio."""
    try:
        import torchaudio.compliance.kaldi as ta_kaldi

        waveform = torch.from_numpy(waveform)
        features = ta_kaldi.fbank(
            waveform, num_mel_bins=n_bins, sample_frequency=sample_rate)
        return features.numpy()
    except ImportError:
        return None
