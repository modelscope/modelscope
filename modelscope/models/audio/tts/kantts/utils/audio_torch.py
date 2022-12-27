# Copyright (c) Alibaba, Inc. and its affiliates.

from distutils.version import LooseVersion

import librosa
import torch

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion('1.7')


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    if is_pytorch_17plus:
        x_stft = torch.stft(
            x, fft_size, hop_size, win_length, window, return_complex=False)
    else:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return 20 * torch.log10(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.pow(10.0, x * 0.05) / C


def spectral_normalize_torch(
    magnitudes,
    min_level_db=-100.0,
    ref_level_db=20.0,
    norm_abs_value=4.0,
    symmetric=True,
):
    output = dynamic_range_compression_torch(magnitudes) - ref_level_db

    if symmetric:
        return torch.clamp(
            2 * norm_abs_value * ((output - min_level_db) /  # noqa W504
                                  (-min_level_db)) - norm_abs_value,
            min=-norm_abs_value,
            max=norm_abs_value)
    else:
        return torch.clamp(
            norm_abs_value * ((output - min_level_db) / (-min_level_db)),
            min=0.0,
            max=norm_abs_value)


def spectral_de_normalize_torch(
    magnitudes,
    min_level_db=-100.0,
    ref_level_db=20.0,
    norm_abs_value=4.0,
    symmetric=True,
):
    if symmetric:
        magnitudes = torch.clamp(
            magnitudes, min=-norm_abs_value, max=norm_abs_value)
        magnitudes = (magnitudes + norm_abs_value) * (-min_level_db) / (
            2 * norm_abs_value) + min_level_db
    else:
        magnitudes = torch.clamp(magnitudes, min=0.0, max=norm_abs_value)
        magnitudes = (magnitudes) * (-min_level_db) / (
            norm_abs_value) + min_level_db

    output = dynamic_range_decompression_torch(magnitudes + ref_level_db)
    return output


class MelSpectrogram(torch.nn.Module):
    """Calculate Mel-spectrogram."""

    def __init__(
        self,
        fs=22050,
        fft_size=1024,
        hop_size=256,
        win_length=None,
        window='hann',
        num_mels=80,
        fmin=80,
        fmax=7600,
        center=True,
        normalized=False,
        onesided=True,
        eps=1e-10,
        log_base=10.0,
        pad_mode='constant',
    ):
        """Initialize MelSpectrogram module."""
        super().__init__()
        self.fft_size = fft_size
        if win_length is None:
            self.win_length = fft_size
        else:
            self.win_length = win_length
        self.hop_size = hop_size
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window is not None and not hasattr(torch, f'{window}_window'):
            raise ValueError(f'{window} window is not implemented')
        self.window = window
        self.eps = eps
        self.pad_mode = pad_mode

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        melmat = librosa.filters.mel(
            sr=fs,
            n_fft=fft_size,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        self.register_buffer('melmat', torch.from_numpy(melmat.T).float())
        self.stft_params = {
            'n_fft': self.fft_size,
            'win_length': self.win_length,
            'hop_length': self.hop_size,
            'center': self.center,
            'normalized': self.normalized,
            'onesided': self.onesided,
            'pad_mode': self.pad_mode,
        }
        if is_pytorch_17plus:
            self.stft_params['return_complex'] = False

        self.log_base = log_base
        if self.log_base is None:
            self.log = torch.log
        elif self.log_base == 2.0:
            self.log = torch.log2
        elif self.log_base == 10.0:
            self.log = torch.log10
        else:
            raise ValueError(f'log_base: {log_base} is not supported.')

    def forward(self, x):
        """Calculate Mel-spectrogram.

        Args:
            x (Tensor): Input waveform tensor (B, T) or (B, 1, T).

        Returns:
            Tensor: Mel-spectrogram (B, #mels, #frames).

        """
        if x.dim() == 3:
            # (B, C, T) -> (B*C, T)
            x = x.reshape(-1, x.size(2))

        if self.window is not None:
            window_func = getattr(torch, f'{self.window}_window')
            window = window_func(
                self.win_length, dtype=x.dtype, device=x.device)
        else:
            window = None

        x_stft = torch.stft(x, window=window, **self.stft_params)
        # (B, #freqs, #frames, 2) -> (B, $frames, #freqs, 2)
        x_stft = x_stft.transpose(1, 2)
        x_power = x_stft[..., 0]**2 + x_stft[..., 1]**2
        x_amp = torch.sqrt(torch.clamp(x_power, min=self.eps))

        x_mel = torch.matmul(x_amp, self.melmat)
        x_mel = torch.clamp(x_mel, min=self.eps)
        x_mel = spectral_normalize_torch(x_mel)

        #  return self.log(x_mel).transpose(1, 2)
        return x_mel.transpose(1, 2)
