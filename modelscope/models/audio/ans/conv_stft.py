# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)**0.5

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(
        window[None, :, None].astype(np.float32))


class ConvSTFT(nn.Module):

    def __init__(self,
                 win_len,
                 win_inc,
                 fft_len=None,
                 win_type='hamming',
                 feature_type='real',
                 fix=True):
        super(ConvSTFT, self).__init__()

        if fft_len is None:
            self.fft_len = int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)

        outputs = F.conv1d(inputs, self.weight, stride=self.stride)

        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real**2 + imag**2)
            phase = torch.atan2(imag, real)
            return mags, phase


class ConviSTFT(nn.Module):

    def __init__(self,
                 win_len,
                 win_inc,
                 fft_len=None,
                 win_type='hamming',
                 feature_type='real',
                 fix=True):
        super(ConviSTFT, self).__init__()
        if fft_len is None:
            self.fft_len = int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(
            win_len, win_inc, self.fft_len, win_type, invers=True)
        self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.win_inc = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:, None, :])

    def forward(self, inputs, phase=None):
        """
        Args:
            inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
            phase: [B, N//2+1, T] (if not none)
        """

        if phase is not None:
            real = inputs * torch.cos(phase)
            imag = inputs * torch.sin(phase)
            inputs = torch.cat([real, imag], 1)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1))**2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs / (coff + 1e-8)
        return outputs
