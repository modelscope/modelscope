# Copyright (c) Alibaba, Inc. and its affiliates.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelScale


class ModulationDomainLossModule(torch.nn.Module):
    """Modulation-domain loss function developed in [1] for supervised speech enhancement

        In our paper, we used the gabor-based STRF kernels as the modulation kernels and used the log-mel spectrogram
        as the input spectrogram representation.
        Specific parameter details are in the paper and in the example below

        Parameters
        ----------
        modulation_kernels: nn.Module
            Differentiable module that transforms a spectrogram representation to the modulation domain

            modulation_domain = modulation_kernels(input_tf_representation)
            Input Spectrogram representation (B, T, F) ---> |(M) modulation_kernels|--->Modulation Domain(B, M, T', F')

        norm: boolean
            Normalizes the modulation domain representation to be 0 mean across time

        [1] T. Vuong, Y. Xia, and R. M. Stern, “A modulation-domain lossfor neural-network-based real-time
         speech enhancement”
            Accepted ICASSP 2021, https://arxiv.org/abs/2102.07330


    """

    def __init__(self, modulation_kernels, norm=True):
        super(ModulationDomainLossModule, self).__init__()

        self.modulation_kernels = modulation_kernels
        self.mse = nn.MSELoss(reduce=False)
        self.norm = norm

    def forward(self, enhanced_spect, clean_spect, weight=None):
        """Calculate modulation-domain loss
        Args:
            enhanced_spect (Tensor): spectrogram representation of enhanced signal (B, #frames, #freq_channels).
            clean_spect (Tensor): spectrogram representation of clean ground-truth signal (B, #frames, #freq_channels).
        Returns:
            Tensor: Modulation-domain loss value.
        """

        clean_mod = self.modulation_kernels(clean_spect)
        enhanced_mod = self.modulation_kernels(enhanced_spect)

        if self.norm:
            mean_clean_mod = torch.mean(clean_mod, dim=2)
            mean_enhanced_mod = torch.mean(enhanced_mod, dim=2)

            clean_mod = clean_mod - mean_clean_mod.unsqueeze(2)
            enhanced_mod = enhanced_mod - mean_enhanced_mod.unsqueeze(2)

        if weight is None:
            alpha = 1
        else:  # TF-mask weight
            alpha = 1 + torch.sum(weight, dim=-1, keepdim=True).unsqueeze(1)
        mod_mse_loss = self.mse(enhanced_mod, clean_mod) * alpha
        mod_mse_loss = torch.mean(
            torch.sum(mod_mse_loss, dim=(1, 2, 3))
            / torch.sum(clean_mod**2, dim=(1, 2, 3)))

        return mod_mse_loss


class ModulationDomainNCCLossModule(torch.nn.Module):
    """Modulation-domain loss function developed in [1] for supervised speech enhancement

        # Speech Intelligibility Prediction Using Spectro-Temporal Modulation Analysis - based off of this

        In our paper, we used the gabor-based STRF kernels as the modulation kernels and used the log-mel spectrogram
        as the input spectrogram representation.
        Specific parameter details are in the paper and in the example below

        Parameters
        ----------
        modulation_kernels: nn.Module
            Differentiable module that transforms a spectrogram representation to the modulation domain

            modulation_domain = modulation_kernels(input_tf_representation)
            Input Spectrogram representation(B, T, F) --- (M) modulation_kernels---> Modulation Domain(B, M, T', F')

        [1]

    """

    def __init__(self, modulation_kernels):
        super(ModulationDomainNCCLossModule, self).__init__()

        self.modulation_kernels = modulation_kernels
        self.mse = nn.MSELoss(reduce=False)

    def forward(self, enhanced_spect, clean_spect):
        """Calculate modulation-domain loss
        Args:
            enhanced_spect (Tensor): spectrogram representation of enhanced signal (B, #frames, #freq_channels).
            clean_spect (Tensor): spectrogram representation of clean ground-truth signal (B, #frames, #freq_channels).
        Returns:
            Tensor: Modulation-domain loss value.
        """

        clean_mod = self.modulation_kernels(clean_spect)
        enhanced_mod = self.modulation_kernels(enhanced_spect)
        mean_clean_mod = torch.mean(clean_mod, dim=2)
        mean_enhanced_mod = torch.mean(enhanced_mod, dim=2)

        normalized_clean = clean_mod - mean_clean_mod.unsqueeze(2)
        normalized_enhanced = enhanced_mod - mean_enhanced_mod.unsqueeze(2)

        inner_product = torch.sum(
            normalized_clean * normalized_enhanced, dim=2)
        normalized_denom = (torch.sum(
            normalized_clean * normalized_clean, dim=2))**.5 * (torch.sum(
                normalized_enhanced * normalized_enhanced, dim=2))**.5

        ncc = inner_product / normalized_denom
        mod_mse_loss = torch.mean((ncc - 1.0)**2)

        return mod_mse_loss


class GaborSTRFConv(nn.Module):
    """Gabor-STRF-based cross-correlation kernel."""

    def __init__(self,
                 supn,
                 supk,
                 nkern,
                 rates=None,
                 scales=None,
                 norm_strf=True,
                 real_only=False):
        """Instantiate a Gabor-based STRF convolution layer.
        Parameters
        ----------
        supn: int
            Time support in number of frames. Also the window length.
        supk: int
            Frequency support in number of channels. Also the window length.
        nkern: int
            Number of kernels, each with a learnable rate and scale.
        rates: list of float, None
            Initial values for temporal modulation.
        scales: list of float, None
            Initial values for spectral modulation.
        norm_strf: Boolean
            Normalize STRF kernels to be unit length
        real_only: Boolean
            If True, nkern REAL gabor-STRF kernels
            If False, nkern//2 REAL and nkern//2 IMAGINARY gabor-STRF kernels
        """
        super(GaborSTRFConv, self).__init__()
        self.numN = supn
        self.numK = supk
        self.numKern = nkern
        self.real_only = real_only
        self.norm_strf = norm_strf

        if not real_only:
            nkern = nkern // 2

        if supk % 2 == 0:  # force odd number
            supk += 1
        self.supk = torch.arange(supk, dtype=torch.float32)
        if supn % 2 == 0:  # force odd number
            supn += 1
        self.supn = torch.arange(supn, dtype=self.supk.dtype)
        self.padding = (supn // 2, supk // 2)
        # Set up learnable parameters
        # for param in (rates, scales):
        #    assert (not param) or len(param) == nkern
        if not rates:

            rates = torch.rand(nkern) * math.pi / 2.0

        if not scales:

            scales = (torch.rand(nkern) * 2.0 - 1.0) * math.pi / 2.0

        self.rates_ = nn.Parameter(torch.Tensor(rates))
        self.scales_ = nn.Parameter(torch.Tensor(scales))

    def strfs(self):
        """Make STRFs using the current parameters."""

        if self.supn.device != self.rates_.device:  # for first run
            self.supn = self.supn.to(self.rates_.device)
            self.supk = self.supk.to(self.rates_.device)
        n0, k0 = self.padding

        nwind = .5 - .5 * \
            torch.cos(2 * math.pi * (self.supn + 1) / (len(self.supn) + 1))
        kwind = .5 - .5 * \
            torch.cos(2 * math.pi * (self.supk + 1) / (len(self.supk) + 1))

        new_wind = torch.matmul((nwind).unsqueeze(-1), (kwind).unsqueeze(0))

        n_n_0 = self.supn - n0
        k_k_0 = self.supk - k0
        n_mult = torch.matmul(
            n_n_0.unsqueeze(1),
            torch.ones((1, len(self.supk))).type(torch.FloatTensor).to(
                self.rates_.device))
        k_mult = torch.matmul(
            torch.ones((len(self.supn),
                        1)).type(torch.FloatTensor).to(self.rates_.device),
            k_k_0.unsqueeze(0))

        inside = self.rates_.unsqueeze(1).unsqueeze(
            1) * n_mult + self.scales_.unsqueeze(1).unsqueeze(1) * k_mult
        real_strf = torch.cos(inside) * new_wind.unsqueeze(0)

        if self.real_only:
            final_strf = real_strf

        else:
            imag_strf = torch.sin(inside) * new_wind.unsqueeze(0)
            final_strf = torch.cat([real_strf, imag_strf], dim=0)

        if self.norm_strf:
            final_strf = final_strf / (torch.sum(
                final_strf**2, dim=(1, 2)).unsqueeze(1).unsqueeze(2))**.5

        return final_strf

    def forward(self, sigspec):
        """Forward pass a batch of (real) spectra [Batch x Time x Frequency]."""
        if len(sigspec.shape) == 2:  # expand batch dimension if single eg
            sigspec = sigspec.unsqueeze(0)
        strfs = self.strfs().unsqueeze(1).type_as(sigspec)
        out = F.conv2d(sigspec.unsqueeze(1), strfs, padding=self.padding)
        return out

    def __repr__(self):
        """Gabor filter"""
        report = """
            +++++ Gabor Filter Kernels [{}], supn[{}], supk[{}] real only [{}] norm strf [{}] +++++

        """.format(self.numKern, self.numN, self.numK, self.real_only,
                   self.norm_strf)

        return report
