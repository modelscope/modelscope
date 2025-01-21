# Copyright (c) Alibaba, Inc. and its affiliates.

import math

import torch


def betas_to_sigmas(betas):
    return torch.sqrt(1 - torch.cumprod(1 - betas, dim=0))


def sigmas_to_betas(sigmas):
    square_alphas = 1 - sigmas**2
    betas = 1 - torch.cat(
        [square_alphas[:1], square_alphas[1:] / square_alphas[:-1]])
    return betas


def logsnrs_to_sigmas(logsnrs):
    return torch.sqrt(torch.sigmoid(-logsnrs))


def sigmas_to_logsnrs(sigmas):
    square_sigmas = sigmas**2
    return torch.log(square_sigmas / (1 - square_sigmas))


def _logsnr_cosine(n, logsnr_min=-15, logsnr_max=15):
    t_min = math.atan(math.exp(-0.5 * logsnr_min))
    t_max = math.atan(math.exp(-0.5 * logsnr_max))
    t = torch.linspace(1, 0, n)
    logsnrs = -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))
    return logsnrs


def _logsnr_cosine_shifted(n, logsnr_min=-15, logsnr_max=15, scale=2):
    logsnrs = _logsnr_cosine(n, logsnr_min, logsnr_max)
    logsnrs += 2 * math.log(1 / scale)
    return logsnrs


def _logsnr_cosine_interp(n,
                          logsnr_min=-15,
                          logsnr_max=15,
                          scale_min=2,
                          scale_max=4):
    t = torch.linspace(1, 0, n)
    logsnrs_min = _logsnr_cosine_shifted(n, logsnr_min, logsnr_max, scale_min)
    logsnrs_max = _logsnr_cosine_shifted(n, logsnr_min, logsnr_max, scale_max)
    logsnrs = t * logsnrs_min + (1 - t) * logsnrs_max
    return logsnrs


def karras_schedule(n, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    ramp = torch.linspace(1, 0, n)
    min_inv_rho = sigma_min**(1 / rho)
    max_inv_rho = sigma_max**(1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho))**rho
    sigmas = torch.sqrt(sigmas**2 / (1 + sigmas**2))
    return sigmas


def logsnr_cosine_interp_schedule(n,
                                  logsnr_min=-15,
                                  logsnr_max=15,
                                  scale_min=2,
                                  scale_max=4):
    return logsnrs_to_sigmas(
        _logsnr_cosine_interp(n, logsnr_min, logsnr_max, scale_min, scale_max))


def noise_schedule(schedule='logsnr_cosine_interp',
                   n=1000,
                   zero_terminal_snr=False,
                   **kwargs):
    # compute sigmas
    sigmas = {
        'logsnr_cosine_interp': logsnr_cosine_interp_schedule
    }[schedule](n, **kwargs)

    # post-processing
    if zero_terminal_snr and sigmas.max() != 1.0:
        scale = (1.0 - sigmas.min()) / (sigmas.max() - sigmas.min())
        sigmas = sigmas.min() + scale * (sigmas - sigmas.min())
    return sigmas
