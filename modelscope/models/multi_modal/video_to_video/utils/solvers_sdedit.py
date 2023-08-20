# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torchsde
from tqdm.auto import trange


def get_ancestral_step(sigma_from,
                       sigma_to,
                       eta=1.):
    """
    Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step.
    """
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (
        sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2
    ) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def get_scalings(sigma):
    c_out = -sigma
    c_in = 1 / (sigma ** 2 + 1. ** 2) ** 0.5
    return c_out, c_in


@torch.no_grad()
def sample_heun(noise,
                model,
                sigmas,
                s_churn=0.,
                s_tmin=0.,
                s_tmax=float('inf'),
                s_noise=1.,
                show_progress=True):
    """
    Implements Algorithm 2 (Heun steps) from Karras et al. (2022).
    """
    x = noise * sigmas[0]
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        gamma = 0.
        if s_tmin <= sigmas[i] <= s_tmax and sigmas[i] < float('inf'):
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, sigma_hat)
            x = denoised + sigmas[i + 1] * (gamma + 1) * noise
        else:
            _, c_in = get_scalings(sigma_hat)
            denoised = model(x * c_in, sigma_hat)
            d = (x - denoised) / sigma_hat
            dt = sigmas[i + 1] - sigma_hat
            if sigmas[i + 1] == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                _, c_in = get_scalings(sigmas[i + 1])
                denoised_2 = model(x_2 * c_in, sigmas[i + 1])
                d_2 = (x_2 - denoised_2) / sigmas[i + 1]
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
    return x
