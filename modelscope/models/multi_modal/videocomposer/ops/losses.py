# Copyright (c) Alibaba, Inc. and its affiliates.

import math

import torch

__all__ = ['kl_divergence', 'discretized_gaussian_log_likelihood']


def kl_divergence(mu1, logvar1, mu2, logvar2):
    return 0.5 * (
        -1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) +  # noqa
        ((mu1 - mu2)**2) * torch.exp(-logvar2))


def standard_normal_cdf(x):
    r"""A fast approximation of the cumulative distribution function of the standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x0, mean, log_scale):
    assert x0.shape == mean.shape == log_scale.shape
    cx = x0 - mean
    inv_stdv = torch.exp(-log_scale)
    cdf_plus = standard_normal_cdf(inv_stdv * (cx + 1.0 / 255.0))
    cdf_min = standard_normal_cdf(inv_stdv * (cx - 1.0 / 255.0))
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x0 < -0.999, log_cdf_plus,
        torch.where(x0 > 0.999, log_one_minus_cdf_min,
                    torch.log(cdf_delta.clamp(min=1e-12))))
    assert log_probs.shape == x0.shape
    return log_probs
