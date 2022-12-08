# Part of the implementation is borrowed and modified from latent-diffusion,
# publicly avaialbe at https://github.com/CompVis/latent-diffusion.
# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import math

import torch

from modelscope.models.multi_modal.dpm_solver_pytorch import (
    DPM_Solver, NoiseScheduleVP, model_wrapper, model_wrapper_guided_diffusion)

__all__ = ['GaussianDiffusion', 'beta_schedule']


def kl_divergence(mu1, logvar1, mu2, logvar2):
    u1 = -1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
    u2 = ((mu1 - mu2)**2) * torch.exp(-logvar2)
    return 0.5 * (u1 + u2)


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


def _i(tensor, t, x):
    r"""Index tensor using t and format the output according to x.
    """
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    return tensor[t].view(shape).to(x)


def beta_schedule(schedule,
                  num_timesteps=1000,
                  init_beta=None,
                  last_beta=None):
    if schedule == 'linear':
        scale = 1000.0 / num_timesteps
        init_beta = init_beta or scale * 0.0001
        last_beta = last_beta or scale * 0.02
        return torch.linspace(
            init_beta, last_beta, num_timesteps, dtype=torch.float64)
    elif schedule == 'quadratic':
        init_beta = init_beta or 0.0015
        last_beta = last_beta or 0.0195
        return torch.linspace(
            init_beta**0.5, last_beta**0.5, num_timesteps,
            dtype=torch.float64)**2
    elif schedule == 'cosine':
        betas = []
        for step in range(num_timesteps):
            t1 = step / num_timesteps
            t2 = (step + 1) / num_timesteps
            fn_t1 = math.cos((t1 + 0.008) / 1.008 * math.pi / 2)**2
            fn_t2 = math.cos((t2 + 0.008) / 1.008 * math.pi / 2)**2
            betas.append(min(1.0 - fn_t2 / fn_t1, 0.999))
        return torch.tensor(betas, dtype=torch.float64)
    else:
        raise ValueError(f'Unsupported schedule: {schedule}')


class GaussianDiffusion(object):

    def __init__(self,
                 betas,
                 mean_type='eps',
                 var_type='learned_range',
                 loss_type='mse',
                 rescale_timesteps=False):
        # check input
        if not isinstance(betas, torch.DoubleTensor):
            betas = torch.tensor(betas, dtype=torch.float64)
        assert min(betas) > 0 and max(betas) <= 1
        assert mean_type in ['x0', 'x_{t-1}', 'eps']
        assert var_type in [
            'learned', 'learned_range', 'fixed_large', 'fixed_small'
        ]
        assert loss_type in [
            'mse', 'rescaled_mse', 'kl', 'rescaled_kl', 'l1', 'rescaled_l1'
        ]
        self.betas = betas
        self.num_timesteps = len(betas)
        self.mean_type = mean_type
        self.var_type = var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # alphas
        alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [alphas.new_ones([1]), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat(
            [self.alphas_cumprod[1:],
             alphas.new_zeros([1])])

        # q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0
                                                        - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0
                                                      - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod
                                                      - 1)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (
            1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(
            self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (
            1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (
                1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        r"""Sample from q(x_t | x_0).
        """
        noise = torch.randn_like(x0) if noise is None else noise
        u1 = _i(self.sqrt_alphas_cumprod, t, x0) * x0
        u2 = _i(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise
        return u1 + u2

    def q_mean_variance(self, x0, t):
        r"""Distribution of q(x_t | x_0).
        """
        mu = _i(self.sqrt_alphas_cumprod, t, x0) * x0
        var = _i(1.0 - self.alphas_cumprod, t, x0)
        log_var = _i(self.log_one_minus_alphas_cumprod, t, x0)
        return mu, var, log_var

    def q_posterior_mean_variance(self, x0, xt, t):
        r"""Distribution of q(x_{t-1} | x_t, x_0).
        """
        mu = _i(self.posterior_mean_coef1, t, xt) * x0 + _i(
            self.posterior_mean_coef2, t, xt) * xt
        var = _i(self.posterior_variance, t, xt)
        log_var = _i(self.posterior_log_variance_clipped, t, xt)
        return mu, var, log_var

    @torch.no_grad()
    def p_sample(self,
                 xt,
                 t,
                 model,
                 model_kwargs={},
                 clamp=None,
                 percentile=None,
                 condition_fn=None,
                 guide_scale=None):
        r"""Sample from p(x_{t-1} | x_t).
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        # predict distribution of p(x_{t-1} | x_t)
        mu, var, log_var, x0 = self.p_mean_variance(xt, t, model, model_kwargs,
                                                    clamp, percentile,
                                                    guide_scale)

        # random sample (with optional conditional function)
        noise = torch.randn_like(xt)
        shape = (-1, *((1, ) * (xt.ndim - 1)))
        mask = t.ne(0).float().view(shape)  # no noise when t == 0
        if condition_fn is not None:
            grad = condition_fn(xt, self._scale_timesteps(t), **model_kwargs)
            mu = mu.float() + var * grad.float()
        xt_1 = mu + mask * torch.exp(0.5 * log_var) * noise
        return xt_1, x0

    @torch.no_grad()
    def p_sample_loop(self,
                      noise,
                      model,
                      model_kwargs={},
                      clamp=None,
                      percentile=None,
                      condition_fn=None,
                      guide_scale=None):
        r"""Sample from p(x_{t-1} | x_t) p(x_{t-2} | x_{t-1}) ... p(x_0 | x_1).
        """
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process
        for step in torch.arange(self.num_timesteps).flip(0):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.p_sample(xt, t, model, model_kwargs, clamp,
                                  percentile, condition_fn, guide_scale)
        return xt

    def p_mean_variance(self,
                        xt,
                        t,
                        model,
                        model_kwargs={},
                        clamp=None,
                        percentile=None,
                        guide_scale=None):
        r"""Distribution of p(x_{t-1} | x_t).
        """
        # predict distribution
        if guide_scale is None:
            out = model(xt, self._scale_timesteps(t), **model_kwargs)
        else:
            # classifier-free guidance
            # (model_kwargs[0]: conditional kwargs; model_kwargs[1]: non-conditional kwargs)
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            y_out = model(xt, self._scale_timesteps(t), **model_kwargs[0])
            u_out = model(xt, self._scale_timesteps(t), **model_kwargs[1])
            cond = self.var_type.startswith('fixed')
            dim = y_out.size(1) if cond else y_out.size(1) // 2
            u1 = u_out[:, :dim]
            u2 = guide_scale * (y_out[:, :dim] - u_out[:, :dim])
            out = torch.cat([u1 + u2, y_out[:, dim:]], dim=1)

        # compute variance
        if self.var_type == 'learned':
            out, log_var = out.chunk(2, dim=1)
            var = torch.exp(log_var)
        elif self.var_type == 'learned_range':
            out, fraction = out.chunk(2, dim=1)
            min_log_var = _i(self.posterior_log_variance_clipped, t, xt)
            max_log_var = _i(torch.log(self.betas), t, xt)
            fraction = (fraction + 1) / 2.0
            log_var = fraction * max_log_var + (1 - fraction) * min_log_var
            var = torch.exp(log_var)
        elif self.var_type == 'fixed_large':
            var = _i(
                torch.cat([self.posterior_variance[1:2], self.betas[1:]]), t,
                xt)
            log_var = torch.log(var)
        elif self.var_type == 'fixed_small':
            var = _i(self.posterior_variance, t, xt)
            log_var = _i(self.posterior_log_variance_clipped, t, xt)

        # compute mean and x0
        if self.mean_type == 'x_{t-1}':
            mu = out  # x_{t-1}
            u1 = _i(1.0 / self.posterior_mean_coef1, t, xt) * mu
            u2 = _i(self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                    xt) * xt
            x0 = u1 - u2
        elif self.mean_type == 'x0':
            x0 = out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        elif self.mean_type == 'eps':
            u1 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt
            u2 = _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out
            x0 = u1 - u2
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)

        # restrict the range of x0
        if percentile is not None:
            assert percentile > 0 and percentile <= 1  # e.g., 0.995
            s = torch.quantile(
                x0.flatten(1).abs(), percentile,
                dim=1).clamp_(1.0).view(-1, 1, 1, 1)
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)
        return mu, var, log_var, x0

    @torch.no_grad()
    def dpm_solver_sample_loop(self,
                               noise,
                               model,
                               skip_type,
                               order,
                               method,
                               model_kwargs={},
                               clamp=None,
                               percentile=None,
                               condition_fn=None,
                               guide_scale=None,
                               dpm_solver_timesteps=20,
                               t_start=None,
                               t_end=None,
                               lower_order_final=True,
                               denoise_to_zero=False,
                               solver_type='dpm_solver'):
        r"""Sample using DPM-Solver-based method.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
            Please check all the parameters in `dpm_solver.sample` before using.
        """
        noise_schedule = NoiseScheduleVP(
            schedule='discrete', betas=self.betas.float())
        model_fn = model_wrapper_guided_diffusion(
            model=model,
            noise_schedule=noise_schedule,
            var_type=self.var_type,
            mean_type=self.mean_type,
            model_kwargs=model_kwargs,
            clamp=clamp,
            percentile=percentile,
            rescale_timesteps=self.rescale_timesteps,
            num_timesteps=self.num_timesteps,
            guide_scale=guide_scale,
            condition_fn=condition_fn,
        )
        dpm_solver = DPM_Solver(
            model_fn=model_fn,
            noise_schedule=noise_schedule,
        )
        xt = dpm_solver.sample(
            noise,
            steps=dpm_solver_timesteps,
            order=order,
            skip_type=skip_type,
            method=method,
            solver_type=solver_type,
            t_start=t_start,
            t_end=t_end,
            lower_order_final=lower_order_final,
            denoise_to_zero=denoise_to_zero)
        return xt

    @torch.no_grad()
    def ddim_sample(self,
                    xt,
                    t,
                    model,
                    model_kwargs={},
                    clamp=None,
                    percentile=None,
                    condition_fn=None,
                    guide_scale=None,
                    ddim_timesteps=20,
                    eta=0.0):
        r"""Sample from p(x_{t-1} | x_t) using DDIM.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // ddim_timesteps

        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp,
                                           percentile, guide_scale)
        if condition_fn is not None:
            # x0 -> eps
            alpha = _i(self.alphas_cumprod, t, xt)
            u1 = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0)
            u2 = _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            eps = u1 / u2
            eps = eps - (1 - alpha).sqrt() * condition_fn(
                xt, self._scale_timesteps(t), **model_kwargs)

            # eps -> x0
            u1 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt
            u2 = _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
            x0 = u1 - u2

        # derive variables
        u1 = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0)
        u2 = _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        eps = u1 / u2
        alphas = _i(self.alphas_cumprod, t, xt)
        alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
        u1 = (1 - alphas_prev) / (1 - alphas)
        u2 = (1 - alphas / alphas_prev)
        sigmas = eta * torch.sqrt(u1 * u2)

        # random sample
        noise = torch.randn_like(xt)
        direction = torch.sqrt(1 - alphas_prev - sigmas**2) * eps
        mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
        xt_1 = torch.sqrt(alphas_prev) * x0 + direction + mask * sigmas * noise
        return xt_1, x0

    @torch.no_grad()
    def ddim_sample_loop(self,
                         noise,
                         model,
                         model_kwargs={},
                         clamp=None,
                         percentile=None,
                         condition_fn=None,
                         guide_scale=None,
                         ddim_timesteps=20,
                         eta=0.0):
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process (TODO: clamp is inaccurate! Consider replacing the stride by explicit prev/next steps)
        steps = (1 + torch.arange(0, self.num_timesteps,
                                  self.num_timesteps // ddim_timesteps)).clamp(
                                      0, self.num_timesteps - 1).flip(0)
        for step in steps:
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.ddim_sample(xt, t, model, model_kwargs, clamp,
                                     percentile, condition_fn, guide_scale,
                                     ddim_timesteps, eta)
        return xt

    @torch.no_grad()
    def ddim_reverse_sample(self,
                            xt,
                            t,
                            model,
                            model_kwargs={},
                            clamp=None,
                            percentile=None,
                            guide_scale=None,
                            ddim_timesteps=20):
        r"""Sample from p(x_{t+1} | x_t) using DDIM reverse ODE (deterministic).
        """
        stride = self.num_timesteps // ddim_timesteps

        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp,
                                           percentile, guide_scale)

        # derive variables
        u1 = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0)
        u2 = _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        eps = u1 / u2

        alphas_next = _i(
            torch.cat(
                [self.alphas_cumprod,
                 self.alphas_cumprod.new_zeros([1])]),
            (t + stride).clamp(0, self.num_timesteps), xt)

        # reverse sample
        mu = torch.sqrt(alphas_next) * x0 + torch.sqrt(1 - alphas_next) * eps
        return mu, x0

    @torch.no_grad()
    def ddim_reverse_sample_loop(self,
                                 x0,
                                 model,
                                 model_kwargs={},
                                 clamp=None,
                                 percentile=None,
                                 guide_scale=None,
                                 ddim_timesteps=20):
        # prepare input
        b = x0.size(0)
        xt = x0

        # reconstruction steps
        steps = torch.arange(0, self.num_timesteps,
                             self.num_timesteps // ddim_timesteps)
        for step in steps:
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.ddim_reverse_sample(xt, t, model, model_kwargs, clamp,
                                             percentile, guide_scale,
                                             ddim_timesteps)
        return xt

    @torch.no_grad()
    def plms_sample(self,
                    xt,
                    t,
                    model,
                    model_kwargs={},
                    clamp=None,
                    percentile=None,
                    condition_fn=None,
                    guide_scale=None,
                    plms_timesteps=20):
        r"""Sample from p(x_{t-1} | x_t) using PLMS.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // plms_timesteps

        # function for compute eps
        def compute_eps(xt, t):
            # predict distribution of p(x_{t-1} | x_t)
            _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs,
                                               clamp, percentile, guide_scale)

            # condition
            if condition_fn is not None:
                # x0 -> eps
                alpha = _i(self.alphas_cumprod, t, xt)
                u1 = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0)
                u2 = _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
                eps = u1 / u2
                eps = eps - (1 - alpha).sqrt() * condition_fn(
                    xt, self._scale_timesteps(t), **model_kwargs)

                # eps -> x0
                u1 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt
                u2 = _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
                x0 = u1 - u2

            # derive eps
            u1 = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0)
            u2 = _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            eps = u1 / u2
            return eps

        # function for compute x_0 and x_{t-1}
        def compute_x0(eps, t):
            # eps -> x0
            u1 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt
            u2 = _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
            x0 = u1 - u2

            # deterministic sample
            alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
            direction = torch.sqrt(1 - alphas_prev) * eps
            # mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
            xt_1 = torch.sqrt(alphas_prev) * x0 + direction
            return xt_1, x0

        # PLMS sample
        eps = compute_eps(xt, t)
        if len(eps_cache) == 0:
            # 2nd order pseudo improved Euler
            xt_1, x0 = compute_x0(eps, t)
            eps_next = compute_eps(xt_1, (t - stride).clamp(0))
            eps_prime = (eps + eps_next) / 2.0
        elif len(eps_cache) == 1:
            # 2nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (3 * eps - eps_cache[-1]) / 2.0
        elif len(eps_cache) == 2:
            # 3nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (23 * eps - 16 * eps_cache[-1]
                         + 5 * eps_cache[-2]) / 12.0
        elif len(eps_cache) >= 3:
            # 4nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (55 * eps - 59 * eps_cache[-1] + 37 * eps_cache[-2]
                         - 9 * eps_cache[-3]) / 24.0
        xt_1, x0 = compute_x0(eps_prime, t)
        return xt_1, x0, eps

    @torch.no_grad()
    def plms_sample_loop(self,
                         noise,
                         model,
                         model_kwargs={},
                         clamp=None,
                         percentile=None,
                         condition_fn=None,
                         guide_scale=None,
                         plms_timesteps=20):
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process
        steps = (1 + torch.arange(0, self.num_timesteps,
                                  self.num_timesteps // plms_timesteps)).clamp(
                                      0, self.num_timesteps - 1).flip(0)
        eps_cache = []
        for step in steps:
            # PLMS sampling step
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _, eps = self.plms_sample(xt, t, model, model_kwargs, clamp,
                                          percentile, condition_fn,
                                          guide_scale, plms_timesteps,
                                          eps_cache)

            # update eps cache
            eps_cache.append(eps)
            if len(eps_cache) >= 4:
                eps_cache.pop(0)
        return xt

    def loss(self, x0, t, model, model_kwargs={}, noise=None, input_x0=None):
        noise = torch.randn_like(x0) if noise is None else noise
        input_x0 = x0 if input_x0 is None else input_x0
        xt = self.q_sample(input_x0, t, noise=noise)

        # compute loss
        if self.loss_type in ['kl', 'rescaled_kl']:
            loss, _ = self.variational_lower_bound(x0, xt, t, model,
                                                   model_kwargs)
            if self.loss_type == 'rescaled_kl':
                loss = loss * self.num_timesteps
        elif self.loss_type in ['mse', 'rescaled_mse', 'l1', 'rescaled_l1']:
            out = model(xt, self._scale_timesteps(t), **model_kwargs)

            # VLB for variation
            loss_vlb = 0.0
            if self.var_type in ['learned', 'learned_range']:
                out, var = out.chunk(2, dim=1)
                frozen = torch.cat([
                    out.detach(), var
                ], dim=1)  # learn var without affecting the prediction of mean
                loss_vlb, _ = self.variational_lower_bound(
                    x0, xt, t, model=lambda *args, **kwargs: frozen)
                if self.loss_type.startswith('rescaled_'):
                    loss_vlb = loss_vlb * self.num_timesteps / 1000.0

            # MSE/L1 for x0/eps
            target = {
                'eps': noise,
                'x0': x0,
                'x_{t-1}': self.q_posterior_mean_variance(x0, xt, t)[0]
            }[self.mean_type]
            loss = (out - target).pow(1 if self.loss_type.endswith('l1') else 2
                                      ).abs().flatten(1).mean(dim=1)

            # total loss
            loss = loss + loss_vlb
        return loss

    def variational_lower_bound(self,
                                x0,
                                xt,
                                t,
                                model,
                                model_kwargs={},
                                clamp=None,
                                percentile=None):
        # compute groundtruth and predicted distributions
        mu1, _, log_var1 = self.q_posterior_mean_variance(x0, xt, t)
        mu2, _, log_var2, x0 = self.p_mean_variance(xt, t, model, model_kwargs,
                                                    clamp, percentile)

        # compute KL loss
        kl = kl_divergence(mu1, log_var1, mu2, log_var2)
        kl = kl.flatten(1).mean(dim=1) / math.log(2.0)

        # compute discretized NLL loss (for p(x0 | x1) only)
        nll = -discretized_gaussian_log_likelihood(
            x0, mean=mu2, log_scale=0.5 * log_var2)
        nll = nll.flatten(1).mean(dim=1) / math.log(2.0)

        # NLL for p(x0 | x1) and KL otherwise
        vlb = torch.where(t == 0, nll, kl)
        return vlb, x0

    @torch.no_grad()
    def variational_lower_bound_loop(self,
                                     x0,
                                     model,
                                     model_kwargs={},
                                     clamp=None,
                                     percentile=None):
        r"""Compute the entire variational lower bound, measured in bits-per-dim.
        """
        # prepare input and output
        b = x0.size(0)
        metrics = {'vlb': [], 'mse': [], 'x0_mse': []}

        # loop
        for step in torch.arange(self.num_timesteps).flip(0):
            # compute VLB
            t = torch.full((b, ), step, dtype=torch.long, device=x0.device)
            noise = torch.randn_like(x0)
            xt = self.q_sample(x0, t, noise)
            vlb, pred_x0 = self.variational_lower_bound(
                x0, xt, t, model, model_kwargs, clamp, percentile)

            # predict eps from x0
            u1 = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0)
            u2 = _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            eps = u1 / u2

            # collect metrics
            metrics['vlb'].append(vlb)
            metrics['x0_mse'].append(
                (pred_x0 - x0).square().flatten(1).mean(dim=1))
            metrics['mse'].append(
                (eps - noise).square().flatten(1).mean(dim=1))
        metrics = {k: torch.stack(v, dim=1) for k, v in metrics.items()}

        # compute the prior KL term for VLB, measured in bits-per-dim
        mu, _, log_var = self.q_mean_variance(x0, t)
        kl_prior = kl_divergence(mu, log_var, torch.zeros_like(mu),
                                 torch.zeros_like(log_var))
        kl_prior = kl_prior.flatten(1).mean(dim=1) / math.log(2.0)

        # update metrics
        metrics['prior_bits_per_dim'] = kl_prior
        metrics['total_bits_per_dim'] = metrics['vlb'].sum(dim=1) + kl_prior
        return metrics

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * 1000.0 / self.num_timesteps
        return t
