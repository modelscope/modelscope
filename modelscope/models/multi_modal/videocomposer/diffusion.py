# Copyright (c) Alibaba, Inc. and its affiliates.

import math

import torch

from .dpm_solver import (DPM_Solver, NoiseScheduleVP, model_wrapper,
                         model_wrapper_guided_diffusion)
from .ops.losses import discretized_gaussian_log_likelihood, kl_divergence

__all__ = ['GaussianDiffusion', 'beta_schedule', 'GaussianDiffusion_style']


def _i(tensor, t, x):
    r"""Index tensor using t and format the output according to x.
    """
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    if tensor.device != x.device:
        tensor = tensor.to(x.device)
    return tensor[t].view(shape).to(x)


def beta_schedule(schedule,
                  num_timesteps=1000,
                  init_beta=None,
                  last_beta=None):
    '''
    This code defines a function beta_schedule that generates a sequence of beta
    values based on the given input parameters.
    These beta values can be used in video diffusion processes. The function has the following parameters:
        schedule(str): Determines the type of beta schedule to be generated.
            It can be 'linear', 'linear_sd', 'quadratic', or 'cosine'.
        num_timesteps(int, optional): The number of timesteps for the generated beta schedule. Default is 1000.
        init_beta(float, optional): The initial beta value.
            If not provided, a default value is used based on the chosen schedule.
        last_beta(float, optional): The final beta value.
            If not provided, a default value is used based on the chosen schedule.
    The function returns a PyTorch tensor containing the generated beta values.
    The beta schedule is determined by the schedule parameter:
        1.Linear: Generates a linear sequence of beta values betweeninit_betaandlast_beta.
        2.Linear_sd: Generates a linear sequence of beta values between the square root of
            init_beta and the square root oflast_beta, and then squares the result.
        3.Quadratic: Similar to the 'linear_sd' schedule, but with different default values forinit_betaandlast_beta.
        4.Cosine: Generates a sequence of beta values based on a cosine function,
            ensuring the values are between 0 and 0.999.
    If an unsupported schedule is provided, a ValueError is raised with a message indicating the issue.
    '''
    if schedule == 'linear':
        scale = 1000.0 / num_timesteps
        init_beta = init_beta or scale * 0.0001
        last_beta = last_beta or scale * 0.02
        return torch.linspace(
            init_beta, last_beta, num_timesteps, dtype=torch.float64)
    elif schedule == 'linear_sd':
        return torch.linspace(
            init_beta**0.5, last_beta**0.5, num_timesteps,
            dtype=torch.float64)**2
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
            fn = lambda u: math.cos(  # noqa
                (u + 0.008) / 1.008 * math.pi / 2)**2  # noqa
            betas.append(min(1.0 - fn(t2) / fn(t1), 0.999))
        return torch.tensor(betas, dtype=torch.float64)
    else:
        raise ValueError(f'Unsupported schedule: {schedule}')


def load_stable_diffusion_pretrained(state_dict, temporal_attention):
    import collections
    sd_new = collections.OrderedDict()
    keys = list(state_dict.keys())

    for k in keys:
        if k.find('diffusion_model') >= 0:
            k_new = k.split('diffusion_model.')[-1]
            if k_new in [
                    'input_blocks.3.0.op.weight', 'input_blocks.3.0.op.bias',
                    'input_blocks.6.0.op.weight', 'input_blocks.6.0.op.bias',
                    'input_blocks.9.0.op.weight', 'input_blocks.9.0.op.bias'
            ]:
                k_new = k_new.replace('0.op', 'op')
            if temporal_attention:
                if k_new.find('middle_block.2') >= 0:
                    k_new = k_new.replace('middle_block.2', 'middle_block.3')
                if k_new.find('output_blocks.5.2') >= 0:
                    k_new = k_new.replace('output_blocks.5.2',
                                          'output_blocks.5.3')
                if k_new.find('output_blocks.8.2') >= 0:
                    k_new = k_new.replace('output_blocks.8.2',
                                          'output_blocks.8.3')
            sd_new[k_new] = state_dict[k]

    return sd_new


class AddGaussianNoise(object):

    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, img):
        assert isinstance(img, torch.Tensor)
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(torch.float32)
        out = img + self.std * torch.randn_like(img) + self.mean
        if out.dtype != dtype:
            out = out.to(dtype)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(
            self.mean, self.std)


class GaussianDiffusion(object):

    def __init__(self,
                 betas,
                 mean_type='eps',
                 var_type='learned_range',
                 loss_type='mse',
                 epsilon=1e-12,
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
            'mse', 'rescaled_mse', 'kl', 'rescaled_kl', 'l1', 'rescaled_l1',
            'charbonnier'
        ]
        self.betas = betas
        self.num_timesteps = len(betas)
        self.mean_type = mean_type  # eps
        self.var_type = var_type  # 'fixed_small'
        self.loss_type = loss_type  # mse
        self.epsilon = epsilon  # 1e-12
        self.rescale_timesteps = rescale_timesteps  # False

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
        return _i(self.sqrt_alphas_cumprod, t, x0) * x0 + \
               _i(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise  # noqa

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
        mask = t.ne(0).float().view(
            -1,
            *((1, ) *  # noqa
              (xt.ndim - 1)))
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
            dim = y_out.size(1) if self.var_type.startswith(
                'fixed') else y_out.size(1) // 2
            out = torch.cat(
                [
                    u_out[:, :dim] + guide_scale *  # noqa
                    (y_out[:, :dim] - u_out[:, :dim]),
                    y_out[:, dim:]
                ],
                dim=1)  # noqa

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
            x0 = _i(1.0 / self.posterior_mean_coef1, t, xt) * mu - \
                 _i(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, xt) * xt  # noqa
        elif self.mean_type == 'x0':
            x0 = out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        elif self.mean_type == 'eps':
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out  # noqa
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
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)  # noqa
            eps = eps - (1 - alpha).sqrt() * condition_fn(
                xt, self._scale_timesteps(t), **model_kwargs)

            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps  # noqa

        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)  # noqa
        alphas = _i(self.alphas_cumprod, t, xt)
        alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
        sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) *  # noqa
                                  (1 - alphas / alphas_prev))

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
        # import ipdb; ipdb.set_trace()
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
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)  # noqa
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
                eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                      _i(self.sqrt_recipm1_alphas_cumprod, t, xt)  # noqa
                eps = eps - (1 - alpha).sqrt() * condition_fn(
                    xt, self._scale_timesteps(t), **model_kwargs)

                # eps -> x0
                x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                     _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps  # noqa

            # derive eps
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)  # noqa
            return eps

        # function for compute x_0 and x_{t-1}
        def compute_x0(eps, t):
            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps  # noqa

            # deterministic sample
            alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
            direction = torch.sqrt(1 - alphas_prev) * eps
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

    def loss(self,
             x0,
             t,
             model,
             model_kwargs={},
             noise=None,
             weight=None,
             use_div_loss=False):
        noise = torch.randn_like(
            x0) if noise is None else noise  # [80, 4, 8, 32, 32]
        xt = self.q_sample(x0, t, noise=noise)

        # compute loss
        if self.loss_type in ['kl', 'rescaled_kl']:
            loss, _ = self.variational_lower_bound(x0, xt, t, model,
                                                   model_kwargs)
            if self.loss_type == 'rescaled_kl':
                loss = loss * self.num_timesteps
        elif self.loss_type in ['mse', 'rescaled_mse', 'l1',
                                'rescaled_l1']:  # self.loss_type: mse
            out = model(xt, self._scale_timesteps(t), **model_kwargs)

            # VLB for variation
            loss_vlb = 0.0
            if self.var_type in ['learned', 'learned_range'
                                 ]:  # self.var_type: 'fixed_small'
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
            if weight is not None:
                loss = loss * weight

            # div loss
            if use_div_loss and self.mean_type == 'eps' and x0.shape[2] > 1:

                # derive  x0
                x0_ = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                    _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out

                # ncfhw, std on f
                div_loss = 0.001 / (
                    x0_.std(dim=2).flatten(1).mean(dim=1) + 1e-4)
                loss = loss + div_loss

            # total loss
            loss = loss + loss_vlb
        elif self.loss_type in ['charbonnier']:
            out = model(xt, self._scale_timesteps(t), **model_kwargs)

            # VLB for variation
            loss_vlb = 0.0
            if self.var_type in ['learned', 'learned_range']:
                out, var = out.chunk(2, dim=1)
                frozen = torch.cat([out.detach(), var], dim=1)
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
            loss = torch.sqrt((out - target)**2 + self.epsilon)
            if weight is not None:
                loss = loss * weight
            loss = loss.flatten(1).mean(dim=1)

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
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)  # noqa

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
        if self.rescale_timesteps:  # noqa
            return t.float() * 1000.0 / self.num_timesteps
        return t


class GaussianDiffusion_style(object):

    def __init__(self,
                 betas,
                 mean_type='eps',
                 var_type='fixed_small',
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
        xt = _i(self.sqrt_alphas_cumprod, t, x0) * x0 + \
             _i(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise  # noqa
        return xt.type_as(x0)

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
        dtype = xt.dtype

        # predict distribution of p(x_{t-1} | x_t)
        mu, var, log_var, x0 = self.p_mean_variance(xt, t, model, model_kwargs,
                                                    clamp, percentile,
                                                    guide_scale)

        # random sample (with optional conditional function)
        noise = torch.randn_like(xt)
        t_mask = t.ne(0).float().view(
            -1,
            *((1, ) *  # noqa
              (xt.ndim - 1)))
        if condition_fn is not None:
            grad = condition_fn(xt, self._scale_timesteps(t), **model_kwargs)
            mu = mu.float() + var * grad.float()
        xt_1 = mu + t_mask * torch.exp(0.5 * log_var) * noise
        return xt_1.type(dtype), x0.type(dtype)

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
            out = model(xt, t=self._scale_timesteps(t), **model_kwargs)
        else:
            # classifier-free guidance
            # (model_kwargs[0]: conditional kwargs; model_kwargs[1]: non-conditional kwargs)
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            y_out = model(xt, t=self._scale_timesteps(t), **model_kwargs[0])
            if guide_scale != 1.0:
                u_out = model(
                    xt, t=self._scale_timesteps(t), **model_kwargs[1])
                dim = y_out.size(1) if self.var_type.startswith(
                    'fixed') else y_out.size(1) // 2
                out = torch.cat(
                    [
                        u_out[:, :dim] + guide_scale *  # noqa
                        (y_out[:, :dim] - u_out[:, :dim]),
                        y_out[:, dim:]
                    ],
                    dim=1)  # noqa
            else:
                out = y_out

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
            x0 = _i(1.0 / self.posterior_mean_coef1, t, xt) * mu - \
                 _i(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, xt) * xt  # noqa
        elif self.mean_type == 'x0':
            x0 = out
        elif self.mean_type == 'eps':
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out  # noqa

        # restrict the range of x0
        if percentile is not None:
            assert percentile > 0 and percentile <= 1  # e.g., 0.995
            s = torch.quantile(
                x0.flatten(1).abs(), percentile,
                dim=1).clamp_(1.0).view(-1, 1, 1, 1, 1)
            # s = torch.quantile(x0.flatten(1).abs(), percentile, dim=1).clamp_(1.0).view(-1, 1, 1, 1) # old
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)

        # recompute mu using the restricted x0
        mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        return mu, var, log_var, x0

    @torch.no_grad()
    def ddim_sample(self,
                    xt,
                    t,
                    t_prev,
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
        dtype = xt.dtype

        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp,
                                           percentile, guide_scale)
        if condition_fn is not None:
            # x0 -> eps
            alpha = _i(self.alphas_cumprod, t, xt)
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)  # noqa
            eps = eps - (1 - alpha).sqrt() * condition_fn(
                xt, self._scale_timesteps(t), **model_kwargs)

            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps  # noqa

        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)  # noqa
        alphas = _i(self.alphas_cumprod, t, xt)
        alphas_prev = _i(self.alphas_cumprod, t_prev, xt)
        sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) *  # noqa
                                  (1 - alphas / alphas_prev))

        # random sample
        noise = torch.randn_like(xt)
        direction = torch.sqrt(1 - alphas_prev - sigmas**2) * eps
        t_mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
        xt_1 = torch.sqrt(
            alphas_prev) * x0 + direction + t_mask * sigmas * noise
        return xt_1.type(dtype), x0.type(dtype)

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

        # diffusion process
        steps = (1 + torch.arange(0, self.num_timesteps,
                                  self.num_timesteps // ddim_timesteps)).clamp(
                                      0, self.num_timesteps - 1).flip(0)
        for i, step in enumerate(steps):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            t_prev = torch.full((b, ),
                                steps[i + 1] if i < len(steps) - 1 else 0,
                                dtype=torch.long,
                                device=xt.device)
            xt, _ = self.ddim_sample(xt, t, t_prev, model, model_kwargs, clamp,
                                     percentile, condition_fn, guide_scale,
                                     ddim_timesteps, eta)
        return xt

    @torch.no_grad()
    def ddim_reverse_sample(self,
                            xt,
                            t,
                            t_next,
                            model,
                            model_kwargs={},
                            clamp=None,
                            percentile=None,
                            guide_scale=None,
                            ddim_timesteps=20):
        r"""Sample from p(x_{t+1} | x_t) using DDIM reverse ODE (deterministic).
        """
        dtype = xt.dtype

        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp,
                                           percentile, guide_scale)

        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)  # noqa
        alphas_next = _i(
            torch.cat(
                [self.alphas_cumprod,
                 self.alphas_cumprod.new_zeros([1])]), t_next, xt)

        # reverse sample
        mu = torch.sqrt(alphas_next) * x0 + torch.sqrt(1 - alphas_next) * eps
        return mu.type(dtype), x0.type(dtype)

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
        steps = (1 + torch.arange(0, self.num_timesteps,
                                  self.num_timesteps // ddim_timesteps)).clamp(
                                      0, self.num_timesteps - 1)
        for i, step in enumerate(steps):
            t = torch.full((b, ),
                           steps[i - 1] if i > 0 else 0,
                           dtype=torch.long,
                           device=xt.device)
            t_next = torch.full((b, ),
                                step,
                                dtype=torch.long,
                                device=xt.device)
            xt, _ = self.ddim_reverse_sample(xt, t, t_next, model,
                                             model_kwargs, clamp, percentile,
                                             guide_scale, ddim_timesteps)
        return xt

    @torch.no_grad()
    def plms_sample(self,
                    xt,
                    t,
                    t_prev,
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

        # function for compute eps
        def compute_eps(xt, t):
            dtype = xt.dtype

            # predict distribution of p(x_{t-1} | x_t)
            _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs,
                                               clamp, percentile, guide_scale)

            # condition
            if condition_fn is not None:
                # x0 -> eps
                alpha = _i(self.alphas_cumprod, t, xt)
                eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                      _i(self.sqrt_recipm1_alphas_cumprod, t, xt)  # noqa
                eps = eps - (1 - alpha).sqrt() * condition_fn(
                    xt, self._scale_timesteps(t), **model_kwargs)

                # eps -> x0
                x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                     _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps  # noqa

            # derive eps
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)  # noqa
            return eps.type(dtype)

        # function for compute x_0 and x_{t-1}
        def compute_x0(eps, t):
            dtype = eps.dtype

            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps  # noqa

            # deterministic sample
            alphas_prev = _i(self.alphas_cumprod, t_prev, xt)
            direction = torch.sqrt(1 - alphas_prev) * eps
            xt_1 = torch.sqrt(alphas_prev) * x0 + direction
            return xt_1.type(dtype), x0.type(dtype)

        # PLMS sample
        eps = compute_eps(xt, t)
        if len(eps_cache) == 0:
            # 2nd order pseudo improved Euler
            xt_1, x0 = compute_x0(eps, t)
            eps_next = compute_eps(xt_1, t_prev)
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
        for i, step in enumerate(steps):
            # PLMS sampling step
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            t_prev = torch.full((b, ),
                                steps[i + 1] if i < len(steps) - 1 else 0,
                                dtype=torch.long,
                                device=xt.device)
            xt, _, eps = self.plms_sample(xt, t, t_prev, model, model_kwargs,
                                          clamp, percentile, condition_fn,
                                          guide_scale, plms_timesteps,
                                          eps_cache)

            # update eps cache
            eps_cache.append(eps)
            if len(eps_cache) >= 4:
                eps_cache.pop(0)
        return xt

    @torch.no_grad()
    def dpm_solver_sample_loop(self,
                               noise,
                               model,
                               model_kwargs={},
                               order=2,
                               skip_type='logSNR',
                               method='multistep',
                               clamp=None,
                               percentile=None,
                               condition_fn=None,
                               guide_scale=None,
                               dpm_solver_timesteps=20,
                               algorithm_type='dpmsolver++',
                               t_start=None,
                               t_end=None,
                               lower_order_final=True,
                               denoise_to_zero=False,
                               solver_type='dpmsolver'):
        r"""Sample using DPM-Solver-based method.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).

            Please check all the parameters in `dpm_solver.sample` before using.
        """
        assert self.mean_type in ('eps', 'x0')
        assert percentile in (None, 0.995)
        assert clamp is None or percentile is None
        noise_schedule = NoiseScheduleVP(
            schedule='discrete', betas=self.betas.float())
        model_fn = model_wrapper_guided_diffusion(
            model=model,
            noise_schedule=noise_schedule,
            var_type=self.var_type,
            mean_type=self.mean_type,
            model_kwargs=model_kwargs,
            rescale_timesteps=self.rescale_timesteps,
            num_timesteps=self.num_timesteps,
            guide_scale=guide_scale,
            condition_fn=condition_fn)
        dpm_solver = DPM_Solver(
            model_fn=model_fn,
            noise_schedule=noise_schedule,
            algorithm_type=algorithm_type,
            percentile=percentile,
            clamp=clamp)
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
    def inpaint_p_sample(self,
                         xt,
                         t,
                         y,
                         mask,
                         model,
                         model_kwargs={},
                         clamp=None,
                         percentile=None,
                         guide_scale=None):
        r"""DDPM sampling step for inpainting.
        """
        dtype = xt.dtype

        # predict distribution of p(x_{t-1} | x_t), conditioned on y and mask
        xt = self.q_sample(y, t) * mask + xt * (1 - mask)
        mu, var, log_var, x0 = self.p_mean_variance(xt, t, model, model_kwargs,
                                                    clamp, percentile,
                                                    guide_scale)

        # random sample
        t_mask = t.ne(0).float().view(
            -1,
            *((1, ) *  # noqa
              (xt.ndim - 1)))
        xt_1 = mu + t_mask * torch.exp(0.5 * log_var) * torch.randn_like(xt)
        return xt_1.type(dtype), x0.type(dtype)

    @torch.no_grad()
    def inpaint_p_sample_loop(self,
                              noise,
                              y,
                              mask,
                              model,
                              model_kwargs={},
                              clamp=None,
                              percentile=None,
                              guide_scale=None):
        r"""DDPM sampling loop for inpainting.
        """
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process
        for step in torch.arange(self.num_timesteps).flip(0):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.inpaint_p_sample(xt, t, y, mask, model, model_kwargs,
                                          clamp, percentile, guide_scale)
        return xt

    @torch.no_grad()
    def inpaint_mcg_p_sample(self,
                             xt,
                             t,
                             y,
                             mask,
                             model,
                             model_kwargs={},
                             clamp=None,
                             percentile=None,
                             guide_scale=None,
                             mcg_scale=1.0):
        r"""DDPM sampling step for inpainting, with Manifold Constrained Gradient (MCG) correction.
        """
        dtype = xt.dtype

        # predict distribution of p(x_{t-1} | x_t), conditioned on y and mask
        with torch.enable_grad():
            xt.requires_grad_(True)
            mu, var, log_var, x0 = self.p_mean_variance(
                xt, t, model, model_kwargs, clamp, percentile, guide_scale)
            loss = (y * mask - x0 * mask).square().mean()
            grad = torch.autograd.grad(loss, xt)[0]

        # random sample
        t_mask = t.ne(0).float().view(
            -1,
            *((1, ) *  # noqa
              (xt.ndim - 1)))
        xt_1 = mu + t_mask * torch.exp(0.5 * log_var) * torch.randn_like(xt)
        xt_1 = xt_1 - mcg_scale * grad

        # merge foreground and background
        xt_1 = self.q_sample(y, t) * mask + xt_1 * (1 - mask)
        return xt_1.type(dtype), x0.type(dtype)

    @torch.no_grad()
    def inpaint_mcg_p_sample_loop(self,
                                  noise,
                                  y,
                                  mask,
                                  model,
                                  model_kwargs={},
                                  clamp=None,
                                  percentile=None,
                                  guide_scale=None,
                                  mcg_scale=1.0):
        r"""DDPM sampling loop for inpainting, with Manifold Constrained Gradient (MCG) correction.
        """
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process
        for step in torch.arange(self.num_timesteps).flip(0):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.inpaint_mcg_p_sample(xt, t, y, mask, model,
                                              model_kwargs, clamp, percentile,
                                              guide_scale, mcg_scale)
        return xt

    def loss(self,
             x0,
             t,
             model,
             model_kwargs={},
             noise=None,
             input_x0=None,
             reduction='mean'):
        assert reduction in ['mean', 'none']
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
            out = model(xt, t=self._scale_timesteps(t), **model_kwargs)

            # VLB for variation
            loss_vlb = 0.0
            if self.var_type in ['learned', 'learned_range']:
                out, var = out.chunk(2, dim=1)
                frozen = torch.cat([
                    out.detach(), var
                ], dim=1)  # learn var without affecting the prediction of mean
                loss_vlb, _ = self.variational_lower_bound(
                    x0,
                    xt,
                    t,
                    model=lambda *args, **kwargs: frozen,
                    reduction=reduction)
                if self.loss_type.startswith('rescaled_'):
                    loss_vlb = loss_vlb * self.num_timesteps / 1000.0

            # MSE/L1 for x0/eps
            target = {
                'eps': noise,
                'x0': x0,
                'x_{t-1}': self.q_posterior_mean_variance(x0, xt, t)[0]
            }[self.mean_type]
            loss = (
                out
                - target).pow(1 if self.loss_type.endswith('l1') else 2).abs()
            if reduction == 'mean':
                loss = loss.flatten(1).mean(dim=1)

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
                                percentile=None,
                                reduction='mean'):
        assert reduction in ['mean', 'none']

        # compute groundtruth and predicted distributions
        mu1, _, log_var1 = self.q_posterior_mean_variance(x0, xt, t)
        mu2, _, log_var2, x0 = self.p_mean_variance(xt, t, model, model_kwargs,
                                                    clamp, percentile)

        # compute KL loss
        kl = kl_divergence(mu1, log_var1, mu2, log_var2) / math.log(2.0)
        if reduction == 'mean':
            kl = kl.flatten(1).mean(dim=1)

        # compute discretized NLL loss (for p(x0 | x1) only)
        nll = -discretized_gaussian_log_likelihood(
            x0, mean=mu2, log_scale=0.5 * log_var2) / math.log(2.0)
        if reduction == 'mean':
            nll = nll.flatten(1).mean(dim=1)

        # NLL for p(x0 | x1) and KL otherwise
        t = t.view(-1, *(1, ) * (nll.ndim - 1))
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
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)  # noqa

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
