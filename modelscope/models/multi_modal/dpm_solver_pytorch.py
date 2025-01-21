# The implementation is borrowed and modified from dpm-solver,
# publicly available at https://github.com/LuChengTHU/dpm-solver.
# Copyright LuChengTHU Authors.
# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors.
# All rights reserved.
import math

import torch
import torch.nn.functional as F


def _i(tensor, t, x):
    r"""Index tensor using t and format the output according to x.
    """
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    return tensor[t].view(shape).to(x)


class NoiseScheduleVP:

    def __init__(
        self,
        schedule='discrete',
        betas=None,
        alphas_cumprod=None,
        continuous_beta_0=0.1,
        continuous_beta_1=20.,
    ):
        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'"
                .format(schedule))

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.
            self.t_array = torch.linspace(0., 1.,
                                          self.total_N + 1)[1:].reshape(
                                              (1, -1))
            self.log_alpha_array = log_alphas.reshape((
                1,
                -1,
            ))
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.
            self.cosine_t_max = math.atan(
                self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (
                    1. + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_log_alpha_0 = math.log(
                math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            self.schedule = schedule
            if schedule == 'cosine':
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else:
                self.T = 1.

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            return interpolate_fn(
                t.reshape((-1, 1)), self.t_array.to(t.device),
                self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t**2 * (self.beta_1
                                   - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':

            def log_alpha_fn(s):
                _a = (s + self.cosine_s)
                _b = (1. + self.cosine_s)
                _c = math.pi / 2.
                return torch.log(torch.cos(_a / _b * _c))

            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(
                -2. * lamb,
                torch.zeros((1, )).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (
                self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(
                torch.zeros((1, )).to(lamb.device), -2. * lamb)
            t = interpolate_fn(
                log_alpha.reshape((-1, 1)),
                torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1, ))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb,
                                               torch.zeros((1, )).to(lamb))

            def t_fn(log_alpha_t):
                return torch.arccos(
                    torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (
                        1. + self.cosine_s) / math.pi - self.cosine_s

            t = t_fn(log_alpha)
            return t


def model_wrapper(
    model,
    noise_schedule,
    model_type='noise',
    model_kwargs={},
    guidance_type='uncond',
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.,
    classifier_fn=None,
    classifier_kwargs={},
):

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        if t_continuous.reshape((-1, )).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0]))
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == 'noise':
            return output
        elif model_type == 'x_start':
            alpha_t, sigma_t = noise_schedule.marginal_alpha(
                t_continuous), noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return (x - expand_dims(alpha_t, dims) * output) / expand_dims(
                sigma_t, dims)
        elif model_type == 'v':
            alpha_t, sigma_t = noise_schedule.marginal_alpha(
                t_continuous), noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return expand_dims(alpha_t, dims) * output + expand_dims(
                sigma_t, dims) * x
        elif model_type == 'score':
            sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return -expand_dims(sigma_t, dims) * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition,
                                     **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if t_continuous.reshape((-1, )).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0]))
        if guidance_type == 'uncond':
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == 'classifier':
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * expand_dims(
                sigma_t, dims=cond_grad.dim()) * cond_grad
        elif guidance_type == 'classifier-free':
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(
                    x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ['noise', 'x_start', 'v']
    assert guidance_type in ['uncond', 'classifier', 'classifier-free']
    return model_fn


def model_wrapper_guided_diffusion(
    model,
    noise_schedule,
    var_type,
    mean_type,
    model_kwargs={},
    clamp=None,
    percentile=None,
    rescale_timesteps=False,
    num_timesteps=1000,
    guide_scale=None,
    condition_fn=None,
):

    def _scale_timesteps(t):
        if rescale_timesteps:
            return t.float() * 1000.0 / num_timesteps
        return t

    def get_model_input_time(t_continuous):
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        else:
            return t_continuous

    def noise_pred_fn(xt, t_continuous):
        if t_continuous.reshape((-1, )).shape[0] == 1:
            t_continuous = t_continuous.expand((xt.shape[0]))
        t_input = get_model_input_time(_scale_timesteps(t_continuous))
        # predict distribution
        if guide_scale is None:
            out = model(xt, t_input, **model_kwargs)
        else:
            # classifier-free guidance
            # (model_kwargs[0]: conditional kwargs; model_kwargs[1]: non-conditional kwargs)
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            y_out = model(xt, t_input, **model_kwargs[0])
            u_out = model(xt, t_input, **model_kwargs[1])
            dim = y_out.size(1) if var_type.startswith(
                'fixed') else y_out.size(1) // 2
            _a = u_out[:, :dim]
            _b = guide_scale * (y_out[:, :dim] - u_out[:, :dim])
            _c = [_a + _b, y_out[:, dim:]]
            out = torch.cat(_c, dim=1)
        if var_type == 'learned':
            out, _ = out.chunk(2, dim=1)
        elif var_type == 'learned_range':
            out, _ = out.chunk(2, dim=1)

        if mean_type == 'eps':
            alpha_t, sigma_t = noise_schedule.marginal_alpha(
                t_continuous), noise_schedule.marginal_std(t_continuous)
            dims = xt.dim()
            x0 = (xt - expand_dims(sigma_t, dims) * out) / expand_dims(
                alpha_t, dims)
        elif mean_type == 'x_{t-1}':
            assert noise_schedule.schedule == 'discrete'
            mu = out
            posterior_mean_coef1 = None
            posterior_mean_coef2 = None
            x0 = expand_dims(
                1. / posterior_mean_coef1, dims) * mu - expand_dims(
                    posterior_mean_coef2 / posterior_mean_coef1, dims) * xt
        elif mean_type == 'x0':
            x0 = out

        # restrict the range of x0
        if percentile is not None:
            assert percentile > 0 and percentile <= 1  # e.g., 0.995
            s = torch.quantile(
                x0.flatten(1).abs(), percentile,
                dim=1).clamp_(1.0).view(-1, 1, 1, 1)
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)

        if condition_fn is not None:
            alpha_t = noise_schedule.marginal_alpha(t_continuous)
            eps = (xt - expand_dims(alpha_t, dims) * x0) / expand_dims(
                sigma_t, dims)

            eps = eps - (1 - alpha_t).sqrt() * condition_fn(
                xt, t_input, **model_kwargs)
            x0 = (xt - expand_dims(sigma_t, dims) * eps) / expand_dims(
                alpha_t, dims)

        eps = (xt - expand_dims(alpha_t, dims) * x0) / expand_dims(
            sigma_t, dims)
        return eps

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if t_continuous.reshape((-1, )).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0]))
        return noise_pred_fn(x, t_continuous)

    return model_fn


class DPM_Solver:

    def __init__(self,
                 model_fn,
                 noise_schedule,
                 predict_x0=False,
                 thresholding=False,
                 max_val=1.):
        self.model = model_fn
        self.noise_schedule = noise_schedule
        self.predict_x0 = predict_x0
        self.thresholding = thresholding
        self.max_val = max_val

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with thresholding).
        """
        noise = self.noise_prediction_fn(x, t)
        dims = x.dim()
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(
            t), self.noise_schedule.marginal_std(t)
        x0 = (x - expand_dims(sigma_t, dims) * noise) / expand_dims(
            alpha_t, dims)
        if self.thresholding:
            p = 0.995  # A hyperparameter in the paper of "Imagen" [1].
            s = torch.quantile(
                torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
            s = expand_dims(
                torch.maximum(s,
                              self.max_val * torch.ones_like(s).to(s.device)),
                dims)
            x0 = torch.clamp(x0, -s, s) / s
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(
                torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(
                torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(),
                                          lambda_0.cpu().item(),
                                          N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order),
                               N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'"
                .format(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order,
                                                       skip_type, t_T, t_0,
                                                       device):
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [
                    3,
                ] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [
                    3,
                ] * (K - 1) + [1]
            else:
                orders = [
                    3,
                ] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [
                    2,
                ] * K
            else:
                K = steps // 2 + 1
                orders = [
                    2,
                ] * (K - 1) + [1]
        elif order == 1:
            # Bug here.
            # K = 1
            K = steps
            orders = [
                1,
            ] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K,
                                                  device)
        # TODO: bug here.
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps,
                                                  device)[torch.cumsum(
                                                      torch.tensor([
                                                          0,
                                                      ] + orders)).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        return self.data_prediction_fn(x, s)

    def dpm_solver_first_update(self,
                                x,
                                s,
                                t,
                                model_s=None,
                                return_intermediate=False):
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(
            s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.predict_x0:
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                expand_dims(sigma_t / sigma_s, dims) * x
                - expand_dims(alpha_t * phi_1, dims) * model_s)
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                expand_dims(torch.exp(log_alpha_t - log_alpha_s), dims) * x
                - expand_dims(sigma_t * phi_1, dims) * model_s)
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t

    def singlestep_dpm_solver_second_update(self,
                                            x,
                                            s,
                                            t,
                                            r1=0.5,
                                            model_s=None,
                                            return_intermediate=False,
                                            solver_type='dpm_solver'):
        if solver_type not in ['dpm_solver', 'taylor']:
            raise ValueError(
                "'solver_type' must be either 'dpm_solver' or 'taylor', got {}"
                .format(solver_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(
            s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(
            s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)

        if self.predict_x0:
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            _a = expand_dims(sigma_s1 / sigma_s, dims)
            _b = expand_dims(alpha_s1 * phi_11, dims)
            x_s1 = (_a * x - _b * model_s)
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpm_solver':
                _a = expand_dims(sigma_t / sigma_s, dims)
                _b = expand_dims(alpha_t * phi_1, dims)
                _c = expand_dims(alpha_t * phi_1, dims)
                _d = (model_s1 - model_s)
                x_t = (_a * x - _b * model_s - (0.5 / r1) * _c * _d)
            elif solver_type == 'taylor':
                _a = expand_dims(sigma_t / sigma_s, dims)
                _b = expand_dims(alpha_t * phi_1, dims)
                _c = expand_dims(alpha_t * ((torch.exp(-h) - 1.) / h + 1.),
                                 dims)
                _d = (model_s1 - model_s)
                x_t = (_a * x - _b * model_s + (1. / r1) * _c * _d)
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            _a = expand_dims(torch.exp(log_alpha_s1 - log_alpha_s), dims)
            _b = expand_dims(sigma_s1 * phi_11, dims)
            x_s1 = (_a * x - _b * model_s)
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpm_solver':
                _a = expand_dims(torch.exp(log_alpha_t - log_alpha_s), dims)
                _b = expand_dims(sigma_t * phi_1, dims)
                _c = expand_dims(sigma_t * phi_1, dims)
                _d = (model_s1 - model_s)
                x_t = (_a * x - _b * model_s - (0.5 / r1) * _c * _d)
            elif solver_type == 'taylor':
                _a = expand_dims(torch.exp(log_alpha_t - log_alpha_s), dims)
                _b = expand_dims(sigma_t * phi_1, dims)
                _c = expand_dims(sigma_t * ((torch.exp(h) - 1.) / h - 1.),
                                 dims)
                _d = (model_s1 - model_s)
                x_t = (_a * x - _b * model_s - (1. / r1) * _c * _d)
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1}
        else:
            return x_t

    def singlestep_dpm_solver_third_update(self,
                                           x,
                                           s,
                                           t,
                                           r1=1. / 3.,
                                           r2=2. / 3.,
                                           model_s=None,
                                           model_s1=None,
                                           return_intermediate=False,
                                           solver_type='dpm_solver'):
        if solver_type not in ['dpm_solver', 'taylor']:
            raise ValueError(
                "'solver_type' must be either 'dpm_solver' or 'taylor', got {}"
                .format(solver_type))
        if r1 is None:
            r1 = 1. / 3.
        if r2 is None:
            r2 = 2. / 3.
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(
            s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(
                s2), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(
            s), ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(
            log_alpha_s2), torch.exp(log_alpha_t)

        if self.predict_x0:
            phi_11 = torch.expm1(-r1 * h)
            phi_12 = torch.expm1(-r2 * h)
            phi_1 = torch.expm1(-h)
            phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                _a = expand_dims(sigma_s1 / sigma_s, dims)
                _b = expand_dims(alpha_s1 * phi_11, dims)
                x_s1 = (_a * x - _b * model_s)
                model_s1 = self.model_fn(x_s1, s1)
            _a = expand_dims(sigma_s2 / sigma_s, dims)
            _b = expand_dims(alpha_s2 * phi_12, dims)
            _c = expand_dims(alpha_s2 * phi_22, dims)
            x_s2 = (
                _a * x - _b * model_s + r2 / r1 * _c * (model_s1 - model_s))
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpm_solver':
                _a = expand_dims(sigma_t / sigma_s, dims)
                _b = expand_dims(alpha_t * phi_1, dims)
                _c = expand_dims(alpha_t * phi_2, dims)
                _d = (model_s2 - model_s)
                x_t = (_a * x - _b * model_s + (1. / r2) * _c * _d)
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                _a = expand_dims(sigma_t / sigma_s, dims)
                _b = expand_dims(alpha_t * phi_1, dims)
                _c = expand_dims(alpha_t * phi_2, dims)
                _d = expand_dims(alpha_t * phi_3, dims)
                x_t = (_a * x - _b * model_s + _c * D1 - _d * D2)
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_12 = torch.expm1(r2 * h)
            phi_1 = torch.expm1(h)
            phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                _a = expand_dims(torch.exp(log_alpha_s1 - log_alpha_s), dims)
                _b = expand_dims(sigma_s1 * phi_11, dims)
                x_s1 = (_a * x - _b * model_s)
                model_s1 = self.model_fn(x_s1, s1)
            _a = expand_dims(torch.exp(log_alpha_s2 - log_alpha_s), dims)
            _b = expand_dims(sigma_s2 * phi_12, dims)
            _c = expand_dims(sigma_s2 * phi_22, dims)
            x_s2 = (
                _a * x - _b * model_s - r2 / r1 * _c * (model_s1 - model_s))
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpm_solver':
                _a = expand_dims(torch.exp(log_alpha_t - log_alpha_s), dims)
                _b = expand_dims(sigma_t * phi_1, dims)
                _c = expand_dims(sigma_t * phi_2, dims)
                _d = (model_s2 - model_s)
                x_t = (_a * x - _b * model_s - (1. / r2) * _c * _d)
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    expand_dims(torch.exp(log_alpha_t - log_alpha_s), dims) * x
                    - expand_dims(sigma_t * phi_1, dims) * model_s
                    - expand_dims(sigma_t * phi_2, dims) * D1
                    - expand_dims(sigma_t * phi_3, dims) * D2)

        if return_intermediate:
            return x_t, {
                'model_s': model_s,
                'model_s1': model_s1,
                'model_s2': model_s2
            }
        else:
            return x_t

    def multistep_dpm_solver_second_update(self,
                                           x,
                                           model_prev_list,
                                           t_prev_list,
                                           t,
                                           solver_type='dpm_solver'):
        if solver_type not in ['dpm_solver', 'taylor']:
            raise ValueError(
                "'solver_type' must be either 'dpm_solver' or 'taylor', got {}"
                .format(solver_type))
        ns = self.noise_schedule
        dims = x.dim()
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(
            t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(
            t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = expand_dims(1. / r0, dims) * (model_prev_0 - model_prev_1)
        if self.predict_x0:
            if solver_type == 'dpm_solver':
                _a = expand_dims(sigma_t / sigma_prev_0, dims)
                _b = expand_dims(alpha_t * (torch.exp(-h) - 1.), dims)
                _c = expand_dims(alpha_t * (torch.exp(-h) - 1.), dims)
                x_t = (_a * x - _b * model_prev_0 - 0.5 * _c * D1_0)
            elif solver_type == 'taylor':
                _a = expand_dims(sigma_t / sigma_prev_0, dims)
                _b = expand_dims(alpha_t * (torch.exp(-h) - 1.), dims)
                _c = expand_dims(alpha_t * ((torch.exp(-h) - 1.) / h + 1.),
                                 dims)
                x_t = (_a * x - _b * model_prev_0 + _c * D1_0)
        else:
            if solver_type == 'dpm_solver':
                _a = expand_dims(
                    torch.exp(log_alpha_t - log_alpha_prev_0), dims)
                _b = expand_dims(sigma_t * (torch.exp(h) - 1.), dims)
                _c = expand_dims(sigma_t * (torch.exp(h) - 1.), dims)
                x_t = (_a * x - _b * model_prev_0 - 0.5 * _c * D1_0)
            elif solver_type == 'taylor':
                _a = expand_dims(
                    torch.exp(log_alpha_t - log_alpha_prev_0), dims)
                _b = expand_dims(sigma_t * (torch.exp(h) - 1.), dims)
                _c = expand_dims(sigma_t * ((torch.exp(h) - 1.) / h - 1.),
                                 dims)
                x_t = (_a * x - _b * model_prev_0 - _c * D1_0)
        return x_t

    def multistep_dpm_solver_third_update(self,
                                          x,
                                          model_prev_list,
                                          t_prev_list,
                                          t,
                                          solver_type='dpm_solver'):
        ns = self.noise_schedule
        dims = x.dim()
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(
            t_prev_2), ns.marginal_lambda(t_prev_1), ns.marginal_lambda(
                t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(
            t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = expand_dims(1. / r0, dims) * (model_prev_0 - model_prev_1)
        D1_1 = expand_dims(1. / r1, dims) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + expand_dims(r0 / (r0 + r1), dims) * (D1_0 - D1_1)
        D2 = expand_dims(1. / (r0 + r1), dims) * (D1_0 - D1_1)
        if self.predict_x0:
            _a = expand_dims(sigma_t / sigma_prev_0, dims)
            _b = expand_dims(alpha_t * (torch.exp(-h) - 1.), dims)
            _c = expand_dims(alpha_t * ((torch.exp(-h) - 1.) / h + 1.), dims)
            _d = expand_dims(alpha_t * ((torch.exp(-h) - 1. + h) / h**2 - 0.5),
                             dims)
            x_t = (_a * x - _b * model_prev_0 + _c * D1 - _d * D2)
        else:
            _a = expand_dims(torch.exp(log_alpha_t - log_alpha_prev_0), dims)
            _b = expand_dims(sigma_t * (torch.exp(h) - 1.), dims)
            _c = expand_dims(sigma_t * ((torch.exp(h) - 1.) / h - 1.), dims)
            _d = expand_dims(sigma_t * ((torch.exp(h) - 1. - h) / h**2 - 0.5),
                             dims)
            x_t = (_a * x - _b * model_prev_0 - _c * D1 - _d * D2)
        return x_t

    def singlestep_dpm_solver_update(self,
                                     x,
                                     s,
                                     t,
                                     order,
                                     return_intermediate=False,
                                     solver_type='dpm_solver',
                                     r1=None,
                                     r2=None):
        if order == 1:
            return self.dpm_solver_first_update(
                x, s, t, return_intermediate=return_intermediate)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(
                x,
                s,
                t,
                return_intermediate=return_intermediate,
                solver_type=solver_type,
                r1=r1)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(
                x,
                s,
                t,
                return_intermediate=return_intermediate,
                solver_type=solver_type,
                r1=r1,
                r2=r2)
        else:
            raise ValueError(
                'Solver order must be 1 or 2 or 3, got {}'.format(order))

    def multistep_dpm_solver_update(self,
                                    x,
                                    model_prev_list,
                                    t_prev_list,
                                    t,
                                    order,
                                    solver_type='dpm_solver'):
        if order == 1:
            return self.dpm_solver_first_update(
                x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(
                x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(
                x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError(
                'Solver order must be 1 or 2 or 3, got {}'.format(order))

    def dpm_solver_adaptive(self,
                            x,
                            order,
                            t_T,
                            t_0,
                            h_init=0.05,
                            atol=0.0078,
                            rtol=0.05,
                            theta=0.9,
                            t_err=1e-5,
                            solver_type='dpm_solver'):
        ns = self.noise_schedule
        s = t_T * torch.ones((x.shape[0], )).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5

            def lower_update(x, s, t):
                return self.dpm_solver_first_update(
                    x, s, t, return_intermediate=True)

            def higher_update(x, s, t, **kwargs):
                self.singlestep_dpm_solver_second_update(
                    x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.

            def lower_update(x, s, t):
                self.singlestep_dpm_solver_second_update(
                    x,
                    s,
                    t,
                    r1=r1,
                    return_intermediate=True,
                    solver_type=solver_type)

            def higher_update(x, s, t, **kwargs):
                self.singlestep_dpm_solver_third_update(
                    x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs)
        else:
            raise ValueError(
                'For adaptive step size solver, order must be 2 or 3, got {}'.
                format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(
                torch.ones_like(x).to(x) * atol,
                rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))

            def norm_fn(v):
                return torch.sqrt(
                    torch.square(v.reshape(
                        (v.shape[0], -1))).mean(dim=-1, keepdim=True))

            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(
                theta * h * torch.float_power(E, -1. / order).float(),
                lambda_0 - lambda_s)
            nfe += order
        print('adaptive solver nfe', nfe)
        return x

    def sample(
        self,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=3,
        skip_type='time_uniform',
        method='singlestep',
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type='dpm_solver',
        atol=0.0078,
        rtol=0.05,
    ):
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device
        if method == 'adaptive':
            with torch.no_grad():
                x = self.dpm_solver_adaptive(
                    x,
                    order=order,
                    t_T=t_T,
                    t_0=t_0,
                    atol=atol,
                    rtol=rtol,
                    solver_type=solver_type)
        elif method == 'multistep':
            assert steps >= order
            timesteps = self.get_time_steps(
                skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            assert timesteps.shape[0] - 1 == steps
            with torch.no_grad():
                vec_t = timesteps[0].expand((x.shape[0]))
                model_prev_list = [self.model_fn(x, vec_t)]
                t_prev_list = [vec_t]
                # Init the first `order` values by lower order multistep DPM-Solver.
                for init_order in range(1, order):
                    vec_t = timesteps[init_order].expand(x.shape[0])
                    x = self.multistep_dpm_solver_update(
                        x,
                        model_prev_list,
                        t_prev_list,
                        vec_t,
                        init_order,
                        solver_type=solver_type)
                    model_prev_list.append(self.model_fn(x, vec_t))
                    t_prev_list.append(vec_t)
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in range(order, steps + 1):
                    vec_t = timesteps[step].expand(x.shape[0])
                    if lower_order_final and steps < 15:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(
                        x,
                        model_prev_list,
                        t_prev_list,
                        vec_t,
                        step_order,
                        solver_type=solver_type)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = vec_t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, vec_t)
        elif method in ['singlestep', 'singlestep_fixed']:
            if method == 'singlestep':
                timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(
                    steps=steps,
                    order=order,
                    skip_type=skip_type,
                    t_T=t_T,
                    t_0=t_0,
                    device=device)
            elif method == 'singlestep_fixed':
                K = steps // order
                orders = [
                    order,
                ] * K
                timesteps_outer = self.get_time_steps(
                    skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
            for i, order in enumerate(orders):
                t_T_inner, t_0_inner = timesteps_outer[i], timesteps_outer[i
                                                                           + 1]
                timesteps_inner = self.get_time_steps(
                    skip_type=skip_type,
                    t_T=t_T_inner.item(),
                    t_0=t_0_inner.item(),
                    N=order,
                    device=device)
                lambda_inner = self.noise_schedule.marginal_lambda(
                    timesteps_inner)
                vec_s, vec_t = t_T_inner.tile(x.shape[0]), t_0_inner.tile(
                    x.shape[0])
                h = lambda_inner[-1] - lambda_inner[0]
                r1 = None if order <= 1 else (lambda_inner[1]
                                              - lambda_inner[0]) / h
                r2 = None if order <= 2 else (lambda_inner[2]
                                              - lambda_inner[0]) / h
                x = self.singlestep_dpm_solver_update(
                    x,
                    vec_s,
                    vec_t,
                    order,
                    solver_type=solver_type,
                    r1=r1,
                    r2=r2)
        if denoise_to_zero:
            x = self.denoise_to_zero_fn(
                x,
                torch.ones((x.shape[0], )).to(device) * t_0)
        return x


def interpolate_fn(x, xp, yp):
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat(
        [x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    end_idx = torch.where(
        torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(
        sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(
        sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(
        y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(
        y_positions_expanded, dim=2,
        index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    return v[(..., ) + (None, ) * (dims - 1)]
