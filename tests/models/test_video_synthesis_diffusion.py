# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import torch

from modelscope.models.multi_modal.video_synthesis.diffusion import (
    GaussianDiffusion, beta_schedule)


class VideoSynthesisDiffusionTest(unittest.TestCase):

    def setUp(self):
        self.diffusion = GaussianDiffusion(
            beta_schedule('linear_sd', 1000, 0.00085, 0.012),
            mean_type='eps',
            var_type='fixed_small')

    def test_loop_follows_selected_timesteps(self):
        for count in (27, 50, 1000):
            with self.subTest(ddim_timesteps=count):
                calls = []

                def zero_predictor(x, t):
                    calls.append((t.clone(), x.clone()))
                    return torch.zeros_like(x)

                initial = torch.ones((2, 1, 2, 2, 2), dtype=torch.float64)
                result = self.diffusion.ddim_sample_loop(
                    initial, zero_predictor, ddim_timesteps=count, eta=0.0)
                alphas = self.diffusion.alphas_cumprod
                for (t, x), (next_t, next_x) in zip(calls, calls[1:]):
                    # With epsilon=0, DDIM preserves x_t / sqrt(alpha_t).
                    ratio = (alphas[next_t] / alphas[t]).sqrt()
                    expected = ratio.view(2, 1, 1, 1, 1) * x
                    torch.testing.assert_close(
                        next_x, expected, atol=1e-10, rtol=0)

                # Preserve the sampler's existing endpoint at alpha[0].
                first_t = calls[0][0]
                expected = (alphas[0] / alphas[first_t]).sqrt().view(
                    2, 1, 1, 1, 1) * initial
                torch.testing.assert_close(
                    result, expected, atol=1e-10, rtol=0)

    def test_explicit_predecessor_supports_batched_timesteps(self):
        x = torch.ones((2, 1, 2, 2, 2), dtype=torch.float64)
        t = torch.tensor([999, 999])
        prev_t = torch.tensor([963, 999])
        result, _ = self.diffusion.ddim_sample(
            x, t, lambda x, t: torch.zeros_like(x), prev_t=prev_t)
        alphas = self.diffusion.alphas_cumprod
        expected = (alphas[prev_t] / alphas[t]).sqrt().view(2, 1, 1, 1, 1) * x
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=0)

    def test_repeated_timestep_is_identity(self):
        x = torch.ones((2, 1, 2, 2, 2), dtype=torch.float64)
        t = torch.full((2, ), 999, dtype=torch.long)
        for eta in (0.0, 1.0):
            with self.subTest(eta=eta):
                result, _ = self.diffusion.ddim_sample(
                    x,
                    t,
                    lambda x, t: torch.full_like(x, 0.2),
                    eta=eta,
                    prev_t=t)
                torch.testing.assert_close(result, x, atol=1e-10, rtol=0)

    def test_direct_sample_keeps_default_stride(self):
        x = torch.ones((2, 1, 2, 2, 2), dtype=torch.float64)
        t = torch.full((2, ), 80, dtype=torch.long)
        result, _ = self.diffusion.ddim_sample(
            x, t, lambda x, t: torch.zeros_like(x), ddim_timesteps=50)
        alphas = self.diffusion.alphas_cumprod
        expected = (alphas[60] / alphas[80]).sqrt() * x
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=0)


if __name__ == '__main__':
    unittest.main()
