# ------------------------------------------------------------------------
# Modified from https://github.com/Totoro97/NeuS/blob/main/models/renderer.py
# Copyright (c) 2021 Peng Wang.
# Copyright (c) Alibaba, Inc. and its affiliates.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork
from .utils import extract_geometry, sample_pdf


class SurfaceRenderer(nn.Module):

    def __init__(self, conf, device):
        super().__init__()
        self.conf = conf
        self.device = device
        self.sdf_network = SDFNetwork(**self.conf['sdf_network']).to(
            self.device)
        self.variance_network = SingleVarianceNetwork(
            **self.conf['variance_network']).to(self.device)
        self.color_network = RenderingNetwork(
            **self.conf['rendering_network']).to(self.device)
        self.light_network = RenderingNetwork(**self.conf['light_network']).to(
            self.device)
        self.n_samples = self.conf['neus_renderer']['n_samples']
        self.n_importance = self.conf['neus_renderer']['n_importance']
        self.n_outside = self.conf['neus_renderer']['n_outside']
        self.up_sample_steps = self.conf['neus_renderer']['up_sample_steps']
        self.perturb = self.conf['neus_renderer']['perturb']

    def extract_geometry(self,
                         bound_min,
                         bound_max,
                         resolution,
                         threshold=0.0,
                         device='cuda'):
        return extract_geometry(
            bound_min,
            bound_max,
            resolution=resolution,
            threshold=threshold,
            query_func=lambda pts: -self.sdf_network.sdf(pts),
            device=device)

    def render_core_outside(self,
                            rays_o,
                            rays_d,
                            z_vals,
                            sample_dist,
                            nerf,
                            background_rgb=None):
        batch_size, n_samples = z_vals.shape

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists,
             torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[
            ..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(
            pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center],
                        dim=-1)  # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        alpha = 1.0 - torch.exp(
            -F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1),
            -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (
                1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat(
            [torch.zeros([batch_size, 1]).to(self.device), cos_val[:, :-1]],
            dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([
                torch.ones([batch_size, 1]).to(self.device), 1. - alpha + 1e-7
            ], -1), -1)[:, :-1]

        z_samples = sample_pdf(
            z_vals, weights, n_importance, det=True,
            device=self.device).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:,
                     None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(
                batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(
                batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size,
                                           n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    light_network,
                    depth_z=None,
                    background_alpha=None,
                    bg_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([
            dists,
            torch.Tensor([sample_dist]).expand(dists[..., :1].shape).to(
                self.device)
        ], -1)
        mid_z_vals = z_vals + dists * 0.5

        pts = rays_o[:,
                     None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts).squeeze()
        sampled_albedo = color_network(pts, gradients, dirs,
                                       feature_vector).reshape(
                                           batch_size, n_samples, 3)
        sampled_light = light_network(pts, gradients, dirs,
                                      feature_vector).reshape(
                                          batch_size, n_samples, 3)
        sampled_color = sampled_albedo * sampled_light

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)
        iter_cos_p1 = F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
        iter_cos = -(iter_cos_p1 + F.relu(-true_cos) * cos_anneal_ratio)

        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size,
                                                  n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(
            pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (
                1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            foreground_color = sampled_color * inside_sphere[:, :, None]
            background_color = bg_sampled_color[:, :n_samples] * (
                1.0 - inside_sphere)[:, :, None]
            sampled_color = foreground_color + background_color

            sampled_color = torch.cat(
                [sampled_color, bg_sampled_color[:, n_samples:]], dim=1)

        beta = torch.cat([
            torch.ones([batch_size, 1], device=alpha.device), 1. - alpha + 1e-7
        ], -1)
        weights = alpha * torch.cumprod(beta, -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights_sum)

        albedo = (sampled_albedo * weights[:, :, None]).sum(dim=1)

        depth = (mid_z_vals * weights).sum(dim=1)
        if depth_z is not None:
            pts_depth = rays_o[:, None, :] + rays_d[:, None, :] * depth_z[
                ..., :, None]  # n_rays, n_samples, 3
            pts_depth = pts_depth.reshape(-1, 3)
            sdf_depth = sdf_network(pts_depth)[:, :1]
        else:
            sdf_depth = None

        gradients_norm = torch.linalg.norm(
            gradients.reshape(batch_size, n_samples, 3), ord=2, dim=-1)
        gradient_error = (gradients_norm - 1.0)**2
        gradient_error = (relax_inside_sphere * gradient_error).sum()
        gradient_error = gradient_error / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'albedo': albedo,
            'depth': depth,
            'sdf': sdf,
            'sdf_depth': sdf_depth,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }

    def render(self,
               rays_o,
               rays_d,
               near,
               far,
               depth_z=None,
               perturb_overwrite=-1,
               background_rgb=None,
               cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples  # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(self.device)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_end = 1.0 - 1.0 / (self.n_outside + 1.0)
            z_vals_outside = torch.linspace(1e-3, z_vals_end, self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]).to(self.device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (
                    z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper
                                                   - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(
                z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :,
                                                                       None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(
                    batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(
                        rays_o, rays_d, z_vals, sdf,
                        self.n_importance // self.up_sample_steps, 64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(
                        rays_o,
                        rays_d,
                        z_vals,
                        new_z_vals,
                        sdf,
                        last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed,
                                                   sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        ret_fine = self.render_core(
            rays_o,
            rays_d,
            z_vals,
            sample_dist,
            self.sdf_network,
            self.variance_network,
            self.color_network,
            self.light_network,
            depth_z=depth_z,
            background_rgb=background_rgb,
            background_alpha=background_alpha,
            background_sampled_color=background_sampled_color,
            cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']
        albedo_fine = ret_fine['albedo']
        depth_fine = ret_fine['depth']
        sdf_depth = ret_fine['sdf_depth']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(
            dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            'albedo_fine': albedo_fine,
            'depth_fine': depth_fine,
            'sdf_depth': sdf_depth,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'mid_z_vals': ret_fine['mid_z_vals'],
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }
