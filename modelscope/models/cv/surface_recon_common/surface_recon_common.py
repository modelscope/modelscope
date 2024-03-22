# Copyright (c) Alibaba, Inc. and its affiliates.

import os

import numpy as np
import torch
import trimesh

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .dataset import Dataset
from .renderer import SurfaceRenderer

logger = get_logger()

__all__ = ['SurfaceReconCommon']


@MODELS.register_module(
    Tasks.surface_recon_common, module_name=Models.surface_recon_common)
class SurfaceReconCommon(TorchModel):

    def __init__(self, model_dir, network_cfg, **kwargs):
        """initialize the surface reconstruction model for common objects.

        Args:
            model_dir (str): the model path.
            network_cfg (dict): args of network config
        """
        super().__init__(model_dir, **kwargs)
        logger.info('model params:{}'.format(kwargs))

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            raise Exception('GPU is required')

        logger.info(network_cfg)

        self.renderer = SurfaceRenderer(network_cfg, device=self.device)
        self.ckpt_path = os.path.join(model_dir, 'model.pth')
        if not os.path.exists(self.ckpt_path):
            raise Exception('model path not found')
        self.load_checkpoint(self.ckpt_path)
        logger.info('load models from %s' % self.ckpt_path)

        self.n_rays = network_cfg['n_rays']

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        for name, module in self.renderer.named_modules():
            saved_name = name + '_fine'
            if saved_name in checkpoint:
                module.load_state_dict(checkpoint[saved_name])

    def surface_reconstruction(self,
                               data_dir,
                               save_dir,
                               color=False,
                               n_directions=8):

        self.dataset = Dataset(data_dir, self.device)

        bound_min = torch.tensor(
            self.dataset.object_bbox_min, dtype=torch.float32).to(self.device)
        bound_max = torch.tensor(
            self.dataset.object_bbox_max, dtype=torch.float32).to(self.device)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=512, threshold=0.0,
                                           device=self.device)
        if color:
            pt_vertices = torch.from_numpy(vertices).cuda().reshape(-1, 1,
                                                                    3).float()
            idx_list = np.linspace(
                0,
                self.dataset.n_images,
                n_directions,
                endpoint=False,
                dtype=int)
            rays_o_list = []
            for idx in idx_list:
                rays_o = self.dataset.pose_all[idx, :3, 3]
                rays_o_list.append(rays_o)

            rgb_final = None
            diff_final = None
            for rays_o in rays_o_list:
                rays_o = rays_o.reshape(1, 3).repeat(vertices.shape[0],
                                                     1).float()

                rays_d = pt_vertices.reshape(-1, 3) - rays_o
                rays_d = rays_d / torch.norm(rays_d, dim=-1).reshape(-1, 1)
                dist = torch.norm(pt_vertices.reshape(-1, 3) - rays_o, dim=-1)

                rays_o = rays_o.reshape(-1, 3).split(self.n_rays)
                rays_d = rays_d.reshape(-1, 3).split(self.n_rays)
                dist = dist.reshape(-1).split(self.n_rays)
                out_rgb_fine = []
                depth_diff = []
                for i, (rays_o_batch,
                        rays_d_batch) in enumerate(zip(rays_o, rays_d)):
                    near, far = self.dataset.near_far_from_sphere(
                        rays_o_batch, rays_d_batch)
                    render_out = self.renderer.render(
                        rays_o_batch,
                        rays_d_batch,
                        near,
                        far,
                        cos_anneal_ratio=1.0,
                        background_rgb=None)

                    # out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                    out_rgb_fine.append(
                        render_out['albedo_fine'].detach().cpu().numpy())

                    weights = render_out['weights']
                    mid_z_vals = render_out['mid_z_vals']
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                    depth_batch = (mid_z_vals[:, :n_samples]
                                   * weights[:, :n_samples]).sum(
                                       dim=1).detach().cpu().numpy()
                    dist_batch = dist[i].detach().cpu().numpy()
                    depth_diff.append(np.abs(depth_batch - dist_batch))

                    del render_out

                out_rgb_fine = np.concatenate(
                    out_rgb_fine, axis=0).reshape(vertices.shape[0], 3)
                depth_diff = np.concatenate(
                    depth_diff, axis=0).reshape(vertices.shape[0])

                if rgb_final is None:
                    rgb_final = out_rgb_fine.copy()
                    diff_final = depth_diff.copy()
                else:
                    ind = diff_final > depth_diff
                    ind = ind.reshape(-1)
                    rgb_final[ind] = out_rgb_fine[ind]
                    diff_final[ind] = depth_diff[ind]

        vertices = vertices * self.dataset.scale_mats_np[0][
            0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        if color:
            logger.info('save mesh with color')
            vert_colors = (255 * np.clip(rgb_final[..., ::-1], 0, 1)).astype(
                np.uint8)
            mesh = trimesh.Trimesh(
                vertices, triangles, vertex_colors=vert_colors)
        else:
            mesh = trimesh.Trimesh(vertices, triangles)

        outpath = os.path.join(save_dir, 'mesh.ply')
        mesh.export(outpath)

        logger.info('surface econstruction done, export mesh to %s' % outpath)
