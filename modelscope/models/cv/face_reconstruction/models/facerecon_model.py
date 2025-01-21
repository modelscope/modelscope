# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch

from modelscope.models import MODELS, TorchModel
from modelscope.models.cv.face_reconstruction.models import opt
from .. import utils
from . import networks
from .bfm import ParametricFaceModel
from .de_retouching_module import DeRetouchingModule
from .losses import TVLoss, photo_loss
from .nv_diffrast import MeshRenderer
from .pix2pix.pix2pix_model import Pix2PixModel
from .pix2pix.pix2pix_options import Pix2PixOptions


@MODELS.register_module('face-reconstruction', 'face_reconstruction')
class FaceReconModel(TorchModel):

    def __init__(self,
                 model_dir,
                 w_color=1.92,
                 tex_iters=100,
                 w_tex_smooth=10,
                 *args,
                 **kwargs):
        """The FaceReconModel is implemented based on Deep3DFaceRecon_pytorch, publicly available at
        https://github.com/sicxu/Deep3DFaceRecon_pytorch

        Args:
            model_dir: the root directory of the model files
            w_color: the weight of color loss
        """
        super().__init__(model_dir, *args, **kwargs)

        opt.bfm_folder = os.path.join(model_dir, 'assets')
        self.opt = opt
        self.w_color = w_color
        self.w_tex_smooth = w_tex_smooth
        self.tex_iters = tex_iters
        self.isTrain = opt.isTrain
        self.visual_names = ['output_vis']
        self.model_names = ['net_recon', 'mid_net', 'high_net']
        self.parallel_names = self.model_names + [
            'renderer', 'renderer_high_res'
        ]

        # networks
        self.net_recon = networks.define_net_recon(
            net_recon=opt.net_recon,
            use_last_fc=opt.use_last_fc,
            init_path=None)

        de_retouching_model_path = os.path.join(model_dir,
                                                'de_retouching_model.pth')
        self.de_retouching_module = DeRetouchingModule(
            de_retouching_model_path)

        self.mid_opt = Pix2PixOptions()
        self.mid_opt.input_nc = 6
        self.mid_opt.output_nc = 3
        self.mid_opt.name = 'mid_net'
        self.mid_net = Pix2PixModel(self.mid_opt).netG

        self.high_opt = Pix2PixOptions()
        self.high_opt.input_nc = 9
        self.high_opt.output_nc = 1
        self.high_opt.name = 'high_net'
        self.high_net = Pix2PixModel(self.high_opt).netG

        self.alpha = (torch.ones(1, dtype=torch.float32) * 0.01)
        self.alpha.requires_grad = True
        self.beta = (torch.ones(1, dtype=torch.float32) * 0.01)
        self.beta.requires_grad = True

        # assets
        self.facemodel_front = ParametricFaceModel(
            assets_folder=opt.bfm_folder,
            camera_distance=opt.camera_d,
            focal=opt.focal,
            center=opt.center,
            is_train=self.isTrain,
            default_name='BFM_model_front.mat')

        bfm_uvs_path = os.path.join(opt.bfm_folder, 'bfm_uvs2.npy')
        self.bfm_UVs = np.load(bfm_uvs_path)
        self.bfm_UVs = torch.from_numpy(self.bfm_UVs).float()

        # renderer
        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        self.renderer_high_res = MeshRenderer(
            rasterize_fov=fov,
            znear=opt.z_near,
            zfar=opt.z_far,
            rasterize_size=int(2 * opt.center))

        self.renderer = MeshRenderer(
            rasterize_fov=fov,
            znear=opt.z_near,
            zfar=opt.z_far,
            rasterize_size=int(2 * opt.center))

        self.compute_color_loss = photo_loss

    def set_device(self, device):
        self.device = device
        self.mid_net = self.mid_net.to(self.device)
        self.high_net = self.high_net.to(self.device)
        self.alpha = self.alpha.to(self.device)
        self.beta = self.beta.to(self.device)
        self.bfm_UVs = self.bfm_UVs.to(self.device)
        self.facemodel_front.to(self.device)

    def load_networks(self, load_path):
        state_dict = torch.load(load_path, map_location=self.device)
        print('loading the model from %s' % load_path)

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                net.load_state_dict(state_dict[name], strict=False)

        self.alpha = state_dict['alpha']
        self.beta = state_dict['beta']

        if self.opt.phase != 'test':
            if self.opt.continue_train:

                try:
                    for i, sched in enumerate(self.schedulers):
                        sched.load_state_dict(state_dict['sched_%02d' % i])
                except Exception as e:
                    print(e)
                    for i, sched in enumerate(self.schedulers):
                        sched.last_epoch = self.opt.epoch_count - 1

    def setup(self, checkpoint_path):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.load_networks(checkpoint_path)

    def parallelize(self, convert_sync_batchnorm=True):
        if not self.opt.use_ddp:
            for name in self.parallel_names:
                if isinstance(name, str):
                    module = getattr(self, name)
                    setattr(self, name, module.to(self.device))
        else:
            for name in self.model_names:
                if isinstance(name, str):
                    module = getattr(self, name)
                    if convert_sync_batchnorm:
                        module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                            module)
                    setattr(
                        self, name,
                        torch.nn.parallel.DistributedDataParallel(
                            module.to(self.device),
                            device_ids=[self.device.index],
                            find_unused_parameters=True,
                            broadcast_buffers=True))

            # DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.
            for name in self.parallel_names:
                if isinstance(name, str) and name not in self.model_names:
                    module = getattr(self, name)
                    setattr(self, name, module.to(self.device))

        # put state_dict of optimizer to gpu device
        if self.opt.phase != 'test':
            if self.opt.continue_train:
                for optim in self.optimizers:
                    for state in optim.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)

    def eval(self):
        """Make models eval mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

        self.alpha.requires_grad = False
        self.beta.requires_grad = False

    def set_render(self, image_res):
        fov = 2 * np.arctan(self.opt.center / self.opt.focal) * 180 / np.pi
        if image_res is None:
            image_res = int(2 * self.opt.center)

        self.renderer_high_res = MeshRenderer(
            rasterize_fov=fov,
            znear=self.opt.z_near,
            zfar=self.opt.z_far,
            rasterize_size=image_res)

    def set_input_base(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['imgs'].to(self.device)
        self.input_img_hd = input['imgs_hd'].to(
            self.device) if 'imgs_hd' in input else None
        self.gt_lm = input['lms'].to(self.device) if 'lms' in input else None
        self.gt_lm_hd = input['lms_hd'].to(
            self.device) if 'lms_hd' in input else None
        self.face_mask = input['face_mask'].to(
            self.device) if 'face_mask' in input else None

    def set_input_hrn(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['input_img'].to(self.device)
        self.input_img_for_tex = input['input_img_for_tex'].to(self.device)
        self.input_img_hd = input['input_img_hd'].to(self.device)
        self.face_mask = input['face_mask'].to(self.device)
        self.gt_lm = input['gt_lm'].to(self.device)
        self.coeffs = input['coeffs'].to(self.device)
        self.position_map = input['position_map'].to(self.device)
        self.texture_map = input['texture_map'].to(self.device)
        self.tex_valid_mask = input['tex_valid_mask'].to(self.device)
        self.de_retouched_albedo_map = input['de_retouched_albedo_map'].to(
            self.device)

    def predict_results_base(self):
        # predict low-frequency coefficients
        with torch.no_grad():
            output_coeff = self.net_recon(self.input_img)

        # 3DMM
        face_vertex, face_albedo_map, face_color_map, landmark, face_vertex_noTrans, position_map = \
            self.facemodel_front.compute_for_render(output_coeff)

        # get texture map
        texture_map = self.facemodel_front.get_texture_map(
            face_vertex, self.input_img_hd)

        # de-retouch
        texture_map_input_high = texture_map.permute(
            0, 3, 1, 2).detach()  # (1, 3, 256, 256)
        texture_map_input_high = (texture_map_input_high - 0.5) * 2
        de_retouched_face_albedo_map = self.de_retouching_module.run(
            face_albedo_map, texture_map_input_high)

        # get valid texture mask to deal with occlusion
        valid_mask = self.facemodel_front.get_texture_map(
            face_vertex, self.face_mask)  # (256, 256, 1)
        valid_mask = valid_mask.permute(0, 3, 1,
                                        2).detach()  # (1, 1, 256, 256)

        # render
        pred_mask, _, pred_face = self.renderer.render_uv_texture(
            face_vertex, self.facemodel_front.face_buf, self.bfm_UVs.clone(),
            face_color_map)

        input_img_numpy = 255. * (self.input_img).detach().cpu().permute(
            0, 2, 3, 1).numpy()
        input_img_numpy = np.squeeze(input_img_numpy)
        output_vis = pred_face * pred_mask + (1 - pred_mask) * self.input_img
        output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(
            0, 2, 3, 1).numpy()
        output_vis_numpy_raw = np.squeeze(output_vis_numpy_raw)
        output_vis_numpy = np.concatenate(
            (input_img_numpy, output_vis_numpy_raw), axis=-2)
        output_vis = np.squeeze(output_vis_numpy)
        output_vis = output_vis[..., ::-1]  # rgb->bgr
        output_face_mask = pred_mask.detach().cpu().permute(
            0, 2, 3, 1).squeeze().numpy() * 255.0
        output_vis = np.column_stack(
            (output_vis, cv2.cvtColor(output_face_mask, cv2.COLOR_GRAY2BGR)))
        output_input_vis = output_vis[:, :224]
        output_pred_vis = output_vis[:, 224:448]

        input_img_hd = 255. * (self.input_img_hd).detach().cpu().permute(
            0, 2, 3, 1).numpy()[..., ::-1]
        input_img_hd = np.squeeze(input_img_hd)

        # from camera space to world space
        recon_vertices = face_vertex  # [1, 35709, 3]
        recon_vertices[..., -1] = 10 - recon_vertices[..., -1]

        recon_shape = face_vertex_noTrans  # [1, 35709, 3]
        recon_shape[..., -1] = 10 - recon_shape[..., -1]

        tri = self.facemodel_front.face_buf

        # output
        output = {}
        output['coeffs'] = output_coeff.detach()  # [B, 257]
        output['vertices'] = recon_vertices.detach()  # [B, 35709, 3]
        output['vertices_noTrans'] = recon_shape.detach()  # [B, 35709, 3]
        output['triangles'] = tri.detach()  # [n_faces, 3], start from 0
        output['UVs'] = self.bfm_UVs.detach()  # [35709, 2]
        output['texture_map'] = texture_map.detach()  # [B, h, w, 3], RGB
        output['albedo_map'] = face_albedo_map.detach()  # [B, 3, h, w]
        output['color_map'] = face_color_map.detach()  # [B, 3, h, w]
        output['position_map'] = position_map.detach()  # [B, 3, h, w]

        output['input_face'] = output_input_vis
        output['pred_face'] = output_pred_vis
        output['input_face_hd'] = input_img_hd

        output['input_img'] = self.input_img
        output['input_img_hd'] = self.input_img_hd
        output['gt_lm'] = self.gt_lm
        output['face_mask'] = self.face_mask
        output['tex_valid_mask'] = valid_mask

        output[
            'de_retouched_albedo_map'] = de_retouched_face_albedo_map.detach()

        return output

    def forward(self, visualize=False):
        self.bfm_UVs = self.bfm_UVs

        # get valid mask to deal with occlusion
        tex_valid_mask = self.tex_valid_mask  # (B, 1, 256, 256)
        if visualize:
            tex_valid_mask = self.smooth_valid_mask(tex_valid_mask)
        tex_valid_mask_mid = torch.nn.functional.interpolate(
            tex_valid_mask, (64, 64), mode='bilinear')

        # mid frequency
        texture_map_input = self.texture_map.permute(0, 3, 1,
                                                     2).to(self.device)
        texture_map_input = (texture_map_input - 0.5) * 2
        texture_map_input_mid = torch.nn.functional.interpolate(
            texture_map_input, (64, 64), mode='bilinear')
        position_map_input_mid = torch.nn.functional.interpolate(
            self.position_map, (64, 64), mode='bilinear')
        input_mid = torch.cat([position_map_input_mid, texture_map_input_mid],
                              dim=1)
        self.deformation_map = self.mid_net(
            input_mid) * 0.1 * self.alpha  # ori * 0.1 * self.alpha
        self.deformation_map = self.deformation_map * tex_valid_mask_mid
        self.deformation_map = self.deformation_map.permute(0, 2, 3, 1)

        # render
        self.pred_vertex, self.pred_color, self.pred_lm, self.verts_proj, self.face_albedo_map, \
            face_shape_transformed, face_norm_roted, self.extra_results = \
            self.facemodel_front.compute_for_render_hierarchical_mid(self.coeffs, self.deformation_map, self.bfm_UVs,
                                                                     visualize=visualize,
                                                                     de_retouched_albedo_map=
                                                                     self.de_retouched_albedo_map)
        self.pred_mask, _, self.pred_face_mid = self.renderer.render_uv_texture(
            self.pred_vertex, self.facemodel_front.face_buf,
            self.bfm_UVs.clone(), self.pred_color)
        self.deformation_map = self.deformation_map.permute(0, 3, 1, 2)

        # get re-aligned texture
        texture_map_input_high = self.facemodel_front.get_texture_map(
            self.pred_vertex, self.input_img_hd)  # (1, 256, 256, 3)
        texture_map_input_high = texture_map_input_high.permute(
            0, 3, 1, 2).detach()  # (1, 3, 256, 256)
        texture_map_input_high = (texture_map_input_high - 0.5) * 2

        # high frequency
        position_map_input_high = torch.nn.functional.interpolate(
            self.position_map, (256, 256), mode='bilinear')
        deformation_map_input_high = torch.nn.functional.interpolate(
            self.deformation_map, (256, 256), mode='bilinear')
        value = [
            position_map_input_high, texture_map_input_high,
            deformation_map_input_high
        ]
        input_high = torch.cat(value, dim=1)
        self.displacement_map = self.high_net(
            input_high) * 0.1 * self.beta  # ori * 0.1 * self.alpha
        self.displacement_map = self.displacement_map * tex_valid_mask

        # render
        self.pred_color_high, self.extra_results = self.facemodel_front.compute_for_render_hierarchical_high(
            self.coeffs,
            self.displacement_map,
            self.de_retouched_albedo_map,
            face_shape_transformed,
            face_norm_roted,
            extra_results=self.extra_results)
        _, _, self.pred_face_high = self.renderer.render_uv_texture(
            self.pred_vertex, self.facemodel_front.face_buf,
            self.bfm_UVs.clone(), self.pred_color_high)

        self.pred_coeffs_dict = self.facemodel_front.split_coeff(self.coeffs)

        if visualize:
            # high
            self.extra_results['pred_mask_high'] = self.pred_mask
            self.extra_results['pred_face_high_color'] = self.pred_face_high
            _, _, self.extra_results[
                'pred_face_high_gray'] = self.renderer.render_uv_texture(
                    self.pred_vertex, self.facemodel_front.face_buf,
                    self.bfm_UVs.clone(), self.extra_results['tex_high_gray'])

            # mid
            self.extra_results['pred_mask_mid'] = self.pred_mask
            self.extra_results['pred_face_mid_color'] = self.pred_face_mid
            _, _, self.extra_results[
                'pred_face_mid_gray'] = self.renderer.render_uv_texture(
                    self.pred_vertex, self.facemodel_front.face_buf,
                    self.bfm_UVs.clone(), self.extra_results['tex_mid_gray'])

            # base
            self.extra_results['pred_mask_base'], _, self.extra_results[
                'pred_face_base_color'] = self.renderer.render_uv_texture(
                    self.extra_results['pred_vertex_base'],
                    self.facemodel_front.face_buf, self.bfm_UVs.clone(),
                    self.extra_results['tex_base_color'])
            _, _, self.extra_results[
                'pred_face_base_gray'] = self.renderer.render_uv_texture(
                    self.extra_results['pred_vertex_base'],
                    self.facemodel_front.face_buf, self.bfm_UVs.clone(),
                    self.extra_results['tex_base_gray'])

            # fit texture
            with torch.enable_grad():
                texture_offset = torch.zeros(
                    (1, 3, 256, 256), dtype=torch.float32).to(self.device)
                texture_offset.requires_grad = True

                optim = torch.optim.Adam([texture_offset], lr=1e-2)

                for i in range(self.tex_iters):
                    pred_color_high = self.pred_color_high.detach(
                    ) + texture_offset
                    _, _, pred_face_high = self.renderer.render_uv_texture(
                        self.pred_vertex.detach(),
                        self.facemodel_front.face_buf, self.bfm_UVs.clone(),
                        pred_color_high)

                    loss_color_high = self.w_color * self.compute_color_loss(
                        pred_face_high, self.input_img_for_tex,
                        self.pred_mask.detach())
                    loss_smooth = TVLoss()(texture_offset) * self.w_tex_smooth
                    loss_all = loss_color_high + loss_smooth
                    optim.zero_grad()
                    loss_all.backward()
                    optim.step()

                self.pred_color_high = (self.pred_color_high
                                        + texture_offset).detach()

            # for video
            self.extra_results['pred_face_high_gray_list'] = []
            self.extra_results['pred_face_high_color_list'] = []
            for i in range(len(self.extra_results['tex_high_gray_list'])):
                _, _, pred_face_high_gray_i = self.renderer_high_res.render_uv_texture(
                    self.extra_results['face_vertex_list'][i],
                    self.facemodel_front.face_buf, self.bfm_UVs.clone(),
                    self.extra_results['tex_high_gray_list'][i])
                self.extra_results['pred_face_high_gray_list'].append(
                    pred_face_high_gray_i)

                _, _, pred_face_high_color_i = self.renderer_high_res.render_uv_texture(
                    self.extra_results['face_vertex_list'][i],
                    self.facemodel_front.face_buf, self.bfm_UVs.clone(),
                    self.pred_color_high)
                self.extra_results['pred_face_high_color_list'].append(
                    pred_face_high_color_i)

    def get_edge_points_horizontal(self):
        left_points_list = []
        right_points_list = []
        for k in range(self.face_mask.shape[0]):
            left_points = []
            right_points = []
            for i in range(self.face_mask.shape[2]):
                inds = torch.where(self.face_mask[k, 0, i, :] > 0.5)  # 0.9
                if len(inds[0]) > 0:  # i > 112 and len(inds[0]) > 0
                    left_points.append(int(inds[0][0]) + 1)
                    right_points.append(int(inds[0][-1]))
                else:
                    left_points.append(0)
                    right_points.append(self.face_mask.shape[3] - 1)
            left_points_list.append(
                torch.tensor(left_points).long().to(self.device))
            right_points_list.append(
                torch.tensor(right_points).long().to(self.device))
        self.left_points = torch.stack(
            left_points_list, dim=0).long().to(self.device)
        self.right_points = torch.stack(
            right_points_list, dim=0).long().to(self.device)

    def smooth_valid_mask(self, tex_valid_mask):
        """

        :param tex_valid_mask: torch.tensor, (B, 1, 256, 256), value: 0~1
        :return:
        """
        batch_size = tex_valid_mask.shape[0]
        tex_valid_mask_ = tex_valid_mask.detach().cpu().numpy()
        result_list = []
        for i in range(batch_size):
            mask = tex_valid_mask_[i, 0]
            mask = cv2.erode(mask, np.ones(shape=(3, 3), dtype=np.float32))
            mask = cv2.blur(mask, (11, 11), 0)
            result_list.append(
                torch.from_numpy(mask)[None].float().to(tex_valid_mask.device))
        smoothed_mask = torch.stack(result_list, dim=0)
        return smoothed_mask

    def compute_visuals_hrn(self):
        with torch.no_grad():

            input_img_vis = 255. * self.input_img.detach().cpu().permute(
                0, 2, 3, 1).numpy()
            output_vis_mid = self.pred_face_mid * self.pred_mask + (
                1 - self.pred_mask) * self.input_img
            output_vis_mid = 255. * output_vis_mid.detach().cpu().permute(
                0, 2, 3, 1).numpy()
            output_vis_high = self.pred_face_high * self.pred_mask + (
                1 - self.pred_mask) * self.input_img
            output_vis_high = 255. * output_vis_high.detach().cpu().permute(
                0, 2, 3, 1).numpy()

            deformation_map_vis = torch.nn.functional.interpolate(
                self.deformation_map,
                input_img_vis.shape[1:3],
                mode='bilinear').permute(0, 2, 3, 1)
            deformation_map_vis = (
                deformation_map_vis - deformation_map_vis.min()) / (
                    deformation_map_vis.max() - deformation_map_vis.min())
            deformation_map_vis = 255. * deformation_map_vis.detach().cpu(
            ).numpy()

            displacement_map_vis = torch.nn.functional.interpolate(
                self.displacement_map,
                input_img_vis.shape[1:3],
                mode='bilinear').permute(0, 2, 3, 1)
            displacement_vis = (
                displacement_map_vis - displacement_map_vis.min()) / (
                    displacement_map_vis.max() - displacement_map_vis.min())
            displacement_vis = 255. * displacement_vis.detach().cpu().numpy()
            displacement_vis = np.concatenate(
                [displacement_vis, displacement_vis, displacement_vis],
                axis=-1)

            face_albedo_map_vis = torch.nn.functional.interpolate(
                self.face_albedo_map,
                input_img_vis.shape[1:3],
                mode='bilinear').permute(0, 2, 3, 1)
            face_albedo_map_vis = 255. * face_albedo_map_vis.detach().cpu(
            ).numpy()

            de_retouched_face_albedo_map_vis = torch.nn.functional.interpolate(
                self.de_retouched_albedo_map,
                input_img_vis.shape[1:3],
                mode='bilinear').permute(0, 2, 3, 1)
            de_retouched_face_albedo_map_vis = 255. * de_retouched_face_albedo_map_vis.detach(
            ).cpu().numpy()

            if self.extra_results is not None:
                pred_mask_base = self.extra_results['pred_mask_base']
                output_vis_base = self.extra_results[
                    'pred_face_base_color'] * pred_mask_base + (
                        1 - pred_mask_base) * self.input_img
                output_vis_base = 255. * output_vis_base.detach().cpu(
                ).permute(0, 2, 3, 1).numpy()

                output_vis_base_gray = self.extra_results[
                    'pred_face_base_gray'] * pred_mask_base + (
                        1 - pred_mask_base) * self.input_img
                output_vis_base_gray = 255. * output_vis_base_gray.detach(
                ).cpu().permute(0, 2, 3, 1).numpy()
                output_vis_mid_gray = self.extra_results[
                    'pred_face_mid_gray'] * self.pred_mask + (
                        1 - self.pred_mask) * self.input_img
                output_vis_mid_gray = 255. * output_vis_mid_gray.detach().cpu(
                ).permute(0, 2, 3, 1).numpy()
                output_vis_high_gray = self.extra_results[
                    'pred_face_high_gray'] * self.pred_mask + (
                        1 - self.pred_mask) * self.input_img
                output_vis_high_gray = 255. * output_vis_high_gray.detach(
                ).cpu().permute(0, 2, 3, 1).numpy()

                output_vis_numpy = np.concatenate(
                    (input_img_vis, output_vis_high, output_vis_base_gray,
                     output_vis_mid_gray, output_vis_high_gray,
                     deformation_map_vis, displacement_vis,
                     face_albedo_map_vis, de_retouched_face_albedo_map_vis),
                    axis=-2)

            elif self.gt_lm is not None:
                gt_lm_numpy = self.gt_lm.cpu().numpy()
                pred_lm_numpy = self.pred_lm.detach().cpu().numpy()
                output_vis_high_lm = utils.draw_landmarks(
                    output_vis_high, gt_lm_numpy, 'b')
                output_vis_high_lm = utils.draw_landmarks(
                    output_vis_high_lm, pred_lm_numpy, 'r')

                output_vis_numpy = np.concatenate(
                    (input_img_vis, output_vis_mid, output_vis_high,
                     output_vis_high_lm, deformation_map_vis,
                     displacement_vis),
                    axis=-2)
            else:
                output_vis_numpy = np.concatenate(
                    (input_img_vis, output_vis_mid, output_vis_high,
                     deformation_map_vis, displacement_vis),
                    axis=-2)

            self.output_vis = torch.tensor(
                output_vis_numpy / 255.,
                dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)[:, :3, ...]
        return visual_ret

    def save_results_hrn(self):
        self.compute_visuals_hrn()
        results = self.get_current_visuals()

        hrn_output_vis_batch = (255.0 * results['output_vis']).permute(
            0, 2, 3, 1).detach().cpu().numpy()[..., ::-1]

        vertices_batch = self.pred_vertex.detach(
        )  # get reconstructed shape, [1, 35709, 3]
        vertices_batch[..., -1] = 10 - vertices_batch[
            ..., -1]  # from camera space to world space
        vertices_batch = vertices_batch.cpu().numpy()

        texture_map_batch = (255.0 * self.pred_color_high).permute(
            0, 2, 3, 1).detach().cpu().numpy()[..., ::-1]

        output = {}

        output['vis_image'] = hrn_output_vis_batch[0]

        texture_map = texture_map_batch[0]
        vertices = vertices_batch[0]
        normals = utils.estimate_normals(
            vertices,
            self.facemodel_front.face_buf.cpu().numpy())
        face_mesh = {
            'vertices': vertices,
            'faces': self.facemodel_front.face_buf.cpu().numpy() + 1,
            'UVs': self.bfm_UVs.detach().cpu().numpy(),
            'faces_uv': self.facemodel_front.face_buf.cpu().numpy() + 1,
            'normals': normals,
            'faces_normal': self.facemodel_front.face_buf.cpu().numpy() + 1,
        }
        output['face_mesh'] = face_mesh
        output['texture_map'] = texture_map

        frame_list = []
        if 'pred_face_high_gray_list' in self.extra_results:
            input_img_vis = 255. * self.input_img_hd.detach().cpu().permute(
                0, 2, 3, 1).numpy()[0]
            for j in range(
                    len(self.extra_results['pred_face_high_gray_list'])):
                pred_face_high_gray_j = self.extra_results[
                    'pred_face_high_gray_list'][j][0, ...]
                pred_face_high_gray_j = 255. * pred_face_high_gray_j.detach(
                ).cpu().permute(1, 2, 0).numpy()
                pred_face_high_color_j = self.extra_results[
                    'pred_face_high_color_list'][j][0, ...]
                pred_face_high_color_j = 255. * pred_face_high_color_j.detach(
                ).cpu().permute(1, 2, 0).numpy()

                value = [
                    input_img_vis, pred_face_high_gray_j,
                    pred_face_high_color_j
                ]
                vis_j = np.concatenate(value, axis=1)

                frame_list.append(vis_j.clip(0, 255).astype(np.uint8))
            output['frame_list'] = frame_list

        return output
