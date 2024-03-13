# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch

from modelscope.models import MODELS, TorchModel
from modelscope.models.cv.face_reconstruction.utils import (estimate_normals,
                                                            read_obj)
from . import networks, opt
from .bfm import ParametricFaceModel
from .losses import (BinaryDiceLoss, TVLoss, TVLoss_std, landmark_loss,
                     perceptual_loss, photo_loss, points_loss_horizontal,
                     reflectance_loss, reg_loss)
from .nv_diffrast import MeshRenderer


@MODELS.register_module('head-reconstruction', 'head_reconstruction')
class HeadReconModel(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        """The HeadReconModel is implemented based on HRN, publicly available at
        https://github.com/youngLBW/HRN

        Args:
            model_dir: the root directory of the model files
        """
        super().__init__(model_dir, *args, **kwargs)

        self.model_dir = model_dir
        opt.bfm_folder = os.path.join(model_dir, 'assets')
        self.opt = opt
        self.isTrain = opt.isTrain
        self.visual_names = ['output_vis']
        self.model_names = ['net_recon']
        self.parallel_names = self.model_names + [
            'renderer', 'renderer_fitting'
        ]

        # networks
        self.net_recon = networks.define_net_recon(
            net_recon=opt.net_recon,
            use_last_fc=opt.use_last_fc,
            init_path=None)

        # assets
        self.headmodel = ParametricFaceModel(
            assets_root=opt.bfm_folder,
            camera_distance=opt.camera_d,
            focal=opt.focal,
            center=opt.center,
            is_train=self.isTrain,
            default_name='ourRefineBFMEye0504_model.mat')

        self.headmodel_for_fitting = ParametricFaceModel(
            assets_root=opt.bfm_folder,
            camera_distance=opt.camera_d,
            focal=opt.focal,
            center=opt.center,
            is_train=self.isTrain,
            default_name='ourRefineFull_model.mat')

        # renderer
        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        self.renderer = MeshRenderer(
            rasterize_fov=fov,
            znear=opt.z_near,
            zfar=opt.z_far,
            rasterize_size=int(2 * opt.center))

        self.renderer_fitting = MeshRenderer(
            rasterize_fov=fov,
            znear=opt.z_near,
            zfar=opt.z_far,
            rasterize_size=int(2 * opt.center))

        template_obj_path = os.path.join(
            model_dir,
            'assets/3dmm/template_mesh/template_ourFull_bfmEyes.obj')
        self.template_output_mesh = read_obj(template_obj_path)

        self.nonlinear_UVs = self.template_output_mesh['uvs']
        self.nonlinear_UVs = torch.from_numpy(self.nonlinear_UVs)

        self.jaw_edge_mask = cv2.imread(
            os.path.join(model_dir,
                         'assets/texture/jaw_edge_mask2.png'))[..., 0].astype(
                             np.float32) / 255.0
        self.jaw_edge_mask = cv2.resize(self.jaw_edge_mask, (300, 300))[...,
                                                                        None]

        self.input_imgs = []
        self.input_img_hds = []
        self.input_fat_img_hds = []
        self.atten_masks = []
        self.gt_lms = []
        self.gt_lm_hds = []
        self.trans_ms = []
        self.img_names = []
        self.face_masks = []
        self.head_masks = []
        self.input_imgs_coeff = []
        self.gt_lms_coeff = []

        self.loss_names = [
            'all', 'feat', 'color', 'lm', 'reg', 'gamma', 'reflc'
        ]

        self.compute_feat_loss = perceptual_loss
        self.compute_color_loss = photo_loss
        self.compute_lm_loss = landmark_loss
        self.compute_reg_loss = reg_loss
        self.compute_reflc_loss = reflectance_loss

        if opt.isTrain:
            self.optimizer = torch.optim.Adam(
                self.net_recon.parameters(), lr=opt.lr)
            self.optimizers = [self.optimizer]
            self.parallel_names += ['net_recog']

    def set_device(self, device):
        self.device = device
        self.net_recon = self.net_recon.to(self.device)
        self.headmodel.to(self.device)
        self.headmodel_for_fitting.to(self.device)
        self.nonlinear_UVs = self.nonlinear_UVs.to(self.device)

    def load_networks(self, load_path):
        state_dict = torch.load(load_path, map_location=self.device)
        print('loading the model from %s' % load_path)

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                net.load_state_dict(state_dict[name], strict=False)

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

    def set_render(self, image_res=1024):
        fov = 2 * np.arctan(self.opt.center / self.opt.focal) * 180 / np.pi
        if image_res is None:
            image_res = int(2 * self.opt.center)

        self.renderer = MeshRenderer(
            rasterize_fov=fov,
            znear=self.opt.z_near,
            zfar=self.opt.z_far,
            rasterize_size=image_res)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['imgs'].to(self.device)
        self.input_img_hd = input['imgs_hd'].to(
            self.device) if 'imgs_hd' in input else None

        if 'imgs_fat_hd' not in input or input['imgs_fat_hd'] is None:
            self.input_fat_img_hd = self.input_img_hd
        else:
            self.input_fat_img_hd = input['imgs_fat_hd'].to(self.device)

        self.atten_mask = input['msks'].to(
            self.device) if 'msks' in input else None
        self.gt_lm = input['lms'].to(self.device) if 'lms' in input else None
        self.gt_lm_hd = input['lms_hd'].to(
            self.device) if 'lms_hd' in input else None
        self.trans_m = input['M'].to(self.device) if 'M' in input else None
        self.image_paths = input['im_paths'] if 'im_paths' in input else None
        self.img_name = input['img_name'] if 'img_name' in input else None
        self.face_mask = input['face_mask'].to(
            self.device) if 'face_mask' in input else None
        self.head_mask = input['head_mask'].to(
            self.device) if 'head_mask' in input else None
        self.gt_normals = input['normals'].to(
            self.device) if 'normals' in input else None
        self.input_img_coeff = input['imgs_coeff'].to(
            self.device) if 'imgs_coeff' in input else None
        self.gt_lm_coeff = input['lms_coeff'].to(
            self.device) if 'lms_coeff' in input else None

    def check_head_pose(self, coeffs):
        pi = 3.14
        if coeffs[0, 225] > pi / 6 or coeffs[0, 225] < -pi / 6:
            return False
        elif coeffs[0, 224] > pi / 6 or coeffs[0, 224] < -pi / 6:
            return False
        elif coeffs[0, 226] > pi / 6 or coeffs[0, 226] < -pi / 6:
            return False
        else:
            return True

    def get_fusion_mask(self, keep_forehead=True):
        self.without_forehead_inds = torch.from_numpy(
            np.load(
                os.path.join(self.model_dir,
                             'assets/3dmm/inds/bfm_withou_forehead_inds.npy'))
        ).long().to(self.device)

        h, w = self.shape_offset_uv.shape[1:3]
        self.fusion_mask = torch.zeros((h, w)).to(self.device).float()
        if keep_forehead:
            UVs_coords = self.nonlinear_UVs.clone()[:35709][
                self.without_forehead_inds]
        else:
            UVs_coords = self.nonlinear_UVs.clone()[:35709]
        UVs_coords[:, 0] *= w
        UVs_coords[:, 1] *= h
        UVs_coords_int = torch.floor(UVs_coords)
        UVs_coords_int = UVs_coords_int.long()

        self.fusion_mask[h - 1 - UVs_coords_int[:, 1], UVs_coords_int[:,
                                                                      0]] = 1

        # blur mask
        self.fusion_mask = self.fusion_mask.cpu().numpy()
        new_kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        new_kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        self.fusion_mask = cv2.dilate(self.fusion_mask, new_kernel1, 1)
        self.fusion_mask = cv2.erode(self.fusion_mask, new_kernel2, 1)
        self.fusion_mask = cv2.blur(self.fusion_mask, (17, 17))
        self.fusion_mask = torch.from_numpy(self.fusion_mask).float().to(
            self.device)

    def get_edge_mask(self):

        h, w = self.shape_offset_uv.shape[1:3]
        self.edge_mask = torch.zeros((h, w)).to(self.device).float()
        UVs_coords = self.nonlinear_UVs.clone()[self.edge_points_inds]
        UVs_coords[:, 0] *= w
        UVs_coords[:, 1] *= h
        UVs_coords_int = torch.floor(UVs_coords)
        UVs_coords_int = UVs_coords_int.long()

        self.edge_mask[h - 1 - UVs_coords_int[:, 1], UVs_coords_int[:, 0]] = 1

        # blur mask
        self.edge_mask = self.edge_mask.cpu().numpy()
        new_kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        self.edge_mask = cv2.dilate(self.edge_mask, new_kernel1, 1)
        self.edge_mask = cv2.blur(self.edge_mask, (5, 5))
        self.edge_mask = torch.from_numpy(self.edge_mask).float().to(
            self.device)

    def blur_shape_offset_uv(self, global_blur=False, blur_size=3):
        if self.edge_mask is not None:
            shape_offset_uv_blur = self.shape_offset_uv[0].detach().cpu(
            ).numpy()
            shape_offset_uv_blur = cv2.blur(shape_offset_uv_blur, (15, 15))
            shape_offset_uv_blur = torch.from_numpy(
                shape_offset_uv_blur).float().to(self.device)[None, ...]
            self.shape_offset_uv = shape_offset_uv_blur * self.edge_mask[
                None, ..., None] + self.shape_offset_uv * (
                    1 - self.edge_mask[None, ..., None])

        self.shape_offset_uv = self.shape_offset_uv * self.fusion_mask[None,
                                                                       ...,
                                                                       None]

        if global_blur and blur_size > 0:
            shape_offset_uv_blur = self.shape_offset_uv[0].detach().cpu(
            ).numpy()
            shape_offset_uv_blur = cv2.blur(shape_offset_uv_blur,
                                            (blur_size, blur_size))
            shape_offset_uv_blur = torch.from_numpy(
                shape_offset_uv_blur).float().to(self.device)[None, ...]
            self.shape_offset_uv = shape_offset_uv_blur

    def blur_offset_edge(self):
        shape_offset_uv = self.shape_offset_uv[0].detach().cpu().numpy()
        shape_offset_uv_head = self.shape_offset_uv_head[0].detach().cpu(
        ).numpy()
        shape_offset_uv_head = cv2.resize(shape_offset_uv_head, (300, 300))
        shape_offset_uv_head = shape_offset_uv_head * (
            1 - self.jaw_edge_mask) + shape_offset_uv * self.jaw_edge_mask
        shape_offset_uv_head = cv2.resize(shape_offset_uv_head, (100, 100))

        self.shape_offset_uv_head = torch.from_numpy(
            shape_offset_uv_head).float().to(self.device)[None, ...]

    def fitting_nonlinear(self, coeff, n_iters=250):
        output_coeff = coeff.detach().clone()

        output_coeff = self.headmodel_for_fitting.split_coeff(output_coeff)
        output_coeff['id'].requires_grad = True
        output_coeff['exp'].requires_grad = True
        output_coeff['tex'].requires_grad = True
        output_coeff['angle'].requires_grad = True
        output_coeff['gamma'].requires_grad = True
        output_coeff['trans'].requires_grad = True

        self.shape_offset_uv = torch.zeros((1, 300, 300, 3),
                                           dtype=torch.float32).to(self.device)
        self.shape_offset_uv.requires_grad = True

        self.texture_offset_uv = torch.zeros(
            (1, 300, 300, 3), dtype=torch.float32).to(self.device)
        self.texture_offset_uv.requires_grad = True

        self.shape_offset_uv_head = torch.zeros(
            (1, 100, 100, 3), dtype=torch.float32).to(self.device)
        self.shape_offset_uv_head.requires_grad = True

        self.texture_offset_uv_head = torch.zeros(
            (1, 100, 100, 3), dtype=torch.float32).to(self.device)
        self.texture_offset_uv_head.requires_grad = True

        head_face_inds = np.load(
            os.path.join(self.model_dir,
                         'assets/3dmm/inds/ours_head_face_inds.npy'))
        head_face_inds = torch.from_numpy(head_face_inds).to(self.device)
        head_faces = self.headmodel_for_fitting.face_buf[head_face_inds]

        # print('before fitting', output_coeff)

        opt_parameters = [
            self.shape_offset_uv, self.texture_offset_uv,
            self.shape_offset_uv_head, self.texture_offset_uv_head,
            output_coeff['id'], output_coeff['exp'], output_coeff['tex'],
            output_coeff['gamma']
        ]
        optim = torch.optim.Adam(opt_parameters, lr=1e-3)

        optim_pose = torch.optim.Adam([output_coeff['trans']], lr=1e-1)

        self.get_edge_points_horizontal()

        for i in range(n_iters):  # 500
            self.pred_vertex_head, self.pred_tex, self.pred_color_head, self.pred_lm, face_shape, \
                face_shape_offset, self.verts_proj_head = \
                self.headmodel_for_fitting.compute_for_render_head_fitting(output_coeff, self.shape_offset_uv,
                                                                           self.texture_offset_uv,
                                                                           self.shape_offset_uv_head,
                                                                           self.texture_offset_uv_head,
                                                                           self.nonlinear_UVs)
            self.pred_vertex = self.pred_vertex_head[:, :35241]
            self.pred_color = self.pred_color_head[:, :35241]
            self.verts_proj = self.verts_proj_head[:, :35241]
            self.pred_mask_head, _, self.pred_head, self.occ_head = self.renderer_fitting(
                self.pred_vertex_head, head_faces, feat=self.pred_color_head)
            self.pred_mask, _, self.pred_face, self.occ_face = self.renderer_fitting(
                self.pred_vertex,
                self.headmodel_for_fitting.face_buf[:69732],
                feat=self.pred_color)

            self.pred_coeffs_dict = self.headmodel_for_fitting.split_coeff(
                output_coeff)
            self.compute_losses_fitting()

            if i < 150:
                optim_pose.zero_grad()
                (self.loss_lm + self.loss_color * 0.1).backward()
                optim_pose.step()
            else:
                optim.zero_grad()
                self.loss_all.backward()
                optim.step()

        output_coeff = self.headmodel_for_fitting.merge_coeff(output_coeff)

        self.get_edge_mask()
        self.get_fusion_mask(keep_forehead=False)
        self.blur_shape_offset_uv(global_blur=True)
        self.blur_offset_edge()
        return output_coeff

    def forward(self):
        with torch.no_grad():
            output_coeff = self.net_recon(self.input_img_coeff)

        if not self.check_head_pose(output_coeff):
            return None

        with torch.enable_grad():
            output_coeff = self.fitting_nonlinear(output_coeff)

        output_coeff = self.headmodel.split_coeff(output_coeff)
        eye_coeffs = output_coeff['exp'][0, 16] + output_coeff['exp'][
            0, 17] + output_coeff['exp'][0, 19]
        if eye_coeffs > 1.0:
            degree = 0.5
        else:
            degree = 1.0
        # degree = 0.5
        output_coeff['exp'][0, 16] += 1 * degree
        output_coeff['exp'][0, 17] += 1 * degree
        output_coeff['exp'][0, 19] += 1.5 * degree
        output_coeff = self.headmodel.merge_coeff(output_coeff)

        self.pred_vertex, _, _, _, face_shape_ori, face_shape, _ = \
            self.headmodel.compute_for_render_head(output_coeff,
                                                   self.shape_offset_uv.detach(),
                                                   self.texture_offset_uv.detach(),
                                                   self.shape_offset_uv_head.detach() * 0,
                                                   self.texture_offset_uv_head.detach(),
                                                   self.nonlinear_UVs,
                                                   nose_coeff=0.1,
                                                   neck_coeff=0.3,
                                                   neckSlim_coeff=0.5,
                                                   neckStretch_coeff=0.5)

        UVs = np.array(self.template_output_mesh['uvs'])
        UVs_tensor = torch.tensor(UVs, dtype=torch.float32)
        UVs_tensor = torch.unsqueeze(UVs_tensor, 0).to(self.pred_vertex.device)

        target_img = self.input_fat_img_hd
        target_img = target_img.permute(0, 2, 3, 1)
        face_buf = self.headmodel.face_buf
        # get texture map
        with torch.enable_grad():
            pred_mask, _, pred_face, texture_map, texture_mask = self.renderer.pred_shape_and_texture(
                self.pred_vertex, face_buf, UVs_tensor, target_img, None)
        self.pred_coeffs_dict = self.headmodel.split_coeff(output_coeff)

        recon_shape = face_shape  # get reconstructed shape, [1, 35709, 3]
        recon_shape[
            ...,
            -1] = 10 - recon_shape[..., -1]  # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        tri = self.headmodel.face_buf.cpu().numpy()

        output = {}
        output['flag'] = 0

        output['tex_map'] = texture_map
        output['tex_mask'] = texture_mask * 255.0
        '''
        coeffs
         {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }
        '''
        output['coeffs'] = self.pred_coeffs_dict

        normals = estimate_normals(recon_shape, tri)

        output['vertices'] = recon_shape
        output['triangles'] = tri
        output['uvs'] = UVs
        output['faces_uv'] = self.template_output_mesh['faces_uv']
        output['normals'] = normals

        return output

    def get_edge_points_horizontal(self):
        left_points = []
        right_points = []
        for i in range(self.face_mask.shape[2]):
            inds = torch.where(self.face_mask[0, 0, i, :] > 0.5)  # 0.9
            if len(inds[0]) > 0:  # i > 112 and len(inds[0]) > 0
                left_points.append(int(inds[0][0]) + 1)
                right_points.append(int(inds[0][-1]))
            else:
                left_points.append(0)
                right_points.append(self.face_mask.shape[3] - 1)
        self.left_points = torch.tensor(left_points).long().to(self.device)
        self.right_points = torch.tensor(right_points).long().to(self.device)

    def compute_losses_fitting(self):
        face_mask = self.pred_mask
        face_mask = face_mask.detach()
        self.loss_color = self.opt.w_color * self.compute_color_loss(
            self.pred_face, self.input_img, face_mask)  # 1.0

        loss_reg, loss_gamma = self.compute_reg_loss(
            self.pred_coeffs_dict,
            w_id=self.opt.w_id,
            w_exp=self.opt.w_exp,
            w_tex=self.opt.w_tex)
        self.loss_reg = self.opt.w_reg * loss_reg  # 1.0
        self.loss_gamma = self.opt.w_gamma * loss_gamma  # 1.0

        self.loss_lm = self.opt.w_lm * self.compute_lm_loss(
            self.pred_lm, self.gt_lm) * 0.1  # 0.1

        self.loss_smooth_offset = TVLoss()(self.shape_offset_uv.permute(
            0, 3, 1, 2)) * 10000  # 10000

        self.loss_reg_textureOff = torch.mean(
            torch.abs(self.texture_offset_uv)) * 10  # 10

        self.loss_smooth_offset_std = TVLoss_std()(
            self.shape_offset_uv.permute(0, 3, 1, 2)) * 50000  # 50000

        self.loss_points_horizontal, self.edge_points_inds = points_loss_horizontal(
            self.verts_proj, self.left_points, self.right_points)  # 20
        self.loss_points_horizontal *= 20

        self.loss_all = self.loss_color + self.loss_lm + self.loss_reg + self.loss_gamma
        self.loss_all += self.loss_smooth_offset + self.loss_smooth_offset_std + self.loss_reg_textureOff
        self.loss_all += self.loss_points_horizontal

        head_mask = self.pred_mask_head
        head_mask = head_mask.detach()
        self.loss_color_head = self.opt.w_color * self.compute_color_loss(
            self.pred_head, self.input_img, head_mask)  # 1.0
        self.loss_smooth_offset_head = TVLoss()(
            self.shape_offset_uv_head.permute(0, 3, 1, 2)) * 100  # 10000
        self.loss_smooth_offset_std_head = TVLoss_std()(
            self.shape_offset_uv_head.permute(0, 3, 1, 2)) * 500  # 50000
        self.loss_mask = BinaryDiceLoss()(self.occ_head, self.head_mask) * 20

        self.loss_all += self.loss_mask + self.loss_color_head
        self.loss_all += self.loss_smooth_offset_head + self.loss_smooth_offset_std_head
