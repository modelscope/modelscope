# Part of the implementation is borrowed and modified from Deep3DFaceRecon_pytorch,
# publicly available at https://github.com/sicxu/Deep3DFaceRecon_pytorch
import os

import cv2
import numpy as np
import torch

from modelscope.models import MODELS, TorchModel
from modelscope.models.cv.face_reconstruction.models import opt
from .. import utils
from . import networks
from .bfm import ParametricFaceModel
from .losses import (CLIPLoss_relative, TVLoss, TVLoss_std, landmark_loss,
                     perceptual_loss, photo_loss, points_loss_horizontal,
                     reflectance_loss, reg_loss)
from .nv_diffrast import MeshRenderer


@MODELS.register_module('face-reconstruction', 'face_reconstruction')
class FaceReconModel(TorchModel):

    def __init__(self,
                 model_dir,
                 w_color=1.92,
                 w_exp=0.8,
                 w_gamma=10.0,
                 w_id=1.0,
                 w_lm=0.0016,
                 w_reg=0.0003,
                 w_tex=0.017,
                 *args,
                 **kwargs):
        """The FaceReconModel is implemented based on Deep3DFaceRecon_pytorch, publicly available at
        https://github.com/sicxu/Deep3DFaceRecon_pytorch

        Args:
            model_dir: the root directory of the model files
            w_color: the weight of color loss
            w_exp: the regularization weight of expression
            w_gamma: the regularization weight of lighting
            w_id: the regularization weight of identity
            w_lm: the weight of landmark loss
            w_reg: the weight of regularization loss
            w_tex: the regularization weight of texture
        """
        super().__init__(model_dir, *args, **kwargs)

        opt.bfm_folder = os.path.join(model_dir, 'assets')
        self.opt = opt
        self.w_color = w_color
        self.w_exp = w_exp
        self.w_gamma = w_gamma
        self.w_id = w_id
        self.w_lm = w_lm
        self.w_reg = w_reg
        self.w_tex = w_tex
        self.device = torch.device('cpu')
        self.isTrain = opt.isTrain
        self.visual_names = ['output_vis']
        self.model_names = ['net_recon']
        self.parallel_names = self.model_names + ['renderer']

        self.net_recon = networks.define_net_recon(
            net_recon=opt.net_recon,
            use_last_fc=opt.use_last_fc,
            init_path=None)

        self.facemodel = ParametricFaceModel(
            bfm_folder=opt.bfm_folder,
            camera_distance=opt.camera_d,
            focal=opt.focal,
            center=opt.center,
            is_train=self.isTrain,
            default_name=opt.bfm_model)

        self.facemodel_front = ParametricFaceModel(
            bfm_folder=opt.bfm_folder,
            camera_distance=opt.camera_d,
            focal=opt.focal,
            center=opt.center,
            is_train=self.isTrain,
            default_name='face_model_for_maas.mat')

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

        self.nonlinear_UVs = self.facemodel.nonlinear_UVs
        self.nonlinear_UVs = torch.from_numpy(self.nonlinear_UVs).to(
            torch.device('cuda'))

        template_obj_path = os.path.join(opt.bfm_folder, 'template_mesh.obj')
        self.template_mesh = utils.read_obj(template_obj_path)

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

        # loss func name: (compute_%s_loss) % loss_name
        self.compute_feat_loss = perceptual_loss
        self.comupte_color_loss = photo_loss
        self.compute_lm_loss = landmark_loss
        self.compute_reg_loss = reg_loss
        self.compute_reflc_loss = reflectance_loss

    def load_networks(self, load_path):
        state_dict = torch.load(load_path, map_location=self.device)
        print('loading the model from %s' % load_path)

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                net.load_state_dict(state_dict[name], strict=False)

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

    def set_render(self, image_res):
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

    def get_edge_points_vertical(self):
        top_points = []
        bottom_points = []
        for i in range(self.face_mask.shape[3]):
            inds = torch.where(self.face_mask[0, 0, :, i] > 0.9)
            if len(inds[0]) > 0:
                top_points.append(int(inds[0][0]))
                bottom_points.append(int(inds[0][-1]))
            else:
                top_points.append(0)
                bottom_points.append(self.face_mask.shape[2] - 1)
        self.top_points = torch.tensor(top_points).long().to(self.device)
        self.bottom_points = torch.tensor(bottom_points).long().to(self.device)

    def blur_shape_offset_uv(self, global_blur=False, blur_size=3):
        if self.edge_mask is not None:
            shape_offset_uv_blur = self.shape_offset_uv[0].detach().cpu(
            ).numpy()
            shape_offset_uv_blur = cv2.blur(shape_offset_uv_blur, (15, 15))
            shape_offset_uv_blur = torch.from_numpy(
                shape_offset_uv_blur).float().to(self.device)[None, ...]
            value_1 = shape_offset_uv_blur * self.edge_mask[None, ..., None]
            value_2 = self.shape_offset_uv * (
                1 - self.edge_mask[None, ..., None])
            self.shape_offset_uv = value_1 + value_2

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

    def get_fusion_mask(self):

        h, w = self.shape_offset_uv.shape[1:3]
        self.fusion_mask = torch.zeros((h, w)).to(self.device).float()
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

    def fitting_nonlinear(self, coeff, debug=False, n_iters=100, out_dir=None):
        output_coeff = coeff.detach().clone()

        output_coeff = self.facemodel_front.split_coeff(output_coeff)
        output_coeff['id'].requires_grad = True
        output_coeff['exp'].requires_grad = True
        output_coeff['tex'].requires_grad = True
        output_coeff['angle'].requires_grad = True
        output_coeff['gamma'].requires_grad = True
        output_coeff['trans'].requires_grad = True

        self.shape_offset_uv = torch.zeros(
            (1, 300, 300, 3),
            dtype=torch.float32).to(self.device)  # (1, 180, 256, 3)
        self.shape_offset_uv.requires_grad = True

        self.texture_offset_uv = torch.zeros(
            (1, 300, 300, 3),
            dtype=torch.float32).to(self.device)  # (1, 180, 256, 3)
        self.texture_offset_uv.requires_grad = True

        value_list = [
            self.shape_offset_uv, self.texture_offset_uv, output_coeff['id'],
            output_coeff['exp'], output_coeff['tex'], output_coeff['angle'],
            output_coeff['gamma'], output_coeff['trans']
        ]
        optim = torch.optim.Adam(value_list, lr=1e-3)

        self.get_edge_points_horizontal()
        self.get_edge_points_vertical()

        self.cur_iter = 0
        for i in range(n_iters):  # 500
            self.pred_vertex, _, self.pred_color, self.pred_lm, _, face_shape_offset, self.verts_proj = \
                self.facemodel_front.compute_for_render_train_nonlinear(output_coeff, self.shape_offset_uv,
                                                                        self.texture_offset_uv,
                                                                        self.nonlinear_UVs[:35709, ...])
            self.pred_mask, _, self.pred_face, self.occ = self.renderer_fitting(
                self.pred_vertex,
                self.facemodel_front.face_buf,
                feat=self.pred_color)

            self.pred_coeffs_dict = self.facemodel_front.split_coeff(
                output_coeff)
            self.compute_losses_fitting()
            if debug and i % 10 == 0:
                print('{}: total loss: {:.6f}'.format(i, self.loss_all.item()))

            optim.zero_grad()
            self.loss_all.backward()
            optim.step()

            self.cur_iter += 1

        output_coeff = self.facemodel_front.merge_coeff(output_coeff)

        self.get_edge_mask()
        self.get_fusion_mask()
        self.blur_shape_offset_uv()

        self.pred_vertex, _, self.pred_color, self.pred_lm, _, face_shape_offset, self.verts_proj = \
            self.facemodel_front.compute_for_render_train_nonlinear(output_coeff, self.shape_offset_uv,
                                                                    self.texture_offset_uv,
                                                                    self.nonlinear_UVs[:35709, ...])

        if out_dir is not None:
            input_img_numpy = 255. * (self.input_img).detach().cpu().permute(
                0, 2, 3, 1).numpy()
            input_img_numpy = np.squeeze(input_img_numpy)

            output_vis = self.pred_face
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(
                0, 2, 3, 1).numpy()
            output_vis_numpy_raw = np.squeeze(output_vis_numpy_raw)

            output_vis_numpy = np.concatenate(
                (input_img_numpy, output_vis_numpy_raw), axis=-2)

            output_vis = np.squeeze(output_vis_numpy)
            output_vis = output_vis[..., ::-1]  # rgb->bgr
            output_face_mask = self.pred_mask.detach().cpu().permute(
                0, 2, 3, 1).squeeze().numpy() * 255.0
            output_vis = np.column_stack(
                (output_vis, cv2.cvtColor(output_face_mask,
                                          cv2.COLOR_GRAY2BGR)))
            output_input_vis = output_vis[:, :224]
            output_pred_vis = output_vis[:, 224:448]
            output_mask_vis = output_vis[:, 448:]

            face_mask_vis = 255. * self.face_mask.detach().cpu()[0, 0].numpy()

            shape_offset_vis = self.shape_offset_uv.detach().cpu().numpy()[0]
            shape_offset_vis = (shape_offset_vis - shape_offset_vis.min()) / (
                shape_offset_vis.max() - shape_offset_vis.min()) * 255.0

            cv2.imwrite(
                os.path.join(out_dir, 'fitting_01_input.jpg'),
                output_input_vis)
            cv2.imwrite(
                os.path.join(out_dir, 'fitting_02_pred.jpg'), output_pred_vis)
            cv2.imwrite(
                os.path.join(out_dir, 'fitting_03_mask.jpg'), output_mask_vis)
            cv2.imwrite(
                os.path.join(out_dir, 'fitting_04_facemask.jpg'),
                face_mask_vis)
            cv2.imwrite(
                os.path.join(out_dir, 'fitting_05_shape_offset.jpg'),
                shape_offset_vis)

        recon_shape_offset = face_shape_offset
        recon_shape_offset[..., -1] = 10 - recon_shape_offset[
            ..., -1]  # from camera space to world space
        recon_shape_offset = recon_shape_offset.detach().cpu().numpy()[0]

        tri = self.facemodel_front.face_buf.cpu().numpy()
        pred_color = self.pred_color.detach().cpu().numpy()[0].clip(0, 1)

        output = {
            'coeffs': output_coeff,
            'face_vertices': recon_shape_offset,
            'face_faces': tri + 1,
            'face_colors': pred_color
        }
        return output

    def forward(self, out_dir=None):
        self.facemodel.to(self.device)
        self.facemodel_front.to(self.device)
        with torch.no_grad():

            output_coeff = self.net_recon(self.input_img)

        with torch.enable_grad():
            output = self.fitting_nonlinear(
                output_coeff, debug=True, out_dir=out_dir)

        output_coeff = output['coeffs']
        output_coeff = self.facemodel.split_coeff(output_coeff)
        eye_coeffs = output_coeff['exp'][0, 16] + output_coeff['exp'][
            0, 17] + output_coeff['exp'][0, 19]
        if eye_coeffs > 1.0:
            degree = 0.5
        else:
            degree = 1.0
        output_coeff['exp'][0, 16] += 1 * degree
        output_coeff['exp'][0, 17] += 1 * degree
        output_coeff['exp'][0, 19] += 1.5 * degree
        output_coeff = self.facemodel.merge_coeff(output_coeff)

        self.pred_vertex, face_shape_ori, head_shape = \
            self.facemodel.compute_for_render_nonlinear_full(output_coeff, self.shape_offset_uv.detach(),
                                                             self.nonlinear_UVs, nose_coeff=0.1)

        UVs_tensor = torch.tensor(
            self.template_mesh['uvs'],
            dtype=torch.float32)[None, ...].to(self.pred_vertex.device)
        target_img = self.input_fat_img_hd.permute(0, 2, 3, 1)
        with torch.enable_grad():
            _, _, _, texture_map, _ = self.renderer.pred_shape_and_texture(
                self.pred_vertex, self.facemodel.face_buf, UVs_tensor,
                target_img)

        recon_shape = head_shape
        recon_shape[
            ...,
            -1] = 10 - recon_shape[..., -1]  # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        tri = self.facemodel.face_buf.cpu().numpy()
        normals = utils.estimate_normals(recon_shape, tri)

        output['head_vertices'] = recon_shape
        output['head_faces'] = tri + 1
        output['head_tex_map'] = texture_map
        output['head_UVs'] = self.template_mesh['uvs']
        output['head_faces_uv'] = self.template_mesh['faces_uv']
        output['head_normals'] = normals

        return output

    def compute_losses_fitting(self):
        face_mask = self.pred_mask

        face_mask = face_mask.detach()
        self.loss_color = self.w_color * self.comupte_color_loss(
            self.pred_face, self.input_img, face_mask)  # 1.0

        self.loss_color_nose = torch.tensor(0.0).float().to(self.device)

        loss_reg, loss_gamma = self.compute_reg_loss(self.pred_coeffs_dict,
                                                     self.w_id, self.w_exp,
                                                     self.w_tex)
        self.loss_reg = self.w_reg * loss_reg  # 1.0
        self.loss_gamma = self.w_gamma * loss_gamma  # 1.0

        self.loss_lm = self.w_lm * self.compute_lm_loss(
            self.pred_lm, self.gt_lm) * 0.1  # 0.1

        self.loss_smooth_offset = TVLoss()(self.shape_offset_uv.permute(
            0, 3, 1, 2)) * 10000  # 10000

        self.loss_reg_offset = torch.tensor(0.0).float().to(self.device)

        self.loss_reg_textureOff = torch.mean(
            torch.abs(self.texture_offset_uv)) * 10  # 10

        self.loss_smooth_offset_std = TVLoss_std()(
            self.shape_offset_uv.permute(0, 3, 1, 2)) * 50000  # 50000

        self.loss_points_horizontal, self.edge_points_inds = points_loss_horizontal(
            self.verts_proj, self.left_points, self.right_points)  # 20
        self.loss_points_horizontal *= 20
        self.loss_points_horizontal_jaw = torch.tensor(0.0).float().to(
            self.device)
        self.loss_points_vertical = torch.tensor(0.0).float().to(self.device)
        self.loss_normals = torch.tensor(0.0).float().to(self.device)

        self.loss_all = self.loss_color + self.loss_lm + self.loss_reg + self.loss_gamma + self.loss_smooth_offset
        self.loss_all += self.loss_reg_offset + self.loss_smooth_offset_std + self.loss_points_horizontal
        self.loss_all += self.loss_points_vertical + self.loss_reg_textureOff
        self.loss_all += self.loss_color_nose + self.loss_normals + self.loss_points_horizontal_jaw

        self.loss_mask = torch.tensor(0.0).float().to(self.device)
