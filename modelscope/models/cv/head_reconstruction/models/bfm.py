# Part of the implementation is borrowed and modified from Deep3DFaceRecon_pytorch,
# publicly available at https://github.com/sicxu/Deep3DFaceRecon_pytorch

import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from modelscope.models.cv.face_reconstruction.utils import read_obj


def perspective_projection(focal, center):
    # return p.T (N, 3) @ (3, 3)
    return np.array([focal, 0, center, 0, focal, center, 0, 0,
                     1]).reshape([3, 3]).astype(np.float32).transpose()


class SH:

    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        self.c = [
            1 / np.sqrt(4 * np.pi),
            np.sqrt(3.) / np.sqrt(4 * np.pi),
            3 * np.sqrt(5.) / np.sqrt(12 * np.pi)
        ]


class ParametricFaceModel:

    def __init__(self,
                 assets_root='assets',
                 recenter=True,
                 camera_distance=10.,
                 init_lit=np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0]),
                 focal=1015.,
                 center=112.,
                 is_train=True,
                 default_name='BFM_model_front.mat'):

        model = loadmat(os.path.join(assets_root, '3dmm/BFM', default_name))
        model_bfm_front = loadmat(
            os.path.join(assets_root, '3dmm/BFM/BFM_model_front.mat'))
        self.mean_shape_ori = model_bfm_front['meanshape'].astype(np.float32)
        # mean face shape. [3*N,1]
        self.mean_shape = model['meanshape'].astype(np.float32)  # (1, 107127)

        # identity basis. [3*N,80]
        self.id_base = model['idBase'].astype(np.float32)  # (107127, 80)

        # expression basis. [3*N,64]
        self.exp_base = model['exBase'].astype(np.float32)  # (107127, 64)

        # mean face texture. [3*N,1] (0-255)
        self.mean_tex = model['meantex'].astype(np.float32)  # (1, 107127)

        # texture basis. [3*N,80]
        self.tex_base = model['texBase'].astype(np.float32)  # (107127, 80)

        self.bfm_keep_inds = np.load(
            os.path.join(assets_root, '3dmm/inds/bfm_keep_inds.npy'))

        self.ours_hair_area_inds = np.load(
            os.path.join(assets_root, '3dmm/inds/ours_hair_area_inds.npy'))

        if default_name == 'ourRefineFull_model.mat':
            self.mean_tex = self.mean_tex.reshape(1, -1, 3)
            mean_tex_keep = self.mean_tex[:, self.bfm_keep_inds]
            self.mean_tex[:, :len(self.bfm_keep_inds)] = mean_tex_keep
            self.mean_tex[:,
                          len(self.bfm_keep_inds):] = np.array([200, 146,
                                                                118])[None,
                                                                      None]
            self.mean_tex[:, self.ours_hair_area_inds] = 40.0
            self.mean_tex = self.mean_tex.reshape(1, -1)
            self.mean_tex = np.ascontiguousarray(self.mean_tex)

            self.tex_base = self.tex_base.reshape(-1, 3, 80)
            tex_base_keep = self.tex_base[self.bfm_keep_inds]
            self.tex_base[:len(self.bfm_keep_inds)] = tex_base_keep
            self.tex_base[len(self.bfm_keep_inds):] = 0.0
            self.tex_base = self.tex_base.reshape(-1, 80)
            self.tex_base = np.ascontiguousarray(self.tex_base)

        # face indices for each vertex that lies in. starts from 0. [N,8]
        self.point_buf = model['point_buf'].astype(np.int64) - 1  # (35709, 8)

        # vertex indices for each face. starts from 0. [F,3]
        self.face_buf = model['tri'].astype(np.int64) - 1  # (70789, 3)

        # vertex indices for 68 landmarks. starts from 0. [68,1]
        self.keypoints = np.squeeze(model['keypoints']).astype(np.int64) - 1

        if default_name == 'ourRefineFull_model.mat':
            self.keypoints = np.load(
                os.path.join(
                    assets_root,
                    '3dmm/inds/our_refine0223_basis_withoutEyes_withUV_keypoints_inds.npy'
                )).astype(np.int64)
            self.point_buf = self.point_buf[:, :8] + 1

        if is_train:
            # vertex indices for small face region to compute photometric error. starts from 0.
            self.front_mask = np.squeeze(model['frontmask2_idx']).astype(
                np.int64) - 1
            # vertex indices for each face from small face region. starts from 0. [f,3]
            self.front_face_buf = model['tri_mask2'].astype(np.int64) - 1
            # vertex indices for pre-defined skin region to compute reflectance loss
            self.skin_mask = np.squeeze(model['skinmask'])

        if default_name == 'ourRefineFull_model.mat':
            nose_reduced_mesh = read_obj(
                os.path.join(assets_root,
                             '3dmm/adjust_part/our_full/145_nose.obj'))
            self.nose_reduced_part = nose_reduced_mesh['vertices'].reshape(
                (1, -1)) - self.mean_shape

            neck_mesh = read_obj(
                os.path.join(assets_root,
                             '3dmm/adjust_part/our_full/154_neck.obj'))
            self.neck_adjust_part = neck_mesh['vertices'].reshape(
                (1, -1)) - self.mean_shape

            eyes_mesh = read_obj(
                os.path.join(
                    assets_root,
                    '3dmm/adjust_part/our_full/our_mean_adjust_eyes.obj'))
            self.eyes_adjust_part = eyes_mesh['vertices'].reshape(
                (1, -1)) - self.mean_shape

            self.neck_slim_part = None
            self.neck_stretch_part = None
        elif default_name == 'ourRefineBFMEye0504_model.mat':
            nose_reduced_mesh = read_obj(
                os.path.join(assets_root,
                             '3dmm/adjust_part/our_full_bfmEyes/145_nose.obj'))
            self.nose_reduced_part = nose_reduced_mesh['vertices'].reshape(
                (1, -1)) - self.mean_shape

            neck_mesh = read_obj(
                os.path.join(assets_root,
                             '3dmm/adjust_part/our_full_bfmEyes/146_neck.obj'))
            self.neck_adjust_part = neck_mesh['vertices'].reshape(
                (1, -1)) - self.mean_shape

            self.eyes_adjust_part = None

            neck_slim_mesh = read_obj(
                os.path.join(
                    assets_root,
                    '3dmm/adjust_part/our_full_bfmEyes/147_neckSlim2.obj'))
            self.neck_slim_part = neck_slim_mesh['vertices'].reshape(
                (1, -1)) - self.mean_shape

            neck_stretch_mesh = read_obj(
                os.path.join(
                    assets_root,
                    '3dmm/adjust_part/our_full_bfmEyes/148_neckLength.obj'))
            self.neck_stretch_part = neck_stretch_mesh['vertices'].reshape(
                (1, -1)) - self.mean_shape
        else:
            self.nose_reduced_part = None

            self.neck_adjust_part = None
            self.eyes_adjust_part = None
            self.neck_slim_part = None
            self.neck_stretch_part = None

        if recenter:
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape_ori = self.mean_shape_ori.reshape([-1, 3])
            mean_shape = mean_shape - np.mean(
                mean_shape_ori[:35709, ...], axis=0, keepdims=True)
            self.mean_shape = mean_shape.reshape([-1, 1])

        eye_corner_inds = np.load(
            os.path.join(assets_root, '3dmm/inds/eye_corner_inds.npy'))
        self.eye_corner_inds = torch.from_numpy(eye_corner_inds).long()
        eye_lines = np.load(
            os.path.join(assets_root, '3dmm/inds/eye_corner_lines.npy'))
        self.eye_lines = torch.from_numpy(eye_lines).long()

        self.center = center
        self.persc_proj = perspective_projection(focal, self.center)
        self.camera_distance = camera_distance
        self.SH = SH()
        self.init_lit = init_lit.reshape([1, 1, -1]).astype(np.float32)

    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value).to(device))

    def compute_shape(self,
                      id_coeff,
                      exp_coeff,
                      nose_coeff=0.0,
                      neck_coeff=0.0,
                      eyes_coeff=0.0,
                      neckSlim_coeff=0.0,
                      neckStretch_coeff=0.0):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3)

        Parameters:
            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs
        """
        batch_size = id_coeff.shape[0]
        id_part = torch.einsum('ij,aj->ai', self.id_base, id_coeff)
        exp_part = torch.einsum('ij,aj->ai', self.exp_base, exp_coeff)
        face_shape = id_part + exp_part + self.mean_shape.reshape([1, -1])

        if nose_coeff != 0:
            face_shape = face_shape + nose_coeff * self.nose_reduced_part
        if neck_coeff != 0:
            face_shape = face_shape + neck_coeff * self.neck_adjust_part
        if eyes_coeff != 0 and self.eyes_adjust_part is not None:
            face_shape = face_shape + eyes_coeff * self.eyes_adjust_part
        if neckSlim_coeff != 0 and self.neck_slim_part is not None:
            face_shape = face_shape + neckSlim_coeff * self.neck_slim_part
        if neckStretch_coeff != 0 and self.neck_stretch_part is not None:

            neck_stretch_part = self.neck_stretch_part.reshape(1, -1, 3)
            neck_stretch_part_top = neck_stretch_part[0, 37476, 1]
            neck_stretch_part_bottom = neck_stretch_part[0, 37357, 1]
            neck_stretch_height = neck_stretch_part_top - neck_stretch_part_bottom

            face_shape_ = face_shape.reshape(1, -1, 3)
            face_shape_top = face_shape_[0, 37476, 1]
            face_shape_bottom = face_shape_[0, 37357, 1]
            face_shape_height = face_shape_top - face_shape_bottom

            target_neck_height = 0.72  # top ind 37476, bottom ind 37357
            neckStretch_coeff = (target_neck_height
                                 - face_shape_height) / neck_stretch_height

            face_shape = face_shape + neckStretch_coeff * self.neck_stretch_part

        return face_shape.reshape([batch_size, -1, 3])

    def compute_texture(self, tex_coeff, normalize=True):
        """
        Return:
            face_texture     -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.)

        Parameters:
            tex_coeff        -- torch.tensor, size (B, 80)
        """
        batch_size = tex_coeff.shape[0]
        face_texture = torch.einsum('ij,aj->ai', self.tex_base,
                                    tex_coeff) + self.mean_tex
        if normalize:
            face_texture = face_texture / 255.
        return face_texture.reshape([batch_size, -1, 3])

    def compute_norm(self, face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """

        v1 = face_shape[:, self.face_buf[:, 0]]
        v2 = face_shape[:, self.face_buf[:, 1]]
        v3 = face_shape[:, self.face_buf[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat(
            [face_norm,
             torch.zeros(face_norm.shape[0], 1, 3).to(self.device)],
            dim=1)

        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm

    def compute_color(self, face_texture, face_norm, gamma):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_texture     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_norm        -- torch.tensor, size (B, N, 3), rotated face normal
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        batch_size = gamma.shape[0]
        a, c = self.SH.a, self.SH.c
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + self.init_lit
        gamma = gamma.permute(0, 2, 1)

        y1 = a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.device)
        y2 = -a[1] * c[1] * face_norm[..., 1:2]
        y3 = a[1] * c[1] * face_norm[..., 2:]
        y4 = -a[1] * c[1] * face_norm[..., :1]
        y5 = a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2]
        y6 = -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:]
        y7 = 0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:]**2 - 1)
        y8 = -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:]
        y9 = 0.5 * a[2] * c[2] * (
            face_norm[..., :1]**2 - face_norm[..., 1:2]**2)
        Y = torch.cat([y1, y2, y3, y4, y5, y6, y7, y8, y9], dim=-1)
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        face_color = torch.cat([r, g, b], dim=-1) * face_texture
        return face_color

    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

        value_list = [
            ones, zeros, zeros, zeros,
            torch.cos(x), -torch.sin(x), zeros,
            torch.sin(x),
            torch.cos(x)
        ]
        rot_x = torch.cat(value_list, dim=1).reshape([batch_size, 3, 3])

        value_list = [
            torch.cos(y), zeros,
            torch.sin(y), zeros, ones, zeros, -torch.sin(y), zeros,
            torch.cos(y)
        ]
        rot_y = torch.cat(value_list, dim=1).reshape([batch_size, 3, 3])

        value_list = [
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z),
            torch.cos(z), zeros, zeros, zeros, ones
        ]
        rot_z = torch.cat(value_list, dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        """
        Return:
            face_proj        -- torch.tensor, size (B, N, 2), y direction is opposite to v direction

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        face_proj = face_shape @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]

        return face_proj

    def transform(self, face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + trans.unsqueeze(1)

    def get_landmarks(self, face_proj):
        """
        Return:
            face_lms         -- torch.tensor, size (B, 68, 2)

        Parameters:
            face_proj       -- torch.tensor, size (B, N, 2)
        """
        return face_proj[:, self.keypoints]

    def split_coeff(self, coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        if type(coeffs) == dict and 'id' in coeffs:
            return coeffs

        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80:144]
        tex_coeffs = coeffs[:, 144:224]
        angles = coeffs[:, 224:227]
        gammas = coeffs[:, 227:254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }

    def merge_coeff(self, coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        names = ['id', 'exp', 'tex', 'angle', 'gamma', 'trans']
        coeffs_merge = []
        for name in names:
            coeffs_merge.append(coeffs[name].detach())
        coeffs_merge = torch.cat(coeffs_merge, dim=1)

        return coeffs_merge

    def reverse_recenter(self, face_shape):
        batch_size = face_shape.shape[0]
        face_shape = face_shape.reshape([-1, 3])
        mean_shape_ori = self.mean_shape_ori.reshape([-1, 3])
        face_shape = face_shape + torch.mean(
            mean_shape_ori[:35709, ...], dim=0, keepdim=True)
        face_shape = face_shape.reshape([batch_size, -1, 3])
        return face_shape

    def add_nonlinear_offset_eyes(self, face_shape, shape_offset):
        assert face_shape.shape[0] == 1 and shape_offset.shape[0] == 1
        face_shape = face_shape[0]
        shape_offset = shape_offset[0]

        corner_shape = face_shape[-625:, :]
        corner_offset = shape_offset[self.eye_corner_inds]
        for i in range(len(self.eye_lines)):
            corner_shape[self.eye_lines[i]] += corner_offset[i][None, ...]
        face_shape[-625:, :] = corner_shape

        l_eye_landmarks = [11540, 11541]
        r_eye_landmarks = [4271, 4272]

        l_eye_offset = torch.mean(
            shape_offset[l_eye_landmarks], dim=0, keepdim=True)
        face_shape[37082:37082 + 609] += l_eye_offset

        r_eye_offset = torch.mean(
            shape_offset[r_eye_landmarks], dim=0, keepdim=True)
        face_shape[37082 + 609:37082 + 609 + 608] += r_eye_offset

        face_shape = face_shape[None, ...]

        return face_shape

    def add_nonlinear_offset(self, face_shape, shape_offset_uv, UVs):
        """

        Args:
            face_shape: torch.tensor, size (1, N, 3)
            shape_offset_uv: torch.tensor, size (1, h, w, 3)
            UVs: torch.tensor, size (N, 2)

        Returns:

        """
        assert face_shape.shape[0] == 1 and shape_offset_uv.shape[0] == 1
        face_shape = face_shape[0]
        shape_offset_uv = shape_offset_uv[0]

        h, w = shape_offset_uv.shape[:2]
        UVs_coords = UVs.clone()
        UVs_coords[:, 0] *= w
        UVs_coords[:, 1] *= h
        UVs_coords_int = torch.floor(UVs_coords)
        UVs_coords_float = UVs_coords - UVs_coords_int
        UVs_coords_int = UVs_coords_int.long()

        shape_lt = shape_offset_uv[(h - 1 - UVs_coords_int[:, 1]).clamp(
            0, h - 1), UVs_coords_int[:, 0].clamp(0, w - 1)]  # (N, 3)
        shape_lb = shape_offset_uv[(h - UVs_coords_int[:, 1]).clamp(0, h - 1),
                                   UVs_coords_int[:, 0].clamp(0, w - 1)]
        shape_rt = shape_offset_uv[(h - 1
                                    - UVs_coords_int[:, 1]).clamp(0, h - 1),
                                   (UVs_coords_int[:, 0] + 1).clamp(0, w - 1)]
        shape_rb = shape_offset_uv[(h - UVs_coords_int[:, 1]).clamp(0, h - 1),
                                   (UVs_coords_int[:, 0] + 1).clamp(0, w - 1)]

        value_1 = shape_lt * (
            1 - UVs_coords_float[:, :1]) * UVs_coords_float[:, 1:]
        value_2 = shape_lb * (1 - UVs_coords_float[:, :1]) * (
            1 - UVs_coords_float[:, 1:])
        value_3 = shape_rt * UVs_coords_float[:, :1] * UVs_coords_float[:, 1:]
        value_4 = shape_rb * UVs_coords_float[:, :1] * (
            1 - UVs_coords_float[:, 1:])

        offset_shape = value_1 + value_2 + value_3 + value_4  # (B, N, 3)

        face_shape = (face_shape + offset_shape)[None, ...]

        return face_shape, offset_shape[None, ...]

    def compute_for_render_head_fitting(self,
                                        coeffs,
                                        shape_offset_uv,
                                        texture_offset_uv,
                                        shape_offset_uv_head,
                                        texture_offset_uv_head,
                                        UVs,
                                        reverse_recenter=True,
                                        get_eyes=False,
                                        get_neck=False,
                                        nose_coeff=0.0,
                                        neck_coeff=0.0,
                                        eyes_coeff=0.0):
        if type(coeffs) == dict:
            coef_dict = coeffs
        elif type(coeffs) == torch.Tensor:
            coef_dict = self.split_coeff(coeffs)

        face_shape = self.compute_shape(
            coef_dict['id'],
            coef_dict['exp'],
            nose_coeff=nose_coeff,
            neck_coeff=neck_coeff,
            eyes_coeff=eyes_coeff)  # (1, n, 3)
        if reverse_recenter:
            face_shape_ori_noRecenter = self.reverse_recenter(
                face_shape.clone())
        else:
            face_shape_ori_noRecenter = face_shape.clone()
        face_vertex_ori = self.to_camera(face_shape_ori_noRecenter)

        face_shape[:, :35241, :], shape_offset = self.add_nonlinear_offset(
            face_shape[:, :35241, :], shape_offset_uv,
            UVs[:35709, ...][self.bfm_keep_inds])  # (1, n, 3)
        if get_eyes:
            face_shape = self.add_nonlinear_offset_eyes(
                face_shape, shape_offset)
        if get_neck:
            face_shape[:, 35241:37082, ...], _ = self.add_nonlinear_offset(
                face_shape[:, 35241:37082, ...], shape_offset_uv_head,
                UVs[35709:, ...])  # (1, n, 3)
        else:
            face_shape[:, self.ours_hair_area_inds,
                       ...], _ = self.add_nonlinear_offset(
                           face_shape[:, self.ours_hair_area_inds,
                                      ...], shape_offset_uv_head,
                           UVs[self.ours_hair_area_inds + (35709 - 35241),
                               ...])  # (1, n, 3)

        if reverse_recenter:
            face_shape_offset_noRecenter = self.reverse_recenter(
                face_shape.clone())
        else:
            face_shape_offset_noRecenter = face_shape.clone()
        face_vertex_offset = self.to_camera(face_shape_offset_noRecenter)

        rotation = self.compute_rotation(coef_dict['angle'])

        face_shape_transformed = self.transform(face_shape, rotation,
                                                coef_dict['trans'])
        face_vertex = self.to_camera(face_shape_transformed)

        face_proj = self.to_image(face_vertex)
        landmark = self.get_landmarks(face_proj)

        face_texture = self.compute_texture(coef_dict['tex'])  # (1, n, 3)
        face_texture[:, :35241, :], texture_offset = self.add_nonlinear_offset(
            face_texture[:, :35241, :], texture_offset_uv,
            UVs[:35709, ...][self.bfm_keep_inds])
        face_texture[:, 35241:37082, :], _ = self.add_nonlinear_offset(
            face_texture[:, 35241:37082, :], texture_offset_uv_head,
            UVs[35709:, ...])

        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_color = self.compute_color(face_texture, face_norm_roted,
                                        coef_dict['gamma'])

        return face_vertex, face_texture, face_color, landmark, face_vertex_ori, face_vertex_offset, face_proj

    def compute_for_render_head(self,
                                coeffs,
                                shape_offset_uv,
                                texture_offset_uv,
                                shape_offset_uv_head,
                                texture_offset_uv_head,
                                UVs,
                                reverse_recenter=True,
                                nose_coeff=0.0,
                                neck_coeff=0.0,
                                eyes_coeff=0.0,
                                neckSlim_coeff=0.0,
                                neckStretch_coeff=0.0):
        if type(coeffs) == dict:
            coef_dict = coeffs
        elif type(coeffs) == torch.Tensor:
            coef_dict = self.split_coeff(coeffs)

        face_shape = self.compute_shape(
            coef_dict['id'],
            coef_dict['exp'],
            nose_coeff=nose_coeff,
            neck_coeff=neck_coeff,
            eyes_coeff=eyes_coeff,
            neckSlim_coeff=neckSlim_coeff,
            neckStretch_coeff=neckStretch_coeff)  # (1, n, 3)
        if reverse_recenter:
            face_shape_ori_noRecenter = self.reverse_recenter(
                face_shape.clone())
        else:
            face_shape_ori_noRecenter = face_shape.clone()
        face_vertex_ori = self.to_camera(face_shape_ori_noRecenter)

        face_shape[:, :35709, :], shape_offset = self.add_nonlinear_offset(
            face_shape[:, :35709, :], shape_offset_uv, UVs[:35709,
                                                           ...])  # (1, n, 3)
        face_shape[:, 35709:,
                   ...], _ = self.add_nonlinear_offset(face_shape[:, 35709:,
                                                                  ...],
                                                       shape_offset_uv_head,
                                                       UVs[35709:,
                                                           ...])  # (1, n, 3)

        if reverse_recenter:
            face_shape_offset_noRecenter = self.reverse_recenter(
                face_shape.clone())
        else:
            face_shape_offset_noRecenter = face_shape.clone()
        face_vertex_offset = self.to_camera(face_shape_offset_noRecenter)

        rotation = self.compute_rotation(coef_dict['angle'])

        face_shape_transformed = self.transform(face_shape, rotation,
                                                coef_dict['trans'])
        face_vertex = self.to_camera(face_shape_transformed)

        face_proj = self.to_image(face_vertex)
        landmark = self.get_landmarks(face_proj)

        face_texture = self.compute_texture(coef_dict['tex'])  # (1, n, 3)
        face_texture[:, :35709, :], texture_offset = self.add_nonlinear_offset(
            face_texture[:, :35709, :], texture_offset_uv, UVs[:35709, ...])
        face_texture[:, 35709:, :], _ = self.add_nonlinear_offset(
            face_texture[:, 35709:, :], texture_offset_uv_head, UVs[35709:,
                                                                    ...])

        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_color = self.compute_color(face_texture, face_norm_roted,
                                        coef_dict['gamma'])

        return face_vertex, face_texture, face_color, landmark, face_vertex_ori, face_vertex_offset, face_proj
