# Part of the implementation is borrowed and modified from Deep3DFaceRecon_pytorch,
# publicly available at https://github.com/sicxu/Deep3DFaceRecon_pytorch
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from ..utils import read_obj, transferBFM09


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
                 bfm_folder='./asset/BFM',
                 recenter=True,
                 camera_distance=10.,
                 init_lit=np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0]),
                 focal=1015.,
                 center=112.,
                 is_train=True,
                 default_name='BFM_model_front.mat'):

        if not os.path.isfile(os.path.join(bfm_folder, default_name)):
            transferBFM09(bfm_folder)
        model = loadmat(os.path.join(bfm_folder, default_name))
        # mean face shape. [3*N,1]
        self.mean_shape = model['meanshape'].astype(np.float32)

        # identity basis. [3*N,80]
        self.id_base = model['idBase'].astype(np.float32)

        # expression basis. [3*N,64]
        self.exp_base = model['exBase'].astype(np.float32)

        # mean face texture. [3*N,1] (0-255)
        self.mean_tex = model['meantex'].astype(np.float32)

        # texture basis. [3*N,80]
        self.tex_base = model['texBase'].astype(np.float32)

        # face indices for each vertex that lies in. starts from 0. [N,8]
        self.point_buf = model['point_buf'].astype(np.int64) - 1

        # vertex indices for each face. starts from 0. [F,3]
        self.face_buf = model['tri'].astype(np.int64) - 1

        # vertex indices for 68 landmarks. starts from 0. [68,1]
        self.keypoints = np.squeeze(model['keypoints']).astype(np.int64) - 1

        self.mean_shape_ori = model['meanshape_ori'].astype(np.float32)
        self.bfm_keep_inds = model['bfm_keep_inds'][0]
        self.nose_reduced_part = model['nose_reduced_part'].reshape(
            (1, -1)) - self.mean_shape
        self.nonlinear_UVs = model['nonlinear_UVs']

        if default_name == 'head_model_for_maas.mat':
            self.ours_hair_area_inds = model['hair_area_inds'][0]

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

            self.point_buf = self.point_buf[:, :8] + 1

            self.neck_adjust_part = model['neck_adjust_part'].reshape(
                (1, -1)) - self.mean_shape
            self.eyes_adjust_part = model['eyes_adjust_part'].reshape(
                (1, -1)) - self.mean_shape

            self.eye_corner_inds = model['eye_corner_inds'][0]
            self.eye_corner_lines = model['eye_corner_lines']

        if recenter:
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape_ori = self.mean_shape_ori.reshape([-1, 3])
            mean_shape = mean_shape - np.mean(
                mean_shape_ori[:35709, ...], axis=0, keepdims=True)
            self.mean_shape = mean_shape.reshape([-1, 1])

        self.center = center
        self.persc_proj = perspective_projection(focal, self.center)
        self.device = 'cpu'
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
                      eyes_coeff=0.0):
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

    def compute_for_render(self, coeffs, coeffs_mvs=None):
        """
        Return:
            face_vertex     -- torch.tensor, size (B, N, 3), in camera coordinate
            face_color      -- torch.tensor, size (B, N, 3), in RGB order
            landmark        -- torch.tensor, size (B, 68, 2), y direction is opposite to v direction
        Parameters:
            coeffs          -- torch.tensor, size (B, 257)
        """
        if type(coeffs) == dict:
            coef_dict = coeffs
        elif type(coeffs) == torch.Tensor:
            coef_dict = self.split_coeff(coeffs)

        face_shape = self.compute_shape(
            coef_dict['id'], coef_dict['exp'], nose_coeff=0.4, neck_coeff=0.6)

        rotation = self.compute_rotation(coef_dict['angle'])

        face_shape_transformed = self.transform(face_shape, rotation,
                                                coef_dict['trans'])
        face_vertex = self.to_camera(face_shape_transformed)
        face_vertex_ori = self.to_camera(face_shape)

        face_proj = self.to_image(face_vertex)
        landmark = self.get_landmarks(face_proj)

        face_texture = self.compute_texture(coef_dict['tex'])
        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_color = self.compute_color(face_texture, face_norm_roted,
                                        coef_dict['gamma'])

        if coeffs_mvs is not None:
            mvs_face_shape = self.compute_shape(coeffs_mvs['id'],
                                                coeffs_mvs['exp'])

            mvs_face_shape_transformed = self.transform(
                mvs_face_shape, rotation, coef_dict['trans'])
            mvs_face_vertex = self.to_camera(mvs_face_shape_transformed)
            return face_vertex, face_texture, face_color, landmark, mvs_face_vertex
        else:
            return face_vertex, face_texture, face_color, landmark, face_vertex_ori

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

        corner_inds = self.eye_corner_inds
        lines = self.eye_corner_lines

        corner_shape = face_shape[-625:, :]
        corner_offset = shape_offset[corner_inds]
        for i in range(len(lines)):
            corner_shape[lines[i]] += corner_offset[i][None, ...]
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

        shape_lt = shape_offset_uv[(h - 1
                                    - UVs_coords_int[:, 1]).clamp(0, h - 1),
                                   UVs_coords_int[:, 0].clamp(0, w - 1)]
        shape_lb = shape_offset_uv[(h - UVs_coords_int[:, 1]).clamp(0, h - 1),
                                   UVs_coords_int[:, 0].clamp(0, w - 1)]
        shape_rt = shape_offset_uv[(h - 1
                                    - UVs_coords_int[:, 1]).clamp(0, h - 1),
                                   (UVs_coords_int[:, 0] + 1).clamp(0, w - 1)]
        shape_rb = shape_offset_uv[(h - UVs_coords_int[:, 1]).clamp(0, h - 1),
                                   (UVs_coords_int[:, 0] + 1).clamp(0, w - 1)]

        value1 = shape_lt * (
            1 - UVs_coords_float[:, :1]) * UVs_coords_float[:, 1:]
        value2 = shape_lb * (1 - UVs_coords_float[:, :1]) * (
            1 - UVs_coords_float[:, 1:])
        value3 = shape_rt * UVs_coords_float[:, :1] * UVs_coords_float[:, 1:]
        value4 = shape_rb * UVs_coords_float[:, :1] * (
            1 - UVs_coords_float[:, 1:])
        offset_shape = value1 + value2 + value3 + value4  # (N, 3)

        face_shape = (face_shape + offset_shape)[None, ...]

        return face_shape, offset_shape[None, ...]

    def compute_for_render_train_nonlinear(self,
                                           coeffs,
                                           shape_offset_uv,
                                           tex_offset_uv,
                                           UVs,
                                           reverse_recenter=True):
        if type(coeffs) == dict:
            coef_dict = coeffs
        elif type(coeffs) == torch.Tensor:
            coef_dict = self.split_coeff(coeffs)

        face_shape = self.compute_shape(coef_dict['id'],
                                        coef_dict['exp'])  # (1, n, 3)
        if reverse_recenter:
            face_shape_ori_noRecenter = self.reverse_recenter(
                face_shape.clone())
        else:
            face_shape_ori_noRecenter = face_shape.clone()
        face_vertex_ori = self.to_camera(face_shape_ori_noRecenter)

        face_shape, shape_offset = self.add_nonlinear_offset(
            face_shape, shape_offset_uv, UVs[:35709, :])  # (1, n, 3)
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
        face_texture, texture_offset = self.add_nonlinear_offset(
            face_texture, tex_offset_uv, UVs[:35709, :])
        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_color = self.compute_color(face_texture, face_norm_roted,
                                        coef_dict['gamma'])

        return face_vertex, face_texture, face_color, landmark, face_vertex_ori, face_vertex_offset, face_proj

    def compute_for_render_nonlinear_full(self,
                                          coeffs,
                                          shape_offset_uv,
                                          UVs,
                                          nose_coeff=0.0,
                                          eyes_coeff=0.0):
        if type(coeffs) == dict:
            coef_dict = coeffs
        elif type(coeffs) == torch.Tensor:
            coef_dict = self.split_coeff(coeffs)

        face_shape = self.compute_shape(
            coef_dict['id'],
            coef_dict['exp'],
            nose_coeff=nose_coeff,
            neck_coeff=0.6,
            eyes_coeff=eyes_coeff)  # (1, n, 3)
        face_vertex_ori = self.to_camera(face_shape.clone())

        face_shape[:, :35241, :], shape_offset = self.add_nonlinear_offset(
            face_shape[:, :35241, :], shape_offset_uv,
            UVs[:35709, :][self.bfm_keep_inds])
        face_shape = self.add_nonlinear_offset_eyes(face_shape, shape_offset)
        face_shape_noRecenter = self.reverse_recenter(face_shape.clone())
        face_vertex_offset = self.to_camera(face_shape_noRecenter)

        rotation = self.compute_rotation(coef_dict['angle'])

        face_shape_transformed = self.transform(face_shape, rotation,
                                                coef_dict['trans'])
        face_vertex = self.to_camera(face_shape_transformed)

        return face_vertex, face_vertex_ori, face_vertex_offset

    def compute_for_render_train(self, coeffs):
        """
        Return:
            face_vertex     -- torch.tensor, size (B, N, 3), in camera coordinate
            face_color      -- torch.tensor, size (B, N, 3), in RGB order
            landmark        -- torch.tensor, size (B, 68, 2), y direction is opposite to v direction
        Parameters:
            coeffs          -- torch.tensor, size (B, 257)
        """
        if type(coeffs) == dict:
            coef_dict = coeffs
        elif type(coeffs) == torch.Tensor:
            coef_dict = self.split_coeff(coeffs)

        face_shape = self.compute_shape(coef_dict['id'], coef_dict['exp'])
        uv_geometry = self.render.world2uv(face_shape)

        rotation = self.compute_rotation(coef_dict['angle'])

        face_shape_transformed = self.transform(face_shape, rotation,
                                                coef_dict['trans'])
        face_vertex = self.to_camera(face_shape_transformed)

        face_proj = self.to_image(face_vertex)
        landmark = self.get_landmarks(face_proj)

        face_texture = self.compute_texture(coef_dict['tex'])
        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_color = self.compute_color(face_texture, face_norm_roted,
                                        coef_dict['gamma'])

        return face_vertex, face_texture, face_color, landmark, uv_geometry
