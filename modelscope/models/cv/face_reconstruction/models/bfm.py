# Part of the implementation is borrowed and modified from Deep3DFaceRecon_pytorch,
# publicly available at https://github.com/sicxu/Deep3DFaceRecon_pytorch

import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from .. import utils
from ..utils import read_obj, transferBFM09
from .renderer import SRenderY, set_rasterizer


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
                 assets_folder='assets',
                 recenter=True,
                 camera_distance=10.,
                 init_lit=np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0]),
                 focal=1015.,
                 center=112.,
                 is_train=True,
                 default_name='BFM_model_front.mat'):

        if not os.path.isfile(os.path.join(assets_folder, default_name)):
            transferBFM09(assets_folder)
        model = loadmat(os.path.join(assets_folder, default_name))
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

        mean_tex_uv_path = os.path.join(assets_folder, 'bfm_tex_mean2.npy')
        tex_base_uv_path = os.path.join(assets_folder, 'bfm_texmap_base2.npy')
        self.mean_tex_uv = np.load(mean_tex_uv_path)
        self.mean_tex_uv = self.mean_tex_uv.reshape((1, -1))
        self.tex_base_uv = np.load(tex_base_uv_path)
        self.tex_base_uv = self.tex_base_uv.reshape((-1, 80))
        set_rasterizer()
        template_obj_path = os.path.join(assets_folder, 'template_bfm.obj')
        uvcoords_path = os.path.join(assets_folder, 'bfm_uvs2.npy')
        self.render = SRenderY(
            224, template_obj_path, uvcoords_path, uv_size=256)

        # face indices for each vertex that lies in. starts from 0. [N,8]
        self.point_buf = model['point_buf'].astype(np.int64) - 1  # (35709, 8)

        # vertex indices for each face. starts from 0. [F,3]
        self.face_buf = model['tri'].astype(np.int64) - 1  # (70789, 3)

        # vertex indices for 68 landmarks. starts from 0. [68,1]
        self.keypoints = np.squeeze(model['keypoints']).astype(np.int64) - 1

        if recenter:
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape = mean_shape - np.mean(
                mean_shape[:35709, ...], axis=0, keepdims=True)
            self.mean_shape = mean_shape.reshape([-1, 1])

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
        self.render = self.render.to(device)

    def compute_shape(self, id_coeff, exp_coeff):
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

        return face_shape.reshape([batch_size, -1, 3])

    def compute_albedo(self, tex_coeff, normalize=True):
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

    def compute_albedo_map(self, tex_coeff, normalize=True):
        """
        Return:
            face_texture     -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.)

        Parameters:
            tex_coeff        -- torch.tensor, size (B, 80)
        """
        batch_size = tex_coeff.shape[0]
        face_texture = torch.einsum('ij,aj->ai', self.tex_base_uv,
                                    tex_coeff) + self.mean_tex_uv
        if normalize:
            face_texture = face_texture / 255.
        return face_texture.reshape([batch_size, 512, 512, 3])

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

    def compute_color_map(self, face_texture_uv, face_norm_uv, gamma):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_texture_uv     -- torch.tensor, (B, 3, 256, 256)
            face_norm_uv        -- torch.tensor, (B, 3, 256, 256)
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        face_texture_uv = face_texture_uv.permute(
            0, 2, 3, 1).contiguous()  # (B, 256, 256, 3)
        face_norm_uv = face_norm_uv.permute(0, 2, 3,
                                            1).contiguous()  # (B, 256, 256, 3)
        size = face_texture_uv.shape[1]

        batch_size = gamma.shape[0]
        a, c = self.SH.a, self.SH.c
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + self.init_lit
        gamma = gamma.permute(0, 2, 1)  # (B, 9, 3)
        y1 = a[0] * c[0] * torch.ones_like(face_norm_uv[..., :1]).to(
            self.device)
        y2 = -a[1] * c[1] * face_norm_uv[..., 1:2]
        y3 = a[1] * c[1] * face_norm_uv[..., 2:]
        y4 = -a[1] * c[1] * face_norm_uv[..., :1]
        y5 = a[2] * c[2] * face_norm_uv[..., :1] * face_norm_uv[..., 1:2]
        y6 = -a[2] * c[2] * face_norm_uv[..., 1:2] * face_norm_uv[..., 2:]
        y7 = 0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm_uv[..., 2:]**2
                                                - 1)
        y8 = -a[2] * c[2] * face_norm_uv[..., :1] * face_norm_uv[..., 2:]
        y9 = 0.5 * a[2] * c[2] * (
            face_norm_uv[..., :1]**2 - face_norm_uv[..., 1:2]**2)
        Y = torch.cat([y1, y2, y3, y4, y5, y6, y7, y8, y9],
                      dim=-1)  # (B, 256, 256, 9)
        Y = Y.reshape(batch_size, -1, 9)
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        face_color = torch.cat([r, g, b], dim=-1).reshape(
            batch_size, size, size, 3) * face_texture_uv  # (B, 256, 256, 3)
        face_color = face_color.permute(0, 3, 1,
                                        2).contiguous()  # (B, 3, 256, 256)
        return face_color

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
        ''' Convert displacement map into detail normal map
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts)
        uv_coarse_normals = self.render.world2uv(coarse_normals)

        uv_detail_vertices = uv_coarse_vertices
        dense_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape(
            [batch_size, -1, 3])
        uv_detail_normals = utils.vertex_normals(
            dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape([
            batch_size, uv_coarse_vertices.shape[2],
            uv_coarse_vertices.shape[3], 3
        ]).permute(0, 3, 1, 2)
        uv_detail_normals[:, :2, ...] = -uv_detail_normals[:, :2, ...]
        offset = uv_coarse_normals - uv_detail_normals

        uv_detail_vertices = uv_coarse_vertices + uv_z * uv_coarse_normals
        dense_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape(
            [batch_size, -1, 3])
        uv_detail_normals = utils.vertex_normals(
            dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape([
            batch_size, uv_coarse_vertices.shape[2],
            uv_coarse_vertices.shape[3], 3
        ]).permute(0, 3, 1, 2)
        uv_detail_normals[:, :2, ...] = -uv_detail_normals[:, :2, ...]
        uv_detail_normals = uv_detail_normals + offset
        return uv_detail_normals

    def compute_color_with_displacement(self, face_texture_uv, verts, normals,
                                        displacement_uv, gamma):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_texture     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_norm        -- torch.tensor, size (B, N, 3), rotated face normal
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        uv_detail_normals = self.displacement2normal(
            displacement_uv, verts, normals
        )  # verts: (B, n, 3), ops['normals']: (B, n, 3), uv_detail_normals: (B, 3, 256, 256)
        uv_texture = self.compute_color_map(face_texture_uv, uv_detail_normals,
                                            gamma)
        return uv_texture

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
            coeffs          -- torch.tensor, size (B, 257)
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

    def get_texture_map(self, face_vertex, input_img):
        batch_size = input_img.shape[0]
        h, w = input_img.shape[2:]
        face_vertex_uv = self.render.world2uv(face_vertex)  # (B, 3, 256, 256)
        face_vertex_uv = face_vertex_uv.reshape(batch_size, 3, -1).permute(
            0, 2, 1)  # (B, N, 3), N: 256*256
        face_vertex_uv_proj = self.to_image(
            face_vertex_uv)  # (B, N, 2) , project to image (size=224)
        face_vertex_uv_proj[..., 0] *= w / 224
        face_vertex_uv_proj[..., 1] *= h / 224
        face_vertex_uv_proj[torch.isnan(face_vertex_uv_proj)] = 0

        face_vertex_uv_proj[..., -1] = h - 1 - face_vertex_uv_proj[..., -1]

        input_img = input_img.permute(0, 2, 3, 1)  # (B, h, w, 3)

        face_vertex_uv_proj_int = torch.floor(face_vertex_uv_proj)
        face_vertex_uv_proj_float = face_vertex_uv_proj - face_vertex_uv_proj_int  # (B, N, 2)
        face_vertex_uv_proj_float = face_vertex_uv_proj_float.reshape(
            -1, 2)  # (B * N, 2)
        face_vertex_uv_proj_int = face_vertex_uv_proj_int.long()  # (B, N, 2)

        batch_indices = torch.arange(0, batch_size)[:, None, None].repeat(
            1, face_vertex_uv_proj_int.shape[1],
            1).long().to(face_vertex_uv_proj_int.device)  # (B, N, 1)
        indices = torch.cat([face_vertex_uv_proj_int, batch_indices], dim=2)
        indices = indices.reshape(-1, 3)  # (B * N, 3)

        face_vertex_uv_proj_lt = input_img[indices[:, 2], indices[:, 1].clamp(
            0, h - 1), indices[:, 0].clamp(0, w - 1)]  # (B * N, 3)
        face_vertex_uv_proj_lb = input_img[indices[:, 2],
                                           (indices[:, 1] + 1).clamp(0, h - 1),
                                           indices[:, 0].clamp(0, w - 1)]
        face_vertex_uv_proj_rt = input_img[indices[:, 2],
                                           indices[:, 1].clamp(0, h - 1),
                                           (indices[:, 0] + 1).clamp(0, w - 1)]
        face_vertex_uv_proj_rb = input_img[indices[:, 2],
                                           (indices[:, 1] + 1).clamp(0, h - 1),
                                           (indices[:, 0] + 1).clamp(0, w - 1)]

        value_1 = face_vertex_uv_proj_lt * (
            1 - face_vertex_uv_proj_float[:, :1]
        ) * face_vertex_uv_proj_float[:, 1:]
        value_2 = face_vertex_uv_proj_lb * (
            1 - face_vertex_uv_proj_float[:, :1]) * (
                1 - face_vertex_uv_proj_float[:, 1:])
        value_3 = face_vertex_uv_proj_rt * face_vertex_uv_proj_float[:, :
                                                                     1] * face_vertex_uv_proj_float[:,
                                                                                                    1:]
        value_4 = face_vertex_uv_proj_rb * face_vertex_uv_proj_float[:, :1] * (
            1 - face_vertex_uv_proj_float[:, 1:])

        texture_map = value_1 + value_2 + value_3 + value_4  # (B * N, 3)

        texture_map = texture_map.reshape(batch_size, self.render.uv_size,
                                          self.render.uv_size,
                                          -1)  # (B, 256, 256, 3)

        return texture_map

    def compute_for_render(self, coeffs):
        if type(coeffs) == dict:
            coef_dict = coeffs
        elif type(coeffs) == torch.Tensor:
            coef_dict = self.split_coeff(coeffs)

        face_shape = self.compute_shape(coef_dict['id'], coef_dict['exp'])

        rotation = self.compute_rotation(coef_dict['angle'])

        face_shape_transformed = self.transform(face_shape, rotation,
                                                coef_dict['trans'])
        face_vertex = self.to_camera(face_shape_transformed.clone())
        face_vertex_noTrans = self.to_camera(face_shape.clone())

        face_proj = self.to_image(face_vertex)
        landmark = self.get_landmarks(face_proj)

        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation

        face_albedo_map = self.compute_albedo_map(
            coef_dict['tex'])  # (1, 512, 512, 3)
        face_albedo_map = face_albedo_map.permute(0, 3, 1, 2)
        face_albedo_map = torch.nn.functional.interpolate(
            face_albedo_map, [self.render.uv_size, self.render.uv_size],
            mode='bilinear')
        face_norm_roted_uv = self.render.world2uv(face_norm_roted)
        face_color_map = self.compute_color_map(face_albedo_map,
                                                face_norm_roted_uv,
                                                coef_dict['gamma'])

        position_map = self.render.world2uv(face_shape)

        return face_vertex, face_albedo_map, face_color_map, landmark, face_vertex_noTrans, position_map

    def recolor_texture(self, shaded_texture):
        rgb_mean = torch.mean(shaded_texture, dim=(2, 3), keepdim=True)
        target_mean = torch.ones_like(rgb_mean) * 0.6
        shaded_texture = shaded_texture * target_mean / rgb_mean
        return shaded_texture

    def compute_for_render_hierarchical_mid(self,
                                            coeffs,
                                            deformation_map,
                                            UVs,
                                            visualize=False,
                                            de_retouched_albedo_map=None):
        if type(coeffs) == dict:
            coef_dict = coeffs
        elif type(coeffs) == torch.Tensor:
            coef_dict = self.split_coeff(coeffs)

        face_shape = self.compute_shape(coef_dict['id'],
                                        coef_dict['exp'])  # (B, n, 3)
        face_shape_base = face_shape.clone()

        face_shape, shape_offset = self.add_nonlinear_offset(
            face_shape, deformation_map, UVs)  # (B, n, 3)

        rotation = self.compute_rotation(coef_dict['angle'])

        face_shape_transformed = self.transform(face_shape, rotation,
                                                coef_dict['trans'])
        face_vertex = self.to_camera(face_shape_transformed.clone())

        face_proj = self.to_image(face_vertex)
        landmark = self.get_landmarks(face_proj)

        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation

        face_albedo_map = self.compute_albedo_map(
            coef_dict['tex'])  # (B, 512, 512, 3)
        face_albedo_map = face_albedo_map.permute(0, 3, 1, 2)
        face_albedo_map = torch.nn.functional.interpolate(
            face_albedo_map, [self.render.uv_size, self.render.uv_size],
            mode='bilinear')
        face_norm_roted_uv = self.render.world2uv(face_norm_roted)

        albedo_for_render = face_albedo_map if de_retouched_albedo_map is None else de_retouched_albedo_map
        face_color_map = self.compute_color_map(albedo_for_render,
                                                face_norm_roted_uv,
                                                coef_dict['gamma'])

        extra_results = None
        if visualize:
            extra_results = {}
            extra_results['tex_mid_color'] = face_color_map

            face_shape_transformed_base = self.transform(
                face_shape_base, rotation, coef_dict['trans'])
            face_vertex_base = self.to_camera(
                face_shape_transformed_base.clone())
            extra_results['pred_vertex_base'] = face_vertex_base
            face_norm_base = self.compute_norm(face_shape_base)
            face_norm_roted_base = face_norm_base @ rotation

            batch_size = albedo_for_render.shape[0]
            size = albedo_for_render.shape[2]
            gray_tex = torch.ones((batch_size, 3, size, size),
                                  dtype=torch.float32).to(self.device) * 0.8
            zero_displacement = torch.zeros(
                (batch_size, 1, size, size),
                dtype=torch.float32).to(self.device)

            tex_mid_gray = self.compute_color_with_displacement(
                gray_tex.detach(), face_shape_transformed, face_norm_roted,
                zero_displacement, coef_dict['gamma'])
            tex_mid_gray = self.recolor_texture(tex_mid_gray)
            extra_results['tex_mid_gray'] = tex_mid_gray

            tex_base_color = self.compute_color_with_displacement(
                albedo_for_render, face_shape_transformed_base,
                face_norm_roted_base, zero_displacement, coef_dict['gamma'])
            extra_results['tex_base_color'] = tex_base_color

            tex_base_gray = self.compute_color_with_displacement(
                gray_tex.detach(), face_shape_transformed_base,
                face_norm_roted_base, zero_displacement, coef_dict['gamma'])
            tex_base_gray = self.recolor_texture(tex_base_gray)
            extra_results['tex_base_gray'] = tex_base_gray

            # to export rotate video
            init_angle = torch.zeros_like(coef_dict['angle']).to(
                coef_dict['angle'].device)

            pi = 3.14
            n_frame = 30
            y_angles = torch.linspace(-pi / 6, pi / 6, steps=n_frame).float()

            extra_results['face_shape_transformed_list'] = []
            extra_results['face_norm_roted_list'] = []
            extra_results['face_vertex_list'] = []
            for y_angle in y_angles:
                cur_angle = init_angle.clone()
                cur_angle[:, 1] = y_angle
                cur_angle[:, 0] = pi / 36
                cur_rotation = self.compute_rotation(cur_angle)
                cur_face_shape_transformed = self.transform(
                    face_shape, cur_rotation, coef_dict['trans'] * 0)
                cur_face_norm_roted = face_norm @ cur_rotation
                cur_face_vertex = self.to_camera(
                    cur_face_shape_transformed.clone())

                extra_results['face_shape_transformed_list'].append(
                    cur_face_shape_transformed)
                extra_results['face_norm_roted_list'].append(
                    cur_face_norm_roted)
                extra_results['face_vertex_list'].append(cur_face_vertex)

        return (face_vertex, face_color_map, landmark, face_proj,
                face_albedo_map, face_shape_transformed, face_norm_roted,
                extra_results)

    def compute_for_render_hierarchical_high(self,
                                             coeffs,
                                             displacement_uv,
                                             face_albedo_map,
                                             face_shape_transformed,
                                             face_norm_roted,
                                             extra_results=None):
        if type(coeffs) == dict:
            coef_dict = coeffs
        elif type(coeffs) == torch.Tensor:
            coef_dict = self.split_coeff(coeffs)

        face_color_map = self.compute_color_with_displacement(
            face_albedo_map, face_shape_transformed, face_norm_roted,
            displacement_uv, coef_dict['gamma'])

        if extra_results is not None:
            extra_results['tex_high_color'] = face_color_map

            batch_size = face_albedo_map.shape[0]
            size = face_albedo_map.shape[2]
            gray_tex = torch.ones((batch_size, 3, size, size),
                                  dtype=torch.float32).to(self.device) * 0.8

            tex_high_gray = self.compute_color_with_displacement(
                gray_tex.detach(), face_shape_transformed, face_norm_roted,
                displacement_uv, coef_dict['gamma'])
            tex_high_gray = self.recolor_texture(tex_high_gray)
            extra_results['tex_high_gray'] = tex_high_gray

            if 'face_shape_transformed_list' in extra_results:
                extra_results['tex_high_gray_list'] = []
                extra_results['tex_high_color_list'] = []
                for i in range(
                        len(extra_results['face_shape_transformed_list'])):
                    tex_high_gray_i = self.compute_color_with_displacement(
                        gray_tex.detach(),
                        extra_results['face_shape_transformed_list'][i],
                        extra_results['face_norm_roted_list'][i],
                        displacement_uv, coef_dict['gamma'])
                    extra_results['tex_high_gray_list'].append(tex_high_gray_i)

                    tex_high_color_i = self.compute_color_with_displacement(
                        face_albedo_map.detach(),
                        extra_results['face_shape_transformed_list'][i],
                        extra_results['face_norm_roted_list'][i],
                        displacement_uv, coef_dict['gamma'])
                    extra_results['tex_high_color_list'].append(
                        tex_high_color_i)

        return face_color_map, extra_results

    def reverse_recenter(self, face_shape):
        batch_size = face_shape.shape[0]
        face_shape = face_shape.reshape([-1, 3])
        mean_shape_ori = self.mean_shape_ori.reshape([-1, 3])
        face_shape = face_shape + torch.mean(
            mean_shape_ori[:35709, ...], dim=0, keepdim=True)
        face_shape = face_shape.reshape([batch_size, -1, 3])
        return face_shape

    def add_nonlinear_offset(self, face_shape, shape_offset_uv, UVs):
        """

        Args:
            face_shape: torch.tensor, size (B, N, 3)
            shape_offset_uv: torch.tensor, size (B, h, w, 3)
            UVs: torch.tensor, size (N, 2)

        Returns:

        """
        h, w = shape_offset_uv.shape[1:3]
        UVs_coords = UVs.clone()
        UVs_coords[:, 0] *= w
        UVs_coords[:, 1] *= h
        UVs_coords_int = torch.floor(UVs_coords)
        UVs_coords_float = UVs_coords - UVs_coords_int
        UVs_coords_int = UVs_coords_int.long()

        shape_lt = shape_offset_uv[:, (h - 1 - UVs_coords_int[:, 1]).clamp(
            0, h - 1), UVs_coords_int[:, 0].clamp(0, w - 1)]  # (B, N, 3)
        shape_lb = shape_offset_uv[:,
                                   (h - UVs_coords_int[:, 1]).clamp(0, h - 1),
                                   UVs_coords_int[:, 0].clamp(0, w - 1)]
        shape_rt = shape_offset_uv[:, (h - 1
                                       - UVs_coords_int[:, 1]).clamp(0, h - 1),
                                   (UVs_coords_int[:, 0] + 1).clamp(0, w - 1)]
        shape_rb = shape_offset_uv[:,
                                   (h - UVs_coords_int[:, 1]).clamp(0, h - 1),
                                   (UVs_coords_int[:, 0] + 1).clamp(0, w - 1)]

        value_1 = shape_lt * (
            1 - UVs_coords_float[:, :1]) * UVs_coords_float[:, 1:]
        value_2 = shape_lb * (1 - UVs_coords_float[:, :1]) * (
            1 - UVs_coords_float[:, 1:])
        value_3 = shape_rt * UVs_coords_float[:, :1] * UVs_coords_float[:, 1:]
        value_4 = shape_rb * UVs_coords_float[:, :1] * (
            1 - UVs_coords_float[:, 1:])

        offset_shape = value_1 + value_2 + value_3 + value_4  # (B, N, 3)

        face_shape = face_shape + offset_shape

        return face_shape, offset_shape
