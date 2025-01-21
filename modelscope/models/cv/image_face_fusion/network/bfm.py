# The implementation is adopted from Deep3DFaceRecon_pytorch, made publicly available under the MIT License
# at https://github.com/sicxu/Deep3DFaceRecon_pytorch/blob/master/models/bfm.py
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat


def perspective_projection(focal, center):
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


class ParametricFaceModel():

    def __init__(self,
                 bfm_folder='./BFM',
                 recenter=True,
                 camera_distance=10.,
                 init_lit=np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0]),
                 focal=1015.,
                 center=112.,
                 is_train=True,
                 default_name='BFM_model_front.mat'):

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

        if is_train:
            # vertex indices for small face region to compute photometric error. starts from 0.
            self.front_mask = np.squeeze(model['frontmask2_idx']).astype(
                np.int64) - 1
            # vertex indices for each face from small face region. starts from 0. [f,3]
            self.front_face_buf = model['tri_mask2'].astype(np.int64) - 1
            # vertex indices for pre-defined skin region to compute reflectance loss
            self.skin_mask = np.squeeze(model['skinmask'])

        if recenter:
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape = mean_shape - np.mean(
                mean_shape, axis=0, keepdims=True)
            self.mean_shape = mean_shape.reshape([-1, 1])

        self.persc_proj = perspective_projection(focal, center)
        self.device = 'cpu'
        self.camera_distance = camera_distance
        self.SH = SH()
        self.init_lit = init_lit.reshape([1, 1, -1]).astype(np.float32)

    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value).to(device))

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
        batch_size = gamma.shape[0]
        a, c = self.SH.a, self.SH.c
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + self.init_lit
        gamma = gamma.permute(0, 2, 1)
        face_norm_p1 = face_norm[..., :1]
        face_norm_p2 = face_norm[..., 1:2]
        face_norm_p3 = face_norm[..., 2:]
        face_norm_diff = face_norm_p1**2 - face_norm_p2**2
        temp = [
            a[0] * c[0] * torch.ones_like(face_norm_p1).to(self.device),
            -a[1] * c[1] * face_norm_p2, a[1] * c[1] * face_norm_p3,
            -a[1] * c[1] * face_norm_p1,
            a[2] * c[2] * face_norm_p1 * face_norm_p2,
            -a[2] * c[2] * face_norm_p2 * face_norm_p3,
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm_p3**2 - 1),
            -a[2] * c[2] * face_norm_p1 * face_norm_p3,
            0.5 * a[2] * c[2] * face_norm_diff
        ]

        Y = torch.cat(temp, dim=-1)
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        face_color = torch.cat([r, g, b], dim=-1) * face_texture
        return face_color

    def compute_rotation(self, angles):
        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

        temp_x = [
            ones, zeros, zeros, zeros,
            torch.cos(x), -torch.sin(x), zeros,
            torch.sin(x),
            torch.cos(x)
        ]
        rot_x = torch.cat(temp_x, dim=1).reshape([batch_size, 3, 3])

        temp_y = [
            torch.cos(y), zeros,
            torch.sin(y), zeros, ones, zeros, -torch.sin(y), zeros,
            torch.cos(y)
        ]
        rot_y = torch.cat(temp_y, dim=1).reshape([batch_size, 3, 3])

        temp_z = [
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z),
            torch.cos(z), zeros, zeros, zeros, ones
        ]
        rot_z = torch.cat(temp_z, dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        # to image_plane
        face_proj = face_shape @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]

        return face_proj

    def transform(self, face_shape, rot, trans):
        return face_shape @ rot + trans.unsqueeze(1)

    def get_landmarks(self, face_proj):
        return face_proj[:, self.keypoints]

    def split_coeff(self, coeffs):
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

    def compute_for_render(self, coeffs):
        coef_dict = self.split_coeff(coeffs)
        face_shape = self.compute_shape(coef_dict['id'], coef_dict['exp'])
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

        return face_vertex, face_texture, face_color, landmark
