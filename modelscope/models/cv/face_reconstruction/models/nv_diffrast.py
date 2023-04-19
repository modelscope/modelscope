# Part of the implementation is borrowed and modified from Deep3DFaceRecon_pytorch,
# publicly available at https://github.com/sicxu/Deep3DFaceRecon_pytorch
import warnings
from typing import List

import numpy as np
import nvdiffrast
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from torch import nn

from .losses import TVLoss, TVLoss_std

warnings.filterwarnings('ignore')


def ndc_projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n / x, 0, 0, 0], [0, n / -x, 0, 0],
                     [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
                     [0, 0, -1, 0]]).astype(np.float32)


def to_image(face_shape):
    """
    Return:
        face_proj        -- torch.tensor, size (B, N, 2), y direction is opposite to v direction

    Parameters:
        face_shape       -- torch.tensor, size (B, N, 3)
    """

    focal = 1015.
    center = 112.
    persc_proj = np.array([focal, 0, center, 0, focal, center, 0, 0,
                           1]).reshape([3, 3]).astype(np.float32).transpose()

    persc_proj = torch.tensor(persc_proj).to(face_shape.device)

    face_proj = face_shape @ persc_proj
    face_proj = face_proj[..., :2] / face_proj[..., 2:]

    return face_proj


class MeshRenderer(nn.Module):

    def __init__(self, rasterize_fov, znear=0.1, zfar=10, rasterize_size=224):
        super(MeshRenderer, self).__init__()

        x = np.tan(np.deg2rad(rasterize_fov * 0.5)) * znear
        self.ndc_proj = torch.tensor(ndc_projection(
            x=x, n=znear,
            f=zfar)).matmul(torch.diag(torch.tensor([1., -1, -1, 1])))
        self.rasterize_size = rasterize_size
        self.glctx = None

    def forward(self, vertex, tri, feat=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)  -- torch.tensor, size (B, C), features
        """
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)
        verts_proj = to_image(vertex)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat(
                [vertex, torch.ones([*vertex.shape[:2], 1]).to(device)],
                dim=-1)
            vertex[..., 1] = -vertex[..., 1]

        vertex_ndc = vertex @ ndc_proj.t()
        if self.glctx is None:
            if nvdiffrast.__version__ == '0.2.7':
                self.glctx = dr.RasterizeGLContext(device=device)
            else:
                self.glctx = dr.RasterizeCudaContext(device=device)

        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0]
                                 for f in tri]).unsqueeze(1).to(device)

            print('fnum shape:{}'.format(fnum.shape))

            fstartidx = torch.cumsum(fnum, dim=0) - fnum
            ranges = torch.cat([fstartidx, fnum],
                               axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i * vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(
            self.glctx,
            vertex_ndc.contiguous(),
            tri,
            resolution=[rsize, rsize],
            ranges=ranges)

        depth, _ = dr.interpolate(
            vertex.reshape([-1, 4])[..., 2].unsqueeze(1).contiguous(),
            rast_out, tri)
        depth = depth.permute(0, 3, 1, 2)
        mask = (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth

        image = None

        verts_x = verts_proj[0, :, 0]
        verts_y = 224 - verts_proj[0, :, 1]
        verts_int = torch.ceil(verts_proj[0]).long()  # (n, 2)
        verts_xr_int = verts_int[:, 0].clamp(1, 224 - 1)
        verts_yt_int = 224 - verts_int[:, 1].clamp(2, 224)
        verts_right_float = verts_xr_int - verts_x
        verts_left_float = 1 - verts_right_float
        verts_top_float = verts_y - verts_yt_int
        verts_bottom_float = 1 - verts_top_float

        rast_lt = rast_out[0, verts_yt_int, verts_xr_int - 1, 3]
        rast_lb = rast_out[0, verts_yt_int + 1, verts_xr_int - 1, 3]
        rast_rt = rast_out[0, verts_yt_int, verts_xr_int, 3]
        rast_rb = rast_out[0, verts_yt_int + 1, verts_xr_int, 3]

        occ_feat = (rast_lt > 0) * 1.0 * (verts_left_float + verts_top_float) + \
                   (rast_lb > 0) * 1.0 * (verts_left_float + verts_bottom_float) + \
                   (rast_rt > 0) * 1.0 * (verts_right_float + verts_top_float) + \
                   (rast_rb > 0) * 1.0 * (verts_right_float + verts_bottom_float)
        occ_feat = occ_feat[None, :, None] / 4.0

        occ, _ = dr.interpolate(occ_feat, rast_out, tri)
        occ = occ.permute(0, 3, 1, 2)

        if feat is not None:
            image, _ = dr.interpolate(feat, rast_out, tri)
            image = image.permute(0, 3, 1, 2)
            image = mask * image

        return mask, depth, image, occ

    def render_uv_texture(self, vertex, tri, uv, uv_texture):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (M, 3), triangles
            uv                -- torch.tensor, size (B,N, 2),  uv mapping
            base_tex   -- torch.tensor, size (B,H,W,C)
        """
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat(
                [vertex, torch.ones([*vertex.shape[:2], 1]).to(device)],
                dim=-1)
            vertex[..., 1] = -vertex[..., 1]

        vertex_ndc = vertex @ ndc_proj.t()
        if self.glctx is None:
            if nvdiffrast.__version__ == '0.2.7':
                self.glctx = dr.RasterizeGLContext(device=device)
            else:
                self.glctx = dr.RasterizeCudaContext(device=device)

        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0]
                                 for f in tri]).unsqueeze(1).to(device)

            print('fnum shape:{}'.format(fnum.shape))

            fstartidx = torch.cumsum(fnum, dim=0) - fnum
            ranges = torch.cat([fstartidx, fnum],
                               axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i * vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(
            self.glctx,
            vertex_ndc.contiguous(),
            tri,
            resolution=[rsize, rsize],
            ranges=ranges)

        depth, _ = dr.interpolate(
            vertex.reshape([-1, 4])[..., 2].unsqueeze(1).contiguous(),
            rast_out, tri)
        depth = depth.permute(0, 3, 1, 2)
        mask = (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth
        uv[..., -1] = 1.0 - uv[..., -1]

        rast_out, rast_db = dr.rasterize(
            self.glctx,
            vertex_ndc.contiguous(),
            tri,
            resolution=[rsize, rsize],
            ranges=ranges)

        interp_out, uv_da = dr.interpolate(
            uv, rast_out, tri, rast_db, diff_attrs='all')

        uv_texture = uv_texture.permute(0, 2, 3, 1).contiguous()
        img = dr.texture(
            uv_texture, interp_out, filter_mode='linear')  # , uv_da)
        img = img * torch.clamp(rast_out[..., -1:], 0,
                                1)  # Mask out background.

        image = img.permute(0, 3, 1, 2)

        return mask, depth, image
