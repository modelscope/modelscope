# Part of the implementation is borrowed and modified from Deep3DFaceRecon_pytorch,
# publicly available at https://github.com/sicxu/Deep3DFaceRecon_pytorch
import warnings
from typing import List

import numpy as np
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

        tex_map = uv_texture[0].detach().cpu().numpy()[..., ::-1] * 255.0

        image = img.permute(0, 3, 1, 2)

        return mask, depth, image, tex_map

    def pred_shape_and_texture(self,
                               vertex,
                               tri,
                               uv,
                               target_img,
                               base_tex=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            uv                -- torch.tensor, size (B,N, 2),  uv mapping
            base_tex   -- torch.tensor, size (B,H,W,C)
        """
        vertex = torch.cat([vertex[:, :35241, :], vertex[:, 37082:, :]],
                           dim=1)  # BFM front
        tri = torch.cat([tri[:69732, :], tri[73936:, ]], dim=0)
        uv = torch.cat([uv[:, :35241, :], uv[:, 37082:, :]], dim=1)
        tri[69732:, :] = tri[69732:, :] - (37082 - 35241)

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
            self.glctx = dr.RasterizeCudaContext(device=device)

        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0]
                                 for f in tri]).unsqueeze(1).to(device)

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

        mask_3c = mask.permute(0, 2, 3, 1)
        mask_3c = torch.cat((mask_3c, mask_3c, mask_3c), dim=-1)
        maskout_img = mask_3c * target_img
        mean_color = torch.sum(maskout_img, dim=(1, 2))
        valid_pixel_count = torch.sum(mask)

        mean_color = mean_color / valid_pixel_count

        tex = torch.zeros((1, 128 * 5 // 4, 128, 3), dtype=torch.float32)
        tex[:, :, :, 0] = mean_color[0, 0]
        tex[:, :, :, 1] = mean_color[0, 1]
        tex[:, :, :, 2] = mean_color[0, 2]

        tex = tex.cuda()

        tex_mask = torch.zeros((1, 2048 * 5 // 4, 2048, 3),
                               dtype=torch.float32)
        tex_mask[:, :, :, 1] = 1.0
        tex_mask = tex_mask.cuda()
        tex_mask.requires_grad = True
        tex_mask = tex_mask.contiguous()

        criterionTV = TVLoss()

        if base_tex is not None:
            base_tex = base_tex.cuda()

        for tex_resolution in [64, 128, 256, 512, 1024, 2048]:
            tex = tex.detach()
            tex = tex.permute(0, 3, 1, 2)
            tex = F.interpolate(tex, (tex_resolution * 5 // 4, tex_resolution))
            tex = tex.permute(0, 2, 3, 1).contiguous()

            if base_tex is not None:
                _base_tex = base_tex.permute(0, 3, 1, 2)
                _base_tex = F.interpolate(
                    _base_tex, (tex_resolution * 5 // 4, tex_resolution))
                _base_tex = _base_tex.permute(0, 2, 3, 1).contiguous()
                tex += _base_tex

            tex.requires_grad = True
            optim = torch.optim.Adam([tex], lr=1e-2)

            texture_opt_iters = 100

            if tex_resolution == 2048:
                optim_mask = torch.optim.Adam([tex_mask], lr=1e-2)

            for i in range(int(texture_opt_iters)):

                if tex_resolution == 2048:
                    optim_mask.zero_grad()
                    rendered = dr.texture(
                        tex_mask, interp_out, filter_mode='linear')  # , uv_da)
                    rendered = rendered * torch.clamp(
                        rast_out[..., -1:], 0, 1)  # Mask out background.
                    tex_loss = torch.mean((target_img - rendered)**2)

                    tex_loss.backward()
                    optim_mask.step()

                optim.zero_grad()

                img = dr.texture(
                    tex, interp_out, filter_mode='linear')  # , uv_da)
                img = img * torch.clamp(rast_out[..., -1:], 0,
                                        1)  # Mask out background.
                recon_loss = torch.mean((target_img - img)**2)

                if tex_resolution < 2048:
                    tv_loss = criterionTV(tex.permute(0, 3, 1, 2))

                    total_loss = recon_loss + tv_loss * 0.01
                else:

                    total_loss = recon_loss

                total_loss.backward()
                optim.step()

        tex_map = tex[0].detach().cpu().numpy()[..., ::-1] * 255.0

        image = img.permute(0, 3, 1, 2)

        tex_mask = tex_mask[0].detach().cpu().numpy() * 255.0
        tex_mask = np.where(tex_mask[..., 1] > 250, 1.0, 0.0) * np.where(
            tex_mask[..., 0] < 10, 1.0, 0) * np.where(tex_mask[..., 2] < 10,
                                                      1.0, 0)
        tex_mask = 1.0 - tex_mask

        return mask, depth, image, tex_map, tex_mask
