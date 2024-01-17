# BSD 3-Clause License

# Copyright (c) 2022, Zhao Xiaoming
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Authors:
# Xiaoming Zhao, Xingming Wu, Weihai Chen, Peter C.Y. Chen, Qingsong Xu, and Zhengguo Li
# Code from https://github.com/Shiaoming/ALIKED

from typing import Callable, Optional

import torch
import torch.nn.functional as F
import torchvision
from kornia.color import grayscale_to_rgb
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.models import resnet

from .utils import Extractor


def get_patches(
    tensor: torch.Tensor, required_corners: torch.Tensor, ps: int
) -> torch.Tensor:
    c, h, w = tensor.shape
    corner = (required_corners - ps / 2 + 1).long()
    corner[:, 0] = corner[:, 0].clamp(min=0, max=w - 1 - ps)
    corner[:, 1] = corner[:, 1].clamp(min=0, max=h - 1 - ps)
    offset = torch.arange(0, ps)

    kw = {"indexing": "ij"} if torch.__version__ >= "1.10" else {}
    x, y = torch.meshgrid(offset, offset, **kw)
    patches = torch.stack((x, y)).permute(2, 1, 0).unsqueeze(2)
    patches = patches.to(corner) + corner[None, None]
    pts = patches.reshape(-1, 2)
    sampled = tensor.permute(1, 2, 0)[tuple(pts.T)[::-1]]
    sampled = sampled.reshape(ps, ps, -1, c)
    assert sampled.shape[:3] == patches.shape[:3]
    return sampled.permute(2, 3, 0, 1)


def simple_nms(scores: torch.Tensor, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""

    zeros = torch.zeros_like(scores)
    max_mask = scores == torch.nn.functional.max_pool2d(
        scores, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
    )

    for _ in range(2):
        supp_mask = (
            torch.nn.functional.max_pool2d(
                max_mask.float(),
                kernel_size=nms_radius * 2 + 1,
                stride=1,
                padding=nms_radius,
            )
            > 0
        )
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == torch.nn.functional.max_pool2d(
            supp_scores, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


class DKD(nn.Module):
    def __init__(
        self,
        radius: int = 2,
        top_k: int = 0,
        scores_th: float = 0.2,
        n_limit: int = 20000,
    ):
        """
        Args:
            radius: soft detection radius, kernel size is (2 * radius + 1)
            top_k: top_k > 0: return top k keypoints
            scores_th: top_k <= 0 threshold mode:
                scores_th > 0: return keypoints with scores>scores_th
                else: return keypoints with scores > scores.mean()
            n_limit: max number of keypoint in threshold mode
        """
        super().__init__()
        self.radius = radius
        self.top_k = top_k
        self.scores_th = scores_th
        self.n_limit = n_limit
        self.kernel_size = 2 * self.radius + 1
        self.temperature = 0.1  # tuned temperature
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=self.radius)
        # local xy grid
        x = torch.linspace(-self.radius, self.radius, self.kernel_size)
        # (kernel_size*kernel_size) x 2 : (w,h)
        kw = {"indexing": "ij"} if torch.__version__ >= "1.10" else {}
        self.hw_grid = (
            torch.stack(torch.meshgrid([x, x], **kw)).view(2, -1).t()[:, [1, 0]]
        )

    def forward(
        self,
        scores_map: torch.Tensor,
        sub_pixel: bool = True,
        image_size: Optional[torch.Tensor] = None,
    ):
        """
        :param scores_map: Bx1xHxW
        :param descriptor_map: BxCxHxW
        :param sub_pixel: whether to use sub-pixel keypoint detection
        :return: kpts: list[Nx2,...]; kptscores: list[N,....] normalised position: -1~1
        """
        b, c, h, w = scores_map.shape
        scores_nograd = scores_map.detach()
        nms_scores = simple_nms(scores_nograd, self.radius)

        # remove border
        nms_scores[:, :, : self.radius, :] = 0
        nms_scores[:, :, :, : self.radius] = 0
        if image_size is not None:
            for i in range(scores_map.shape[0]):
                w, h = image_size[i].long()
                nms_scores[i, :, h.item() - self.radius :, :] = 0
                nms_scores[i, :, :, w.item() - self.radius :] = 0
        else:
            nms_scores[:, :, -self.radius :, :] = 0
            nms_scores[:, :, :, -self.radius :] = 0

        # detect keypoints without grad
        if self.top_k > 0:
            topk = torch.topk(nms_scores.view(b, -1), self.top_k)
            indices_keypoints = [topk.indices[i] for i in range(b)]  # B x top_k
        else:
            if self.scores_th > 0:
                masks = nms_scores > self.scores_th
                if masks.sum() == 0:
                    th = scores_nograd.reshape(b, -1).mean(dim=1)  # th = self.scores_th
                    masks = nms_scores > th.reshape(b, 1, 1, 1)
            else:
                th = scores_nograd.reshape(b, -1).mean(dim=1)  # th = self.scores_th
                masks = nms_scores > th.reshape(b, 1, 1, 1)
            masks = masks.reshape(b, -1)

            indices_keypoints = []  # list, B x (any size)
            scores_view = scores_nograd.reshape(b, -1)
            for mask, scores in zip(masks, scores_view):
                indices = mask.nonzero()[:, 0]
                if len(indices) > self.n_limit:
                    kpts_sc = scores[indices]
                    sort_idx = kpts_sc.sort(descending=True)[1]
                    sel_idx = sort_idx[: self.n_limit]
                    indices = indices[sel_idx]
                indices_keypoints.append(indices)

        wh = torch.tensor([w - 1, h - 1], device=scores_nograd.device)

        keypoints = []
        scoredispersitys = []
        kptscores = []
        if sub_pixel:
            # detect soft keypoints with grad backpropagation
            patches = self.unfold(scores_map)  # B x (kernel**2) x (H*W)
            self.hw_grid = self.hw_grid.to(scores_map)  # to device
            for b_idx in range(b):
                patch = patches[b_idx].t()  # (H*W) x (kernel**2)
                indices_kpt = indices_keypoints[
                    b_idx
                ]  # one dimension vector, say its size is M
                patch_scores = patch[indices_kpt]  # M x (kernel**2)
                keypoints_xy_nms = torch.stack(
                    [indices_kpt % w, torch.div(indices_kpt, w, rounding_mode="trunc")],
                    dim=1,
                )  # Mx2

                # max is detached to prevent undesired backprop loops in the graph
                max_v = patch_scores.max(dim=1).values.detach()[:, None]
                x_exp = (
                    (patch_scores - max_v) / self.temperature
                ).exp()  # M * (kernel**2), in [0, 1]

                # \frac{ \sum{(i,j) \times \exp(x/T)} }{ \sum{\exp(x/T)} }
                xy_residual = (
                    x_exp @ self.hw_grid / x_exp.sum(dim=1)[:, None]
                )  # Soft-argmax, Mx2

                hw_grid_dist2 = (
                    torch.norm(
                        (self.hw_grid[None, :, :] - xy_residual[:, None, :])
                        / self.radius,
                        dim=-1,
                    )
                    ** 2
                )
                scoredispersity = (x_exp * hw_grid_dist2).sum(dim=1) / x_exp.sum(dim=1)

                # compute result keypoints
                keypoints_xy = keypoints_xy_nms + xy_residual
                keypoints_xy = keypoints_xy / wh * 2 - 1  # (w,h) -> (-1~1,-1~1)

                kptscore = torch.nn.functional.grid_sample(
                    scores_map[b_idx].unsqueeze(0),
                    keypoints_xy.view(1, 1, -1, 2),
                    mode="bilinear",
                    align_corners=True,
                )[
                    0, 0, 0, :
                ]  # CxN

                keypoints.append(keypoints_xy)
                scoredispersitys.append(scoredispersity)
                kptscores.append(kptscore)
        else:
            for b_idx in range(b):
                indices_kpt = indices_keypoints[
                    b_idx
                ]  # one dimension vector, say its size is M
                # To avoid warning: UserWarning: __floordiv__ is deprecated
                keypoints_xy_nms = torch.stack(
                    [indices_kpt % w, torch.div(indices_kpt, w, rounding_mode="trunc")],
                    dim=1,
                )  # Mx2
                keypoints_xy = keypoints_xy_nms / wh * 2 - 1  # (w,h) -> (-1~1,-1~1)
                kptscore = torch.nn.functional.grid_sample(
                    scores_map[b_idx].unsqueeze(0),
                    keypoints_xy.view(1, 1, -1, 2),
                    mode="bilinear",
                    align_corners=True,
                )[
                    0, 0, 0, :
                ]  # CxN
                keypoints.append(keypoints_xy)
                scoredispersitys.append(kptscore)  # for jit.script compatability
                kptscores.append(kptscore)

        return keypoints, scoredispersitys, kptscores


class InputPadder(object):
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, h: int, w: int, divis_by: int = 8):
        self.ht = h
        self.wd = w
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2,
        ]

    def pad(self, x: torch.Tensor):
        assert x.ndim == 4
        return F.pad(x, self._pad, mode="replicate")

    def unpad(self, x: torch.Tensor):
        assert x.ndim == 4
        ht = x.shape[-2]
        wd = x.shape[-1]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        mask=False,
    ):
        super(DeformableConv2d, self).__init__()

        self.padding = padding
        self.mask = mask

        self.channel_num = (
            3 * kernel_size * kernel_size if mask else 2 * kernel_size * kernel_size
        )
        self.offset_conv = nn.Conv2d(
            in_channels,
            self.channel_num,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True,
        )

        self.regular_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.0

        out = self.offset_conv(x)
        if self.mask:
            o1, o2, mask = torch.chunk(out, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)
        else:
            offset = out
            mask = None
        offset = offset.clamp(-max_offset, max_offset)
        x = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=mask,
        )
        return x


def get_conv(
    inplanes,
    planes,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False,
    conv_type="conv",
    mask=False,
):
    if conv_type == "conv":
        conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    elif conv_type == "dcn":
        conv = DeformableConv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=_pair(padding),
            bias=bias,
            mask=mask,
        )
    else:
        raise TypeError
    return conv


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        gate: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_type: str = "conv",
        mask: bool = False,
    ):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = get_conv(
            in_channels, out_channels, kernel_size=3, conv_type=conv_type, mask=mask
        )
        self.bn1 = norm_layer(out_channels)
        self.conv2 = get_conv(
            out_channels, out_channels, kernel_size=3, conv_type=conv_type, mask=mask
        )
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x


# modified based on torchvision\models\resnet.py#27->BasicBlock
class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        gate: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_type: str = "conv",
        mask: bool = False,
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("ResBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers
        # downsample the input when stride != 1
        self.conv1 = get_conv(
            inplanes, planes, kernel_size=3, conv_type=conv_type, mask=mask
        )
        self.bn1 = norm_layer(planes)
        self.conv2 = get_conv(
            planes, planes, kernel_size=3, conv_type=conv_type, mask=mask
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gate(out)

        return out


class SDDH(nn.Module):
    def __init__(
        self,
        dims: int,
        kernel_size: int = 3,
        n_pos: int = 8,
        gate=nn.ReLU(),
        conv2D=False,
        mask=False,
    ):
        super(SDDH, self).__init__()
        self.kernel_size = kernel_size
        self.n_pos = n_pos
        self.conv2D = conv2D
        self.mask = mask

        self.get_patches_func = get_patches

        # estimate offsets
        self.channel_num = 3 * n_pos if mask else 2 * n_pos
        self.offset_conv = nn.Sequential(
            nn.Conv2d(
                dims,
                self.channel_num,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=True,
            ),
            gate,
            nn.Conv2d(
                self.channel_num,
                self.channel_num,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        # sampled feature conv
        self.sf_conv = nn.Conv2d(
            dims, dims, kernel_size=1, stride=1, padding=0, bias=False
        )

        # convM
        if not conv2D:
            # deformable desc weights
            agg_weights = torch.nn.Parameter(torch.rand(n_pos, dims, dims))
            self.register_parameter("agg_weights", agg_weights)
        else:
            self.convM = nn.Conv2d(
                dims * n_pos, dims, kernel_size=1, stride=1, padding=0, bias=False
            )

    def forward(self, x, keypoints):
        # x: [B,C,H,W]
        # keypoints: list, [[N_kpts,2], ...] (w,h)
        b, c, h, w = x.shape
        wh = torch.tensor([[w - 1, h - 1]], device=x.device)
        max_offset = max(h, w) / 4.0

        offsets = []
        descriptors = []
        # get offsets for each keypoint
        for ib in range(b):
            xi, kptsi = x[ib], keypoints[ib]
            kptsi_wh = (kptsi / 2 + 0.5) * wh
            N_kpts = len(kptsi)

            if self.kernel_size > 1:
                patch = self.get_patches_func(
                    xi, kptsi_wh.long(), self.kernel_size
                )  # [N_kpts, C, K, K]
            else:
                kptsi_wh_long = kptsi_wh.long()
                patch = (
                    xi[:, kptsi_wh_long[:, 1], kptsi_wh_long[:, 0]]
                    .permute(1, 0)
                    .reshape(N_kpts, c, 1, 1)
                )

            offset = self.offset_conv(patch).clamp(
                -max_offset, max_offset
            )  # [N_kpts, 2*n_pos, 1, 1]
            if self.mask:
                offset = (
                    offset[:, :, 0, 0].view(N_kpts, 3, self.n_pos).permute(0, 2, 1)
                )  # [N_kpts, n_pos, 3]
                offset = offset[:, :, :-1]  # [N_kpts, n_pos, 2]
                mask_weight = torch.sigmoid(offset[:, :, -1])  # [N_kpts, n_pos]
            else:
                offset = (
                    offset[:, :, 0, 0].view(N_kpts, 2, self.n_pos).permute(0, 2, 1)
                )  # [N_kpts, n_pos, 2]
            offsets.append(offset)  # for visualization

            # get sample positions
            pos = kptsi_wh.unsqueeze(1) + offset  # [N_kpts, n_pos, 2]
            pos = 2.0 * pos / wh[None] - 1
            pos = pos.reshape(1, N_kpts * self.n_pos, 1, 2)

            # sample features
            features = F.grid_sample(
                xi.unsqueeze(0), pos, mode="bilinear", align_corners=True
            )  # [1,C,(N_kpts*n_pos),1]
            features = features.reshape(c, N_kpts, self.n_pos, 1).permute(
                1, 0, 2, 3
            )  # [N_kpts, C, n_pos, 1]
            if self.mask:
                features = torch.einsum("ncpo,np->ncpo", features, mask_weight)

            features = torch.selu_(self.sf_conv(features)).squeeze(
                -1
            )  # [N_kpts, C, n_pos]
            # convM
            if not self.conv2D:
                descs = torch.einsum(
                    "ncp,pcd->nd", features, self.agg_weights
                )  # [N_kpts, C]
            else:
                features = features.reshape(N_kpts, -1)[
                    :, :, None, None
                ]  # [N_kpts, C*n_pos, 1, 1]
                descs = self.convM(features).squeeze()  # [N_kpts, C]

            # normalize
            descs = F.normalize(descs, p=2.0, dim=1)
            descriptors.append(descs)

        return descriptors, offsets


class ALIKED(Extractor):
    default_conf = {
        "model_name": "aliked-n16",
        "max_num_keypoints": -1,
        "detection_threshold": 0.2,
        "nms_radius": 2,
    }

    checkpoint_url = "https://github.com/Shiaoming/ALIKED/raw/main/models/{}.pth"

    n_limit_max = 20000

    # c1, c2, c3, c4, dim, K, M
    cfgs = {
        "aliked-t16": [8, 16, 32, 64, 64, 3, 16],
        "aliked-n16": [16, 32, 64, 128, 128, 3, 16],
        "aliked-n16rot": [16, 32, 64, 128, 128, 3, 16],
        "aliked-n32": [16, 32, 64, 128, 128, 3, 32],
    }
    preprocess_conf = {
        "resize": 1024,
    }

    required_data_keys = ["image"]

    def __init__(self, **conf):
        super().__init__(**conf)  # Update with default configuration.
        conf = self.conf
        c1, c2, c3, c4, dim, K, M = self.cfgs[conf.model_name]
        conv_types = ["conv", "conv", "dcn", "dcn"]
        conv2D = False
        mask = False

        # build model
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d
        self.gate = nn.SELU(inplace=True)
        self.block1 = ConvBlock(3, c1, self.gate, self.norm, conv_type=conv_types[0])
        self.block2 = self.get_resblock(c1, c2, conv_types[1], mask)
        self.block3 = self.get_resblock(c2, c3, conv_types[2], mask)
        self.block4 = self.get_resblock(c3, c4, conv_types[3], mask)

        self.conv1 = resnet.conv1x1(c1, dim // 4)
        self.conv2 = resnet.conv1x1(c2, dim // 4)
        self.conv3 = resnet.conv1x1(c3, dim // 4)
        self.conv4 = resnet.conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )
        self.upsample8 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=True
        )
        self.upsample32 = nn.Upsample(
            scale_factor=32, mode="bilinear", align_corners=True
        )
        self.score_head = nn.Sequential(
            resnet.conv1x1(dim, 8),
            self.gate,
            resnet.conv3x3(8, 4),
            self.gate,
            resnet.conv3x3(4, 4),
            self.gate,
            resnet.conv3x3(4, 1),
        )
        self.desc_head = SDDH(dim, K, M, gate=self.gate, conv2D=conv2D, mask=mask)
        self.dkd = DKD(
            radius=conf.nms_radius,
            top_k=-1 if conf.detection_threshold > 0 else conf.max_num_keypoints,
            scores_th=conf.detection_threshold,
            n_limit=conf.max_num_keypoints
            if conf.max_num_keypoints > 0
            else self.n_limit_max,
        )

        state_dict = torch.hub.load_state_dict_from_url(
            self.checkpoint_url.format(conf.model_name), map_location="cpu"
        )
        self.load_state_dict(state_dict, strict=True)

    def get_resblock(self, c_in, c_out, conv_type, mask):
        return ResBlock(
            c_in,
            c_out,
            1,
            nn.Conv2d(c_in, c_out, 1),
            gate=self.gate,
            norm_layer=self.norm,
            conv_type=conv_type,
            mask=mask,
        )

    def extract_dense_map(self, image):
        # Pads images such that dimensions are divisible by
        div_by = 2**5
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)
        image = padder.pad(image)

        # ================================== feature encoder
        x1 = self.block1(image)  # B x c1 x H x W
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2
        x3 = self.pool4(x2)
        x3 = self.block3(x3)  # B x c3 x H/8 x W/8
        x4 = self.pool4(x3)
        x4 = self.block4(x4)  # B x dim x H/32 x W/32
        # ================================== feature aggregation
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//8 x W//8
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//32 x W//32
        x2_up = self.upsample2(x2)  # B x dim//4 x H x W
        x3_up = self.upsample8(x3)  # B x dim//4 x H x W
        x4_up = self.upsample32(x4)  # B x dim//4 x H x W
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)
        # ================================== score head
        score_map = torch.sigmoid(self.score_head(x1234))
        feature_map = torch.nn.functional.normalize(x1234, p=2, dim=1)

        # Unpads images
        feature_map = padder.unpad(feature_map)
        score_map = padder.unpad(score_map)

        return feature_map, score_map

    def forward(self, data: dict) -> dict:
        image = data["image"]
        if image.shape[1] == 1:
            image = grayscale_to_rgb(image)
        feature_map, score_map = self.extract_dense_map(image)
        keypoints, kptscores, scoredispersitys = self.dkd(
            score_map, image_size=data.get("image_size")
        )
        descriptors, offsets = self.desc_head(feature_map, keypoints)

        _, _, h, w = image.shape
        wh = torch.tensor([w - 1, h - 1], device=image.device)
        # no padding required
        # we can set detection_threshold=-1 and conf.max_num_keypoints > 0
        return {
            "keypoints": wh * (torch.stack(keypoints) + 1) / 2.0,  # B x N x 2
            "descriptors": torch.stack(descriptors),  # B x N x D
            "keypoint_scores": torch.stack(kptscores),  # B x N
        }
