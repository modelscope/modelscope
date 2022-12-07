from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from .decoder import Projection, RecurrentDecoder
from .deep_guided_filter import DeepGuidedFilterRefiner
from .effv2 import EfficientNet
from .lraspp import LRASPP


class MattingNetwork(torch.nn.Module):

    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        self.backbone = EfficientNet(pretrained_backbone)
        self.aspp = LRASPP(160, 64)
        self.decoder = RecurrentDecoder([24, 48, 64, 128], [64, 32, 24, 16])
        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)
        self.refiner = DeepGuidedFilterRefiner()

    def forward(self,
                src: Tensor,
                r0: Optional[Tensor] = None,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):

        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src

        f1, f2, f3, f4 = self.backbone(src_sm)
        f4 = self.aspp(f4)
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r0, r1, r2, r3)

        if not segmentation_pass:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            if downsample_ratio != 1:
                _, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            pha = pha.clamp(0., 1.)
            return [pha, *rec]
        else:
            seg = self.project_seg(hid)
            return [seg, *rec]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(
                x.flatten(0, 1),
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(
                x,
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=False)
        return x
