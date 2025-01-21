# The implementation is adopted from MTTR,
# made publicly available under the Apache 2.0 License at https://github.com/mttr2021/MTTR
# Modified from DETR https://github.com/facebookresearch/detr

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FPNSpatialDecoder(nn.Module):
    """
    An FPN-like spatial decoder. Generates high-res, semantically rich features which serve as the base for creating
    instance segmentation masks.
    """

    def __init__(self, context_dim, fpn_dims, mask_kernels_dim=8):
        super().__init__()

        inter_dims = [
            context_dim, context_dim // 2, context_dim // 4, context_dim // 8,
            context_dim // 16
        ]
        self.lay1 = torch.nn.Conv2d(context_dim, inter_dims[0], 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, inter_dims[0])
        self.lay2 = torch.nn.Conv2d(inter_dims[0], inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.context_dim = context_dim

        self.add_extra_layer = len(fpn_dims) == 3
        if self.add_extra_layer:
            self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)
            self.lay5 = torch.nn.Conv2d(
                inter_dims[3], inter_dims[4], 3, padding=1)
            self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
            self.out_lay = torch.nn.Conv2d(
                inter_dims[4], mask_kernels_dim, 3, padding=1)
        else:
            self.out_lay = torch.nn.Conv2d(
                inter_dims[3], mask_kernels_dim, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, layer_features: List[Tensor]):
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(layer_features[0])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode='nearest')
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(layer_features[1])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode='nearest')
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        if self.add_extra_layer:
            cur_fpn = self.adapter3(layer_features[2])
            x = cur_fpn + F.interpolate(
                x, size=cur_fpn.shape[-2:], mode='nearest')
            x = self.lay5(x)
            x = self.gn5(x)
            x = F.relu(x)

        x = self.out_lay(x)
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs,
                       targets,
                       num_masks,
                       alpha: float = 0.25,
                       gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks
