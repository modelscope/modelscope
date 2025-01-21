"""
Part of the implementation is borrowed and modified from LaMa, publicly available at
https://github.com/saic-mdal/lama
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .ade20k import ModelBuilder

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]


class ResNetPL(nn.Module):

    def __init__(self,
                 weight=1,
                 weights_path=None,
                 arch_encoder='resnet50dilated',
                 segmentation=True):
        super().__init__()
        self.impl = ModelBuilder.get_encoder(
            weights_path=weights_path,
            arch_encoder=arch_encoder,
            arch_decoder='ppm_deepsup',
            fc_dim=2048,
            segmentation=segmentation)
        self.impl.eval()
        for w in self.impl.parameters():
            w.requires_grad_(False)

        self.weight = weight

    def forward(self, pred, target):
        pred = (pred - IMAGENET_MEAN.to(pred)) / IMAGENET_STD.to(pred)
        target = (target - IMAGENET_MEAN.to(target)) / IMAGENET_STD.to(target)

        pred_feats = self.impl(pred, return_feature_maps=True)
        target_feats = self.impl(target, return_feature_maps=True)

        result = torch.stack([
            F.mse_loss(cur_pred, cur_target)
            for cur_pred, cur_target in zip(pred_feats, target_feats)
        ]).sum() * self.weight
        return result
