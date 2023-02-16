# Part of the implementation is borrowed and modified from ViTSTR,
# publicly available at https://github.com/roatienza/deep-text-recognition-benchmark
from __future__ import absolute_import, division, print_function
import logging
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .timm_tinyc import VisionTransformer


class ViTSTR(VisionTransformer):
    '''
    ViTSTR is basically a ViT that uses DeiT weights.
    Modified head to support a sequence of characters prediction for STR.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        b, s, e = x.size()
        x = x.reshape(b * s, e)
        x = self.head(x).view(b, s, self.num_classes)
        return x


def vitstr_tiny(num_tokens):
    vitstr = ViTSTR(
        patch_size=1,
        in_chans=512,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True)
    vitstr.reset_classifier(num_classes=num_tokens)
    return vitstr
