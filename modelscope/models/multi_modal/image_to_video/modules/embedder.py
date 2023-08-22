# Copyright (c) Alibaba, Inc. and its affiliates.

import os

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torchvision.transforms as T


class FrozenOpenCLIPVisualEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = ['last', 'penultimate']

    def __init__(self,
                 pretrained,
                 vit_resolution=(224, 224),
                 arch='ViT-H-14',
                 device='cuda',
                 max_length=77,
                 freeze=True,
                 layer='last',
                 **kwargs):
        super().__init__()
        assert layer in self.LAYERS
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, device=torch.device('cpu'), pretrained=pretrained)

        del model.transformer
        self.model = model
        data_white = np.ones(
            (vit_resolution[0], vit_resolution[1], 3), dtype=np.uint8) * 255
        self.white_image = preprocess(T.ToPILImage()(data_white)).unsqueeze(0)

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == 'last':
            self.layer_idx = 0
        elif self.layer == 'penultimate':
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image):
        z = self.model.encode_image(image.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x)

        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting(
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)
