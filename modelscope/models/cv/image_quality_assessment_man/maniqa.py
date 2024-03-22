# This implementation is adopted from MANIQA, made publicly available under the Apache License 2.0 at
# https://github.com/IIGROUP/MANIQA/blob/master/models/maniqa.py

import timm
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Block

from .swin import SwinTransformer


class TABlock(nn.Module):

    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class SaveOutput:

    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


class MANIQA(nn.Module):

    def __init__(self,
                 embed_dim=768,
                 num_outputs=1,
                 patch_size=8,
                 drop=0.1,
                 depths=[2, 2],
                 window_size=4,
                 dim_mlp=768,
                 num_heads=[4, 4],
                 img_size=224,
                 num_tab=2,
                 scale=0.13,
                 **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size,
                                   img_size // patch_size)

        self.vit = timm.create_model('vit_base_patch8_224', pretrained=False)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size**2)
            self.tablock1.append(tab)

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale)

        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size**2)
            self.tablock2.append(tab)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale)

        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU())
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid())

    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def forward(self, x):
        self.vit(x)
        x = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()

        # stage 1
        x = rearrange(
            x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(
            x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x = self.swintransformer1(x)

        # stage2
        x = rearrange(
            x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(
            x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = self.swintransformer2(x)

        x = rearrange(
            x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score
