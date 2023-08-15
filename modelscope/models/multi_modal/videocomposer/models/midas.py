r"""A much cleaner re-implementation of ``https://github.com/isl-org/MiDaS''.
    Image augmentation: T.Compose([
        Resize(
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            interpolation=cv2.INTER_CUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])]).
    Fast inference:
        model = model.to(memory_format=torch.channels_last).half()
        input = input.to(memory_format=torch.channels_last).half()
        output = model(input)
"""
import math
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MiDaS', 'midas_v3']


class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads):
        assert dim % num_heads == 0
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # layers
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, l, c, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        q, k, v = self.to_qkv(x).view(b, l, n * 3, d).chunk(3, dim=2)

        # compute attention
        attn = self.scale * torch.einsum('binc,bjnc->bnij', q, k)
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)

        # gather context
        x = torch.einsum('bnij,bjnc->binc', attn, v)
        x = x.reshape(b, l, c)

        # output
        x = self.proj(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, dim, num_heads):
        super(AttentionBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # layers
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):

    def __init__(self,
                 image_size=384,
                 patch_size=16,
                 dim=1024,
                 out_dim=1000,
                 num_heads=16,
                 num_layers=24):
        assert image_size % patch_size == 0
        super(VisionTransformer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_patches = (image_size // patch_size)**2

        # embeddings
        self.patch_embedding = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(
            torch.empty(1, self.num_patches + 1, dim).normal_(std=0.02))

        # blocks
        self.blocks = nn.Sequential(
            *[AttentionBlock(dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)

        # head
        self.head = nn.Linear(dim, out_dim)

    def forward(self, x):
        b = x.size(0)

        # embeddings
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([self.cls_embedding.repeat(b, 1, 1), x], dim=1)
        x = x + self.pos_embedding

        # blocks
        x = self.blocks(x)
        x = self.norm(x)

        # head
        x = self.head(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.dim = dim

        # layers
        self.residual = nn.Sequential(
            nn.ReLU(inplace=False),  # NOTE: avoid modifying the input
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1))

    def forward(self, x):
        return x + self.residual(x)


class FusionBlock(nn.Module):

    def __init__(self, dim):
        super(FusionBlock, self).__init__()
        self.dim = dim

        # layers
        self.layer1 = ResidualBlock(dim)
        self.layer2 = ResidualBlock(dim)
        self.conv_out = nn.Conv2d(dim, dim, 1)

    def forward(self, *xs):
        assert len(xs) in (1, 2), 'invalid number of inputs'
        if len(xs) == 1:
            x = self.layer2(xs[0])
        else:
            x = self.layer2(xs[0] + self.layer1(xs[1]))
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_out(x)
        return x


class MiDaS(nn.Module):
    r"""MiDaS v3.0 DPT-Large from ``https://github.com/isl-org/MiDaS''.
        Monocular depth estimation using dense prediction transformers.
    """

    def __init__(self,
                 image_size=384,
                 patch_size=16,
                 dim=1024,
                 neck_dims=[256, 512, 1024, 1024],
                 fusion_dim=256,
                 num_heads=16,
                 num_layers=24):
        assert image_size % patch_size == 0
        assert num_layers % 4 == 0
        super(MiDaS, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.neck_dims = neck_dims
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_patches = (image_size // patch_size)**2

        # embeddings
        self.patch_embedding = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(
            torch.empty(1, self.num_patches + 1, dim).normal_(std=0.02))

        # blocks
        stride = num_layers // 4
        self.blocks = nn.Sequential(
            *[AttentionBlock(dim, num_heads) for _ in range(num_layers)])
        self.slices = [slice(i * stride, (i + 1) * stride) for i in range(4)]

        # stage1 (4x)
        self.fc1 = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU())
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, neck_dims[0], 1),
            nn.ConvTranspose2d(neck_dims[0], neck_dims[0], 4, stride=4),
            nn.Conv2d(neck_dims[0], fusion_dim, 3, padding=1, bias=False))
        self.fusion1 = FusionBlock(fusion_dim)

        # stage2 (8x)
        self.fc2 = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, neck_dims[1], 1),
            nn.ConvTranspose2d(neck_dims[1], neck_dims[1], 2, stride=2),
            nn.Conv2d(neck_dims[1], fusion_dim, 3, padding=1, bias=False))
        self.fusion2 = FusionBlock(fusion_dim)

        # stage3 (16x)
        self.fc3 = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, neck_dims[2], 1),
            nn.Conv2d(neck_dims[2], fusion_dim, 3, padding=1, bias=False))
        self.fusion3 = FusionBlock(fusion_dim)

        # stage4 (32x)
        self.fc4 = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(dim, neck_dims[3], 1),
            nn.Conv2d(neck_dims[3], neck_dims[3], 3, stride=2, padding=1),
            nn.Conv2d(neck_dims[3], fusion_dim, 3, padding=1, bias=False))
        self.fusion4 = FusionBlock(fusion_dim)

        # head
        self.head = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim // 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(fusion_dim // 2, 32, 3, padding=1),
            nn.ReLU(inplace=True), nn.ConvTranspose2d(32, 1, 1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b, _, h, w, p = *x.size(), self.patch_size
        assert h % p == 0 and w % p == 0, f'Image size ({w}, {h}) is not divisible by patch size ({p}, {p})'
        hp, wp, grid = h // p, w // p, self.image_size // p

        # embeddings
        pos_embedding = torch.cat([
            self.pos_embedding[:, :1],
            F.interpolate(
                self.pos_embedding[:, 1:].reshape(1, grid, grid, -1).permute(
                    0, 3, 1, 2),
                size=(hp, wp),
                mode='bilinear',
                align_corners=False).permute(0, 2, 3, 1).reshape(
                    1, hp * wp, -1)
        ],
                                  dim=1)  # noqa
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([self.cls_embedding.repeat(b, 1, 1), x], dim=1)
        x = x + pos_embedding

        # stage1
        x = self.blocks[self.slices[0]](x)
        x1 = torch.cat([x[:, 1:], x[:, :1].expand_as(x[:, 1:])], dim=-1)
        x1 = self.fc1(x1).permute(0, 2, 1).unflatten(2, (hp, wp))
        x1 = self.conv1(x1)

        # stage2
        x = self.blocks[self.slices[1]](x)
        x2 = torch.cat([x[:, 1:], x[:, :1].expand_as(x[:, 1:])], dim=-1)
        x2 = self.fc2(x2).permute(0, 2, 1).unflatten(2, (hp, wp))
        x2 = self.conv2(x2)

        # stage3
        x = self.blocks[self.slices[2]](x)
        x3 = torch.cat([x[:, 1:], x[:, :1].expand_as(x[:, 1:])], dim=-1)
        x3 = self.fc3(x3).permute(0, 2, 1).unflatten(2, (hp, wp))
        x3 = self.conv3(x3)

        # stage4
        x = self.blocks[self.slices[3]](x)
        x4 = torch.cat([x[:, 1:], x[:, :1].expand_as(x[:, 1:])], dim=-1)
        x4 = self.fc4(x4).permute(0, 2, 1).unflatten(2, (hp, wp))
        x4 = self.conv4(x4)

        # fusion
        x4 = self.fusion4(x4)
        x3 = self.fusion3(x4, x3)
        x2 = self.fusion2(x3, x2)
        x1 = self.fusion1(x2, x1)

        # head
        x = self.head(x1)
        return x


def midas_v3(model_dir, pretrained=False, **kwargs):
    cfg = dict(
        image_size=384,
        patch_size=16,
        dim=1024,
        neck_dims=[256, 512, 1024, 1024],
        fusion_dim=256,
        num_heads=16,
        num_layers=24)
    cfg.update(**kwargs)
    model = MiDaS(**cfg)
    if pretrained:
        model.load_state_dict(
            torch.load(
                os.path.join(model_dir, 'midas_v3_dpt_large.pth'),
                map_location='cpu'))
    return model
