# The implementation is adopted from PASS-reID, made pubicly available under the Apache-2.0 License at
# https://github.com/CASIA-IVA-Lab/PASS-reID

import collections.abc as container_abcs
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F


# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def vit_base_patch16_224_TransReID(
        img_size=(256, 128),
        stride_size=16,
        drop_path_rate=0.1,
        camera=0,
        view=0,
        local_feature=False,
        sie_xishu=1.5,
        **kwargs):
    model = TransReID(
        img_size=img_size,
        patch_size=16,
        stride_size=stride_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        camera=camera,
        view=view,
        drop_path_rate=drop_path_rate,
        sie_xishu=sie_xishu,
        local_feature=local_feature,
        **kwargs)
    return model


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class TransReID(nn.Module):
    """Transformer-based Object Re-Identification
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 stride_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 camera=0,
                 view=0,
                 drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 local_feature=False,
                 sie_xishu=1.0,
                 hw_ratio=1,
                 gem_pool=False,
                 stem_conv=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.local_feature = local_feature
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride_size=stride_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            stem_conv=stem_conv)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part_token3 = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part1_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part2_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part3_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        self.in_planes = 768
        self.gem_pool = gem_pool

        # Initialize SIE Embedding
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(
                torch.zeros(camera * view, 1, embed_dim))
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer) for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc = nn.Linear(embed_dim,
                            num_classes) if num_classes > 0 else nn.Identity()

        self.gem = GeneralizedMeanPooling()

    def forward_features(self, x, camera_id, view_id):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        part_tokens1 = self.part_token1.expand(B, -1, -1)
        part_tokens2 = self.part_token2.expand(B, -1, -1)
        part_tokens3 = self.part_token3.expand(B, -1, -1)
        x = torch.cat(
            (cls_tokens, part_tokens1, part_tokens2, part_tokens3, x), dim=1)

        if self.cam_num > 0 and self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[
                camera_id * self.view_num + view_id]
        elif self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        elif self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
        else:
            x = x + torch.cat((self.cls_pos, self.part1_pos, self.part2_pos,
                               self.part3_pos, self.pos_embed),
                              dim=1)

        x = self.pos_drop(x)

        if self.local_feature:
            for blk in self.blocks[:-1]:
                x = blk(x)
            return x
        else:
            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)
        if self.gem_pool:
            gf = self.gem(x[:, 1:].permute(0, 2, 1)).squeeze()
            return x[:, 0] + gf
        return x[:, 0], x[:, 1], x[:, 2], x[:, 3]

    def forward(self, x, cam_label=None, view_label=None):
        global_feat, local_feat_1, local_feat_2, local_feat_3 = self.forward_features(
            x, cam_label, view_label)
        return global_feat, local_feat_1, local_feat_2, local_feat_3


class PatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping patches
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 stride_size=16,
                 in_chans=3,
                 embed_dim=768,
                 stem_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        self.num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size

        self.stem_conv = stem_conv
        if self.stem_conv:
            hidden_dim = 64
            stem_stride = 2
            stride_size = patch_size = patch_size[0] // stem_stride
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    hidden_dim,
                    kernel_size=7,
                    stride=stem_stride,
                    padding=3,
                    bias=False),
                IBN(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                IBN(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            in_chans = hidden_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)

    def forward(self, x):
        if self.stem_conv:
            x = self.conv(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768]

        return x


class GeneralizedMeanPooling(nn.Module):
    """Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool1d(x, self.output_size).pow(1. / self.p)


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
