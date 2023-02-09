# The implementation here is modified based on timm,
# originally Apache 2.0 License and publicly available at
# https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .petl import Adapter, LoRA, Prefix, Prompt
from .timm_vision_transformer import (Attention, Block, DropPath, LayerScale,
                                      Mlp, PatchEmbed, VisionTransformer)


class AttentionPETL(nn.Module):
    """Extend the parameter-efficient transfer learning (PETL) method to the original Attention.

    Prefix tuning optimizes the task-specific vector in the multi-head attention layer.
    'Prefix-tuning: Optimizing continuous prompts for generation' by Li & Liang(2021)
    See https://arxiv.org/abs/2101.00190

    LoRA constructs an additional layer with low-rank decomposition matrices of the weights in the network.
    'LoRA: Low-Rank Adaptation of Large Language Models' by Hu et al.(2021)
    See https://arxiv.org/abs/2106.09685

    Attributes:
        prefix_length: An integer indicating the length of prefix tuning.
        prefix_type: A string indicating the type of prefix tuning.
        lora_length: An integer indicating the length of LoRA tuning.
        lora_type: A string indicating the type of LoRA tuning.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        prefix_length=None,
        prefix_type=None,
        lora_length=None,
        lora_type=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if lora_length and lora_length > 0:
            self.lora = LoRA(
                dim=dim,
                num_heads=num_heads,
                lora_length=lora_length,
                lora_type=lora_type)
        else:
            self.lora = None

        if prefix_length and prefix_length > 0:
            self.prefix = Prefix(
                dim=dim,
                num_heads=num_heads,
                prefix_length=prefix_length,
                prefix_type=prefix_type)
        else:
            self.prefix = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.lora is not None:
            q, k, v = self.lora(x, q, k, v)

        if self.prefix is not None:
            q, k, v = self.prefix(x, q, k, v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockPETL(nn.Module):
    """Extend the parameter-efficient transfer learning (PETL) method to the original Block.

    Visual prompt tuning (VPT) is proposed to initialize tunable prompt tokens
    and prepend to the original tokens in the first layer or multiple layers.
    'Visual Prompt Tuning' by Jia et al.(2022)
    See https://arxiv.org/abs/2203.12119

    Adapters project input tokens by an MLP layer.
    'Parameter-Efficient Transfer Learning for NLP' by Houlsby et al.(2019)
    See http://arxiv.org/abs/1902.00751

    Attributes:
        adapter_length: An integer indicating the length of adapter tuning.
        adapter_type: A string indicating the type of adapter tuning.
        prompt_length: An integer indicating the length of prompt tuning.
        prompt_type: A string indicating the type of prompt tuning.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        init_values=None,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_layer=Attention,
        layer_num=-1,
        prompt_length=None,
        prompt_type=None,
        prefix_length=None,
        prefix_type=None,
        adapter_length=None,
        adapter_type=None,
        lora_length=None,
        lora_type=None,
    ):
        super().__init__()
        self.layer_num = layer_num
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            prefix_length=prefix_length,
            prefix_type=prefix_type,
            lora_length=lora_length,
            lora_type=lora_type,
        )
        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()

        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop)
        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.adapter_length = adapter_length
        self.adapter_type = adapter_type
        if adapter_length and adapter_length > 0:
            self.adapter = Adapter(
                dim=dim,
                adapter_length=adapter_length,
                adapter_type=adapter_type,
                act_layer=act_layer)
        else:
            self.adapter = None

        self.prompt_length = prompt_length
        self.prompt_type = prompt_type
        if prompt_length and prompt_length > 0:
            self.prompt = Prompt(
                dim=dim,
                layer_num=layer_num,
                prompt_length=prompt_length,
                prompt_type=prompt_type)
        else:
            self.prompt = None

    def forward(self, x):
        if self.prompt is not None:
            x = self.prompt(x)

        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))

        if self.adapter is not None:
            x = x + self.adapter(
                self.drop_path2(self.ls2(self.mlp(self.norm2(x)))))
        else:
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformerPETL(VisionTransformer):
    """ Extend the parameter-efficient transfer learning (PETL) method to the original Vision Transformer.

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    The implementation of several tuning methods (prompt, prefix, adapter, and LoRA) based on ViT.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool='token',
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        init_values=None,
        class_token=True,
        no_embed_class=False,
        pre_norm=False,
        fc_norm=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        weight_init='',
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        block_fn=Block,
        prompt_length=None,
        prompt_type=None,
        prefix_length=None,
        prefix_type=None,
        adapter_length=None,
        adapter_type=None,
        lora_length=None,
        lora_type=None,
    ):

        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.depth = depth
        self.img_size = img_size
        self.class_token = class_token

        self.prompt_length = prompt_length
        self.prompt_type = prompt_type

        self.prefix_length = prefix_length
        self.prefix_type = prefix_type

        self.adapter_length = adapter_length
        self.adapter_type = adapter_type

        self.lora_length = lora_length
        self.lora_type = lora_type

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(
            1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(
            torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if prompt_length is not None or prefix_length is not None \
           or adapter_length is not None or lora_length is not None:
            attn_layer = AttentionPETL
            block_fn = BlockPETL
            self.blocks = nn.Sequential(*[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    attn_layer=attn_layer,
                    layer_num=i,
                    prompt_length=prompt_length[i] if isinstance(
                        prompt_length, list) else prompt_length,
                    prompt_type=prompt_type,
                    prefix_length=prefix_length[i] if isinstance(
                        prefix_length, list) else prefix_length,
                    prefix_type=prefix_type,
                    adapter_length=adapter_length[i] if isinstance(
                        adapter_length, list) else adapter_length,
                    adapter_type=adapter_type,
                    lora_length=lora_length[i] if isinstance(
                        lora_length, list) else lora_length,
                    lora_type=lora_type) for i in range(depth)
            ])
        else:
            self.blocks = nn.Sequential(*[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer) for i in range(depth)
            ])

        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)
