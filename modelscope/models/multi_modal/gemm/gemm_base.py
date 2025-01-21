# Copyright 2021 The OpenAI Team Authors.
# Copyright 2022 Phil Wang.
# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
#
# The implementation here is modified based on OpenAI CLIP,
# originally MIT License, Copyright (c) 2021 OpenAI,
# and publicly available at https://github.com/openai/CLIP/.
# The implementation here is modified based on Coca-pytorch,
# originally MIT License, Copyright (c) 2022 Phil Wang,
# and publicly available at https://github.com/lucidrains/CoCa-pytorch/,
""" Generative Multimodal Model Architecture."""

import os
from collections import OrderedDict
from typing import Tuple, Union

import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm

from modelscope.models.multi_modal.gemm.tokenizer import (SimpleTokenizer,
                                                          clip_tokenize)


class Bottleneck(nn.Module):
    """ ResNet style bottleneck module
    From https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                OrderedDict([('-1', nn.AvgPool2d(stride)),
                             ('0',
                              nn.Conv2d(
                                  inplanes,
                                  planes * self.expansion,
                                  1,
                                  stride=1,
                                  bias=False)),
                             ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class QuickGELU(nn.Module):
    """ A quick version of GELU module
    From https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """ Multihead attention block with residual link
    Adapted from https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype,
            device=x.device) if self.attn_mask is not None else None
        attn_mask = self.attn_mask
        if attn_mask is not None and attn_mask.shape[0] > x.shape[0]:
            attn_mask = self.attn_mask[:x.shape[0], :x.shape[0]]
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """ Transformer encoder module
    Adapted from https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None,
                 use_gc: bool = False):
        super().__init__()
        self.use_gc = use_gc
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class AttentionPool2d(nn.Module):
    """ Pool layer with attention module
    Adapted from https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1],
                      x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)
        return x.permute(1, 0, 2).contiguous()


class CrossAttention(nn.Module):
    """ Cross attention module with query and context as input
    Adapted from https://github.com/lucidrains/CoCa-pytorch/blob/main/coca_pytorch/coca_pytorch.py
    """

    def __init__(self,
                 dim,
                 *,
                 context_dim=None,
                 dim_head=64,
                 heads=8,
                 parallel_ff=False,
                 ff_mult=4,
                 norm_context=False):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head
        context_dim = dim if context_dim is None else context_dim
        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(
            context_dim) if norm_context else nn.Identity()
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        ff_inner_dim = ff_mult * dim
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False), SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)) if parallel_ff else None

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        x = self.norm(x)
        context = self.context_norm(context)

        q = self.to_q(x)
        q = q.view(q.shape[0], q.shape[1], self.heads,
                   -1).permute(0, 2, 1, 3).contiguous()
        q = q * self.scale
        k, v = self.to_kv(context).chunk(2, dim=-1)
        sim = torch.einsum('b h i d, b j d -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b j d -> b h i d', attn, v)
        out = out.permute(0, 2, 1,
                          3).contiguous().reshape(out.shape[0], out.shape[2],
                                                  -1)
        out = self.to_out(out)
        if self.ff is not None:
            out = out + self.ff(x)
        return out


class ModifiedResNet(nn.Module):
    """ Modified ResNet backbone
    From https://github.com/openai/CLIP/blob/main/clip/model.py
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self,
                 layers,
                 output_dim,
                 heads,
                 input_resolution=224,
                 width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(
            width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim,
                                        heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2),
                             (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class VisualTransformer(nn.Module):
    """ ViT transformer backbone
    From https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(self, input_resolution: int, patch_size: int, width: int,
                 layers: int, heads: int, output_dim: int, use_gc: bool):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False)
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, use_gc=use_gc)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        z = torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([self.class_embedding.to(x.dtype) + z, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
        return x


class GEVL(nn.Module):
    """ Generative vision-language model
    Support learning from both generative and contrastive loss.
    Given image and text input, it could output the features of
    image and text respectively. Furthermore, caption could also
    be produced when image input is available.
    """

    def __init__(self, embed_dim: int, image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int],
                                      int], vision_width: int,
                 vision_patch_size: int, context_length: int, vocab_size: int,
                 transformer_width: int, transformer_heads: int,
                 transformer_layers: int, use_gc: bool, tokenizer):
        nn.Module.__init__(self)
        self.context_length = context_length
        self.vis_token_size = context_length
        self.tokenizer = tokenizer

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width)
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                use_gc=use_gc)

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            use_gc=use_gc)

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.vis_token_projection = nn.Parameter(
            torch.empty(embed_dim, transformer_width))
        nn.init.normal_(
            self.vis_token_projection, std=self.transformer.width**-0.5)
        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.decoder = Transformer(
            width=transformer_width,
            layers=4,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(
                self.vis_token_size + self.context_length,
                self.vis_token_size),
            use_gc=use_gc)
        self.to_logits = nn.Sequential(
            LayerNorm(transformer_width),
            nn.Linear(transformer_width, transformer_width),
            nn.Linear(transformer_width, vocab_size, bias=False))
        self.gen_logit_scale = nn.Parameter(
            torch.ones([]) * np.log(np.log(vocab_size)))
        self.bias = nn.Parameter(torch.ones(vocab_size))
        self.to_logits[-1].weight = self.token_embedding.weight
        self.to_logits[-1].bias = self.bias
        self.img_queries = nn.Parameter(
            torch.randn(self.vis_token_size, transformer_width))
        self.img_attn_pool = CrossAttention(
            dim=transformer_width, norm_context=True)
        self.img_attn_pool_norm = LayerNorm(transformer_width)

    def build_attention_mask(self, seq_length=None, prefix_length=0):
        seq_length = self.context_length if seq_length is None else seq_length
        mask = torch.empty(seq_length, seq_length)
        mask.fill_(torch.tensor(torch.finfo(torch.float16).min))
        mask.triu_(1)
        if prefix_length > 0:
            mask[:prefix_length, :prefix_length] = 0
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_tokens=False):
        image_outputs = self.visual(image)
        image_features = image_outputs[:, 0, :]
        image_features = image_features / image_features.norm(
            dim=-1, p=2, keepdim=True)
        if return_tokens:
            image_tokens = image_outputs[:, 1:, :] @ self.vis_token_projection
            return image_features, image_tokens
        else:
            return image_features

    def encode_text(self, text, return_tokens=False):
        x = self.token_embedding(text)
        x = x + self.positional_embedding[:x.shape[1], :]
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        text_features = x[torch.arange(x.shape[0]),
                          text.argmax(dim=-1), ...] @ self.text_projection
        text_features = text_features / text_features.norm(
            dim=-1, p=2, keepdim=True)
        if return_tokens:
            text_tokens = x
            return text_features, text_tokens
        else:
            return text_features

    def image_to_text(self, image):
        image_features, image_tokens = self.encode_image(
            image, return_tokens=True)
        img_queries = self.img_queries.expand(image_tokens.shape[0], -1, -1)
        img_token_features = self.img_attn_pool(img_queries, image_tokens)
        img_token_features = self.img_attn_pool_norm(img_token_features)
        sot_token = self.tokenizer.encoder['<|startoftext|>']
        eot_token = self.tokenizer.encoder['<|endoftext|>']
        text_input = image.new_ones(
            image.shape[0], 1, dtype=torch.long) * sot_token
        input_tokens = img_token_features
        pred_tokens = []
        for text_idx in range(self.context_length):
            text_features, text_tokens = self.encode_text(
                text_input, return_tokens=True)
            input_tokens = torch.cat([img_token_features, text_tokens], axis=1)
            out_embs = self.decoder(input_tokens.permute(1, 0, 2).contiguous())
            gen_logits = self.to_logits(out_embs[-1:, ...])
            probs = F.softmax(self.gen_logit_scale.exp() * gen_logits, dim=-1)
            pred = torch.argmax(
                probs * (2.0 + torch.rand_like(probs)), axis=-1)
            if int(pred) >= eot_token or int(pred) <= 0:
                break
            pred_tokens.append(pred)
            text_input = torch.cat(
                [text_input, pred.permute(1, 0).contiguous()], axis=1)
        pred_text_tokens = torch.cat(pred_tokens, axis=0).permute(1, 0)
        text_list = []
        for out_tokens in pred_text_tokens:
            tokens = []
            for x in out_tokens:
                tokens.append(int(x))
            out_text = self.tokenizer.decode(tokens)
            out_text = out_text.strip()
            text_list.append(out_text)
        return image_features, text_list[0]


class GEMMModel(nn.Module):
    """ Generative multi-modal model, wrapper of GEVL module.
    It takes image or text or both of them as input, and output
    features of input or caption when image input is available.
    """

    def __init__(self, model_dir):
        super().__init__()
        with open(
                '{}/encoder_config.json'.format(model_dir), 'r',
                encoding='utf-8') as f:
            model_config = json.loads(f.read())
        model_name = list(model_config.keys())[0]
        config_args = model_config[model_name]
        bpe_path = os.path.join(model_dir, 'bpe_vocab_16e6.txt.gz')
        self.tokenizer = SimpleTokenizer(bpe_path)
        self.model = GEVL(*config_args, self.tokenizer)

    def tokenize(self, text_str):
        text_tensor = clip_tokenize(self.tokenizer, [text_str])[0]
        return text_tensor

    def parse_feat(self, feat):
        out = feat.cpu().numpy()
        return out

    @torch.no_grad()
    def forward(self, image=None, text=None, captioning=True):
        img_feature, text_feature, caption = None, None, None
        if captioning and image is not None:
            img_feature, caption = self.model.image_to_text(image)
            img_feature = self.parse_feat(img_feature)
        elif image is not None:
            img_feature = self.parse_feat(self.model.encode_image(image))
        if text is not None:
            text_feature = self.parse_feat(self.model.encode_text(text))
        out = {
            'image_feature': img_feature,
            'text_feature': text_feature,
            'caption': caption,
        }
        return out
