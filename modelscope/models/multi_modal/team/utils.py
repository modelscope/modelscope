# Copyright 2021 The OpenAI Team Authors.
# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
#
# The implementation here is modified based on OpenAI CLIP,
# originally MIT License, Copyright (c) 2021 OpenAI,
# and publicly available at https://github.com/openai/CLIP/.

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import nn
from transformers import BertConfig, BertForMaskedLM

from modelscope.utils.compatible_with_transformers import \
    compatible_position_ids


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

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
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None,
                 use_gc=False):
        super().__init__()
        self.use_gc = use_gc
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        if self.use_gc:
            for each_block in self.resblocks:
                x = checkpoint.checkpoint(each_block, x)
            return x
        else:
            return self.resblocks(x)


class VisionTransformer(nn.Module):

    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 use_gc=False):
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
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        class_embedding = self.class_embedding.to(x.dtype) + \
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([class_embedding, x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIPVisionWrapper(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.vision_transformer = VisionTransformer(
            input_resolution=224,
            patch_size=14,
            width=1024,
            layers=24,
            heads=16,
            output_dim=768)

    def forward(self, x):
        x = self.vision_transformer.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        class_embedding = self.vision_transformer.class_embedding.to(x.dtype) + \
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([class_embedding, x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.vision_transformer.positional_embedding.to(x.dtype)
        x = self.vision_transformer.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.vision_transformer.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x_tensor = x.clone()
        x = self.vision_transformer.ln_post(x[:, 0, :])

        if self.vision_transformer.proj is not None:
            x = x @ self.vision_transformer.proj

        return x, x_tensor


class BertWrapper(nn.Module):

    def __init__(self, config_json, feat_dim, token_dim):
        super(BertWrapper, self).__init__()
        bert_config = BertConfig.from_json_file(config_json)
        self.bert = BertForMaskedLM(bert_config).bert

        self.projector = nn.Linear(768, feat_dim, bias=False)
        self.projector_token_embeds = nn.Linear(768, token_dim)

    def forward(self, input_ids, attention_mask):
        trans_features = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        output_states = self.bert(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token

        return self.projector(cls_tokens), self.projector_token_embeds(
            output_tokens)


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


class CrossLayer(nn.Module):

    def __init__(self, feat_dim, mlp_ratio):
        super(CrossLayer, self).__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.norm3 = nn.LayerNorm(feat_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=16)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=16)
        self.ffn = Mlp(
            in_features=feat_dim,
            hidden_features=feat_dim * mlp_ratio,
            drop=0.1)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, text_tensors, text_masks, image_tensors,
                retrieved_tensors):
        retrieved_tensors_res = self.norm1(retrieved_tensors)
        retrieved_tensors_res = self.self_attn(
            (text_tensors + retrieved_tensors_res).permute(1, 0, 2),
            (text_tensors + retrieved_tensors_res).permute(1, 0, 2),
            retrieved_tensors_res.permute(1, 0, 2),
            key_padding_mask=(text_masks == 0),
        )[0].permute(1, 0, 2)
        retrieved_tensors = retrieved_tensors + self.dropout1(
            retrieved_tensors_res)

        retrieved_tensors_res = self.norm2(retrieved_tensors)
        retrieved_tensors_res = self.cross_attn(
            (text_tensors + retrieved_tensors_res).permute(1, 0, 2),
            image_tensors.permute(1, 0, 2),
            image_tensors.permute(1, 0, 2))[0].permute(1, 0, 2)
        retrieved_tensors = retrieved_tensors + self.dropout2(
            retrieved_tensors_res)

        retrieved_tensors_res = self.norm3(retrieved_tensors)
        retrieved_tensors = retrieved_tensors + self.dropout3(
            self.ffn(retrieved_tensors_res))

        return retrieved_tensors


class TEAM(nn.Module):

    def __init__(self, text_model, image_model, pretrained):
        super(TEAM, self).__init__()
        self.text_model = text_model
        self.image_model = image_model

        self.cross_model = nn.ModuleList(
            [CrossLayer(feat_dim=1024, mlp_ratio=2)])

        self.image_tensor_fc = nn.Linear(1024, 768)
        self.text_tensor_fc = nn.Linear(1024, 768)

        params = torch.load(pretrained, 'cpu')
        compatible_position_ids(params,
                                'text_model.bert.embeddings.position_ids')
        self.load_state_dict(params, strict=True)

    def get_feature(self, text_data=None, text_mask=None, img_tensor=None):
        if text_data is not None:
            text_feature, text_tensors = self.text_model(text_data, text_mask)
            text_feature = F.normalize(text_feature, p=2.0, dim=1)
        else:
            text_feature, text_tensors = None, None

        if img_tensor is not None:
            image_feature, image_tensors = self.image_model(img_tensor)
            image_feature = F.normalize(image_feature, p=2.0, dim=1)
        else:
            image_feature, image_tensors = None, None

        return text_feature, text_tensors, image_feature, image_tensors

    def get_cross_score(self, text_tensors, text_mask, image_tensors):
        retrieved_tensors = torch.zeros_like(text_tensors)
        pair_score_list = []
        text_tensors_proj = self.text_tensor_fc(text_tensors)
        text_mask_float = text_mask.type(text_tensors_proj.dtype)
        for each_cross_model in self.cross_model:
            retrieved_tensors = each_cross_model(text_tensors, text_mask,
                                                 image_tensors,
                                                 retrieved_tensors)
            retrieved_tensors_proj = self.image_tensor_fc(retrieved_tensors)

            pair_score = torch.sum(
                F.normalize(retrieved_tensors_proj, p=2.0, dim=2)
                * F.normalize(text_tensors_proj, p=2.0, dim=2),
                dim=2)
            pair_score_reduced = torch.sum(
                pair_score * text_mask_float, dim=1) / torch.clamp(
                    torch.sum(text_mask_float, dim=1), min=1.0)
            pair_score_list.append(pair_score_reduced)
        return pair_score_list
