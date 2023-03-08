# Copyright 2021 The OpenAI Team Authors.
# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
#
# The implementation here is modified based on OpenAI CLIP,
# originally MIT License, Copyright (c) 2021 OpenAI,
# and publicly available at https://github.com/openai/CLIP/.
""" Generative Multimodal Model Architecture."""

import os

import json
import torch
import torch.nn.functional as F
from torch import nn

from modelscope.models.multi_modal.gemm import gemm_base, tokenizer


class ImageEncoder(nn.Module):
    """Image Feature Encoder
    ViT Style Transformer
    """

    def __init__(self, configs):
        super().__init__()
        (embed_dim, image_resolution, vision_layers, vision_width,
         vision_patch_size) = configs[:5]
        self.visual = gemm_base.VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim,
            use_gc=False)

    def forward(self, image, return_tokens=False):
        features = self.visual(image)
        tokens = features[:, 1:, :]
        embedding = features[:, 0, :]
        return (embedding, tokens) if return_tokens else embedding


class TextEncoder(nn.Module):
    """Text Feature Encoder
    BERT style transformer
    """

    def __init__(self, configs):
        super().__init__()
        (context_length, vocab_size, model_width, model_heads,
         model_layers) = configs[-5:]
        # text model
        self.transformer = gemm_base.Transformer(
            width=model_width,
            layers=model_layers,
            heads=model_heads,
            attn_mask=self.build_attention_mask(context_length),
        )
        # others
        self.token_embedding = nn.Embedding(vocab_size, model_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(context_length, model_width))
        self.ln_final = nn.LayerNorm(model_width)
        self.text_projection = nn.Parameter(
            torch.empty(model_width, configs[0]))

    def build_attention_mask(self, seq_length=None):
        mask = torch.ones(seq_length, seq_length) * -1e4
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, return_tokens=False):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        embedding = x[torch.arange(x.shape[0]),
                      text.argmax(dim=-1), ...] @ self.text_projection
        return (embedding, x) if return_tokens else embedding


class RLEGModel(nn.Module):
    """ Generative multi-modal model, trained with RLEG method.
    It takes image or text or both of them as input, and produce
    the corresponding features of inputs.
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
        self.tokenizer = tokenizer.SimpleTokenizer(bpe_path)
        # build model architecture
        self.image_encoder = ImageEncoder(config_args)
        self.text_encoder = TextEncoder(config_args)
        self.logit_scale = nn.Parameter(torch.ones([]))

    def tokenize(self, text_str):
        text_tensor = tokenizer.clip_tokenize(self.tokenizer, [text_str])[0]
        return text_tensor

    def encode_text(self, text):
        feature = self.text_encoder(text)
        feature = F.normalize(feature, p=2, dim=-1)
        return feature

    def encode_image(self, image):
        feature = self.image_encoder(image)
        feature = F.normalize(feature, p=2, dim=-1)
        return feature

    def parse_feat(self, feat):
        out = feat.cpu().numpy()
        return out

    @torch.no_grad()
    def forward(self, image=None, text=None):
        """ It takes image or text as input,
        and extracts the features as output.
        """
        img_feature, text_feature = None, None
        if image is not None:
            img_feature = self.parse_feat(self.encode_image(image))
        if text is not None:
            text_feature = self.parse_feat(self.encode_text(text))
        out = {
            'image_feature': img_feature,
            'text_feature': text_feature,
        }
        return out
