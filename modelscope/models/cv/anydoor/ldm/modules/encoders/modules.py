import os

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                          T5Tokenizer)

from ....dinov2 import hubconf
from ....ldm.util import count_params


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(
            x.to(torch.float32), self.normalized_shape, self.weight, self.bias,
            self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                         self.eps)
        return x.to(orig_type)


class AbstractEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):

    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1 - mask) * torch.ones_like(c) * (
                self.n_classes - 1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device='cuda'):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs, ), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""

    def __init__(self,
                 version='google/t5-v1_1-large',
                 device='cuda',
                 max_length=77,
                 freeze=True
                 ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length  # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding='max_length',
            return_tensors='pt')
        tokens = batch_encoding['input_ids'].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = ['last', 'pooled', 'hidden']

    def __init__(self,
                 version='openai/clip-vit-large-patch14',
                 device='cuda',
                 max_length=77,
                 freeze=True,
                 layer='last',
                 layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == 'hidden':
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding='max_length',
            return_tensors='pt')
        tokens = batch_encoding['input_ids'].to(self.device)
        outputs = self.transformer(
            input_ids=tokens, output_hidden_states=self.layer == 'hidden')
        if self.layer == 'last':
            z = outputs.last_hidden_state
        elif self.layer == 'pooled':
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        'last',
        'penultimate'
    ]

    def __init__(self,
                 arch='ViT-H-14',
                 version='laion2b_s32b_b79k',
                 device='cuda',
                 max_length=77,
                 freeze=True,
                 layer='last'):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

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

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
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


class FrozenCLIPT5Encoder(AbstractEncoder):

    def __init__(self,
                 clip_version='openai/clip-vit-large-patch14',
                 t5_version='google/t5-v1_1-xl',
                 device='cuda',
                 clip_max_length=77,
                 t5_max_length=77):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(
            clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(
            t5_version, device, max_length=t5_max_length)
        print(
            f'{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder)*1.e-6:.2f} M parameters, '
            f'{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder)*1.e-6:.2f} M params.'
        )

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]


class FrozenOpenCLIPImageEncoder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for image
    """

    def __init__(self,
                 arch='ViT-H-14',
                 version='laion2b_s32b_b79k',
                 device='cuda',
                 freeze=True):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, device=torch.device('cpu'), pretrained=version)
        del model.transformer
        self.model = model
        self.model.visual.output_tokens = True
        self.device = device
        if freeze:
            self.freeze()
        self.image_mean = torch.tensor(
            [0.48145466, 0.4578275,
             0.40821073]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.image_std = torch.tensor(
            [0.26862954, 0.26130258,
             0.275777]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.projector_token = nn.Linear(1280, 1024)
        self.projector_embed = nn.Linear(1024, 1024)

    def freeze(self):
        self.model.visual.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        if isinstance(image, list):
            image = torch.cat(image, 0)
        image = (image.to(self.device) - self.image_mean.to(
            self.device)) / self.image_std.to(self.device)
        image_features, tokens = self.model.visual(image)
        image_features = image_features.unsqueeze(1)
        image_features = self.projector_embed(image_features)
        tokens = self.projector_token(tokens)
        hint = torch.cat([image_features, tokens], 1)
        return hint

    def encode(self, image):
        return self(image)


class FrozenDinoV2Encoder(AbstractEncoder):
    """
    Uses the DINOv2 encoder for image
    """

    def __init__(self, model_path, device='cuda', freeze=True):
        DINOv2_weight_path = model_path

        super().__init__()
        dinov2 = hubconf.dinov2_vitg14()
        state_dict = torch.load(DINOv2_weight_path)
        dinov2.load_state_dict(state_dict, strict=False)
        self.model = dinov2.to(device)
        self.device = device
        if freeze:
            self.freeze()
        self.image_mean = torch.tensor(
            [0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.image_std = torch.tensor(
            [0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.projector = nn.Linear(1536, 1024)

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        if isinstance(image, list):
            image = torch.cat(image, 0)

        image = (image.to(self.device) - self.image_mean.to(
            self.device)) / self.image_std.to(self.device)
        features = self.model.forward_features(image)
        tokens = features['x_norm_patchtokens']
        image_features = features['x_norm_clstoken']
        image_features = image_features.unsqueeze(1)
        hint = torch.cat([image_features, tokens], 1)  # 8,257,1024
        hint = self.projector(hint)
        return hint

    def encode(self, image):
        return self(image)
