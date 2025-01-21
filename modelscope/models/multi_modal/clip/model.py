# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import OrderedDict
from typing import Any, Dict, Tuple, Union

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.clip.bert_tokenizer import FullTokenizer
from modelscope.models.multi_modal.clip.configuration_bert import BertConfig
from modelscope.models.multi_modal.clip.modeling_bert import BertModel
from modelscope.utils.constant import ModeKeys, ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['CLIPForMultiModalEmbedding']


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
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
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
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


class AttentionPool2d(nn.Module):

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
                      x.shape[2] * x.shape[3]).permute(2, 0,
                                                       1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
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

        return x[0]


class ModifiedResNet(nn.Module):
    """
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

        # the 3-layer stem
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

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
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

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


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
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):

    def __init__(self, input_resolution: int, patch_size: int, width: int,
                 layers: int, heads: int, output_dim: int):
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

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [  # noqa
                self.class_embedding.to(x.dtype) + torch.zeros(  # noqa
                    x.shape[0],
                    1,
                    x.shape[-1],
                    dtype=x.dtype,
                    device=x.device),
                x  # noqa
            ],
            dim=1)  # noqa shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        vocab_size: int,
        text_attention_probs_dropout_prob: float,
        text_hidden_act: str,
        text_hidden_dropout_prob: float,
        text_hidden_size: int,
        text_initializer_range: float,
        text_intermediate_size: int,
        text_max_position_embeddings: int,
        text_num_attention_heads: int,
        text_num_hidden_layers: int,
        text_type_vocab_size: int,
        tokenizer: FullTokenizer,
        # vision_head_width, added this param for ViT-H
        vision_head_width: int = 64,
    ):
        super().__init__()

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // vision_head_width
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width)
        else:
            vision_heads = vision_width // vision_head_width
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim)

        self.bert_config = BertConfig(
            vocab_size_or_config_json_file=vocab_size,
            hidden_size=text_hidden_size,
            num_hidden_layers=text_num_hidden_layers,
            num_attention_heads=text_num_attention_heads,
            intermediate_size=text_intermediate_size,
            hidden_act=text_hidden_act,
            hidden_dropout_prob=text_hidden_dropout_prob,
            attention_probs_dropout_prob=text_attention_probs_dropout_prob,
            max_position_embeddings=text_max_position_embeddings,
            type_vocab_size=text_type_vocab_size,
            initializer_range=text_initializer_range,
            layer_norm_eps=1e-12,
        )
        self.bert = BertModel(self.bert_config)

        self.text_projection = nn.Parameter(
            torch.empty(text_hidden_size, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.tokenizer = tokenizer

        self.initialize_parameters()

    def initialize_parameters(self):
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features**-0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                    self.visual.layer1, self.visual.layer2, self.visual.layer3,
                    self.visual.layer4
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith('bn3.weight'):
                        nn.init.zeros_(param)

        if self.text_projection is not None:
            nn.init.normal_(
                self.text_projection, std=self.bert_config.hidden_size**-0.5)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        pad_index = self.tokenizer.vocab['[PAD]']
        attn_mask = text.ne(pad_index).type(self.dtype)
        x = self.bert(
            text, attention_mask=attn_mask)[0].type(
                self.dtype)  # [batch_size, seq_length, hidden_size]
        return x[:, 0, :] @ self.text_projection

    def forward(self, image, text):
        assert image is not None or text is not None, 'text and image cannot both be None!'

        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True)

        return image_features, text_features, self.logit_scale.exp()

    def get_similarity(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(
            dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            module.weight.data = module.weight.data.half()
            if module.bias is not None:
                module.bias.data = module.bias.data.half()

        if isinstance(module, nn.MultiheadAttention):
            for attr in [
                    *[f'{s}_proj_weight' for s in ['in', 'q', 'k', 'v']],
                    'in_proj_bias', 'bias_k', 'bias_v'
            ]:
                tensor = getattr(module, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        if isinstance(module, BertModel):
            module.to(torch.half)

        for name in ['text_projection', 'proj']:
            if hasattr(module, name):
                attr = getattr(module, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


@MODELS.register_module(Tasks.multi_modal_embedding, module_name=Models.clip)
class CLIPForMultiModalEmbedding(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir=model_dir, *args, **kwargs)

        # Initialize the model.
        vision_model_config_file = '{}/vision_model_config.json'.format(
            model_dir)
        logger.info(
            f'Loading vision model config from {vision_model_config_file}')
        assert os.path.exists(vision_model_config_file)

        text_model_config_file = '{}/text_model_config.json'.format(model_dir)
        logger.info(f'Loading text model config from {text_model_config_file}')
        assert os.path.exists(text_model_config_file)

        with open(
                vision_model_config_file, 'r',
                encoding='utf-8') as fv,\
                open(text_model_config_file, 'r', encoding='utf-8') as ft:
            self.model_info = json.load(fv)
            for k, v in json.load(ft).items():
                self.model_info[k] = v

        vocab_file = f'{model_dir}/{ModelFile.VOCAB_FILE}'
        self.tokenizer = FullTokenizer(vocab_file=vocab_file)

        # initialize the model
        self.clip_model = CLIP(**self.model_info, tokenizer=self.tokenizer)
        convert_weights(self.clip_model)

        # restore the pretrained weight
        checkpoint = torch.load(
            f'{model_dir}/{ModelFile.TORCH_MODEL_BIN_FILE}', 'cpu')
        sd = checkpoint[
            'state_dict'] if 'state_dict' in checkpoint else checkpoint
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        # support the finetuned model
        if next(iter(sd.items()))[0].startswith('clip_model'):
            sd = {k[len('clip_model.'):]: v for k, v in sd.items()}
        self.clip_model.load_state_dict(sd)
        self.clip_model.eval()

        # place the model
        self.device = 'cuda:{}'.format(int(os.environ.get(
            'LOCAL_RANK', 0))) if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            self.clip_model.to(self.device)
            logger.info('Use GPU {} for finetuning & inference'.format(
                int(os.environ.get('LOCAL_RANK', 0))))
        else:
            self.clip_model.float()
            logger.info('Use CPU for finetuning & inference')

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        from modelscope.outputs import OutputKeys
        output = {
            OutputKeys.IMG_EMBEDDING: None,
            OutputKeys.TEXT_EMBEDDING: None
        }
        mode = input.get('mode', ModeKeys.INFERENCE)

        # encode the image
        if 'img' in input and isinstance(input['img'], torch.Tensor):
            image_tensor = input['img'].to(self.device)
            if image_tensor.dim() == 5 and image_tensor.shape[1] == 1:
                image_tensor = image_tensor.squeeze(1)

            with torch.autograd.set_grad_enabled(mode == ModeKeys.TRAIN):
                image_features = self.clip_model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True)  # l2-normalize

            output[OutputKeys.IMG_EMBEDDING] = image_features

        if 'text' in input and isinstance(input['text'], torch.Tensor):
            text_tensor = input['text'].to(self.device)
            if text_tensor.dim() == 3 and text_tensor.shape[1] == 1:
                text_tensor = text_tensor.squeeze(1)

            with torch.autograd.set_grad_enabled(mode == ModeKeys.TRAIN):
                text_features = self.clip_model.encode_text(text_tensor)
                text_features = text_features / text_features.norm(
                    dim=-1, keepdim=True)  # l2-normalize
            output[OutputKeys.TEXT_EMBEDDING] = text_features

        if mode == ModeKeys.TRAIN:
            output['logit_scale'] = (self.clip_model.logit_scale
                                     * 1.0).exp().mean()

        return output

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    @property
    def temperature(self):
        return 1.0 / self.clip_model.logit_scale.exp()
