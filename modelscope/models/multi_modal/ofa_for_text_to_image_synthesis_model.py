# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict

import json
import numpy as np
import torch
import torch.cuda
from PIL import Image
from pkg_resources import packaging
from taming.models.vqgan import GumbelVQ, VQModel
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)

from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.mmr.models.module_clip import CLIP
from modelscope.models.multi_modal.mmr.models.tokenization_clip import \
    SimpleTokenizer as ClipTokenizer
from modelscope.models.multi_modal.ofa import OFAModel, OFATokenizer
from modelscope.models.multi_modal.ofa.generate import sequence_generator as sg
from modelscope.models.multi_modal.ofa.generate.search import Sampling
from modelscope.models.multi_modal.ofa.generate.utils import move_to_device
from modelscope.utils.constant import Tasks

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

__all__ = ['OfaForTextToImageSynthesis']


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == 'RGB':
        x = x.convert('RGB')
    return x


def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config['model']['params'])
    else:
        model = VQModel(**config['model']['params'])
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location='cpu')['state_dict']
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


def build_clip_model(model_path):
    state_dict = torch.load(model_path, map_location='cpu').state_dict()
    vit = 'visual.proj' in state_dict
    if vit:
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len([
            k for k in state_dict.keys()
            if k.startswith('visual.') and k.endswith('.attn.in_proj_weight')
        ])
        vision_patch_size = state_dict['visual.conv1.weight'].shape[-1]
        grid_size = round(
            (state_dict['visual.positional_embedding'].shape[0] - 1)**0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split('.')[2] for k in state_dict
                    if k.startswith(f'visual.layer{b}')))
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict['visual.layer1.0.conv1.weight'].shape[0]
        output_width = round(
            (state_dict['visual.attnpool.positional_embedding'].shape[0]
             - 1)**0.5)
        vision_patch_size = None
        assert output_width**2 + 1 == state_dict[
            'visual.attnpool.positional_embedding'].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict['text_projection'].shape[1]
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split('.')[2] for k in state_dict
            if k.startswith('transformer.resblocks')))

    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width,
                 vision_patch_size, context_length, vocab_size,
                 transformer_width, transformer_heads, transformer_layers)

    for key in ['input_resolution', 'context_length', 'vocab_size']:
        if key in state_dict:
            del state_dict[key]

    model.load_state_dict(state_dict)
    return model.eval()


def _convert_image_to_rgb(image):
    return image.convert('RGB')


def build_clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


@MODELS.register_module(Tasks.text_to_image_synthesis, module_name=Models.ofa)
class OfaForTextToImageSynthesis(Model):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir=model_dir, *args, **kwargs)
        # Initialize ofa
        model = OFAModel.from_pretrained(model_dir)
        self.model = model.module if hasattr(model, 'module') else model
        self.tokenizer = OFATokenizer.from_pretrained(model_dir)
        self.tokenizer.add_tokens(['<code_{}>'.format(i) for i in range(8192)])
        self.tokenizer.add_tokens(['<bin_{}>'.format(i) for i in range(1000)])
        self._device = torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')
        self.model.to(self._device)

        # Initialize vqgan
        vqgan_config = json.load(
            open(os.path.join(model_dir, 'vqgan_config.json')))
        self.vqgan_model = load_vqgan(
            vqgan_config,
            ckpt_path=os.path.join(model_dir, 'vqgan_model.ckpt'),
            is_gumbel=True).to(self._device)

        # Initialize OpenAI clip

        self.clip_tokenizer = ClipTokenizer(model_dir)
        self.clip_model = build_clip_model(
            os.path.join(model_dir, 'ViT-B-16.pt'))
        self.clip_preprocess = build_clip_transform(
            self.clip_model.visual.input_resolution)

        self.clip_model.to(self._device)
        self.clip_model.eval()

        # Initialize generator
        sampling = Sampling(self.tokenizer, sampling_topp=0.9)
        sg_args = {
            'tokenizer': self.tokenizer,
            'beam_size': 2,
            'max_len_b': 1024,
            'min_len': 1024,
            'search_strategy': sampling,
            'gen_code': True,
            'constraint_range': '50265,58457'
        }
        self.generator = sg.SequenceGenerator(**sg_args)

    def clip_tokenize(self, texts, context_length=77, truncate=False):

        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.clip_tokenizer.encoder['<|startoftext|>']
        eot_token = self.clip_tokenizer.encoder['<|endoftext|>']
        all_tokens = [[sot_token] + self.clip_tokenizer.encode(text)
                      + [eot_token] for text in texts]
        if packaging.version.parse(
                torch.__version__) < packaging.version.parse('1.8.0'):
            result = torch.zeros(
                len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(
                len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(
                        f'Input {texts[i]} is too long for context length {context_length}'
                    )
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def forward(self, input: Dict[str, Any]):

        text = input['samples'][0]['text']
        input = move_to_device(input, self._device)
        clip_text_input = self.clip_tokenize([text]).to(self._device)

        gen_output = self.generator.generate([self.model], input)
        gen_tokens = torch.stack(
            [item['tokens'][:-1] for item in gen_output[0]], dim=0)
        codes = gen_tokens.view(-1, 32, 32) - 50265

        quant_b = self.vqgan_model.quantize.get_codebook_entry(
            codes.view(-1),
            list(codes.size()) + [self.vqgan_model.quantize.embedding_dim])
        imgs = self.vqgan_model.decode(quant_b)

        sample_num = imgs.size()[0]
        pil_imgs = [custom_to_pil(imgs[i]) for i in range(sample_num)]

        clip_image_input = torch.stack(
            [self.clip_preprocess(img) for img in pil_imgs],
            dim=0).to(self._device)

        with torch.no_grad():
            hyp_image_features = self.clip_model.encode_image(clip_image_input)
            hyp_image_features /= hyp_image_features.norm(dim=-1, keepdim=True)
            text_features = self.clip_model.encode_text(clip_text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        ti_similarity = hyp_image_features @ text_features.T

        sorted_score, ti_indices = torch.sort(
            ti_similarity.view(-1), descending=True)

        pil_imgs_orderby_ti = [pil_imgs[index] for index in ti_indices]
        return pil_imgs_orderby_ti[0]
