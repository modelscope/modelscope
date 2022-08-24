import os
from typing import Any, Dict

import json
import numpy as np
import torch
import torch.cuda
from PIL import Image
from taming.models.vqgan import GumbelVQ, VQModel

from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.ofa import OFAModel, OFATokenizer
from modelscope.models.multi_modal.ofa.generate import sequence_generator as sg
from modelscope.models.multi_modal.ofa.generate.search import Sampling
from modelscope.models.multi_modal.ofa.generate.utils import move_to_device
from modelscope.utils.constant import Tasks

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
        # Initialize generator
        sampling = Sampling(self.tokenizer, sampling_topp=0.9)
        sg_args = {
            'tokenizer': self.tokenizer,
            'beam_size': 1,
            'max_len_b': 1024,
            'min_len': 1024,
            'search_strategy': sampling,
            'gen_code': True,
            'constraint_range': '50265,58457'
        }
        self.generator = sg.SequenceGenerator(**sg_args)

    def forward(self, input: Dict[str, Any]):
        input = move_to_device(input, self._device)
        gen_output = self.generator.generate([self.model], input)
        gen_tokens = gen_output[0][0]['tokens'][:-1]
        codes = gen_tokens.view(1, 32, 32) - 50265
        quant_b = self.vqgan_model.quantize.get_codebook_entry(
            codes.view(-1),
            list(codes.size()) + [self.vqgan_model.quantize.embedding_dim])
        dec = self.vqgan_model.decode(quant_b)[0]
        return custom_to_pil(dec)
