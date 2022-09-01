# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch
from PIL import Image
from torchvision import transforms

from modelscope.preprocessors.image import load_image
from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor


class OfaVisualEntailmentPreprocessor(OfaBasePreprocessor):

    def __init__(self,
                 cfg,
                 model_dir,
                 mode=ModeKeys.INFERENCE,
                 *args,
                 **kwargs):
        """preprocess the data

        Args:
            cfg(modelscope.utils.config.ConfigDict) : model config
            model_dir (str): model path,
            mode: preprocessor mode (model mode)
        """
        super(OfaVisualEntailmentPreprocessor,
              self).__init__(cfg, model_dir, mode, *args, **kwargs)
        # Initialize transform
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert('RGB'),
            transforms.Resize(
                (self.patch_image_size, self.patch_image_size),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = data['image'] if isinstance(
            data['image'], Image.Image) else load_image(data['image'])
        patch_image = self.patch_resize_transform(image)
        if 'text2' not in data:
            hypothesis = self.pre_caption(data['text'], self.max_src_length)
            prompt = self.cfg.model.get('prompt',
                                        ' does the image describe " {} "?')
            text = prompt.format(hypothesis)
        else:
            assert 'text' in data, f'text must be in the input {data.keys()}'
            caption = self.pre_caption(data['text2'], self.max_src_length)
            hypothesis = self.pre_caption(data['text'], self.max_src_length)
            prompt = self.cfg.model.get(
                'prompt', ' can image and text1 " {} " imply text2 " {} "?')
            text = prompt.format(caption, hypothesis)
        inputs = self.get_inputs(text)
        if self.prompt_type == 'none':
            decoder_prompt = self.bos_item
        elif self.prompt_type == 'src':
            decoder_prompt = inputs
        elif self.prompt_type == 'prev_output':
            decoder_prompt = inputs[:-1]
        else:
            raise NotImplementedError
        sample = {
            'source': inputs,
            'patch_image': patch_image,
            'patch_mask': torch.tensor([True]),
            'decoder_prompt': decoder_prompt,
        }
        return sample
